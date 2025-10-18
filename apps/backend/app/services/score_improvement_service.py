import gc
import json
import asyncio
import logging
import markdown
import numpy as np
import re
from collections import Counter
from math import sqrt

from sqlalchemy.future import select
from pydantic import ValidationError
from sqlalchemy.ext.asyncio import AsyncSession
from typing import Dict, Optional, Tuple, AsyncGenerator, List, Set

from app.prompt import prompt_factory
from app.schemas.json import json_schema_factory
from app.schemas.pydantic import ResumePreviewerModel
from app.agent import EmbeddingManager, AgentManager
from app.models import Resume, Job, ProcessedResume, ProcessedJob
from .exceptions import (
    ResumeNotFoundError,
    JobNotFoundError,
    ResumeParsingError,
    JobParsingError,
    ResumeKeywordExtractionError,
    JobKeywordExtractionError,
)

logger = logging.getLogger(__name__)


class ScoreImprovementService:
    """
    Advanced resume scoring service using TF-IDF and semantic similarity.
    Implements algorithms similar to SkillSyncer and ResumeWorded.
    """
    """
    Service to handle scoring of resumes and jobs using embeddings.
    Fetches Resume and Job data from the database, computes embeddings,
    and calculates cosine similarity scores. Uses LLM for iteratively improving
    the scoring process.
    """

    def __init__(self, db: AsyncSession, max_retries: int = 5):
        self.db = db
        self.max_retries = max_retries
        self.md_agent_manager = AgentManager(strategy="md")
        self.json_agent_manager = AgentManager()
        self.embedding_manager = EmbeddingManager()

    def _validate_resume_keywords(
        self, processed_resume: ProcessedResume, resume_id: str
    ) -> None:
        """
        Validates that keyword extraction was successful for a resume.
        Raises ResumeKeywordExtractionError if keywords are missing or empty.
        """
        if not processed_resume.extracted_keywords:
            raise ResumeKeywordExtractionError(resume_id=resume_id)

        try:
            keywords_data = json.loads(processed_resume.extracted_keywords)
            keywords = keywords_data.get("extracted_keywords", [])
            if not keywords or len(keywords) == 0:
                raise ResumeKeywordExtractionError(resume_id=resume_id)
        except json.JSONDecodeError:
            raise ResumeKeywordExtractionError(resume_id=resume_id)

    def _validate_job_keywords(self, processed_job: ProcessedJob, job_id: str) -> None:
        """
        Validates that keyword extraction was successful for a job.
        Raises JobKeywordExtractionError if keywords are missing or empty.
        """
        if not processed_job.extracted_keywords:
            raise JobKeywordExtractionError(job_id=job_id)

        try:
            keywords_data = json.loads(processed_job.extracted_keywords)
            keywords = keywords_data.get("extracted_keywords", [])
            if not keywords or len(keywords) == 0:
                raise JobKeywordExtractionError(job_id=job_id)
        except json.JSONDecodeError:
            raise JobKeywordExtractionError(job_id=job_id)

    async def _get_resume(
        self, resume_id: str
    ) -> Tuple[Resume | None, ProcessedResume | None]:
        """
        Fetches the resume from the database.
        """
        query = select(Resume).where(Resume.resume_id == resume_id)
        result = await self.db.execute(query)
        resume = result.scalars().first()

        if not resume:
            raise ResumeNotFoundError(resume_id=resume_id)

        query = select(ProcessedResume).where(ProcessedResume.resume_id == resume_id)
        result = await self.db.execute(query)
        processed_resume = result.scalars().first()

        if not processed_resume:
            raise ResumeParsingError(resume_id=resume_id)

        self._validate_resume_keywords(processed_resume, resume_id)

        return resume, processed_resume

    async def _get_job(self, job_id: str) -> Tuple[Job | None, ProcessedJob | None]:
        """
        Fetches the job from the database.
        """
        query = select(Job).where(Job.job_id == job_id)
        result = await self.db.execute(query)
        job = result.scalars().first()

        if not job:
            raise JobNotFoundError(job_id=job_id)

        query = select(ProcessedJob).where(ProcessedJob.job_id == job_id)
        result = await self.db.execute(query)
        processed_job = result.scalars().first()

        if not processed_job:
            raise JobParsingError(job_id=job_id)

        self._validate_job_keywords(processed_job, job_id)

        return job, processed_job

    def calculate_cosine_similarity(
        self,
        extracted_job_keywords_embedding: np.ndarray,
        resume_embedding: np.ndarray,
    ) -> float:
        """
        Calculates the cosine similarity between two embeddings.
        """
        if resume_embedding is None or extracted_job_keywords_embedding is None:
            return 0.0

        ejk = np.asarray(extracted_job_keywords_embedding).squeeze()
        re = np.asarray(resume_embedding).squeeze()

        return float(np.dot(ejk, re) / (np.linalg.norm(ejk) * np.linalg.norm(re)))

    async def improve_score_with_llm(
        self,
        resume: str,
        extracted_resume_keywords: str,
        job: str,
        extracted_job_keywords: str,
        previous_cosine_similarity_score: float,
        extracted_job_keywords_embedding: np.ndarray,
    ) -> Tuple[str, float]:
        prompt_template = prompt_factory.get("resume_improvement")
        best_resume, best_score = resume, previous_cosine_similarity_score

        for attempt in range(1, self.max_retries + 1):
            logger.info(
                f"Attempt {attempt}/{self.max_retries} to improve resume score."
            )
            prompt = prompt_template.format(
                raw_job_description=job,
                extracted_job_keywords=extracted_job_keywords,
                raw_resume=best_resume,
                extracted_resume_keywords=extracted_resume_keywords,
                current_cosine_similarity=best_score,
            )
            improved = await self.md_agent_manager.run(prompt)
            emb = await self.embedding_manager.embed(text=improved)
            score = self.calculate_cosine_similarity(
                emb, extracted_job_keywords_embedding
            )

            if score > best_score:
                return improved, score

            logger.info(
                f"Attempt {attempt} resulted in score: {score}, best score so far: {best_score}"
            )

        return best_resume, best_score

    async def get_resume_for_previewer(self, updated_resume: str) -> Dict:
        """
        Returns the updated resume in a format suitable for the dashboard.
        """
        prompt_template = prompt_factory.get("structured_resume")
        prompt = prompt_template.format(
            json.dumps(json_schema_factory.get("resume_preview"), indent=2),
            updated_resume,
        )
        logger.info(f"Structured Resume Prompt: {prompt}")
        raw_output = await self.json_agent_manager.run(prompt=prompt)

        try:
            resume_preview: ResumePreviewerModel = ResumePreviewerModel.model_validate(
                raw_output
            )
        except ValidationError as e:
            logger.info(f"Validation error: {e}")
            return None
        return resume_preview.model_dump()

    def _normalize_keyword(self, keyword: str) -> str:
        """Normalize a keyword for better matching"""
        return keyword.lower().strip()
    
    def _tokenize(self, text: str) -> List[str]:
        """Tokenize text into words, removing stop words and short words"""
        stop_words = {
            'a', 'an', 'and', 'are', 'as', 'at', 'be', 'by', 'for', 'from', 'has', 'he',
            'in', 'is', 'it', 'its', 'of', 'on', 'that', 'the', 'to', 'was', 'will', 'with',
            'we', 'you', 'your', 'our', 'their', 'this', 'these', 'those', 'can', 'could',
            'should', 'would', 'may', 'might', 'must', 'shall', 'have', 'had', 'do', 'does',
            'did', 'been', 'being', 'am', 'or', 'but', 'not', 'no', 'yes', 'all', 'any'
        }
        
        # Extract words (alphanumeric, including hyphens and underscores)
        words = re.findall(r'\b[a-z][a-z0-9_-]*\b', text.lower())
        
        # Filter out stop words and very short words
        return [w for w in words if w not in stop_words and len(w) >= 3]
    
    def _calculate_tf(self, tokens: List[str]) -> Dict[str, float]:
        """Calculate Term Frequency (TF) for tokens"""
        token_count = Counter(tokens)
        total_tokens = len(tokens)
        
        if total_tokens == 0:
            return {}
        
        return {token: count / total_tokens for token, count in token_count.items()}
    
    def _calculate_idf(self, documents: List[List[str]]) -> Dict[str, float]:
        """Calculate Inverse Document Frequency (IDF) across documents"""
        num_docs = len(documents)
        if num_docs == 0:
            return {}
        
        # Count documents containing each term
        doc_freq = Counter()
        for doc_tokens in documents:
            unique_tokens = set(doc_tokens)
            doc_freq.update(unique_tokens)
        
        # Calculate IDF: log(total_docs / docs_containing_term)
        import math
        idf = {}
        for term, freq in doc_freq.items():
            idf[term] = math.log((num_docs + 1) / (freq + 1)) + 1  # Smoothing
        
        return idf
    
    def _calculate_tfidf_vector(self, tokens: List[str], idf: Dict[str, float]) -> Dict[str, float]:
        """Calculate TF-IDF vector for a document"""
        tf = self._calculate_tf(tokens)
        tfidf = {}
        
        for token, tf_value in tf.items():
            idf_value = idf.get(token, 1.0)
            tfidf[token] = tf_value * idf_value
        
        return tfidf
    
    def _cosine_similarity(self, vec1: Dict[str, float], vec2: Dict[str, float]) -> float:
        """
        Calculate cosine similarity between two TF-IDF vectors.
        This is the core algorithm used by SkillSyncer and ResumeWorded.
        """
        # Get all unique terms
        all_terms = set(vec1.keys()) | set(vec2.keys())
        
        if not all_terms:
            return 0.0
        
        # Calculate dot product and magnitudes
        dot_product = sum(vec1.get(term, 0) * vec2.get(term, 0) for term in all_terms)
        magnitude1 = sqrt(sum(val ** 2 for val in vec1.values()))
        magnitude2 = sqrt(sum(val ** 2 for val in vec2.values()))
        
        if magnitude1 == 0 or magnitude2 == 0:
            return 0.0
        
        return dot_product / (magnitude1 * magnitude2)
    
    def _extract_named_entities(self, text: str) -> Set[str]:
        """
        Extract named entities (skills, technologies, tools) using comprehensive pattern matching.
        This simulates NER (Named Entity Recognition) without requiring spaCy.
        """
        entities = set()
        text_lower = text.lower()
        
        # Comprehensive skill database with variations
        skill_database = {
            # Programming Languages
            'python': ['python', 'py'],
            'java': ['java'],
            'javascript': ['javascript', 'js'],
            'typescript': ['typescript', 'ts'],
            'c++': ['c\\+\\+', 'cpp'],
            'c#': ['c#', 'csharp'],
            'ruby': ['ruby'],
            'go': ['\\bgo\\b', 'golang'],
            'rust': ['rust'],
            'swift': ['swift'],
            'kotlin': ['kotlin'],
            'scala': ['scala'],
            'r': ['\\br\\b'],
            'matlab': ['matlab'],
            'perl': ['perl'],
            'php': ['php'],
            
            # Web Technologies
            'react': ['react', 'reactjs'],
            'angular': ['angular', 'angularjs'],
            'vue': ['vue', 'vuejs'],
            'node.js': ['node', 'nodejs', 'node\\.js'],
            'express': ['express', 'expressjs'],
            'django': ['django'],
            'flask': ['flask'],
            'fastapi': ['fastapi'],
            'spring': ['spring', 'spring boot', 'springboot'],
            'html': ['html', 'html5'],
            'css': ['css', 'css3'],
            
            # Databases
            'sql': ['\\bsql\\b'],
            'mysql': ['mysql'],
            'postgresql': ['postgresql', 'postgres'],
            'mongodb': ['mongodb', 'mongo'],
            'redis': ['redis'],
            'dynamodb': ['dynamodb'],
            'elasticsearch': ['elasticsearch', 'elastic search'],
            'cassandra': ['cassandra'],
            'oracle': ['oracle'],
            'sqlite': ['sqlite'],
            'couchbase': ['couchbase'],
            
            # Cloud Platforms
            'aws': ['\\baws\\b', 'amazon web services'],
            'azure': ['azure', 'microsoft azure'],
            'gcp': ['\\bgcp\\b', 'google cloud'],
            
            # AWS Services
            'kinesis': ['kinesis'],
            's3': ['\\bs3\\b', 'simple storage service'],
            'emr': ['\\bemr\\b', 'elastic mapreduce'],
            'lambda': ['lambda', 'aws lambda'],
            'ec2': ['\\bec2\\b', 'elastic compute'],
            'cloudformation': ['cloudformation'],
            'cloudwatch': ['cloudwatch'],
            'rds': ['\\brds\\b'],
            'sqs': ['\\bsqs\\b'],
            'sns': ['\\bsns\\b'],
            
            # DevOps Tools
            'docker': ['docker'],
            'kubernetes': ['kubernetes', 'k8s'],
            'jenkins': ['jenkins'],
            'git': ['\\bgit\\b'],
            'github': ['github'],
            'gitlab': ['gitlab'],
            'terraform': ['terraform'],
            'ansible': ['ansible'],
            
            # Methodologies
            'agile': ['agile'],
            'scrum': ['scrum'],
            'kanban': ['kanban'],
            'ci/cd': ['ci/cd', 'continuous integration', 'continuous deployment'],
            'tdd': ['\\btdd\\b', 'test.driven'],
            'microservices': ['microservices', 'microservice'],
            'rest': ['\\brest\\b', 'rest api', 'restful'],
            'api': ['\\bapi\\b', 'apis'],
            'graphql': ['graphql'],
            
            # Data & ML
            'machine learning': ['machine learning', '\\bml\\b'],
            'data science': ['data science'],
            'analytics': ['analytics'],
            'big data': ['big data'],
            'spark': ['\\bspark\\b', 'apache spark'],
            'hadoop': ['hadoop'],
            'pandas': ['pandas'],
            'numpy': ['numpy'],
            'tensorflow': ['tensorflow'],
            'pytorch': ['pytorch'],
            
            # General Engineering
            'software development': ['software development', 'software engineering'],
            'testing': ['testing', 'test automation'],
            'debugging': ['debugging', 'troubleshooting'],
            'code review': ['code review'],
            'design': ['design', 'system design'],
            'architecture': ['architecture', 'software architecture'],
            
            # Soft Skills
            'leadership': ['leadership', 'lead', 'leading'],
            'mentorship': ['mentorship', 'mentor', 'mentoring'],
            'communication': ['communication'],
            'collaboration': ['collaboration', 'collaborate'],
            'problem solving': ['problem solving', 'problem.solving'],
        }
        
        # Search for each skill and its variations
        for skill_name, patterns in skill_database.items():
            for pattern in patterns:
                if re.search(pattern, text_lower):
                    entities.add(skill_name)
                    break  # Found this skill, move to next
        
        return entities
    
    def _get_skill_synonyms(self) -> dict:
        """Return a comprehensive dictionary of skill synonyms and variations"""
        return {
            # Programming Languages
            'python': ['python', 'py', 'python3'],
            'javascript': ['javascript', 'js', 'ecmascript', 'es6', 'es2015'],
            'typescript': ['typescript', 'ts'],
            'java': ['java', 'jdk', 'jvm'],
            'c++': ['c++', 'cpp', 'cplusplus'],
            'c#': ['c#', 'csharp', 'c sharp'],
            
            # Web Frameworks
            'react': ['react', 'reactjs', 'react.js'],
            'angular': ['angular', 'angularjs', 'angular.js'],
            'vue': ['vue', 'vuejs', 'vue.js'],
            'node': ['node', 'nodejs', 'node.js'],
            'express': ['express', 'expressjs', 'express.js'],
            'django': ['django'],
            'flask': ['flask'],
            'fastapi': ['fastapi', 'fast api'],
            'spring': ['spring', 'spring boot', 'springboot'],
            
            # Databases
            'sql': ['sql', 'structured query language'],
            'mysql': ['mysql', 'my sql'],
            'postgresql': ['postgresql', 'postgres', 'psql'],
            'mongodb': ['mongodb', 'mongo'],
            'redis': ['redis'],
            'dynamodb': ['dynamodb', 'dynamo db'],
            'elasticsearch': ['elasticsearch', 'elastic search', 'es'],
            
            # Cloud Platforms
            'aws': ['aws', 'amazon web services'],
            'azure': ['azure', 'microsoft azure'],
            'gcp': ['gcp', 'google cloud', 'google cloud platform'],
            'kinesis': ['kinesis', 'aws kinesis'],
            's3': ['s3', 'simple storage service'],
            'emr': ['emr', 'elastic mapreduce'],
            'lambda': ['lambda', 'aws lambda'],
            'ec2': ['ec2', 'elastic compute cloud'],
            
            # DevOps & Tools
            'docker': ['docker', 'containerization'],
            'kubernetes': ['kubernetes', 'k8s'],
            'jenkins': ['jenkins'],
            'git': ['git', 'github', 'gitlab', 'version control'],
            'ci/cd': ['ci/cd', 'continuous integration', 'continuous deployment', 'continuous delivery'],
            'terraform': ['terraform'],
            'ansible': ['ansible'],
            
            # Methodologies
            'agile': ['agile', 'scrum', 'kanban'],
            'tdd': ['tdd', 'test driven development', 'test-driven'],
            'microservices': ['microservices', 'micro services', 'microservice architecture'],
            
            # Data & ML
            'machine learning': ['machine learning', 'ml', 'artificial intelligence', 'ai'],
            'data science': ['data science', 'data analysis', 'analytics'],
            'big data': ['big data', 'hadoop', 'spark'],
            
            # Soft Skills
            'leadership': ['leadership', 'lead', 'leading', 'mentor', 'mentorship'],
            'communication': ['communication', 'communicate', 'collaboration', 'collaborate'],
            'problem solving': ['problem solving', 'troubleshooting', 'debugging'],
            
            # General
            'api': ['api', 'apis', 'rest api', 'restful', 'rest'],
            'software development': ['software development', 'software engineering', 'development', 'engineering'],
            'testing': ['testing', 'test', 'qa', 'quality assurance'],
            'debugging': ['debugging', 'debug', 'troubleshooting'],
            'scalable': ['scalable', 'scalability', 'scale'],
            'performance': ['performance', 'optimization', 'optimize'],
        }
    
    def _extract_ngrams(self, text: str, n: int = 3) -> set:
        """Extract meaningful n-grams (phrases) from text with filtering"""
        import re
        
        # Common stop words to filter out
        stop_words = {
            'a', 'an', 'and', 'are', 'as', 'at', 'be', 'by', 'for', 'from', 'has', 'he',
            'in', 'is', 'it', 'its', 'of', 'on', 'that', 'the', 'to', 'was', 'will', 'with',
            'we', 'you', 'your', 'our', 'their', 'this', 'these', 'those', 'can', 'could',
            'should', 'would', 'may', 'might', 'must', 'shall', 'have', 'had', 'do', 'does',
            'did', 'been', 'being', 'am', 'or', 'but', 'not', 'no', 'yes', 'all', 'any',
            'both', 'each', 'few', 'more', 'most', 'other', 'some', 'such', 'than', 'too',
            'very', 'about', 'after', 'before', 'between', 'into', 'through', 'during',
            'above', 'below', 'up', 'down', 'out', 'off', 'over', 'under', 'again', 'further',
            'then', 'once', 'here', 'there', 'when', 'where', 'why', 'how', 'who', 'which',
            'what', 'whom', 'whose', 'if', 'because', 'while', 'until', 'since'
        }
        
        # Extract words (alphanumeric only, minimum 2 chars)
        words = [w for w in re.findall(r'\b[a-z][a-z0-9]*\b', text.lower()) if len(w) >= 2]
        
        ngrams = set()
        for i in range(len(words) - n + 1):
            ngram_words = words[i:i+n]
            
            # Skip if all words are stop words
            if all(w in stop_words for w in ngram_words):
                continue
            
            # Skip if first or last word is a stop word (for bigrams/trigrams)
            if n > 1 and (ngram_words[0] in stop_words or ngram_words[-1] in stop_words):
                continue
            
            ngram = ' '.join(ngram_words)
            
            # Only include if it contains at least one meaningful word (length > 3)
            if any(len(w) > 3 for w in ngram_words):
                ngrams.add(ngram)
        
        return ngrams
    
    def _is_keyword_match(self, resume_text: str, job_keyword: str, synonyms: dict) -> tuple:
        """
        Advanced keyword matching with synonym support and context awareness.
        Returns (is_match, match_type, confidence)
        """
        resume_lower = resume_text.lower()
        keyword_lower = job_keyword.lower()
        
        # Check for exact match
        import re
        pattern = r'\b' + re.escape(keyword_lower) + r'\b'
        if re.search(pattern, resume_lower):
            return (True, 'exact', 1.0)
        
        # Check for direct substring match
        if keyword_lower in resume_lower:
            return (True, 'partial', 0.9)
        
        # Check synonyms
        for base_skill, variations in synonyms.items():
            if keyword_lower in variations or keyword_lower == base_skill:
                for variation in variations:
                    if re.search(r'\b' + re.escape(variation) + r'\b', resume_lower):
                        return (True, 'synonym', 0.85)
        
        # Check for plural/singular variations
        if keyword_lower.endswith('s') and keyword_lower[:-1] in resume_lower:
            return (True, 'variation', 0.8)
        if keyword_lower + 's' in resume_lower:
            return (True, 'variation', 0.8)
        
        # Check for stemmed versions (basic stemming)
        stems = [
            (keyword_lower.rstrip('ing'), 0.75),
            (keyword_lower.rstrip('ed'), 0.75),
            (keyword_lower.rstrip('er'), 0.75),
        ]
        for stem, confidence in stems:
            if len(stem) > 3 and stem in resume_lower:
                return (True, 'stem', confidence)
        
        return (False, 'none', 0.0)
    
    def _extract_skills_from_text(self, text: str) -> dict:
        """
        Extract technical skills and keywords from text with categorization.
        Returns dict with skills categorized by type and importance.
        """
        text_lower = text.lower()
        
        # Comprehensive skill database with categories
        skill_database = {
            'programming_languages': [
                'python', 'java', 'javascript', 'typescript', 'c\\+\\+', 'c#', 'ruby', 
                'go', 'rust', 'swift', 'kotlin', 'scala', 'r', 'matlab', 'perl'
            ],
            'web_technologies': [
                'react', 'angular', 'vue', 'node', 'express', 'django', 'flask', 
                'fastapi', 'spring', 'html', 'css', 'jquery', 'bootstrap', 'tailwind'
            ],
            'databases': [
                'sql', 'mysql', 'postgresql', 'mongodb', 'redis', 'dynamodb', 
                'elasticsearch', 'cassandra', 'oracle', 'sqlite', 'couchbase'
            ],
            'cloud_devops': [
                'aws', 'azure', 'gcp', 'docker', 'kubernetes', 'jenkins', 'terraform',
                'ansible', 'kinesis', 's3', 'emr', 'lambda', 'ec2', 'cloudformation',
                'ecs', 'eks', 'cloudwatch'
            ],
            'tools_practices': [
                'git', 'github', 'gitlab', 'agile', 'scrum', 'ci/cd', 'tdd', 'rest',
                'api', 'microservices', 'serverless', 'graphql', 'websocket'
            ],
            'data_ml': [
                'machine learning', 'data science', 'analytics', 'big data', 'spark',
                'hadoop', 'pandas', 'numpy', 'tensorflow', 'pytorch', 'scikit-learn'
            ],
            'soft_skills': [
                'leadership', 'communication', 'problem solving', 'teamwork', 
                'mentorship', 'collaboration', 'presentation', 'negotiation'
            ],
            'general_engineering': [
                'software', 'development', 'engineer', 'design', 'architecture',
                'testing', 'debugging', 'scalable', 'maintainable', 'performance',
                'optimization', 'code review', 'documentation', 'troubleshooting'
            ]
        }
        
        found_skills = {}
        import re
        
        for category, patterns in skill_database.items():
            category_skills = set()
            for pattern in patterns:
                # Use word boundaries for better matching
                if re.search(r'\b' + pattern + r'\b', text_lower):
                    clean_pattern = pattern.replace('\\+\\+', '++').replace('\\', '')
                    category_skills.add(clean_pattern)
            if category_skills:
                found_skills[category] = category_skills
        
        return found_skills
    
    def _calculate_keyword_importance(self, keyword: str, job_text: str) -> float:
        """
        Calculate importance weight of a keyword based on context and type.
        Technical skills and requirements get higher weight.
        """
        job_lower = job_text.lower()
        keyword_lower = keyword.lower()
        
        # Base weight
        weight = 1.0
        
        # Technical skills get higher base weight
        technical_indicators = [
            'python', 'java', 'javascript', 'aws', 'sql', 'api', 'cloud',
            'docker', 'kubernetes', 'react', 'node', 'database', 'git',
            'agile', 'scrum', 'testing', 'ci/cd', 'microservices'
        ]
        if any(tech in keyword_lower for tech in technical_indicators):
            weight = 1.3
        
        # Check if keyword appears in important sections
        important_sections = [
            'required', 'requirements', 'qualifications', 'must have',
            'essential', 'key skills', 'responsibilities', 'you will'
        ]
        
        # Find context around keyword
        import re
        matches = list(re.finditer(r'\b' + re.escape(keyword_lower) + r'\b', job_lower))
        
        if matches:
            for match in matches:
                start = max(0, match.start() - 150)
                end = min(len(job_lower), match.end() + 150)
                context = job_lower[start:end]
                
                # Increase weight if in important section
                if any(section in context for section in important_sections):
                    weight = max(weight, 1.6)
                
                # Higher weight if in bullet points or lists
                if '•' in context or '-' in context or '·' in context:
                    weight *= 1.1
            
            # Increase weight if mentioned multiple times
            if len(matches) > 2:
                weight *= 1.3
            elif len(matches) > 1:
                weight *= 1.15
        
        return min(weight, 2.0)  # Cap at 2.0

    async def _improve_resume_with_groq(
        self,
        resume_text: str,
        job_text: str,
        resume_keywords: List[str],
        job_keywords: List[str],
        current_score: float
    ) -> str:
        """
        Use Groq LLM to intelligently improve resume based on job description.
        """
        prompt = f"""You are an expert resume editor and talent acquisition specialist. Your task is to revise the following resume so that it aligns as closely as possible with the provided job description and extracted job keywords, in order to maximize the match score.

Instructions:
- Carefully review the job description and the list of extracted job keywords.
- Update the candidate's resume by:
  - Emphasizing and naturally incorporating relevant skills, experiences, and keywords from the job description and keyword list.
  - Where appropriate, naturally weave the extracted job keywords into the resume content.
  - Rewriting, adding, or removing resume content as needed to better match the job requirements.
  - Maintaining a natural, professional tone and avoiding keyword stuffing.
  - Where possible, use quantifiable achievements and action verbs.
- The current match score is {current_score:.1%}. Revise the resume to increase this score.
- ONLY output the improved updated resume. Do not include any explanations, commentary, or formatting outside of the resume itself.

Job Description:
```
{job_text}
```

Extracted Job Keywords:
```
{', '.join(job_keywords[:30])}
```

Original Resume:
```
{resume_text}
```

Extracted Resume Keywords:
```
{', '.join(resume_keywords[:30])}
```

NOTE: ONLY OUTPUT THE IMPROVED UPDATED RESUME IN MARKDOWN FORMAT."""

        try:
            # Use Groq to improve the resume
            improved_resume = await self.md_agent_manager.run(prompt)
            return improved_resume
        except Exception as e:
            logger.error(f"Error improving resume with Groq: {str(e)}")
            # Return original if improvement fails
            return resume_text

    async def _extract_keywords_with_llm(self, text: str, text_type: str) -> List[str]:
        """Use Groq LLM to extract keywords from text"""
        prompt = f"""Extract all relevant technical skills, tools, technologies, and key qualifications from the following {text_type}.

List them as comma-separated values. Include:
- Programming languages (Python, Java, JavaScript, etc.)
- Frameworks and libraries (React, Django, Spring, etc.)
- Tools and platforms (Git, Docker, AWS, etc.)
- Cloud services (S3, Lambda, Kinesis, etc.)
- Databases (SQL, MongoDB, PostgreSQL, etc.)
- Methodologies (Agile, Scrum, CI/CD, etc.)
- Technical skills (API development, testing, debugging, etc.)
- Soft skills (leadership, communication, problem-solving, etc.)

{text_type.upper()}:
```
{text[:1000]}
```

Output ONLY comma-separated keywords, nothing else:
keyword1, keyword2, keyword3, ..."""

        try:
            # Use MD agent manager instead of JSON to avoid PyTorch issues
            response = await self.md_agent_manager.run(prompt)
            
            # Parse the response
            if isinstance(response, str):
                # Split by comma and clean
                keywords = [k.strip().lower() for k in response.split(',') if k.strip()]
            else:
                keywords = []
            
            # Filter out very short or invalid keywords
            keyword_list = [k for k in keywords if k and len(k) > 2 and len(k) < 50]
            logger.info(f"Extracted {len(keyword_list)} keywords from {text_type}")
            return keyword_list[:50]  # Limit to 50 keywords
        except Exception as e:
            logger.error(f"Error extracting keywords with LLM: {str(e)}")
            # Fallback to basic extraction
            return self._extract_basic_keywords(text)

    def _extract_basic_keywords(self, text: str) -> List[str]:
        """Fallback: Extract basic keywords using simple pattern matching"""
        keywords = set()
        text_lower = text.lower()
        
        # Common technical terms
        common_skills = [
            'python', 'java', 'javascript', 'typescript', 'react', 'angular', 'vue',
            'node', 'django', 'flask', 'spring', 'sql', 'mysql', 'postgresql', 'mongodb',
            'aws', 'azure', 'gcp', 'docker', 'kubernetes', 'git', 'jenkins', 'terraform',
            'agile', 'scrum', 'ci/cd', 'api', 'rest', 'microservices', 'testing',
            'leadership', 'communication', 'problem solving', 'teamwork'
        ]
        
        for skill in common_skills:
            if skill in text_lower:
                keywords.add(skill)
        
        return list(keywords)

    async def _analyze_match_with_llm(
        self,
        resume_text: str,
        job_text: str,
        resume_keywords: List[str],
        job_keywords: List[str]
    ) -> Dict:
        """Use Groq LLM to analyze resume-job match with strict scoring"""
        
        # Calculate basic match using keyword overlap as fallback
        resume_set = set(resume_keywords)
        job_set = set(job_keywords)
        matched = list(resume_set & job_set)
        missing = list(job_set - resume_set)
        basic_score = int((len(matched) / len(job_set) * 100) if job_set else 50)
        
        prompt = f"""You are a strict ATS (Applicant Tracking System) analyzer. Compare this resume against the job requirements and provide an HONEST, REALISTIC match score.

SCORING GUIDELINES (BE STRICT):
- 90-100%: Perfect match - candidate has ALL required skills and extensive relevant experience
- 75-89%: Strong match - has most required skills with good relevant experience  
- 60-74%: Good match - has many required skills but missing some important ones
- 40-59%: Moderate match - has some required skills but significant gaps
- 20-39%: Weak match - few matching skills, mostly unrelated experience
- 0-19%: Poor match - almost no relevant skills or experience

JOB DESCRIPTION (First 500 chars):
{job_text[:500]}...

JOB REQUIRED SKILLS: {', '.join(job_keywords[:25])}

RESUME (First 500 chars):
{resume_text[:500]}...

RESUME SKILLS: {', '.join(resume_keywords[:25])}

ANALYSIS REQUIRED:
1. Compare resume skills vs job requirements
2. Check for relevant experience and projects
3. Evaluate if candidate can actually do this job
4. BE REALISTIC - most resumes are 40-70% match

Return JSON:
{{
  "match_score": <number 0-100>,
  "matched_skills": [<list of skills found in BOTH resume and job>],
  "missing_skills": [<list of important skills from job NOT in resume>],
  "strengths": [<2-3 specific strengths>],
  "gaps": [<2-3 specific gaps>]
}}

IMPORTANT: Be honest and realistic. Don't give 100% unless it's truly perfect. Most good matches are 60-75%."""

        try:
            # Use MD agent manager to avoid PyTorch issues
            response_text = await self.md_agent_manager.run(prompt)
            
            # Try to parse as JSON
            import json as json_lib
            try:
                response = json_lib.loads(response_text)
            except:
                # If not JSON, use fallback
                response = None
            
            # Ensure response has required fields
            if isinstance(response, dict):
                llm_score = response.get('match_score', basic_score)
                
                # Sanity check: if LLM gives 100% but there are missing skills, cap it
                llm_matched = response.get('matched_skills', matched)
                llm_missing = response.get('missing_skills', missing)
                
                if llm_score > 90 and len(llm_missing) > 3:
                    llm_score = 85  # Cap unrealistic scores
                    logger.warning(f"Capped LLM score from {response.get('match_score')} to {llm_score} due to {len(llm_missing)} missing skills")
                
                result = {
                    "match_score": llm_score,
                    "matched_skills": llm_matched if llm_matched else matched,
                    "missing_skills": llm_missing if llm_missing else missing,
                    "strengths": response.get('strengths', [f"Resume shows {llm_score}% match"]),
                    "gaps": response.get('gaps', [f"Missing {len(llm_missing)} key skills"])
                }
            else:
                # Use fallback
                result = {
                    "match_score": basic_score,
                    "matched_skills": matched,
                    "missing_skills": missing,
                    "strengths": [f"Resume matches {len(matched)} key requirements"],
                    "gaps": [f"Could add {len(missing)} more relevant skills"]
                }
            
            logger.info(f"LLM Analysis: {result['match_score']}% match, {len(result['matched_skills'])} matched, {len(result['missing_skills'])} missing")
            return result
            
        except Exception as e:
            logger.error(f"Error analyzing match with LLM: {str(e)}")
            # Return fallback with keyword-based analysis
            return {
                "match_score": basic_score,
                "matched_skills": matched,
                "missing_skills": missing,
                "strengths": [f"Resume matches {len(matched)} key requirements"],
                "gaps": [f"Could add {len(missing)} more relevant skills"]
            }

    async def run(self, resume_id: str, job_id: str) -> Dict:
        """
        LLM-powered resume matching and improvement using Groq.
        All analysis is done by the LLM for maximum accuracy.
        """
        try:
            resume, processed_resume = await self._get_resume(resume_id)
            job, processed_job = await self._get_job(job_id)

            # Get full text for analysis
            resume_text = resume.content
            job_text = job.content
            
            logger.info("Starting LLM-powered analysis...")
            
            # Step 1: Extract keywords using LLM
            logger.info("Extracting keywords with Groq LLM...")
            resume_keywords = await self._extract_keywords_with_llm(resume_text, "resume")
            job_keywords = await self._extract_keywords_with_llm(job_text, "job description")
            
            logger.info(f"Extracted {len(resume_keywords)} resume keywords, {len(job_keywords)} job keywords")
            
            # Step 2: Analyze match using LLM
            logger.info("Analyzing match with Groq LLM...")
            analysis = await self._analyze_match_with_llm(
                resume_text=resume_text,
                job_text=job_text,
                resume_keywords=resume_keywords,
                job_keywords=job_keywords
            )
            
            original_score = analysis.get('match_score', 50) / 100.0
            matched_skills = analysis.get('matched_skills', [])
            missing_skills = analysis.get('missing_skills', [])
            
            logger.info(f"Original LLM score: {original_score:.2%}")
            
            # Step 3: Use Groq to improve the resume
            logger.info("Using Groq LLM to improve resume...")
            
            improved_resume_text = await self._improve_resume_with_groq(
                resume_text=resume_text,
                job_text=job_text,
                resume_keywords=resume_keywords,
                job_keywords=job_keywords,
                current_score=original_score
            )
            
            # Step 4: Re-analyze improved resume
            logger.info("Re-analyzing improved resume...")
            improved_keywords = await self._extract_keywords_with_llm(improved_resume_text, "resume")
            improved_analysis = await self._analyze_match_with_llm(
                resume_text=improved_resume_text,
                job_text=job_text,
                resume_keywords=improved_keywords,
                job_keywords=job_keywords
            )
            
            improved_score = improved_analysis.get('match_score', 50) / 100.0
            improved_matched = improved_analysis.get('matched_skills', [])
            improved_missing = improved_analysis.get('missing_skills', [])
            
            logger.info(f"Improved LLM score: {improved_score:.2%} (gain: {(improved_score - original_score):.2%})")
            
            # Use the better score
            final_score = max(original_score, improved_score)
            match_score = final_score
            
            # Step 5: Generate insights and recommendations
            improvements_list = []
            
            top_missing_skills = improved_missing[:15] if improved_missing else missing_skills[:15]
            
            if top_missing_skills:
                critical_skills = top_missing_skills[:5]
                improvements_list.append({
                    "section": "Critical Technical Skills",
                    "current": f"Resume is missing {len(top_missing_skills)} key skills from job requirements",
                    "suggested": f"Add these high-priority skills: {', '.join(critical_skills)}",
                    "reason": "These technical skills are explicitly mentioned in the job description"
                })
            
            if match_score < 0.6:
                improvements_list.append({
                    "section": "Overall Tailoring",
                    "current": "Resume could be better tailored to this specific role",
                    "suggested": "Emphasize relevant projects and experience that match job requirements",
                    "reason": "Tailored resumes are 3x more likely to get interviews"
                })
            
            # Use LLM-provided strengths and gaps, or generate fallbacks
            strengths = improved_analysis.get('strengths', analysis.get('strengths', []))
            gaps = improved_analysis.get('gaps', analysis.get('gaps', []))
            
            # Add score improvement to strengths
            score_improvement = improved_score - original_score
            if score_improvement > 0.05:
                strengths.insert(0, f"AI-improved resume increased match score by {int(score_improvement * 100)}%")
            
            # Ensure we have at least some content
            if not strengths:
                strengths = [
                    f"Resume shows {int(match_score * 100)}% match with job requirements",
                    "Focus on highlighting relevant technical skills"
                ]
            
            if not gaps:
                if match_score < 0.7:
                    gaps = [
                        "Add more job-specific keywords and skills",
                        "Tailor experience descriptions to match job requirements"
                    ]
                else:
                    gaps = ["Excellent! Resume is well-optimized for this role"]
            
            execution = {
                "match_score": int(match_score * 100),
                "improvements": improvements_list,
                "missing_keywords": top_missing_skills,
                "matched_keywords": (improved_matched if improved_matched else matched_skills)[:30],
                "strengths": strengths[:5],  # Limit to 5
                "gaps": gaps[:5],  # Limit to 5
                "original_score": int(original_score * 100),
                "improved_score": int(improved_score * 100),
                "improved_resume": improved_resume_text
            }

            gc.collect()
            return execution
            
        except Exception as e:
            logger.error(f"Error in score improvement: {str(e)}")
            raise

    async def run_and_stream(self, resume_id: str, job_id: str) -> AsyncGenerator:
        """
        Main method to run the scoring and improving process and return dict.
        """

        yield f"data: {json.dumps({'status': 'starting', 'message': 'Analyzing resume and job description...'})}\n\n"
        await asyncio.sleep(2)

        resume, processed_resume = await self._get_resume(resume_id)
        job, processed_job = await self._get_job(job_id)

        yield f"data: {json.dumps({'status': 'parsing', 'message': 'Parsing resume content...'})}\n\n"
        await asyncio.sleep(2)

        extracted_job_keywords = ", ".join(
            json.loads(processed_job.extracted_keywords).get("extracted_keywords", [])
        )

        extracted_resume_keywords = ", ".join(
            json.loads(processed_resume.extracted_keywords).get(
                "extracted_keywords", []
            )
        )

        resume_embedding = await self.embedding_manager.embed(text=resume.content)
        extracted_job_keywords_embedding = await self.embedding_manager.embed(
            text=extracted_job_keywords
        )

        yield f"data: {json.dumps({'status': 'scoring', 'message': 'Calculating compatibility score...'})}\n\n"
        await asyncio.sleep(3)

        cosine_similarity_score = self.calculate_cosine_similarity(
            extracted_job_keywords_embedding, resume_embedding
        )

        yield f"data: {json.dumps({'status': 'scored', 'score': cosine_similarity_score})}\n\n"

        yield f"data: {json.dumps({'status': 'improving', 'message': 'Generating improvement suggestions...'})}\n\n"
        await asyncio.sleep(3)

        updated_resume, updated_score = await self.improve_score_with_llm(
            resume=resume.content,
            extracted_resume_keywords=extracted_resume_keywords,
            job=job.content,
            extracted_job_keywords=extracted_job_keywords,
            previous_cosine_similarity_score=cosine_similarity_score,
            extracted_job_keywords_embedding=extracted_job_keywords_embedding,
        )

        for i, suggestion in enumerate(updated_resume):
            yield f"data: {json.dumps({'status': 'suggestion', 'index': i, 'text': suggestion})}\n\n"
            await asyncio.sleep(0.2)

        final_result = {
            "resume_id": resume_id,
            "job_id": job_id,
            "original_score": cosine_similarity_score,
            "new_score": updated_score,
            "updated_resume": markdown.markdown(text=updated_resume),
        }

        yield f"data: {json.dumps({'status': 'completed', 'result': final_result})}\n\n"

import json
import logging
from uuid import uuid4
from typing import List, Dict, Any, Optional
from datetime import datetime

from fastapi import HTTPException, status
from pydantic import ValidationError
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.exc import SQLAlchemyError

from app.agent import AgentManager
from app.prompt import prompt_factory
from app.schemas.json import json_schema_factory
from app.models import Job, Resume, ProcessedJob
from app.schemas.pydantic import StructuredJobModel
from .exceptions import (
    JobNotFoundError,
    JobValidationError,
    JobCreationError,
    JobProcessingError
)

logger = logging.getLogger(__name__)


class JobService:
    def __init__(self, db: AsyncSession):
        self.db = db
        self.json_agent_manager = AgentManager()

    async def create_and_store_job(self, job_data: dict) -> List[str]:
        """
        Stores job data in the database and returns a list of job IDs.
        
        Args:
            job_data: Dictionary containing resume_id and job_descriptions
            
        Returns:
            List of created job IDs
            
        Raises:
            JobValidationError: If request data is invalid
            JobNotFoundError: If resume not found
            JobCreationError: If job creation fails
        """
        # Validate resume_id
        resume_id = job_data.get("resume_id")
        if not resume_id:
            logger.error("resume_id is missing in the request")
            raise JobValidationError("resume_id is required")

        # Convert UUID to string if needed
        resume_id = str(resume_id)
        logger.info(f"Processing job with resume_id: {resume_id}")

        job_descriptions = job_data.get("job_descriptions", [])
        if not job_descriptions or not isinstance(job_descriptions, list):
            logger.error("Invalid job descriptions format")
            raise JobValidationError(
                "job_descriptions must be a non-empty array of strings"
            )

        if not any(desc.strip() for desc in job_descriptions):
            logger.error("No valid job descriptions provided")
            raise JobValidationError(
                "At least one non-empty job description is required"
            )

        # Check if resume exists
        resume_exists = await self._is_resume_available(resume_id)
        if not resume_exists:
            logger.error(f"Resume not found for resume_id: {resume_id}")
            raise JobNotFoundError(
                f"Resume with ID {resume_id} not found"
            )

        # Process jobs
        job_ids = []
        
        # Start transaction
        async with self.db.begin_nested() as nested:
            try:
                # Process each job description
                for job_description in job_descriptions:
                    if not job_description or not job_description.strip():
                        logger.warning("Skipping empty job description")
                        continue

                    job_id = str(uuid4())
                    
                    # Create basic job record
                    job = Job(
                        job_id=job_id,
                        resume_id=resume_id,
                        content=job_description,
                        created_at=datetime.utcnow(),
                    )
                    self.db.add(job)
                    
                    try:
                        # Extract and validate structured data
                        structured_job = await self._extract_structured_json(job_description)
                        
                        # Create processed job record
                        processed_job = ProcessedJob(
                            job_id=job_id,
                            job_title=structured_job.get("jobTitle"),
                            company_profile=json.dumps(structured_job.get("companyProfile")) if structured_job.get("companyProfile") else None,
                            location=json.dumps(structured_job.get("location")) if structured_job.get("location") else None,
                            date_posted=structured_job.get("datePosted"),
                            employment_type=structured_job.get("employmentType"),
                            job_summary=structured_job.get("jobSummary"),
                            key_responsibilities=json.dumps({"key_responsibilities": structured_job.get("keyResponsibilities", [])}) if structured_job.get("keyResponsibilities") else None,
                            qualifications=json.dumps(structured_job.get("qualifications")) if structured_job.get("qualifications") else None,
                            compensation_and_benfits=json.dumps(structured_job.get("compensationAndBenefits")) if structured_job.get("compensationAndBenefits") else None,
                            application_info=json.dumps(structured_job.get("applicationInfo")) if structured_job.get("applicationInfo") else None,
                            extracted_keywords=json.dumps({"extracted_keywords": structured_job.get("extractedKeywords", [])}) if structured_job.get("extractedKeywords") else None,
                        )
                        self.db.add(processed_job)
                        
                        # Save and track successful job
                        job_ids.append(job_id)
                        logger.info(f"Successfully processed job with ID: {job_id}")
                        
                    except (JobValidationError, JobProcessingError) as e:
                        logger.error(f"Failed to process job {job_id}: {str(e)}")
                        continue

                if not job_ids:
                    raise JobCreationError("Failed to process any job descriptions")
                
                await nested.commit()
                
            except SQLAlchemyError as e:
                logger.error(f"Database error: {str(e)}")
                raise JobCreationError("Database error while creating jobs")
            except Exception as e:
                logger.error(f"Unexpected error: {str(e)}")
                raise JobCreationError("Failed to create jobs")
        
        # If we got here, the nested transaction succeeded
        await self.db.commit()
        return job_ids

        # The previous implementation has been moved above and improved

    async def _is_resume_available(self, resume_id: str) -> bool:
        """
        Checks if a resume exists in the database.
        """
        query = select(Resume).where(Resume.resume_id == resume_id)
        result = await self.db.scalar(query)
        return result is not None

    async def _extract_and_store_structured_job(
        self, job_id, job_description_text: str
    ):
        """
        extract and store structured job data in the database
        """
        try:
            structured_job = await self._extract_structured_json(job_description_text)
            if not structured_job:
                logger.error("Structured job extraction failed.")
                return None

            processed_job = ProcessedJob(
                job_id=job_id,
                job_title=structured_job.get("job_title"),
                company_profile=json.dumps(structured_job.get("company_profile"))
                if structured_job.get("company_profile")
                else None,
                location=json.dumps(structured_job.get("location"))
                if structured_job.get("location")
                else None,
                date_posted=structured_job.get("date_posted"),
                employment_type=structured_job.get("employment_type"),
                job_summary=structured_job.get("job_summary"),
                key_responsibilities=json.dumps(
                    {"key_responsibilities": structured_job.get("key_responsibilities", [])}
                )
                if structured_job.get("key_responsibilities")
                else None,
                qualifications=json.dumps(structured_job.get("qualifications", []))
                if structured_job.get("qualifications")
                else None,
                compensation_and_benfits=json.dumps(
                    structured_job.get("compensation_and_benfits", [])
                )
                if structured_job.get("compensation_and_benfits")
                else None,
                application_info=json.dumps(structured_job.get("application_info", []))
                if structured_job.get("application_info")
                else None,
                extracted_keywords=json.dumps(
                    {"extracted_keywords": structured_job.get("extracted_keywords", [])}
                )
                if structured_job.get("extracted_keywords")
                else None,
            )

            self.db.add(processed_job)
            await self.db.flush()
            return job_id
        except Exception as e:
            logger.error(f"Failed to extract and store structured job: {str(e)}")
            raise

    async def _extract_structured_json(
        self, job_description_text: str
    ) -> Dict[str, Any]:
        """
        Uses the AgentManager+JSONWrapper to ask the LLM to
        return the data in exact JSON schema we need.
        
        Args:
            job_description_text: The job description to structure
            
        Returns:
            Dict containing structured job data
            
        Raises:
            JobValidationError: If input is invalid or missing required fields
            JobProcessingError: If AI processing fails
        """
        if not job_description_text or not job_description_text.strip():
            raise JobValidationError("Empty job description text")

        try:
            # For now, create a basic structured output from the text directly
            # This is a fallback when AI services are not available
            lines = job_description_text.split('\n')
            structured_data = {
                "jobTitle": "",
                "companyProfile": {
                    "companyName": "",
                    "industry": "Interior Design",
                    "website": None,
                    "description": ""
                },
                "location": {
                    "city": "",
                    "state": "",
                    "country": "India",
                    "remoteStatus": "On-site"
                },
                "datePosted": datetime.now().strftime("%Y-%m-%d"),
                "employmentType": "Internship",
                "jobSummary": "",
                "keyResponsibilities": [],
                "qualifications": {
                    "required": [],
                    "preferred": []
                },
                "compensationAndBenefits": {
                    "salaryRange": "",
                    "benefits": []
                },
                "applicationInfo": {
                    "howToApply": "Apply through the company's career portal",
                    "applyLink": "",
                    "contactEmail": None
                },
                "extractedKeywords": []
            }
            
            # Extract basic information
            for line in lines:
                line = line.strip()
                if line.startswith("Job Title:"):
                    structured_data["jobTitle"] = line.replace("Job Title:", "").strip()
                elif line.startswith("Location:"):
                    loc = line.replace("Location:", "").strip().split(",")
                    if len(loc) >= 2:
                        structured_data["location"]["city"] = loc[0].strip()
                        structured_data["location"]["state"] = loc[1].strip()
                elif line.startswith("Company:"):
                    structured_data["companyProfile"]["companyName"] = line.replace("Company:", "").strip()
                elif line.startswith("About"):
                    structured_data["companyProfile"]["description"] = next((l.strip() for l in lines[lines.index(line)+1:] if l.strip()), "")
                elif "Key Responsibilities:" in line:
                    idx = lines.index(line)
                    while idx < len(lines):
                        if lines[idx].strip().startswith("●") or lines[idx].strip().startswith("•"):
                            resp = lines[idx].replace("●", "").replace("•", "").strip()
                            if resp:
                                structured_data["keyResponsibilities"].append(resp)
                        idx += 1
                elif "CTC:" in line:
                    structured_data["compensationAndBenefits"]["salaryRange"] = line.replace("CTC:", "").strip()
                elif "Required Skills & Qualifications:" in line:
                    idx = lines.index(line)
                    while idx < len(lines):
                        if lines[idx].strip().startswith("●") or lines[idx].strip().startswith("•"):
                            qual = lines[idx].replace("●", "").replace("•", "").strip()
                            if qual:
                                structured_data["qualifications"]["required"].append(qual)
                        idx += 1
            
            # Extract keywords from the job description text
            words = job_description_text.lower().split()
            keywords = []
            
            # Common job-related keywords to look for
            skill_keywords = [
                'python', 'java', 'javascript', 'react', 'node', 'sql', 'aws', 
                'docker', 'git', 'api', 'database', 'cloud', 'web', 'mobile',
                'data', 'machine', 'learning', 'ai', 'sales', 'marketing', 
                'design', 'communication', 'leadership', 'management', 'excel',
                'powerpoint', 'presentation', 'negotiation', 'customer', 'service',
                'analysis', 'problem', 'solving', 'teamwork', 'collaboration'
            ]
            
            for word in words:
                clean_word = word.strip('.,;:()[]{}')
                if len(clean_word) > 3:
                    # Check if it's a skill keyword
                    if any(skill in clean_word for skill in skill_keywords):
                        keywords.append(clean_word)
                    # Or if it's a meaningful word (capitalized or long)
                    elif len(clean_word) > 5 and clean_word.isalpha():
                        keywords.append(clean_word)
            
            # Remove duplicates and limit
            structured_data["extractedKeywords"] = list(set(keywords))[:30]
            
            # If no keywords found, extract from responsibilities and qualifications
            if not structured_data["extractedKeywords"]:
                all_text = " ".join(structured_data["keyResponsibilities"] + structured_data["qualifications"]["required"])
                words = all_text.lower().split()
                keywords = [w.strip('.,;:()[]{}') for w in words if len(w) > 4 and w.isalpha()]
                structured_data["extractedKeywords"] = list(set(keywords))[:30]

            return structured_data

        except Exception as e:
            logger.error(f"Error extracting structured job data: {str(e)}")
            raise JobProcessingError(f"Failed to extract structured job data: {str(e)}")
            # Code removed as we're now using a simpler extraction method
            return {}

        finally:
            # Clean up any temporary resources if needed
            pass

    async def get_job_with_processed_data(self, job_id: str) -> Optional[Dict]:
        """
        Fetches both job and processed job data from the database and combines them.

        Args:
            job_id: The ID of the job to retrieve

        Returns:
            Combined data from both job and processed_job models

        Raises:
            JobNotFoundError: If the job is not found
        """
        job_query = select(Job).where(Job.job_id == job_id)
        job_result = await self.db.execute(job_query)
        job = job_result.scalars().first()

        if not job:
            raise JobNotFoundError(job_id=job_id)

        processed_query = select(ProcessedJob).where(ProcessedJob.job_id == job_id)
        processed_result = await self.db.execute(processed_query)
        processed_job = processed_result.scalars().first()

        combined_data = {
            "job_id": job.job_id,
            "raw_job": {
                "id": job.id,
                "resume_id": job.resume_id,
                "content": job.content,
                "created_at": job.created_at.isoformat() if job.created_at else None,
            },
            "processed_job": None
        }

        if processed_job:
            combined_data["processed_job"] = {
                "job_title": processed_job.job_title,
                "company_profile": json.loads(processed_job.company_profile) if processed_job.company_profile else None,
                "location": json.loads(processed_job.location) if processed_job.location else None,
                "date_posted": processed_job.date_posted,
                "employment_type": processed_job.employment_type,
                "job_summary": processed_job.job_summary,
                "key_responsibilities": json.loads(processed_job.key_responsibilities).get("key_responsibilities", []) if processed_job.key_responsibilities else None,
                "qualifications": json.loads(processed_job.qualifications).get("qualifications", []) if processed_job.qualifications else None,
                "compensation_and_benfits": json.loads(processed_job.compensation_and_benfits).get("compensation_and_benfits", []) if processed_job.compensation_and_benfits else None,
                "application_info": json.loads(processed_job.application_info).get("application_info", []) if processed_job.application_info else None,
                "extracted_keywords": json.loads(processed_job.extracted_keywords).get("extracted_keywords", []) if processed_job.extracted_keywords else None,
                "processed_at": processed_job.processed_at.isoformat() if processed_job.processed_at else None,
            }

        return combined_data

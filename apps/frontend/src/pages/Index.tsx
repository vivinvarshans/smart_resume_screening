import { useState } from "react";
import { Button } from "@/components/ui/button";
import { Upload, FileText, BarChart3, Sparkles, X } from "lucide-react";
import ResumeUpload from "@/components/ResumeUpload";
import JobDescriptionInput from "@/components/JobDescriptionInput";
import MatchDashboard from "@/components/MatchDashboard";

type Step = "hero" | "upload" | "job" | "match";

const Index = () => {
  const [currentStep, setCurrentStep] = useState<Step>("hero");
  const [showHowItWorks, setShowHowItWorks] = useState(false);
  const [resumeId, setResumeId] = useState<string>(() => {
    // Try to recover resumeId from localStorage on initial load
    return localStorage.getItem('lastResumeId') || "";
  });
  const [jobId, setJobId] = useState<string>("");

  const handleResumeUploaded = (id: string) => {
    if (!id) {
      console.error('No resume ID provided');
      return;
    }
    console.log('Setting resume ID:', id); // Debug log
    setResumeId(id);
    localStorage.setItem('lastResumeId', id);
    setCurrentStep("job");
  };

  const handleJobSaved = (id: string) => {
    setJobId(id);
    setCurrentStep("match");
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-background via-primary/5 to-accent/5">
      {/* Hero Section */}
      {currentStep === "hero" && (
        <div className="relative min-h-screen flex items-center justify-center overflow-hidden">
          {/* Gradient Background */}
          <div className="absolute inset-0 gradient-hero opacity-10" />

          {/* Floating Elements */}
          <div className="absolute top-20 left-10 w-72 h-72 bg-primary/20 rounded-full blur-3xl animate-float" />
          <div className="absolute bottom-20 right-10 w-96 h-96 bg-accent/20 rounded-full blur-3xl animate-float" style={{ animationDelay: "1s" }} />

          {/* Content */}
          <div className="relative z-10 max-w-6xl mx-auto px-6 text-center">
            <div className="animate-fade-in-up">
              <div className="inline-flex items-center gap-2 px-4 py-2 rounded-full glass-card mb-8">
                <Sparkles className="w-4 h-4 text-primary" />
                <span className="text-sm font-medium">AI-Powered Resume Analysis</span>
              </div>

              <h1 className="text-6xl md:text-7xl font-bold mb-6 text-black dark:text-white leading-tight">
                Stop Getting Auto-Rejected by ATS Bots
              </h1>

              <p className="text-xl md:text-2xl text-muted-foreground mb-12 max-w-3xl mx-auto">
                Upload your resume, match it with job descriptions, and get AI-powered insights to beat applicant tracking systems and land your dream job.
              </p>

              <div className="flex flex-col sm:flex-row gap-4 justify-center">
                <Button
                  size="lg"
                  onClick={() => setCurrentStep("upload")}
                  className="gradient-primary text-white hover:opacity-90 transition-opacity text-lg px-8 py-6 rounded-full shadow-[0_0_40px_rgba(139,92,246,0.4)] hover:shadow-[0_0_60px_rgba(139,92,246,0.6)]"
                >
                  <Upload className="mr-2 h-5 w-5" />
                  Upload Your Resume
                </Button>

                <Button
                  size="lg"
                  variant="outline"
                  onClick={() => setShowHowItWorks(true)}
                  className="glass-card hover:bg-white/80 dark:hover:bg-white/10 text-lg px-8 py-6 rounded-full border-2"
                >
                  <FileText className="mr-2 h-5 w-5" />
                  See How It Works
                </Button>
              </div>
            </div>

            {/* Feature Cards */}
            <div className="grid md:grid-cols-3 gap-6 mt-20 animate-fade-in-up" style={{ animationDelay: "0.2s" }}>
              <div className="glass-card p-6 rounded-2xl hover-scale">
                <div className="w-12 h-12 rounded-xl bg-black dark:bg-white flex items-center justify-center mb-4 mx-auto">
                  <Upload className="h-6 w-6 text-white dark:text-black" />
                </div>
                <h3 className="font-semibold text-lg mb-2">Upload Resume</h3>
                <p className="text-sm text-muted-foreground">
                  Drag and drop your PDF or DOCX resume for instant analysis
                </p>
              </div>

              <div className="glass-card p-6 rounded-2xl hover-scale" style={{ animationDelay: "0.1s" }}>
                <div className="w-12 h-12 rounded-xl bg-black dark:bg-white flex items-center justify-center mb-4 mx-auto">
                  <FileText className="h-6 w-6 text-white dark:text-black" />
                </div>
                <h3 className="font-semibold text-lg mb-2">Add Job Description</h3>
                <p className="text-sm text-muted-foreground">
                  Paste the job posting you want to match against
                </p>
              </div>

              <div className="glass-card p-6 rounded-2xl hover-scale" style={{ animationDelay: "0.2s" }}>
                <div className="w-12 h-12 rounded-xl bg-black dark:bg-white flex items-center justify-center mb-4 mx-auto">
                  <BarChart3 className="h-6 w-6 text-white dark:text-black" />
                </div>
                <h3 className="font-semibold text-lg mb-2">Get AI Insights</h3>
                <p className="text-sm text-muted-foreground">
                  Receive match scores and actionable improvement suggestions
                </p>
              </div>
            </div>
          </div>
        </div>
      )}

      {/* How It Works Modal */}
      {showHowItWorks && (
        <div className="fixed inset-0 bg-black/50 backdrop-blur-sm z-50 flex items-center justify-center p-6">
          <div className="glass-card max-w-4xl w-full rounded-3xl p-8 animate-fade-in-up relative max-h-[90vh] overflow-y-auto">
            <button
              onClick={() => setShowHowItWorks(false)}
              className="absolute top-4 right-4 p-2 rounded-full hover:bg-black/10 dark:hover:bg-white/10 transition-colors"
            >
              <X className="h-6 w-6" />
            </button>

            <h2 className="text-4xl font-bold mb-6 text-black dark:text-white">How It Works</h2>
            <p className="text-lg text-muted-foreground mb-8">
              Our AI-powered system helps you optimize your resume to pass ATS (Applicant Tracking Systems) and land more interviews.
            </p>

            <div className="space-y-8">
              {/* Step 1 */}
              <div className="flex gap-6">
                <div className="flex-shrink-0">
                  <div className="w-16 h-16 rounded-xl bg-black dark:bg-white flex items-center justify-center">
                    <Upload className="h-8 w-8 text-white dark:text-black" />
                  </div>
                </div>
                <div>
                  <h3 className="text-2xl font-bold mb-2">1. Upload Your Resume</h3>
                  <p className="text-muted-foreground">
                    Upload your resume in PDF or DOCX format (max 2MB). Our AI will extract and analyze your skills, experience, and qualifications using advanced NLP algorithms.
                  </p>
                </div>
              </div>

              {/* Step 2 */}
              <div className="flex gap-6">
                <div className="flex-shrink-0">
                  <div className="w-16 h-16 rounded-xl bg-black dark:bg-white flex items-center justify-center">
                    <FileText className="h-8 w-8 text-white dark:text-black" />
                  </div>
                </div>
                <div>
                  <h3 className="text-2xl font-bold mb-2">2. Add Job Description</h3>
                  <p className="text-muted-foreground">
                    Paste the complete job description you're applying for. Include requirements, responsibilities, and qualifications for best results.
                  </p>
                </div>
              </div>

              {/* Step 3 */}
              <div className="flex gap-6">
                <div className="flex-shrink-0">
                  <div className="w-16 h-16 rounded-xl bg-black dark:bg-white flex items-center justify-center">
                    <BarChart3 className="h-8 w-8 text-white dark:text-black" />
                  </div>
                </div>
                <div>
                  <h3 className="text-2xl font-bold mb-2">3. Get AI-Powered Analysis</h3>
                  <p className="text-muted-foreground mb-4">
                    Our system uses TF-IDF, Cosine Similarity, and Groq LLM to analyze your resume against the job requirements. You'll receive:
                  </p>
                  <ul className="space-y-2 text-muted-foreground">
                    <li className="flex gap-2">
                      <span className="text-black dark:text-white">✓</span>
                      <span><strong>Match Score (1-10):</strong> Overall compatibility rating</span>
                    </li>
                    <li className="flex gap-2">
                      <span className="text-black dark:text-white">✓</span>
                      <span><strong>Keyword Analysis:</strong> Matched and missing skills</span>
                    </li>
                    <li className="flex gap-2">
                      <span className="text-black dark:text-white">✓</span>
                      <span><strong>Strengths & Gaps:</strong> What you're doing well and what to improve</span>
                    </li>
                    <li className="flex gap-2">
                      <span className="text-black dark:text-white">✓</span>
                      <span><strong>AI-Improved Resume:</strong> Optimized version tailored to the job</span>
                    </li>
                  </ul>
                </div>
              </div>

              {/* Technology */}
              <div className="glass-card p-6 rounded-2xl bg-black/5 dark:bg-white/5">
                <h3 className="text-xl font-bold mb-3">🚀 Powered by Advanced AI</h3>
                <p className="text-muted-foreground mb-3">
                  We use the same algorithms as professional ATS systems:
                </p>
                <ul className="grid md:grid-cols-2 gap-2 text-sm text-muted-foreground">
                  <li>• TF-IDF Vectorization</li>
                  <li>• Cosine Similarity Matching</li>
                  <li>• Groq LLM (Llama 3.3)</li>
                  <li>• Named Entity Recognition</li>
                  <li>• Semantic Analysis</li>
                  <li>• Keyword Importance Weighting</li>
                </ul>
              </div>
            </div>

            <div className="mt-8 flex justify-center">
              <Button
                size="lg"
                onClick={() => {
                  setShowHowItWorks(false);
                  setCurrentStep("upload");
                }}
                className="gradient-primary text-white hover:opacity-90 transition-opacity px-8 py-6 text-lg rounded-full"
              >
                Get Started Now
              </Button>
            </div>
          </div>
        </div>
      )}

      {/* Resume Upload Step */}
      {currentStep === "upload" && (
        <ResumeUpload
          onResumeUploaded={handleResumeUploaded}
          onBack={() => setCurrentStep("hero")}
        />
      )}

      {/* Job Description Step */}
      {currentStep === "job" && (
        <JobDescriptionInput
          resumeId={resumeId}
          onJobSaved={handleJobSaved}
          onBack={() => setCurrentStep("upload")}
        />
      )}

      {/* Match Dashboard Step */}
      {currentStep === "match" && resumeId && jobId && (
        <MatchDashboard resumeId={resumeId} jobId={jobId} onBack={() => setCurrentStep("job")} />
      )}

      {/* Fallback if match step is reached without required data */}
      {currentStep === "match" && (!resumeId || !jobId) && (
        <div className="min-h-screen flex items-center justify-center p-6">
          <div className="text-center">
            <h3 className="text-2xl font-bold mb-4">Missing Information</h3>
            <p className="text-muted-foreground mb-6">
              Please complete all steps before viewing the analysis.
            </p>
            <Button onClick={() => setCurrentStep(resumeId ? "job" : "upload")}>
              Go Back
            </Button>
          </div>
        </div>
      )}
    </div>
  );
};

export default Index;

import { useState } from "react";
import { Button } from "@/components/ui/button";
import { Textarea } from "@/components/ui/textarea";
import { Loader2, Briefcase, ArrowRight, ArrowLeft } from "lucide-react";
import { useToast } from "@/hooks/use-toast";
import axios from "axios";

interface JobDescriptionInputProps {
  resumeId: string;
  onJobSaved: (jobId: string) => void;
  onBack?: () => void;
}

const API_BASE = "http://localhost:8000/api/v1";

const JobDescriptionInput = ({ resumeId, onJobSaved, onBack }: JobDescriptionInputProps) => {
  const [jobDescription, setJobDescription] = useState("");
  const [saving, setSaving] = useState(false);
  const { toast } = useToast();

  const handleSubmit = async () => {
    if (!jobDescription.trim()) {
      toast({
        title: "Job description required",
        description: "Please paste a job description to continue",
        variant: "destructive",
      });
      return;
    }

    if (!resumeId) {
      const savedResumeId = localStorage.getItem('lastResumeId');
      if (!savedResumeId) {
        toast({
          title: "Resume required",
          description: "Please upload your resume first",
          variant: "destructive",
        });
        return;
      }
    }

    setSaving(true);

    try {
      // Validate UUID format
      if (!/^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$/i.test(resumeId)) {
        toast({
          title: "Invalid resume ID",
          description: "Please upload your resume first",
          variant: "destructive",
        });
        setSaving(false);
        return;
      }

      const response = await axios.post(`${API_BASE}/jobs/upload`, {
        job_descriptions: [jobDescription],
        resume_id: resumeId,
      }, {
        headers: {
          'Content-Type': 'application/json'
        }
      });

      toast({
        title: "Job description saved!",
        description: "Analyzing match with your resume...",
      });

      // Get the first job ID from the response
      const jobIds = response.data.job_ids;
      if (!jobIds || jobIds.length === 0) {
        throw new Error("No job IDs returned from server");
      }
      
      const jobId = jobIds[0]; // We're only uploading one job description
      
      setTimeout(() => {
        onJobSaved(jobId);
      }, 1000);
    } catch (error: any) {
      console.error("Save error:", error);
      let errorMessage = "Please try again or check your connection";
      
      if (error.response) {
        // Server returned an error response
        errorMessage = error.response.data?.detail || 
                      error.response.data?.message ||
                      `Server error: ${error.response.status}`;
      } else if (error.request) {
        // Request was made but no response received
        errorMessage = "No response from server. Please check your connection.";
      } else {
        // Something else happened while setting up the request
        errorMessage = error.message || "Failed to send request";
      }
      
      toast({
        title: "Save failed",
        description: errorMessage,
        variant: "destructive",
      });
      setSaving(false);
    }
  };

  return (
    <div className="min-h-screen flex items-center justify-center p-6">
      <div className="max-w-4xl w-full">
        {onBack && (
          <div className="mb-6 animate-fade-in-up">
            <Button variant="outline" onClick={onBack} className="rounded-full px-6">
              <ArrowLeft className="mr-2 h-4 w-4" />
              Back
            </Button>
          </div>
        )}
        
        <div className="text-center mb-8 animate-fade-in-up">
          <div className="w-16 h-16 rounded-full gradient-primary flex items-center justify-center mx-auto mb-6">
            <Briefcase className="h-8 w-8 text-white" />
          </div>
          <h2 className="text-4xl font-bold mb-4">Add Job Description</h2>
          <p className="text-lg text-muted-foreground">
            Paste the job posting you want to match your resume against
          </p>
        </div>

        <div className="glass-card p-8 rounded-3xl animate-fade-in-up" style={{ animationDelay: "0.1s" }}>
          <div className="space-y-6">
            <div>
              <label className="block text-sm font-medium mb-3">
                Job Description
              </label>
              <Textarea
                placeholder="Paste the full job description here including requirements, responsibilities, and qualifications..."
                value={jobDescription}
                onChange={(e) => setJobDescription(e.target.value)}
                className="min-h-[400px] resize-none text-base glass-card border-2"
                disabled={saving}
              />
              <div className="flex justify-between mt-2 text-sm text-muted-foreground">
                <span>Markdown formatting supported</span>
                <span>{jobDescription.length} characters</span>
              </div>
            </div>

            <Button
              onClick={handleSubmit}
              disabled={saving || !jobDescription.trim()}
              size="lg"
              className="w-full gradient-primary text-white hover:opacity-90 transition-opacity py-6 text-lg rounded-full shadow-[0_0_40px_rgba(139,92,246,0.4)]"
            >
              {saving ? (
                <>
                  <Loader2 className="mr-2 h-5 w-5 animate-spin" />
                  Analyzing...
                </>
              ) : (
                <>
                  Continue to Match Analysis
                  <ArrowRight className="ml-2 h-5 w-5" />
                </>
              )}
            </Button>
          </div>
        </div>

        <div className="mt-8 glass-card p-6 rounded-2xl animate-fade-in-up" style={{ animationDelay: "0.2s" }}>
          <h3 className="font-semibold mb-3 flex items-center gap-2">
            <Briefcase className="h-5 w-5 text-primary" />
            Tips for best results:
          </h3>
          <ul className="space-y-2 text-sm text-muted-foreground">
            <li className="flex gap-2">
              <span className="text-primary">•</span>
              <span>Include the complete job description with all sections</span>
            </li>
            <li className="flex gap-2">
              <span className="text-primary">•</span>
              <span>Make sure requirements and qualifications are clearly listed</span>
            </li>
            <li className="flex gap-2">
              <span className="text-primary">•</span>
              <span>You can format using Markdown (headers, bullets, etc.)</span>
            </li>
          </ul>
        </div>
      </div>
    </div>
  );
};

export default JobDescriptionInput;

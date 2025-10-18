import { useState, useCallback } from "react";
import { useDropzone } from "react-dropzone";
import { Upload, FileText, Check, Loader2, ArrowLeft } from "lucide-react";
import { Button } from "@/components/ui/button";
import { useToast } from "@/hooks/use-toast";
import axios from "axios";

interface ResumeUploadProps {
  onResumeUploaded: (resumeId: string) => void;
  onBack?: () => void;
}

const API_BASE = "http://localhost:8000/api/v1";

const ResumeUpload = ({ onResumeUploaded, onBack }: ResumeUploadProps) => {
  const [uploading, setUploading] = useState(false);
  const [uploaded, setUploaded] = useState(false);
  const { toast } = useToast();

  const onDrop = useCallback(async (acceptedFiles: File[]) => {
    const file = acceptedFiles[0];
    
    if (!file) return;

    // Validate file size (2MB limit)
    if (file.size > 2 * 1024 * 1024) {
      toast({
        title: "File too large",
        description: "Please upload a file smaller than 2MB",
        variant: "destructive",
      });
      return;
    }

    // Validate file type
    const validTypes = ["application/pdf", "application/vnd.openxmlformats-officedocument.wordprocessingml.document"];
    if (!validTypes.includes(file.type)) {
      toast({
        title: "Invalid file type",
        description: "Please upload a PDF or DOCX file",
        variant: "destructive",
      });
      return;
    }

    setUploading(true);

    try {
      const formData = new FormData();
      formData.append("file", file);

      console.log('Uploading file:', file.name, 'Type:', file.type); // Debug log

      const response = await axios.post(`${API_BASE}/resumes/upload`, formData, {
        headers: {
          "Content-Type": "multipart/form-data",
        },
        timeout: 300000, // 5 minutes timeout for AI processing
      });

      console.log('Server response:', response.data); // Debug log

      const resumeId = response.data?.resume_id;
      if (!resumeId || typeof resumeId !== 'string') {
        console.error('Invalid resume ID in response:', response.data); // Debug log
        throw new Error('No valid resume ID received from server');
      }

      setUploaded(true);
      toast({
        title: "Resume uploaded successfully!",
        description: "Extracting skills and experience...",
      });

      // Store resume ID in localStorage as a backup
      localStorage.setItem('lastResumeId', resumeId);

      console.log('Resume ID stored:', resumeId); // Debug log

      // Wait a moment to show success state
      setTimeout(() => {
        onResumeUploaded(resumeId);
      }, 1500);
    } catch (error: any) {
      console.error("Upload error:", error);
      let errorMessage = "Please try again or check your connection";
      
      if (error.response) {
        // Get the specific error message from the backend if available
        errorMessage = error.response.data?.detail || errorMessage;
        console.error('Server error response:', error.response.data); // Debug log
      }

      toast({
        title: "Upload failed",
        description: errorMessage,
        variant: "destructive",
      });

      setUploaded(false);
    } finally {
      setUploading(false);
    }
  }, [onResumeUploaded, toast]);

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: {
      "application/pdf": [".pdf"],
      "application/vnd.openxmlformats-officedocument.wordprocessingml.document": [".docx"],
    },
    maxFiles: 1,
    disabled: uploading || uploaded,
  });

  return (
    <div className="min-h-screen flex items-center justify-center p-6">
      <div className="max-w-3xl w-full">
        {onBack && (
          <div className="mb-6 animate-fade-in-up">
            <Button variant="outline" onClick={onBack} className="rounded-full px-6">
              <ArrowLeft className="mr-2 h-4 w-4" />
              Back
            </Button>
          </div>
        )}
        
        <div className="text-center mb-8 animate-fade-in-up">
          <h2 className="text-4xl font-bold mb-4">Upload Your Resume</h2>
          <p className="text-lg text-muted-foreground">
            We'll analyze your skills, experience, and achievements
          </p>
        </div>

        <div
          {...getRootProps()}
          className={`glass-card p-12 rounded-3xl border-2 border-dashed transition-all duration-300 cursor-pointer animate-fade-in-up ${
            isDragActive
              ? "border-primary bg-primary/5 scale-105"
              : "border-border hover:border-primary hover:bg-primary/5"
          } ${uploaded ? "border-success bg-success/5" : ""} ${
            uploading ? "pointer-events-none" : ""
          }`}
        >
          <input {...getInputProps()} />
          
          <div className="flex flex-col items-center text-center space-y-6">
            {uploading ? (
              <>
                <div className="relative">
                  <div className="w-20 h-20 rounded-full gradient-primary flex items-center justify-center animate-pulse">
                    <Loader2 className="h-10 w-10 text-white animate-spin" />
                  </div>
                </div>
                <div>
                  <p className="text-xl font-semibold">Uploading...</p>
                  <p className="text-sm text-muted-foreground mt-2">
                    Extracting resume data with AI
                  </p>
                </div>
              </>
            ) : uploaded ? (
              <>
                <div className="relative">
                  <div className="w-20 h-20 rounded-full bg-success flex items-center justify-center animate-scale-in">
                    <Check className="h-10 w-10 text-white" />
                  </div>
                </div>
                <div>
                  <p className="text-xl font-semibold text-success">Upload Complete!</p>
                  <p className="text-sm text-muted-foreground mt-2">
                    Redirecting to job description...
                  </p>
                </div>
              </>
            ) : (
              <>
                <div className="relative">
                  <div className="w-20 h-20 rounded-full gradient-primary flex items-center justify-center">
                    {isDragActive ? (
                      <FileText className="h-10 w-10 text-white" />
                    ) : (
                      <Upload className="h-10 w-10 text-white" />
                    )}
                  </div>
                </div>
                
                <div>
                  <p className="text-xl font-semibold mb-2">
                    {isDragActive
                      ? "Drop your resume here"
                      : "Drag and drop your resume"}
                  </p>
                  <p className="text-sm text-muted-foreground">
                    or click to browse files
                  </p>
                </div>

                <div className="flex items-center gap-4 text-sm text-muted-foreground">
                  <span className="flex items-center gap-1">
                    <FileText className="h-4 w-4" />
                    PDF or DOCX
                  </span>
                  <span>•</span>
                  <span>Max 2MB</span>
                </div>
              </>
            )}
          </div>
        </div>

        <div className="mt-8 text-center animate-fade-in-up" style={{ animationDelay: "0.2s" }}>
          <p className="text-sm text-muted-foreground">
            Your resume is processed securely and never shared with third parties
          </p>
        </div>
      </div>
    </div>
  );
};

export default ResumeUpload;

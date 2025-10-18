import { useState, useEffect } from "react";
import { Button } from "@/components/ui/button";
import { Loader2, ArrowLeft, TrendingUp, AlertTriangle, CheckCircle2, XCircle } from "lucide-react";
import { useToast } from "@/hooks/use-toast";
import axios from "axios";
import ScoreCircle from "./ScoreCircle";
import ImprovementCard from "./ImprovementCard";
import KeywordAnalysis from "./KeywordAnalysis";

interface MatchDashboardProps {
  resumeId: string;
  jobId: string;
  onBack: () => void;
}

interface MatchData {
  match_score: number;
  improvements: Array<{
    section: string;
    current: string;
    suggested: string;
    reason: string;
  }>;
  missing_keywords: string[];
  matched_keywords: string[];
  strengths: string[];
  gaps: string[];
}

const API_BASE = "http://localhost:8000/api/v1";

const MatchDashboard = ({ resumeId, jobId, onBack }: MatchDashboardProps) => {
  const [loading, setLoading] = useState(true);
  const [matchData, setMatchData] = useState<MatchData | null>(null);
  const { toast } = useToast();

  useEffect(() => {
    const fetchMatchData = async () => {
      // Validate that we have both IDs before making the request
      if (!resumeId || !jobId) {
        console.error("Missing required IDs:", { resumeId, jobId });
        toast({
          title: "Missing information",
          description: "Please upload your resume and job description first",
          variant: "destructive",
        });
        setLoading(false);
        return;
      }

      try {
        const response = await axios.post(`${API_BASE}/resumes/improve?stream=false`, {
          resume_id: resumeId,
          job_id: jobId,
        });

        setMatchData(response.data.data);
        toast({
          title: "Analysis complete!",
          description: "Your match score and insights are ready",
        });
      } catch (error) {
        console.error("Match error:", error);
        toast({
          title: "Analysis failed",
          description: "Please try again or check your connection",
          variant: "destructive",
        });
      } finally {
        setLoading(false);
      }
    };

    fetchMatchData();
  }, [resumeId, jobId, toast]);

  if (loading) {
    return (
      <div className="min-h-screen flex items-center justify-center">
        <div className="text-center space-y-6">
          <div className="relative">
            <div className="w-24 h-24 rounded-full gradient-primary flex items-center justify-center animate-pulse mx-auto">
              <Loader2 className="h-12 w-12 text-white animate-spin" />
            </div>
          </div>
          <div>
            <h3 className="text-2xl font-bold mb-2">Analyzing Your Resume</h3>
            <p className="text-muted-foreground">
              Our AI is comparing your resume with the job requirements...
            </p>
          </div>
        </div>
      </div>
    );
  }

  if (!matchData) {
    return (
      <div className="min-h-screen flex items-center justify-center p-6">
        <div className="text-center">
          <XCircle className="h-16 w-16 text-destructive mx-auto mb-4" />
          <h3 className="text-2xl font-bold mb-2">Analysis Failed</h3>
          <p className="text-muted-foreground mb-6">
            We couldn't analyze your resume. Please try again.
          </p>
          <Button onClick={onBack} className="rounded-full px-6">
            <ArrowLeft className="mr-2 h-4 w-4" />
            Go Back
          </Button>
        </div>
      </div>
    );
  }

  const getScoreColor = (score: number) => {
    if (score >= 80) return "success";
    if (score >= 50) return "warning";
    return "destructive";
  };

  const scoreColor = getScoreColor(matchData.match_score);

  return (
    <div className="min-h-screen p-6">
      <div className="max-w-7xl mx-auto">
        {/* Header */}
        <div className="mb-8 animate-fade-in-up">
          <Button variant="ghost" onClick={onBack} className="mb-4 rounded-full px-6">
            <ArrowLeft className="mr-2 h-4 w-4" />
            Back
          </Button>
          <h1 className="text-4xl font-bold mb-2">Match Analysis</h1>
          <p className="text-lg text-muted-foreground">
            Here's how your resume matches the job requirements
          </p>
        </div>

        {/* Main Score Section */}
        <div className="grid lg:grid-cols-3 gap-6 mb-8">
          <div className="lg:col-span-1 glass-card p-8 rounded-3xl animate-fade-in-up flex flex-col items-center justify-center">
            <ScoreCircle score={matchData.match_score} />
            <h3 className="text-2xl font-bold mt-6 mb-2">Overall Match</h3>
            <p className="text-center text-muted-foreground">
              {scoreColor === "success" && "Excellent match! You're well-qualified."}
              {scoreColor === "warning" && "Good match with room for improvement."}
              {scoreColor === "destructive" && "Consider enhancing your resume."}
            </p>
          </div>

          {/* Quick Stats */}
          <div className="lg:col-span-2 grid sm:grid-cols-2 gap-6">
            <div className="glass-card p-6 rounded-3xl animate-fade-in-up hover-scale" style={{ animationDelay: "0.1s" }}>
              <div className="flex items-center gap-3 mb-3">
                <div className="w-12 h-12 rounded-xl bg-success flex items-center justify-center">
                  <CheckCircle2 className="h-6 w-6 text-white" />
                </div>
                <h4 className="font-semibold">Strengths</h4>
              </div>
              <ul className="space-y-2">
                {matchData.strengths.slice(0, 3).map((strength, idx) => (
                  <li key={idx} className="text-sm text-muted-foreground flex gap-2">
                    <span className="text-success">✓</span>
                    <span>{strength}</span>
                  </li>
                ))}
              </ul>
            </div>

            <div className="glass-card p-6 rounded-3xl animate-fade-in-up hover-scale" style={{ animationDelay: "0.2s" }}>
              <div className="flex items-center gap-3 mb-3">
                <div className="w-12 h-12 rounded-xl bg-warning flex items-center justify-center">
                  <AlertTriangle className="h-6 w-6 text-white" />
                </div>
                <h4 className="font-semibold">Areas to Improve</h4>
              </div>
              <ul className="space-y-2">
                {matchData.gaps.slice(0, 3).map((gap, idx) => (
                  <li key={idx} className="text-sm text-muted-foreground flex gap-2">
                    <span className="text-warning">!</span>
                    <span>{gap}</span>
                  </li>
                ))}
              </ul>
            </div>

            <div className="glass-card p-6 rounded-3xl animate-fade-in-up hover-scale sm:col-span-2" style={{ animationDelay: "0.3s" }}>
              <div className="flex items-center gap-3 mb-3">
                <div className="w-12 h-12 rounded-xl gradient-primary flex items-center justify-center">
                  <TrendingUp className="h-6 w-6 text-white" />
                </div>
                <h4 className="font-semibold">Keyword Coverage</h4>
              </div>
              <div className="flex items-center gap-4">
                <div className="flex-1">
                  <div className="flex justify-between text-sm mb-2">
                    <span className="text-success">Matched: {matchData.matched_keywords.length}</span>
                    <span className="text-destructive">Missing: {matchData.missing_keywords.length}</span>
                  </div>
                  <div className="h-3 bg-secondary rounded-full overflow-hidden">
                    <div
                      className="h-full bg-gradient-to-r from-success to-primary transition-all duration-1000"
                      style={{
                        width: `${(matchData.matched_keywords.length / (matchData.matched_keywords.length + matchData.missing_keywords.length)) * 100}%`,
                      }}
                    />
                  </div>
                </div>
              </div>
            </div>
          </div>
        </div>

        {/* Keyword Analysis */}
        <KeywordAnalysis
          matchedKeywords={matchData.matched_keywords}
          missingKeywords={matchData.missing_keywords}
        />

        {/* Improvements Section */}
        <div className="mb-8">
          <h2 className="text-3xl font-bold mb-6 animate-fade-in-up">Suggested Improvements</h2>
          <div className="space-y-4">
            {matchData.improvements.map((improvement, idx) => (
              <ImprovementCard key={idx} improvement={improvement} index={idx} />
            ))}
          </div>
        </div>

        {/* Action Buttons */}
        <div className="flex gap-4 justify-center animate-fade-in-up">
          <Button
            size="lg"
            className="gradient-primary text-white hover:opacity-90 transition-opacity px-8 py-6 text-lg rounded-full"
          >
            Export Report
          </Button>
          <Button
            size="lg"
            variant="outline"
            className="glass-card hover:bg-white/80 dark:hover:bg-white/10 px-8 py-6 text-lg rounded-full border-2"
            onClick={onBack}
          >
            Try Another Job
          </Button>
        </div>
      </div>
    </div>
  );
};

export default MatchDashboard;

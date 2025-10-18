import { Badge } from "@/components/ui/badge";

interface KeywordAnalysisProps {
  matchedKeywords: string[];
  missingKeywords: string[];
}

const KeywordAnalysis = ({ matchedKeywords, missingKeywords }: KeywordAnalysisProps) => {
  return (
    <div className="mb-8">
      <h2 className="text-3xl font-bold mb-6 animate-fade-in-up">Keyword Analysis</h2>
      
      <div className="grid lg:grid-cols-2 gap-6">
        {/* Matched Keywords */}
        <div className="glass-card p-6 rounded-3xl animate-fade-in-up">
          <div className="flex items-center gap-3 mb-4">
            <div className="w-10 h-10 rounded-lg bg-success flex items-center justify-center">
              <span className="text-white font-bold">✓</span>
            </div>
            <div>
              <h3 className="font-semibold text-lg">Matched Keywords</h3>
              <p className="text-sm text-muted-foreground">
                {matchedKeywords.length} keywords found in your resume
              </p>
            </div>
          </div>
          <div className="flex flex-wrap gap-2">
            {matchedKeywords.map((keyword, idx) => (
              <Badge
                key={idx}
                variant="outline"
                className="px-3 py-1.5 bg-success/10 border-success/30 text-success hover:bg-success/20 transition-colors"
              >
                {keyword}
              </Badge>
            ))}
          </div>
        </div>

        {/* Missing Keywords */}
        <div className="glass-card p-6 rounded-3xl animate-fade-in-up" style={{ animationDelay: "0.1s" }}>
          <div className="flex items-center gap-3 mb-4">
            <div className="w-10 h-10 rounded-lg bg-destructive flex items-center justify-center">
              <span className="text-white font-bold">!</span>
            </div>
            <div>
              <h3 className="font-semibold text-lg">Missing Keywords</h3>
              <p className="text-sm text-muted-foreground">
                {missingKeywords.length} keywords to add
              </p>
            </div>
          </div>
          <div className="flex flex-wrap gap-2">
            {missingKeywords.map((keyword, idx) => (
              <Badge
                key={idx}
                variant="outline"
                className="px-3 py-1.5 bg-destructive/10 border-destructive/30 text-destructive hover:bg-destructive/20 transition-colors"
              >
                {keyword}
              </Badge>
            ))}
          </div>
        </div>
      </div>
    </div>
  );
};

export default KeywordAnalysis;

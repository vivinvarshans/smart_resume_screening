import { useState } from "react";
import { Card } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Copy, Check, ChevronDown, ChevronUp } from "lucide-react";
import { useToast } from "@/hooks/use-toast";

interface Improvement {
  section: string;
  current: string;
  suggested: string;
  reason: string;
}

interface ImprovementCardProps {
  improvement: Improvement;
  index: number;
}

const ImprovementCard = ({ improvement, index }: ImprovementCardProps) => {
  const [expanded, setExpanded] = useState(false);
  const [copied, setCopied] = useState(false);
  const { toast } = useToast();

  const handleCopy = () => {
    navigator.clipboard.writeText(improvement.suggested);
    setCopied(true);
    toast({
      title: "Copied to clipboard!",
      description: "You can now paste this into your resume",
    });
    setTimeout(() => setCopied(false), 2000);
  };

  return (
    <Card
      className="glass-card p-6 rounded-2xl hover-scale animate-fade-in-up"
      style={{ animationDelay: `${index * 0.1}s` }}
    >
      <div className="flex items-start justify-between gap-4 mb-4">
        <div className="flex-1">
          <div className="flex items-center gap-3 mb-2">
            <span className="inline-flex items-center justify-center w-8 h-8 rounded-lg gradient-primary text-white text-sm font-bold">
              {index + 1}
            </span>
            <h3 className="font-semibold text-lg">{improvement.section}</h3>
          </div>
          <p className="text-sm text-muted-foreground">{improvement.reason}</p>
        </div>
        <Button
          variant="ghost"
          size="sm"
          onClick={() => setExpanded(!expanded)}
          className="shrink-0"
        >
          {expanded ? (
            <ChevronUp className="h-5 w-5" />
          ) : (
            <ChevronDown className="h-5 w-5" />
          )}
        </Button>
      </div>

      {expanded && (
        <div className="space-y-4 animate-fade-in">
          <div className="p-4 rounded-xl bg-muted/50 border border-border">
            <p className="text-xs font-medium text-muted-foreground mb-2">CURRENT</p>
            <p className="text-sm">{improvement.current}</p>
          </div>

          <div className="p-4 rounded-xl bg-primary/5 border-2 border-primary/20">
            <div className="flex items-center justify-between mb-2">
              <p className="text-xs font-medium text-primary">SUGGESTED IMPROVEMENT</p>
              <Button
                variant="ghost"
                size="sm"
                onClick={handleCopy}
                className="h-7 px-2"
              >
                {copied ? (
                  <Check className="h-4 w-4 text-success" />
                ) : (
                  <Copy className="h-4 w-4" />
                )}
              </Button>
            </div>
            <p className="text-sm font-medium">{improvement.suggested}</p>
          </div>
        </div>
      )}
    </Card>
  );
};

export default ImprovementCard;

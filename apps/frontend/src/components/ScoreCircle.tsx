import { useEffect, useState } from "react";

interface ScoreCircleProps {
  score: number;
}

const ScoreCircle = ({ score }: ScoreCircleProps) => {
  // Convert 0-100 score to 0-10 scale
  const scoreOutOf10 = score / 10;
  const [displayScore, setDisplayScore] = useState(0);

  useEffect(() => {
    const duration = 2000;
    const steps = 60;
    const increment = scoreOutOf10 / steps;
    let current = 0;

    const timer = setInterval(() => {
      current += increment;
      if (current >= scoreOutOf10) {
        setDisplayScore(scoreOutOf10);
        clearInterval(timer);
      } else {
        setDisplayScore(current);
      }
    }, duration / steps);

    return () => clearInterval(timer);
  }, [scoreOutOf10]);

  const getColor = (score: number) => {
    if (score >= 80) return { stroke: "hsl(var(--success))", bg: "hsl(var(--success) / 0.1)" };
    if (score >= 50) return { stroke: "hsl(var(--warning))", bg: "hsl(var(--warning) / 0.1)" };
    return { stroke: "hsl(var(--destructive))", bg: "hsl(var(--destructive) / 0.1)" };
  };

  const colors = getColor(score);
  const circumference = 2 * Math.PI * 80;
  const offset = circumference - (score / 100) * circumference;

  return (
    <div className="relative w-48 h-48">
      <svg className="transform -rotate-90 w-48 h-48">
        {/* Background circle */}
        <circle
          cx="96"
          cy="96"
          r="80"
          stroke={colors.bg}
          strokeWidth="16"
          fill="none"
        />
        {/* Progress circle */}
        <circle
          cx="96"
          cy="96"
          r="80"
          stroke={colors.stroke}
          strokeWidth="16"
          fill="none"
          strokeLinecap="round"
          strokeDasharray={circumference}
          strokeDashoffset={offset}
          className="transition-all duration-1000 ease-out"
          style={{
            filter: "drop-shadow(0 0 8px currentColor)",
          }}
        />
      </svg>
      <div className="absolute inset-0 flex items-center justify-center">
        <div className="text-center">
          <div className="text-5xl font-bold" style={{ color: colors.stroke }}>
            {displayScore.toFixed(1)}
          </div>
          <div className="text-sm text-muted-foreground">out of 10</div>
        </div>
      </div>
    </div>
  );
};

export default ScoreCircle;

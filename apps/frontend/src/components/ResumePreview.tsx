import { Card } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { User, Mail, Phone, Briefcase, GraduationCap, Code } from "lucide-react";

interface ResumeData {
  personal_data?: {
    name?: string;
    email?: string;
    phone?: string;
    location?: string;
  };
  experiences?: Array<{
    title?: string;
    company?: string;
    duration?: string;
    description?: string;
  }>;
  skills?: string[];
  education?: Array<{
    degree?: string;
    institution?: string;
    year?: string;
  }>;
  extracted_keywords?: string[];
}

interface ResumePreviewProps {
  data: ResumeData;
}

const ResumePreview = ({ data }: ResumePreviewProps) => {
  return (
    <div className="space-y-6">
      {/* Personal Info */}
      {data.personal_data && (
        <Card className="glass-card p-6 rounded-2xl animate-fade-in-up">
          <div className="flex items-center gap-3 mb-4">
            <div className="w-10 h-10 rounded-lg gradient-primary flex items-center justify-center">
              <User className="h-5 w-5 text-white" />
            </div>
            <h3 className="font-semibold text-lg">Personal Information</h3>
          </div>
          <div className="space-y-2">
            {data.personal_data.name && (
              <p className="text-sm">
                <span className="font-medium">Name:</span> {data.personal_data.name}
              </p>
            )}
            {data.personal_data.email && (
              <p className="text-sm flex items-center gap-2">
                <Mail className="h-4 w-4 text-muted-foreground" />
                {data.personal_data.email}
              </p>
            )}
            {data.personal_data.phone && (
              <p className="text-sm flex items-center gap-2">
                <Phone className="h-4 w-4 text-muted-foreground" />
                {data.personal_data.phone}
              </p>
            )}
          </div>
        </Card>
      )}

      {/* Experience */}
      {data.experiences && data.experiences.length > 0 && (
        <Card className="glass-card p-6 rounded-2xl animate-fade-in-up" style={{ animationDelay: "0.1s" }}>
          <div className="flex items-center gap-3 mb-4">
            <div className="w-10 h-10 rounded-lg bg-accent flex items-center justify-center">
              <Briefcase className="h-5 w-5 text-white" />
            </div>
            <h3 className="font-semibold text-lg">Experience</h3>
          </div>
          <div className="space-y-4">
            {data.experiences.map((exp, idx) => (
              <div key={idx} className="border-l-2 border-primary pl-4">
                <h4 className="font-medium">{exp.title}</h4>
                <p className="text-sm text-muted-foreground">
                  {exp.company} {exp.duration && `• ${exp.duration}`}
                </p>
                {exp.description && (
                  <p className="text-sm mt-1">{exp.description}</p>
                )}
              </div>
            ))}
          </div>
        </Card>
      )}

      {/* Skills */}
      {data.skills && data.skills.length > 0 && (
        <Card className="glass-card p-6 rounded-2xl animate-fade-in-up" style={{ animationDelay: "0.2s" }}>
          <div className="flex items-center gap-3 mb-4">
            <div className="w-10 h-10 rounded-lg bg-success flex items-center justify-center">
              <Code className="h-5 w-5 text-white" />
            </div>
            <h3 className="font-semibold text-lg">Skills</h3>
          </div>
          <div className="flex flex-wrap gap-2">
            {data.skills.map((skill, idx) => (
              <Badge key={idx} variant="outline" className="px-3 py-1.5">
                {skill}
              </Badge>
            ))}
          </div>
        </Card>
      )}

      {/* Education */}
      {data.education && data.education.length > 0 && (
        <Card className="glass-card p-6 rounded-2xl animate-fade-in-up" style={{ animationDelay: "0.3s" }}>
          <div className="flex items-center gap-3 mb-4">
            <div className="w-10 h-10 rounded-lg bg-warning flex items-center justify-center">
              <GraduationCap className="h-5 w-5 text-white" />
            </div>
            <h3 className="font-semibold text-lg">Education</h3>
          </div>
          <div className="space-y-3">
            {data.education.map((edu, idx) => (
              <div key={idx}>
                <h4 className="font-medium">{edu.degree}</h4>
                <p className="text-sm text-muted-foreground">
                  {edu.institution} {edu.year && `• ${edu.year}`}
                </p>
              </div>
            ))}
          </div>
        </Card>
      )}

      {/* Extracted Keywords */}
      {data.extracted_keywords && data.extracted_keywords.length > 0 && (
        <Card className="glass-card p-6 rounded-2xl animate-fade-in-up" style={{ animationDelay: "0.4s" }}>
          <h3 className="font-semibold text-lg mb-4">Extracted Keywords</h3>
          <div className="flex flex-wrap gap-2">
            {data.extracted_keywords.map((keyword, idx) => (
              <Badge
                key={idx}
                variant="outline"
                className="px-3 py-1.5 bg-primary/10 border-primary/30 text-primary"
              >
                {keyword}
              </Badge>
            ))}
          </div>
        </Card>
      )}
    </div>
  );
};

export default ResumePreview;

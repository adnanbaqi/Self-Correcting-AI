import { User, Bot } from "lucide-react";
import PipelineStages from "./PipelineStages";
import type { InferenceResponse } from "@/lib/api";

interface ChatMessageProps {
  role: "user" | "assistant";
  content: string;
  response?: InferenceResponse;
  isLoading?: boolean;
}

const ChatMessage = ({ role, content, response, isLoading }: ChatMessageProps) => {
  const isUser = role === "user";

  return (
    <div className={`flex gap-3 ${isUser ? "flex-row-reverse" : ""}`}>
      <div
        className={`flex-shrink-0 h-8 w-8 rounded-full flex items-center justify-center ${
          isUser ? "bg-primary text-primary-foreground" : "bg-secondary text-secondary-foreground"
        }`}
      >
        {isUser ? <User className="h-4 w-4" /> : <Bot className="h-4 w-4" />}
      </div>

      <div className={`flex-1 max-w-[85%] ${isUser ? "flex flex-col items-end" : ""}`}>
        {isUser ? (
          <div className="bg-primary text-primary-foreground rounded-2xl rounded-tr-sm px-4 py-2.5">
            <p className="text-sm">{content}</p>
          </div>
        ) : isLoading ? (
          <div className="bg-card border border-border rounded-2xl rounded-tl-sm px-4 py-3">
            <div className="flex items-center gap-2">
              <div className="flex gap-1">
                <span className="h-2 w-2 rounded-full bg-primary animate-bounce" style={{ animationDelay: "0ms" }} />
                <span className="h-2 w-2 rounded-full bg-primary animate-bounce" style={{ animationDelay: "150ms" }} />
                <span className="h-2 w-2 rounded-full bg-primary animate-bounce" style={{ animationDelay: "300ms" }} />
              </div>
              <span className="text-xs text-muted-foreground">Running pipeline...</span>
            </div>
          </div>
        ) : response ? (
          <div className="w-full">
            <PipelineStages response={response} />
          </div>
        ) : (
          <div className="bg-card border border-border rounded-2xl rounded-tl-sm px-4 py-2.5">
            <p className="text-sm text-card-foreground">{content}</p>
          </div>
        )}
      </div>
    </div>
  );
};

export default ChatMessage;

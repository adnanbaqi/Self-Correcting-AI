import { useState, useRef, useEffect } from "react";
import { Send, Brain, User } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Textarea } from "@/components/ui/textarea";
import HealthStatus from "@/components/HealthStatus";
import ThemeToggle from "@/components/ThemeToggle";
import PipelineStages from "@/components/PipelineStages"; // Make sure this path is correct
import type { InferenceRequest } from "@/lib/api";

interface Message {
  id: string;
  role: "user" | "assistant";
  content: string; // Used for the user's prompt text
  request?: InferenceRequest; // Used to mount the PipelineStages for the assistant
}

const DOMAIN_OPTIONS = [
  { value: "auto", label: "Auto-Detect" },
  { value: "math", label: "Math" },
  { value: "commonsense", label: "Common Sense" },
  { value: "qa", label: "Q&A" },
  { value: "code", label: "Code" },
];


const Index = () => {
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState("");
  const [domain, setDomain] = useState("auto");
  const [isLoading, setIsLoading] = useState(false);
  const scrollRef = useRef<HTMLDivElement>(null);

  // Auto-scroll to bottom as new messages arrive
  useEffect(() => {
    if (scrollRef.current) {
      scrollRef.current.scrollTo({ top: scrollRef.current.scrollHeight, behavior: "smooth" });
    }
  }, [messages]);

  const handleSubmit = (prompt?: string) => {
    const text = (prompt || input).trim();
    if (!text || isLoading) return;

    // 1. Create the user message
    const userMsg: Message = {
      id: crypto.randomUUID(),
      role: "user",
      content: text
    };

    // 2. Format the request for the pipeline. If "auto", we omit the domain field.
    const requestPayload: InferenceRequest = {
      prompt: text,
      domain: domain === "auto" ? undefined : domain
    };

    // 3. Create a placeholder assistant message that holds the request data
    const assistantMsg: Message = {
      id: crypto.randomUUID(),
      role: "assistant",
      content: "",
      request: requestPayload
    };

    setMessages((prev) => [...prev, userMsg, assistantMsg]);
    setInput("");
    setIsLoading(true); // Lock the input until the pipeline stream finishes
  };

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      handleSubmit();
    }
  };

  return (
    <div className="flex flex-col h-screen bg-background">
      {/* Header */}
      <header className="border-b border-border bg-card px-6 py-4">
        <div className="max-w-4xl mx-auto flex items-center justify-between">
          <div className="flex items-center gap-3">
            <div className="h-10 w-10 rounded-xl bg-primary/10 flex items-center justify-center">
              <Brain className="h-5 w-5 text-primary" />
            </div>
            <div>
              <h1 className="text-lg font-bold text-foreground">Self Correcting Neural System</h1>
              <p className="text-xs text-muted-foreground">Autonomous Error Detection & Refinement</p>
            </div>
          </div>
          <div className="flex items-center gap-2">
            <HealthStatus />
            <ThemeToggle />
          </div>
        </div>
      </header>

      {/* Messages Area */}
      <div ref={scrollRef} className="flex-1 overflow-y-auto">
        <div className="max-w-4xl mx-auto px-6 py-6 space-y-8">

          {/* Empty State / Welcome Screen */}
          {messages.length === 0 && (
            <div className="flex flex-col items-center justify-center min-h-[50vh] text-center">
              <div className="h-16 w-16 rounded-2xl bg-primary/10 flex items-center justify-center mb-4">
                <Brain className="h-8 w-8 text-primary" />
              </div>
              <h2 className="text-xl font-bold text-foreground mb-2">Ask anything</h2>
              <p className="text-sm text-muted-foreground mb-6 max-w-md">
                Watch the system generate text, evaluate its own confidence in real-time, and recursively repair its mistakes.
              </p>
              <div className="grid grid-cols-1 sm:grid-cols-2 gap-2 w-full max-w-lg">
              </div>
            </div>
          )}

          {/* Chat Feed */}
          {messages.map((msg) => (
            <div key={msg.id} className="flex w-full">
              {msg.role === "user" ? (
                // User Message UI
                <div className="flex gap-4 w-full justify-end ml-auto max-w-[85%]">
                  <div className="bg-primary text-primary-foreground px-5 py-3 rounded-2xl rounded-tr-sm">
                    <p className="text-sm whitespace-pre-wrap">{msg.content}</p>
                  </div>
                  <div className="h-8 w-8 rounded-full bg-secondary flex items-center justify-center shrink-0 mt-1">
                    <User className="h-4 w-4 text-secondary-foreground" />
                  </div>
                </div>
              ) : (
                // Assistant Message UI (Mounts the Pipeline Stream)
                <div className="flex gap-4 w-full justify-start mr-auto max-w-full">
                  <div className="h-8 w-8 rounded-full bg-primary/10 flex items-center justify-center shrink-0 mt-1">
                    <Brain className="h-4 w-4 text-primary" />
                  </div>
                  <div className="flex-1 min-w-0">
                    <PipelineStages
                      request={msg.request!}
                      onComplete={() => setIsLoading(false)}
                    />
                  </div>
                </div>
              )}
            </div>
          ))}
        </div>
      </div>

      {/* Input Area */}
      <div className="border-t border-border bg-card px-6 py-4">
        <div className="max-w-4xl mx-auto">
          {/* Domain Selector */}
          <div className="flex items-center gap-2 mb-3">
            <span className="text-xs text-muted-foreground">Routing:</span>
            {DOMAIN_OPTIONS.map((d) => (
              <button
                key={d.value}
                onClick={() => setDomain(d.value)}
                disabled={isLoading}
                className={`text-xs px-3 py-1 rounded-full transition-colors ${domain === d.value
                  ? "bg-primary text-primary-foreground"
                  : "bg-secondary text-secondary-foreground hover:bg-muted disabled:opacity-50"
                  }`}
              >
                {d.label}
              </button>
            ))}
          </div>

          {/* Text Input */}
          <div className="flex gap-2 items-end">
            <Textarea
              value={input}
              onChange={(e) => setInput(e.target.value)}
              onKeyDown={handleKeyDown}
              placeholder={isLoading ? "Pipeline is running..." : "Enter your prompt..."}
              disabled={isLoading}
              className="min-h-[44px] max-h-32 resize-none bg-background focus-visible:ring-1"
              rows={1}
            />
            <Button
              onClick={() => handleSubmit()}
              disabled={!input.trim() || isLoading}
              size="icon"
              className="h-11 w-11 shrink-0 rounded-xl"
            >
              <Send className="h-4 w-4" />
            </Button>
          </div>
        </div>
      </div>
    </div>
  );
};

export default Index;
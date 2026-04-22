import React, { useState, useEffect, useRef } from "react";
import { CheckCircle, AlertTriangle, Brain, Wrench, Zap, Clock, Activity, Target } from "lucide-react";
import { streamInference, InferenceRequest, PipelineEvent } from "../lib/api";

interface PipelineStagesProps {
  request: InferenceRequest;
  onComplete?: () => void;
}

interface Token {
  text: string;
  logprob?: number;
}

const PipelineStages = ({ request, onComplete }: PipelineStagesProps) => {
  const [activeStage, setActiveStage] = useState<string>("");
  const [domainDetected, setDomainDetected] = useState<string>("");
  const [baselineTokens, setBaselineTokens] = useState<Token[]>([]);
  const [anomaly, setAnomaly] = useState<string | null>(null);

  const [confidence, setConfidence] = useState<number | null>(null);
  const [wasRefined, setWasRefined] = useState<boolean>(false);

  const [rationaleTokens, setRationaleTokens] = useState<string[]>([]);
  const [correctedTokens, setCorrectedTokens] = useState<string[]>([]);

  const [isDone, setIsDone] = useState(false);
  const [startTime] = useState(Date.now());
  const [totalMs, setTotalMs] = useState<number>(0);

  // Helper to colorize text based on certainty (log probabilities)
  const getLogprobColor = (logprob?: number) => {
    if (logprob === undefined) return "transparent";
    if (logprob < -2.0) return "rgba(239, 68, 68, 0.25)"; // Red (Very uncertain)
    if (logprob < -1.0) return "rgba(234, 179, 8, 0.25)";  // Yellow (Slightly uncertain)
    return "transparent";
  };

  useEffect(() => {
    let isMounted = true;

    const startStream = async () => {
      try {
        for await (const event of streamInference(request)) {
          if (!isMounted) break;

          switch (event.event) {
            case "stage_start":
              setActiveStage(event.stage);
              break;

            case "stage_done":
              if (event.stage === "routing") {
                setDomainDetected(event.content);
              }
              break;

            case "token":
              if (event.stage === "generating") {
                setBaselineTokens((prev) => [...prev, { text: event.token, logprob: event.logprob }]);
              } else if (event.stage === "explaining") {
                setRationaleTokens((prev) => [...prev, event.token]);
              } else if (event.stage === "correcting") {
                setCorrectedTokens((prev) => [...prev, event.token]);
              }
              break;

            case "diagnostic_alert":
              setAnomaly(event.status);
              break;

            case "diagnosis":
              setConfidence(Math.round(event.confidence_score * 100));
              setWasRefined(event.will_refine);
              setActiveStage("diagnosis_complete");
              break;

            case "pipeline_done":
              setIsDone(true);
              setTotalMs(Date.now() - startTime);
              if (onComplete) onComplete();
              break;
          }
        }
      } catch (err) {
        console.error("Stream error:", err);
      }
    };

    startStream();
    return () => { isMounted = false; };
  }, [request]);

  const finalOutput = wasRefined ? correctedTokens.join("") : baselineTokens.map(t => t.text).join("");

  return (
    <div className="space-y-3">
      {/* Stage 0: Routing (Only shows if auto-detected) */}
      {(!request.domain || domainDetected) && (
        <div className="rounded-lg border border-border bg-card p-4">
          <div className="flex items-center gap-2 mb-2">
            <Target className={`h-4 w-4 ${activeStage === "routing" ? "text-blue-500 animate-pulse" : "text-muted-foreground"}`} />
            <span className="text-sm font-semibold">Stage 0 — Auto-Routing</span>
          </div>
          <p className="text-sm text-card-foreground">
            {domainDetected || "Analyzing prompt to determine task domain..."}
          </p>
        </div>
      )}

      {/* Stage 1: Transparent Baseline Generation */}
      <div className="rounded-lg border border-border bg-card p-4">
        <div className="flex items-center justify-between mb-2">
          <div className="flex items-center gap-2">
            <Zap className={`h-4 w-4 ${activeStage === "generating" ? "text-pipeline-baseline animate-pulse" : "text-pipeline-baseline"}`} />
            <span className="text-sm font-semibold text-pipeline-baseline">Initial Output</span>
          </div>
          {anomaly && !isDone && (
            <span className="text-xs flex items-center gap-1 text-destructive bg-destructive/10 px-2 py-0.5 rounded animate-pulse">
              <Activity className="h-3 w-3" /> Live anomaly detected
            </span>
          )}
        </div>

        {/* Render tokens with logprob background highlights */}
        <p className="text-sm text-card-foreground whitespace-pre-wrap leading-relaxed">
          {baselineTokens.length === 0 && activeStage === "generating" ? "Generating..." : null}
          {baselineTokens.map((t, i) => (
            <span key={i} style={{ backgroundColor: getLogprobColor(t.logprob), transition: 'background-color 0.3s ease' }}>
              {t.text}
            </span>
          ))}
        </p>
      </div>

      {/* Stage 2: Diagnoser */}
      {confidence !== null && (
        <div className="rounded-lg border border-border bg-card p-4">
          <div className="flex items-center gap-2 mb-2">
            <Brain className="h-4 w-4 text-pipeline-diagnoser" />
            <span className="text-sm font-semibold text-pipeline-diagnoser">Diagnoser</span>
          </div>
          <div className="flex items-center gap-3">
            <div className="flex-1 h-2 rounded-full bg-muted overflow-hidden">
              <div
                className="h-full rounded-full transition-all duration-700"
                style={{
                  width: `${confidence}%`,
                  backgroundColor: confidence > 70 ? "hsl(var(--accent))" : confidence > 40 ? "hsl(var(--pipeline-diagnoser))" : "hsl(var(--destructive))",
                }}
              />
            </div>
            <span className="text-sm font-mono font-semibold text-card-foreground">{confidence}%</span>
            {confidence > 70 ? (
              <CheckCircle className="h-4 w-4 text-accent" />
            ) : (
              <AlertTriangle className="h-4 w-4 text-pipeline-diagnoser" />
            )}
          </div>
          <p className="text-xs text-muted-foreground mt-1">
            {wasRefined ? "Confidence below threshold — initiating corrective pipeline." : "High confidence — verified."}
          </p>
        </div>
      )}

      {/* Stage 3a: Explainer */}
      {wasRefined && (
        <div className="rounded-lg border border-border bg-card p-4">
          <div className="flex items-center gap-2 mb-2">
            <AlertTriangle className={`h-4 w-4 text-pipeline-explainer ${activeStage === "explaining" ? "animate-pulse" : ""}`} />
            <span className="text-sm font-semibold text-pipeline-explainer">Explainer</span>
          </div>
          <p className="text-sm text-card-foreground whitespace-pre-wrap italic">
            {rationaleTokens.length === 0 && activeStage === "explaining" ? "Diagnosing errors..." : rationaleTokens.join("")}
          </p>
        </div>
      )}

      {/* Stage 3b: Corrector */}
      {wasRefined && (activeStage === "correcting" || isDone) && (
        <div className="rounded-lg border border-border bg-card p-4">
          <div className="flex items-center gap-2 mb-2">
            <Wrench className={`h-4 w-4 text-pipeline-corrector ${activeStage === "correcting" ? "animate-pulse" : ""}`} />
            <span className="text-sm font-semibold text-pipeline-corrector">Corrector</span>
          </div>
          <p className="text-sm text-card-foreground whitespace-pre-wrap">
            {correctedTokens.length === 0 && activeStage === "correcting" ? "Applying fixes..." : correctedTokens.join("")}
          </p>
        </div>
      )}

      {/* Final Output (Only shows when pipeline is complete) */}
      {isDone && (
        <div className="rounded-lg border-2 border-primary bg-primary/5 p-4 mt-6">
          <div className="flex items-center gap-2 mb-2">
            <CheckCircle className="h-4 w-4 text-primary" />
            <span className="text-sm font-semibold text-primary">Final Output</span>
            {wasRefined && (
              <span className="text-xs bg-accent/15 text-accent px-2 py-0.5 rounded-full font-medium">Refined</span>
            )}
            <span className="ml-auto flex items-center gap-1 text-xs text-muted-foreground font-mono">
              <Clock className="h-3 w-3" />
              {totalMs}ms total
            </span>
          </div>
          <p className="text-sm text-card-foreground whitespace-pre-wrap">{finalOutput}</p>
        </div>
      )}
    </div>
  );
};

export default PipelineStages;
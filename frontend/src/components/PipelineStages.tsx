import React, { useState, useEffect, useRef, useCallback } from "react";
import {
  CheckCircle, AlertTriangle, Wrench, Zap, Clock,
  Activity, Target, ChevronRight, Brain, FlaskConical,
  BarChart3, Eye, Copy, Check,
} from "lucide-react";
import { streamInference, InferenceRequest, PipelineEvent } from "../lib/api";

// ─── Types ────────────────────────────────────────────────────────────────────

interface Token {
  text: string;
  logprob?: number;
}

interface StageRecord {
  stage: string;
  label: string;
  startedAt: number;
  endedAt?: number;
  tokenCount: number;
}

interface PipelineStagesProps {
  request: InferenceRequest;
  onComplete?: () => void;
}

// ─── Constants ────────────────────────────────────────────────────────────────

const STAGE_META: Record<string, { icon: React.ElementType; color: string; accent: string; label: string }> = {
  routing: { icon: Target, color: "text-sky-400", accent: "border-sky-500/40", label: "Router" },
  generating: { icon: Zap, color: "text-violet-400", accent: "border-violet-500/40", label: "Generator" },
  diagnosing: { icon: Brain, color: "text-amber-400", accent: "border-amber-500/40", label: "Diagnoser" },
  explaining: { icon: FlaskConical, color: "text-orange-400", accent: "border-orange-500/40", label: "Explainer" },
  correcting: { icon: Wrench, color: "text-emerald-400", accent: "border-emerald-500/40", label: "Corrector" },
};

const PIPELINE_NODES = ["routing", "generating", "diagnosing", "explaining", "correcting"] as const;

// ─── Helpers ──────────────────────────────────────────────────────────────────

function getLogprobStyle(logprob?: number): React.CSSProperties {
  if (logprob === undefined || logprob === null) return {};
  if (logprob < -2.5) return { backgroundColor: "rgba(239,68,68,0.30)", borderBottom: "1px solid rgba(239,68,68,0.6)" };
  if (logprob < -1.8) return { backgroundColor: "rgba(234,179,8,0.22)", borderBottom: "1px solid rgba(234,179,8,0.5)" };
  if (logprob < -1.2) return { backgroundColor: "rgba(251,191,36,0.12)" };
  return {};
}

function formatMs(ms: number): string {
  if (ms < 1000) return `${ms}ms`;
  return `${(ms / 1000).toFixed(1)}s`;
}

function getConfidenceColor(pct: number): string {
  if (pct >= 85) return "#34d399"; // emerald
  if (pct >= 60) return "#fbbf24"; // amber
  return "#f87171"; // red
}

function copyToClipboard(text: string) {
  navigator.clipboard.writeText(text).catch(() => { });
}

// ─── Sub-components ───────────────────────────────────────────────────────────

// Animated scan-line pulse on active stage header
const ScanLinePulse = () => (
  <span
    className="absolute inset-x-0 top-0 h-px opacity-60"
    style={{
      background: "linear-gradient(90deg, transparent, rgba(167,139,250,0.8), transparent)",
      animation: "scanline 2s ease-in-out infinite",
    }}
  />
);

// Mini pipeline graph showing node states
const PipelineGraph = ({
  activeStage,
  completedStages,
  wasRefined,
}: {
  activeStage: string;
  completedStages: Set<string>;
  wasRefined: boolean;
}) => {
  const nodes = wasRefined
    ? PIPELINE_NODES
    : (["routing", "generating", "diagnosing"] as const);

  return (
    <div className="flex items-center gap-0 font-mono text-xs">
      {nodes.map((stage, i) => {
        const meta = STAGE_META[stage];
        const Icon = meta.icon;
        const isDone = completedStages.has(stage);
        const isActive = activeStage === stage;
        const isPending = !isDone && !isActive;

        return (
          <React.Fragment key={stage}>
            <div
              className="flex flex-col items-center gap-1"
              title={meta.label}
            >
              <div
                className={`
                  h-7 w-7 rounded-md flex items-center justify-center border transition-all duration-500
                  ${isDone ? "border-emerald-500/50 bg-emerald-500/10" : ""}
                  ${isActive ? `${meta.accent} bg-white/5 ring-1 ring-inset ring-white/10` : ""}
                  ${isPending ? "border-white/10 bg-white/3 opacity-40" : ""}
                `}
                style={isActive ? { boxShadow: "0 0 12px rgba(167,139,250,0.15)" } : {}}
              >
                {isDone ? (
                  <CheckCircle className="h-3.5 w-3.5 text-emerald-400" />
                ) : (
                  <Icon
                    className={`h-3.5 w-3.5 ${isActive ? meta.color + " animate-pulse" : "text-white/25"}`}
                  />
                )}
              </div>
              <span
                className={`text-[9px] uppercase tracking-widest ${isDone ? "text-emerald-400/70" :
                    isActive ? "text-white/70" : "text-white/20"
                  }`}
              >
                {stage.slice(0, 3)}
              </span>
            </div>
            {i < nodes.length - 1 && (
              <div
                className={`h-px w-5 mb-4 transition-all duration-700 ${completedStages.has(stage) ? "bg-emerald-500/40" : "bg-white/10"
                  }`}
              />
            )}
          </React.Fragment>
        );
      })}
    </div>
  );
};

// Token with logprob background + tooltip
const ColoredToken = ({ token, index }: { token: Token; index: number }) => {
  const style = getLogprobStyle(token.logprob);
  const hasHighlight = Object.keys(style).length > 0;

  if (!hasHighlight) {
    return <span key={index}>{token.text}</span>;
  }

  const certaintyLabel =
    token.logprob !== undefined
      ? token.logprob < -2.5 ? "Very uncertain"
        : token.logprob < -1.8 ? "Uncertain"
          : "Slightly uncertain"
      : "";

  return (
    <span
      key={index}
      style={{ ...style, borderRadius: "2px", cursor: "default" }}
      title={`logprob: ${token.logprob?.toFixed(3)} — ${certaintyLabel}`}
      className="relative inline transition-all duration-300"
    >
      {token.text}
    </span>
  );
};

// Confidence arc meter
const ConfidenceMeter = ({ pct, isCorrect }: { pct: number; isCorrect: boolean }) => {
  const r = 28;
  const circ = 2 * Math.PI * r;
  const dash = (pct / 100) * circ;
  const color = getConfidenceColor(pct);

  return (
    <div className="relative flex items-center justify-center" style={{ width: 72, height: 72 }}>
      <svg width="72" height="72" viewBox="0 0 72 72">
        <circle cx="36" cy="36" r={r} fill="none" stroke="rgba(255,255,255,0.07)" strokeWidth="5" />
        <circle
          cx="36" cy="36" r={r}
          fill="none"
          stroke={color}
          strokeWidth="5"
          strokeDasharray={`${dash} ${circ - dash}`}
          strokeLinecap="round"
          strokeDashoffset={circ * 0.25}
          style={{ transition: "stroke-dasharray 1s ease-out", filter: `drop-shadow(0 0 4px ${color}60)` }}
        />
      </svg>
      <div className="absolute inset-0 flex flex-col items-center justify-center">
        <span className="text-base font-bold font-mono leading-none" style={{ color }}>{pct}%</span>
        <span className="text-[9px] text-white/40 uppercase tracking-wider mt-0.5">
          {isCorrect ? "pass" : "fail"}
        </span>
      </div>
    </div>
  );
};

// Stage timing badge
const TimingBadge = ({ ms }: { ms: number }) => (
  <span className="flex items-center gap-1 text-[10px] font-mono text-white/30">
    <Clock className="h-2.5 w-2.5" />
    {formatMs(ms)}
  </span>
);

// Copy button with "Copied!" flash
const CopyButton = ({ text }: { text: string }) => {
  const [copied, setCopied] = useState(false);
  const handleCopy = () => {
    copyToClipboard(text);
    setCopied(true);
    setTimeout(() => setCopied(false), 1500);
  };
  return (
    <button
      onClick={handleCopy}
      className="flex items-center gap-1 text-[10px] font-mono text-white/30 hover:text-white/60 transition-colors"
    >
      {copied ? <Check className="h-3 w-3 text-emerald-400" /> : <Copy className="h-3 w-3" />}
      {copied ? "copied" : "copy"}
    </button>
  );
};

// Anomaly alert ticker
const AnomalyTicker = ({ anomalies }: { anomalies: string[] }) => {
  const [idx, setIdx] = useState(0);
  useEffect(() => {
    if (anomalies.length <= 1) return;
    const t = setInterval(() => setIdx(i => (i + 1) % anomalies.length), 2000);
    return () => clearInterval(t);
  }, [anomalies.length]);

  if (!anomalies.length) return null;

  return (
    <div className="flex items-center gap-2 px-3 py-1.5 rounded-md border border-red-500/30 bg-red-500/8 text-xs text-red-400">
      <Activity className="h-3 w-3 shrink-0 animate-pulse" />
      <span className="font-mono truncate">{anomalies[idx]}</span>
      {anomalies.length > 1 && (
        <span className="shrink-0 text-red-400/50">+{anomalies.length - 1}</span>
      )}
    </div>
  );
};

// Logprob legend
const LogprobLegend = () => (
  <div className="flex items-center gap-3 text-[10px] text-white/30 font-mono">
    <span className="flex items-center gap-1">
      <span className="inline-block h-2 w-4 rounded-sm" style={{ backgroundColor: "rgba(239,68,68,0.3)" }} />
      high uncertainty
    </span>
    <span className="flex items-center gap-1">
      <span className="inline-block h-2 w-4 rounded-sm" style={{ backgroundColor: "rgba(234,179,8,0.22)" }} />
      moderate
    </span>
  </div>
);

// Stage card shell
const StageCard = ({
  stage,
  isActive,
  isDone,
  children,
  elapsed,
  actions,
}: {
  stage: string;
  isActive: boolean;
  isDone: boolean;
  children: React.ReactNode;
  elapsed?: number;
  actions?: React.ReactNode;
}) => {
  const meta = STAGE_META[stage] ?? STAGE_META.generating;
  const Icon = meta.icon;

  return (
    <div
      className={`
        relative overflow-hidden rounded-xl border transition-all duration-500
        ${isActive ? `${meta.accent} bg-white/[0.03]` : "border-white/8 bg-white/[0.02]"}
        ${isDone ? "border-white/10" : ""}
      `}
      style={isActive ? { boxShadow: "0 0 0 1px rgba(255,255,255,0.04) inset" } : {}}
    >
      {isActive && <ScanLinePulse />}
      <div className="flex items-center justify-between px-4 py-3 border-b border-white/6">
        <div className="flex items-center gap-2.5">
          <div
            className={`
              h-6 w-6 rounded-md flex items-center justify-center
              ${isActive ? "bg-white/8" : "bg-white/4"}
            `}
          >
            <Icon className={`h-3.5 w-3.5 ${isActive ? meta.color : "text-white/30"}`} />
          </div>
          <span className={`text-xs font-semibold uppercase tracking-widest ${isActive ? "text-white/80" : "text-white/35"}`}>
            {meta.label}
          </span>
          {isActive && (
            <span className="flex gap-0.5">
              {[0, 1, 2].map(i => (
                <span
                  key={i}
                  className="h-1 w-1 rounded-full bg-white/30"
                  style={{ animation: `bounce 1.2s ease-in-out ${i * 0.2}s infinite` }}
                />
              ))}
            </span>
          )}
          {isDone && <CheckCircle className="h-3 w-3 text-emerald-400/60" />}
        </div>
        <div className="flex items-center gap-3">
          {actions}
          {elapsed !== undefined && <TimingBadge ms={elapsed} />}
        </div>
      </div>
      <div className="p-4">{children}</div>
    </div>
  );
};

// ─── Main Component ───────────────────────────────────────────────────────────

const PipelineStages = ({ request, onComplete }: PipelineStagesProps) => {
  // Stage state
  const [activeStage, setActiveStage] = useState<string>("routing");
  const [completedStages, setCompletedStages] = useState<Set<string>>(new Set());
  const [domainDetected, setDomainDetected] = useState<string>("");

  // Tokens
  const [baselineTokens, setBaselineTokens] = useState<Token[]>([]);
  const [rationaleTokens, setRationaleTokens] = useState<string[]>([]);
  const [correctedTokens, setCorrectedTokens] = useState<string[]>([]);

  // Anomalies
  const [anomalies, setAnomalies] = useState<string[]>([]);

  // Diagnosis
  const [confidence, setConfidence] = useState<number | null>(null);
  const [isCorrect, setIsCorrect] = useState<boolean | null>(null);
  const [reasoning, setReasoning] = useState<string>("");
  const [wasRefined, setWasRefined] = useState<boolean>(false);

  // Pipeline done
  const [isDone, setIsDone] = useState(false);
  const [stagesExecuted, setStagesExecuted] = useState(0);
  const [startTime] = useState(Date.now());
  const [totalMs, setTotalMs] = useState<number>(0);

  // Stage timing
  const [stageRecords, setStageRecords] = useState<Map<string, StageRecord>>(new Map());
  const stageStartRef = useRef<Map<string, number>>(new Map());

  // View controls
  const [showRawFeed, setShowRawFeed] = useState(false);
  const [showHeatmapLegend, setShowHeatmapLegend] = useState(false);

  const markStageStart = useCallback((stage: string, label: string) => {
    const now = Date.now();
    stageStartRef.current.set(stage, now);
    setStageRecords(prev => {
      const next = new Map(prev);
      next.set(stage, { stage, label, startedAt: now, tokenCount: 0 });
      return next;
    });
  }, []);

  const markStageDone = useCallback((stage: string) => {
    const startedAt = stageStartRef.current.get(stage);
    if (!startedAt) return;
    const endedAt = Date.now();
    setStageRecords(prev => {
      const next = new Map(prev);
      const rec = prev.get(stage);
      if (rec) next.set(stage, { ...rec, endedAt });
      return next;
    });
    setCompletedStages(prev => new Set([...prev, stage]));
  }, []);

  useEffect(() => {
    let isMounted = true;

    const startStream = async () => {
      try {
        for await (const event of streamInference(request)) {
          if (!isMounted) break;

          switch (event.event) {
            case "stage_start":
              setActiveStage(event.stage);
              markStageStart(event.stage, event.label);
              break;

            case "stage_done":
              markStageDone(event.stage);
              if (event.stage === "routing") {
                setDomainDetected(event.content.replace("Domain detected: ", "").toUpperCase());
              }
              break;

            case "token":
              if (event.stage === "generating") {
                setBaselineTokens(prev => [...prev, { text: event.token, logprob: event.logprob }]);
              } else if (event.stage === "explaining") {
                setRationaleTokens(prev => [...prev, event.token]);
              } else if (event.stage === "correcting") {
                setCorrectedTokens(prev => [...prev, event.token]);
              }
              break;

            case "diagnostic_alert":
              setAnomalies(prev => [...prev, event.status]);
              break;

            case "diagnosis":
              setConfidence(Math.round(event.confidence_score * 100));
              setIsCorrect(event.is_correct);
              setReasoning(event.reasoning ?? "");
              setWasRefined(event.will_refine);
              markStageDone("diagnosing");
              setActiveStage(event.will_refine ? "explaining" : "done");
              break;

            case "pipeline_done":
              setIsDone(true);
              setStagesExecuted(event.pipeline_stages_executed);
              setTotalMs(Date.now() - startTime);
              markStageDone(activeStage);
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

  const finalOutput = wasRefined
    ? correctedTokens.join("")
    : baselineTokens.map(t => t.text).join("");

  const highUncertaintyCount = baselineTokens.filter(t =>
    t.logprob !== undefined && t.logprob < -1.8
  ).length;

  const stageElapsed = (stage: string) => {
    const rec = stageRecords.get(stage);
    if (!rec) return undefined;
    if (rec.endedAt) return rec.endedAt - rec.startedAt;
    return Date.now() - rec.startedAt;
  };

  return (
    <>
      {/* Keyframe styles */}
      <style>{`
        @keyframes scanline {
          0% { transform: translateX(-100%); opacity: 0; }
          30% { opacity: 0.6; }
          70% { opacity: 0.6; }
          100% { transform: translateX(100%); opacity: 0; }
        }
        @keyframes bounce {
          0%, 100% { transform: translateY(0); opacity: 0.3; }
          50% { transform: translateY(-3px); opacity: 0.8; }
        }
        @keyframes fadeIn {
          from { opacity: 0; transform: translateY(6px); }
          to { opacity: 1; transform: translateY(0); }
        }
        .stage-enter { animation: fadeIn 0.35s ease-out forwards; }
      `}</style>

      <div className="space-y-2.5 font-mono">

        {/* ── Pipeline Graph Header ── */}
        <div className="rounded-xl border border-white/8 bg-black/20 px-4 py-3">
          <div className="flex items-center justify-between">
            <PipelineGraph
              activeStage={activeStage}
              completedStages={completedStages}
              wasRefined={wasRefined}
            />
            <div className="flex items-center gap-3 text-[10px] text-white/30">
              {domainDetected && (
                <span className="px-2 py-0.5 rounded border border-sky-500/30 text-sky-400/70 uppercase tracking-widest">
                  {domainDetected}
                </span>
              )}
              {isDone && (
                <span className="flex items-center gap-1 text-white/40">
                  <Clock className="h-2.5 w-2.5" />
                  {formatMs(totalMs)}
                </span>
              )}
            </div>
          </div>
        </div>

        {/* ── Stage 1: Generator ── */}
        {(baselineTokens.length > 0 || activeStage === "generating") && (
          <div className="stage-enter">
            <StageCard
              stage="generating"
              isActive={activeStage === "generating"}
              isDone={completedStages.has("generating")}
              elapsed={stageElapsed("generating")}
              actions={
                <div className="flex items-center gap-3">
                  {anomalies.length > 0 && (
                    <span className="text-[10px] text-red-400/60 flex items-center gap-1">
                      <Activity className="h-2.5 w-2.5" />
                      {anomalies.length} anomal{anomalies.length === 1 ? "y" : "ies"}
                    </span>
                  )}
                  {highUncertaintyCount > 0 && (
                    <button
                      onClick={() => setShowHeatmapLegend(v => !v)}
                      className="text-[10px] text-amber-400/60 hover:text-amber-400/90 flex items-center gap-1 transition-colors"
                    >
                      <BarChart3 className="h-2.5 w-2.5" />
                      {highUncertaintyCount} uncertain
                    </button>
                  )}
                  {baselineTokens.length > 0 && (
                    <CopyButton text={baselineTokens.map(t => t.text).join("")} />
                  )}
                </div>
              }
            >
              {showHeatmapLegend && <div className="mb-3"><LogprobLegend /></div>}
              {anomalies.length > 0 && (
                <div className="mb-3">
                  <AnomalyTicker anomalies={anomalies} />
                </div>
              )}
              <p className="text-sm text-white/70 whitespace-pre-wrap leading-relaxed">
                {baselineTokens.length === 0 && (
                  <span className="text-white/20 animate-pulse">Generating…</span>
                )}
                {baselineTokens.map((t, i) => (
                  <ColoredToken key={i} token={t} index={i} />
                ))}
                {activeStage === "generating" && baselineTokens.length > 0 && (
                  <span
                    className="inline-block h-3.5 w-0.5 ml-0.5 bg-violet-400 align-middle"
                    style={{ animation: "bounce 0.8s ease-in-out infinite" }}
                  />
                )}
              </p>
              <div className="mt-3 flex items-center justify-between text-[10px] text-white/20">
                <span>{baselineTokens.length} tokens</span>
                <span className="text-white/15">hover token for logprob</span>
              </div>
            </StageCard>
          </div>
        )}

        {/* ── Stage 2: Diagnoser ── */}
        {confidence !== null && isCorrect !== null && (
          <div className="stage-enter">
            <StageCard
              stage="diagnosing"
              isActive={false}
              isDone={true}
              elapsed={stageElapsed("diagnosing")}
              actions={
                <button
                  onClick={() => setShowRawFeed(v => !v)}
                  className={`flex items-center gap-1 text-[10px] transition-colors ${showRawFeed ? "text-amber-400/80" : "text-white/30 hover:text-white/60"
                    }`}
                >
                  <Eye className="h-2.5 w-2.5" />
                  raw feed
                </button>
              }
            >
              <div className="flex items-start gap-5">
                {/* Arc meter */}
                <div className="shrink-0">
                  <ConfidenceMeter pct={confidence} isCorrect={isCorrect} />
                </div>

                {/* Right side */}
                <div className="flex-1 min-w-0 space-y-3">
                  {/* Verdict */}
                  <div className="flex items-center gap-2">
                    {isCorrect ? (
                      <>
                        <CheckCircle className="h-4 w-4 text-emerald-400 shrink-0" />
                        <span className="text-sm text-emerald-400 font-semibold">Verified — no correction needed</span>
                      </>
                    ) : (
                      <>
                        <AlertTriangle className="h-4 w-4 text-amber-400 shrink-0" />
                        <span className="text-sm text-amber-400 font-semibold">Errors detected — initiating correction</span>
                      </>
                    )}
                  </div>

                  {/* Confidence bar */}
                  <div className="space-y-1">
                    <div className="h-1.5 w-full rounded-full bg-white/6 overflow-hidden">
                      <div
                        className="h-full rounded-full transition-all duration-1000 ease-out"
                        style={{
                          width: `${confidence}%`,
                          backgroundColor: getConfidenceColor(confidence),
                          boxShadow: `0 0 8px ${getConfidenceColor(confidence)}60`,
                        }}
                      />
                    </div>
                    <div className="flex justify-between text-[10px] text-white/20">
                      <span>0%</span>
                      <span className="text-white/30">threshold: 85%</span>
                      <span>100%</span>
                    </div>
                  </div>

                  {/* Raw reasoning feed */}
                  {showRawFeed && reasoning && (
                    <div className="rounded-lg border border-amber-500/20 bg-amber-500/5 p-3 space-y-1">
                      <div className="flex items-center gap-1.5 mb-2">
                        <span className="text-[9px] uppercase tracking-widest text-amber-400/60">
                          Diagnoser raw reasoning
                        </span>
                        <span className="h-px flex-1 bg-amber-500/15" />
                      </div>
                      <p className="text-xs text-amber-200/60 leading-relaxed italic">
                        "{reasoning}"
                      </p>
                    </div>
                  )}

                  {/* Hidden reasoning hint */}
                  {!showRawFeed && reasoning && (
                    <p className="text-xs text-white/25 italic truncate">
                      "{reasoning.slice(0, 80)}{reasoning.length > 80 ? "…" : ""}"
                    </p>
                  )}
                </div>
              </div>
            </StageCard>
          </div>
        )}

        {/* ── Stage 3: Explainer ── */}
        {wasRefined && (rationaleTokens.length > 0 || activeStage === "explaining") && (
          <div className="stage-enter">
            <StageCard
              stage="explaining"
              isActive={activeStage === "explaining"}
              isDone={completedStages.has("explaining")}
              elapsed={stageElapsed("explaining")}
              actions={rationaleTokens.length > 0 ? <CopyButton text={rationaleTokens.join("")} /> : undefined}
            >
              <p className="text-sm text-white/60 whitespace-pre-wrap leading-relaxed italic">
                {rationaleTokens.length === 0 ? (
                  <span className="text-white/20 animate-pulse">Diagnosing errors…</span>
                ) : (
                  rationaleTokens.join("")
                )}
                {activeStage === "explaining" && rationaleTokens.length > 0 && (
                  <span
                    className="inline-block h-3.5 w-0.5 ml-0.5 bg-orange-400 align-middle"
                    style={{ animation: "bounce 0.8s ease-in-out infinite" }}
                  />
                )}
              </p>
            </StageCard>
          </div>
        )}

        {/* ── Stage 4: Corrector ── */}
        {wasRefined && (correctedTokens.length > 0 || activeStage === "correcting") && (
          <div className="stage-enter">
            <StageCard
              stage="correcting"
              isActive={activeStage === "correcting"}
              isDone={completedStages.has("correcting")}
              elapsed={stageElapsed("correcting")}
              actions={correctedTokens.length > 0 ? <CopyButton text={correctedTokens.join("")} /> : undefined}
            >
              <p className="text-sm text-white/75 whitespace-pre-wrap leading-relaxed">
                {correctedTokens.length === 0 ? (
                  <span className="text-white/20 animate-pulse">Applying corrections…</span>
                ) : (
                  correctedTokens.join("")
                )}
                {activeStage === "correcting" && correctedTokens.length > 0 && (
                  <span
                    className="inline-block h-3.5 w-0.5 ml-0.5 bg-emerald-400 align-middle"
                    style={{ animation: "bounce 0.8s ease-in-out infinite" }}
                  />
                )}
              </p>
            </StageCard>
          </div>
        )}

        {/* ── Final Output ── */}
        {isDone && (
          <div className="stage-enter rounded-xl border border-white/15 bg-white/[0.04] overflow-hidden">
            {/* Header */}
            <div className="flex items-center justify-between px-4 py-3 border-b border-white/8">
              <div className="flex items-center gap-2.5">
                <CheckCircle className="h-4 w-4 text-emerald-400" />
                <span className="text-xs font-semibold uppercase tracking-widest text-white/70">
                  Final Output
                </span>
                {wasRefined && (
                  <span className="px-2 py-0.5 rounded text-[9px] uppercase tracking-widest border border-emerald-500/30 text-emerald-400/80 bg-emerald-500/8">
                    Refined
                  </span>
                )}
              </div>
              <div className="flex items-center gap-3">
                <CopyButton text={finalOutput} />
                {/* Stats row */}
                <div className="flex items-center gap-3 text-[10px] text-white/25">
                  <span>{stagesExecuted} stages</span>
                  <span>·</span>
                  <span className="flex items-center gap-1">
                    <Clock className="h-2.5 w-2.5" />
                    {formatMs(totalMs)}
                  </span>
                </div>
              </div>
            </div>

            {/* Content */}
            <div className="p-4">
              <p className="text-sm text-white/90 whitespace-pre-wrap leading-relaxed">
                {finalOutput}
              </p>
            </div>

            {/* Footer metrics */}
            <div className="flex items-center gap-4 px-4 py-2.5 border-t border-white/6 bg-black/10">
              <div className="flex items-center gap-1.5 text-[10px] text-white/25">
                <span className="h-1.5 w-1.5 rounded-full bg-emerald-400" />
                {confidence !== null && `${confidence}% confidence`}
              </div>
              {wasRefined && (
                <div className="flex items-center gap-1.5 text-[10px] text-white/25">
                  <ChevronRight className="h-2.5 w-2.5" />
                  self-corrected
                </div>
              )}
              {highUncertaintyCount > 0 && (
                <div className="flex items-center gap-1.5 text-[10px] text-white/25">
                  <Activity className="h-2.5 w-2.5" />
                  {highUncertaintyCount} uncertain tokens detected
                </div>
              )}
            </div>
          </div>
        )}
      </div>
    </>
  );
};

export default PipelineStages;
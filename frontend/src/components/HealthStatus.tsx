import { useEffect, useState } from "react";
import { checkHealth, type HealthResponse } from "@/lib/api";
import { Activity, Cpu, Server } from "lucide-react";

const HealthStatus = () => {
  const [health, setHealth] = useState<HealthResponse | null>(null);
  const [error, setError] = useState(false);

  useEffect(() => {
    checkHealth()
      .then(setHealth)
      .catch(() => setError(true));

    const interval = setInterval(() => {
      checkHealth()
        .then((h) => { setHealth(h); setError(false); })
        .catch(() => setError(true));
    }, 15000);
    return () => clearInterval(interval);
  }, []);

  if (error) {
    return (
      <div className="flex items-center gap-2 text-xs text-destructive">
        <span className="h-2 w-2 rounded-full bg-destructive" />
        Backend offline
      </div>
    );
  }

  if (!health) {
    return (
      <div className="flex items-center gap-2 text-xs text-muted-foreground">
        <span className="h-2 w-2 rounded-full bg-muted-foreground animate-pulse" />
        Connecting...
      </div>
    );
  }

  return (
    <div className="flex flex-wrap items-center gap-4 text-xs text-muted-foreground">
      <div className="flex items-center gap-1.5">
        <span className={`h-2 w-2 rounded-full ${health.status === "healthy" ? "bg-accent" : "bg-pipeline-diagnoser"}`} />
        {health.status === "healthy" ? "Online" : "Loading model..."}
      </div>
      <div className="flex items-center gap-1">
        <Server className="h-3 w-3" />
        <span className="font-mono">{health.model_id}</span>
      </div>
      <div className="flex items-center gap-1">
        <Cpu className="h-3 w-3" />
        <span className="font-mono">{health.device}</span>
      </div>
      <div className="flex items-center gap-1">
        <Activity className="h-3 w-3" />
        {health.pipeline_components.join(" → ")}
      </div>
    </div>
  );
};

export default HealthStatus;

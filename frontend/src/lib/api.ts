const API_BASE = "http://127.0.0.1:8000/api/v1";

export interface InferenceRequest {
  prompt: string;
  domain?: string;
}

export interface HealthResponse {
  status: string;
  model_loaded: boolean;
  model_id: string;
  model_pt_path: string;
  device: string;
  pipeline_components: string[];
}

// Types for the incoming SSE events
export type PipelineEvent =
  | { event: "stage_start"; stage: string; label: string }
  | { event: "token"; stage: string; token: string; logprob?: number }
  | { event: "stage_done"; stage: string; content: string }
  | { event: "diagnostic_alert"; status: string }
  | { event: "diagnosis"; confidence_score: number; is_correct: boolean; reasoning: string; will_refine: boolean }
  | { event: "pipeline_done"; was_refined: boolean; pipeline_stages_executed: number }
  | { event: "error"; message: string };

/**
 * Consumes the SSE stream from the backend and yields parsed JSON events.
 */
export async function* streamInference(request: InferenceRequest): AsyncGenerator<PipelineEvent, void, unknown> {
  const res = await fetch(`${API_BASE}/infer/stream`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(request),
  });

  if (!res.ok) {
    const detail = await res.text();
    throw new Error(`Inference stream failed (${res.status}): ${detail}`);
  }

  const reader = res.body?.getReader();
  if (!reader) throw new Error("Response body is not readable");

  const decoder = new TextDecoder();
  let buffer = "";

  while (true) {
    const { done, value } = await reader.read();
    if (done) break;

    buffer += decoder.decode(value, { stream: true });

    // SSE messages are separated by double newlines
    const chunks = buffer.split("\n\n");

    // Keep the last incomplete chunk in the buffer
    buffer = chunks.pop() || "";

    for (const chunk of chunks) {
      // Find the data payload line
      const dataLine = chunk.split("\n").find(line => line.startsWith("data: "));
      if (dataLine) {
        const jsonStr = dataLine.slice(6);
        try {
          yield JSON.parse(jsonStr) as PipelineEvent;
        } catch (e) {
          console.error("Failed to parse SSE JSON:", jsonStr);
        }
      }
    }
  }
}

export async function checkHealth(): Promise<HealthResponse> {
  const res = await fetch(`${API_BASE}/health`);
  if (!res.ok) throw new Error("Health check failed");
  return res.json();
}
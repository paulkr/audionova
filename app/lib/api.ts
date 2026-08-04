// Typed client for the local audio API.
//
// THE BROWSER TALKS TO IT DIRECTLY, never through a Next.js route. Server
// Actions cap request bodies at 1 MB and a three-minute 24-bit WAV is about
// 50 MB; worse, adding a proxy gets a 10 MB default that SILENTLY TRUNCATES
// larger bodies with only a console warning — a data-corruption bug that
// returns success. In production this base URL points at the API host and
// uploads go straight to object storage with a presigned URL.

export const API_BASE =
  process.env.NEXT_PUBLIC_API_BASE ?? "http://localhost:8787";

export type Band = {
  filter_type: string;
  center_freq_hz: number;
  gain_db: number;
  q_factor: number;
};

export type GraphNode = {
  id: string;
  type: string;
  mix: number;
  bypass: boolean;
  rationale: string;
  params: Record<string, unknown> & { bands?: Band[] };
  master: boolean;
};

export type Report = {
  source: { channels: number; sample_rate_hz: number; duration_s: number };
  loudness: {
    integrated_lufs: number | null;
    true_peak_dbtp: number | null;
    loudness_range_lu: number | null;
  };
  dynamics: {
    crest_factor_db: number | null;
    peak_to_loudness_ratio_db: number | null;
  };
  spectrum: {
    tilt_db_per_octave_re_pink: number | null;
    hf_rolloff_db_per_octave: number | null;
    third_octave_db: { centre_hz: number; level_db: number | null }[];
  };
  warnings: { code: string; severity: string; message: string }[];
};

export type Project = {
  id: string;
  name: string;
  sample_rate: number;
  channels: number;
  duration_s: number;
  report: Report;
};

export type PlanResult = {
  graph_id: string;
  elapsed_s: number;
  explanation: string;
  nodes: GraphNode[];
  attempts: number;
  repairs: string[];
  declined?: string[];
};

export type RenderResult = {
  ok: boolean;
  render_key?: string;
  source_key?: string;
  // The unmatched render. `render_key` is pulled to the source's loudness so
  // the A/B compares processing rather than level; this one is the chain's
  // actual output, and it is what a download must deliver.
  deliver_key?: string;
  full?: boolean;
  elapsed_s?: number;
  match_offset_db?: number;
  gate?: { passed: boolean; checked: string[] };
  after?: Report;
  // The gate is not advisory: a failing render does not ship, and the reason
  // arrives in words rather than as a correlation coefficient.
  failures?: { code: string; message: string }[];
};

export type RefineResult = {
  graph_id?: string;
  parent?: string;
  diff?: string[];
  applied?: string[];
  rejected?: string[];
  nodes?: GraphNode[];
  note?: string;
};

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(`${API_BASE}${path}`, init);
  if (!response.ok) {
    // The API's own message names the measurement and the rule; rewording it
    // here would only make it vaguer.
    let detail = `${response.status}`;
    try {
      const body = await response.json();
      detail = body.detail ?? detail;
    } catch {
      /* the body was not JSON; the status is all there is */
    }
    throw new Error(detail);
  }
  return response.json() as Promise<T>;
}

export function health() {
  return request<{ ok: boolean; engine: boolean; model: boolean }>(
    "/api/health",
  );
}

export function upload(file: File) {
  const body = new FormData();
  body.append("file", file);
  return request<Project>("/api/projects", { method: "POST", body });
}

export function getProject(id: string) {
  return request<Project & { head: string | null }>(`/api/projects/${id}`);
}

export function plan(
  id: string,
  body: { request: string; planner: "model" | "rules"; reverb_mix?: number },
) {
  return request<PlanResult>(`/api/projects/${id}/plan`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
}

export function render(id: string, graphId: string, full = false) {
  return request<RenderResult>(
    `/api/projects/${id}/graphs/${graphId}/render?full=${full}`,
    { method: "POST" },
  );
}

export function refine(
  id: string,
  graphId: string,
  intent: string,
  quantifier: string,
) {
  return request<RefineResult>(`/api/projects/${id}/graphs/${graphId}/refine`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ intent, quantifier }),
  });
}

export function intents() {
  return request<{
    intents: { id: string; label: string; node_type: string }[];
    quantifiers: string[];
  }>("/api/intents");
}

export function audioUrl(id: string, key: string) {
  return `${API_BASE}/api/projects/${id}/audio/${key}`;
}

export function downloadUrl(id: string, key: string) {
  return `${API_BASE}/api/projects/${id}/download/${key}`;
}

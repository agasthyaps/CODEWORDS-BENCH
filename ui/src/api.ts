import { GameType, ReplaySummary, TeamSelection } from "./types";

const API_BASE =
  import.meta.env.VITE_API_BASE || "http://localhost:8000";

export async function fetchModels() {
  const res = await fetch(`${API_BASE}/models`);
  if (!res.ok) throw new Error("Failed to load models");
  return res.json();
}

export async function startCodenames(payload: {
  team_selection: TeamSelection;
  mode: string;
  seed?: number;
  max_discussion_rounds: number;
  max_turns: number;
  event_delay_ms: number;
}) {
  const res = await fetch(`${API_BASE}/codenames/start`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
  if (!res.ok) throw new Error("Failed to start codenames");
  return res.json();
}

export async function startDecrypto(payload: {
  team_selection: TeamSelection;
  seed?: number;
  max_rounds: number;
  max_discussion_turns_per_guesser: number;
  event_delay_ms: number;
}) {
  const res = await fetch(`${API_BASE}/decrypto/start`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
  if (!res.ok) throw new Error("Failed to start decrypto");
  return res.json();
}

export async function startHanabi(payload: {
  player_models: string[];
  seed?: number;
  event_delay_ms: number;
}) {
  const res = await fetch(`${API_BASE}/hanabi/start`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
  if (!res.ok) throw new Error("Failed to start hanabi");
  return res.json();
}

export async function startBatch(payload: any) {
  const res = await fetch(`${API_BASE}/batch/start`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
  if (!res.ok) throw new Error("Failed to start batch");
  return res.json();
}

export async function fetchReplays(): Promise<ReplaySummary[]> {
  const res = await fetch(`${API_BASE}/replays`);
  if (!res.ok) throw new Error("Failed to load replays");
  return res.json();
}

export async function fetchReplay(gameType: GameType, replayId: string) {
  const res = await fetch(`${API_BASE}/replays/${gameType}/${replayId}`);
  if (!res.ok) throw new Error("Failed to load replay");
  return res.json();
}

export async function fetchStats(replayId: string) {
  const res = await fetch(`${API_BASE}/stats/${replayId}`);
  if (!res.ok) throw new Error("Failed to load stats");
  return res.json();
}

export function openEventStream(path: string) {
  return new EventSource(`${API_BASE}${path}`);
}

// Cloud Benchmark APIs

export async function startBenchmark(payload: {
  experiment_name: string;
  model_ids: string[];
  seed_count?: number;
  seed_list?: number[];
  run_codenames?: boolean;
  run_decrypto?: boolean;
  run_hanabi?: boolean;
  codenames_concurrency?: number;
  decrypto_concurrency?: number;
  hanabi_concurrency?: number;
  codenames_mode?: string;
  codenames_max_turns?: number;
  codenames_max_discussion_rounds?: number;
  decrypto_max_rounds?: number;
  decrypto_max_discussion_turns?: number;
  interim_analysis_batch_size?: number;
  max_retries?: number;
  temperature?: number;
}) {
  const res = await fetch(`${API_BASE}/benchmark/start`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
  if (!res.ok) {
    const error = await res.json().catch(() => ({ detail: "Failed to start benchmark" }));
    throw new Error(error.detail || "Failed to start benchmark");
  }
  return res.json();
}

export async function fetchBenchmarkStatus() {
  const res = await fetch(`${API_BASE}/benchmark/status`);
  if (!res.ok) throw new Error("Failed to get benchmark status");
  return res.json();
}

export async function pauseBenchmark() {
  const res = await fetch(`${API_BASE}/benchmark/pause`, { method: "POST" });
  if (!res.ok) throw new Error("Failed to pause benchmark");
  return res.json();
}

export async function fetchBenchmarkFindings() {
  const res = await fetch(`${API_BASE}/benchmark/findings`);
  if (!res.ok) throw new Error("Failed to get findings");
  return res.json();
}

export async function fetchBenchmarkFinding(findingId: string) {
  const res = await fetch(`${API_BASE}/benchmark/findings/${findingId}`);
  if (!res.ok) throw new Error("Failed to get finding");
  return res.json();
}

export async function fetchExperiments() {
  const res = await fetch(`${API_BASE}/benchmark/experiments`);
  if (!res.ok) throw new Error("Failed to get experiments");
  return res.json();
}

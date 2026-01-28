import { GameType, ReplaySummary, TeamSelection } from "./types";

// In production (same origin), use relative URLs. In dev, use localhost.
const API_BASE =
  import.meta.env.VITE_API_BASE ||
  (window.location.hostname === "localhost" ? "http://localhost:8000" : "");

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

export async function fetchBenchmarkFindings(experimentName?: string) {
  const params = experimentName ? `?experiment_name=${encodeURIComponent(experimentName)}` : "";
  const res = await fetch(`${API_BASE}/benchmark/findings${params}`);
  if (!res.ok) throw new Error("Failed to get findings");
  return res.json();
}

export async function fetchBenchmarkFinding(findingId: string, experimentName?: string) {
  const params = experimentName ? `?experiment_name=${encodeURIComponent(experimentName)}` : "";
  const res = await fetch(`${API_BASE}/benchmark/findings/${findingId}${params}`);
  if (!res.ok) throw new Error("Failed to get finding");
  return res.json();
}

export async function fetchExperiments() {
  const res = await fetch(`${API_BASE}/benchmark/experiments`);
  if (!res.ok) throw new Error("Failed to get experiments");
  return res.json();
}

export async function cancelBenchmark(experimentName?: string) {
  const params = experimentName ? `?experiment_name=${encodeURIComponent(experimentName)}` : "";
  const res = await fetch(`${API_BASE}/benchmark/cancel${params}`, { method: "POST" });
  if (!res.ok) throw new Error("Failed to cancel benchmark");
  return res.json();
}

export async function forceStopBenchmark(experimentName: string) {
  const res = await fetch(`${API_BASE}/benchmark/force-stop/${encodeURIComponent(experimentName)}`, { method: "POST" });
  if (!res.ok) throw new Error("Failed to force-stop benchmark");
  return res.json();
}

export function downloadBenchmarkResults(experimentName: string) {
  // Direct download via browser
  window.location.href = `${API_BASE}/benchmark/download/${experimentName}`;
}

// Running games management

export type RunningGame = {
  game_id: string;
  game_type: "codenames" | "decrypto" | "hanabi";
  matchup_id: string;
  seed: number;
  models: Record<string, string>;
  started_at: string;
  duration_seconds: number;
  current_turn: number | null;
};

export type GamePeek = {
  game_id: string;
  game_type: "codenames" | "decrypto" | "hanabi";
  current_turn: number | null;
  recent_transcript: Record<string, unknown>[];
  agent_scratchpads: Record<string, string>;
  started_at: string;
  duration_seconds: number;
  last_update: string;
  stale_warning: boolean;
};

export async function fetchRunningGames(): Promise<RunningGame[]> {
  const res = await fetch(`${API_BASE}/benchmark/running-games`);
  if (!res.ok) throw new Error("Failed to get running games");
  return res.json();
}

export async function peekGame(gameId: string): Promise<GamePeek> {
  const res = await fetch(`${API_BASE}/benchmark/game-peek?game_id=${encodeURIComponent(gameId)}`);
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: `HTTP ${res.status}` }));
    throw new Error(err.detail || `Failed to peek at game (${res.status})`);
  }
  return res.json();
}

export async function restartGame(gameId: string) {
  const res = await fetch(`${API_BASE}/benchmark/restart-game?game_id=${encodeURIComponent(gameId)}`, {
    method: "POST",
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: `HTTP ${res.status}` }));
    throw new Error(err.detail || `Failed to restart game (${res.status})`);
  }
  return res.json();
}

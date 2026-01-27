import { GameType, ReplaySummary, TeamSelection, ClueGenerationMode, BatchClueGenerationMode } from "./types";

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
  clue_generation_mode?: ClueGenerationMode;
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
  seed: number;
  max_rounds: number;
  max_discussion_turns_per_guesser: number;
  event_delay_ms: number;
  clue_generation_mode?: ClueGenerationMode;
}) {
  const res = await fetch(`${API_BASE}/decrypto/start`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
  if (!res.ok) throw new Error("Failed to start decrypto");
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

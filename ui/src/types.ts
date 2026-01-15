export type GameType = "codenames" | "decrypto";

export interface ModelInfo {
  model_id: string;
  name: string;
  provider: string;
  base_url?: string | null;
}

export interface TeamRoleConfig {
  cluer: string;
  guesser_1: string;
  guesser_2?: string;
}

export interface TeamSelection {
  red: TeamRoleConfig;
  blue: TeamRoleConfig;
}

export interface ReplaySummary {
  replay_id: string;
  game_type: GameType;
  filename: string;
  timestamp?: string | null;
}

export interface CodenamesEventPayload {
  event: any;
  revealed: Record<string, string>;
  current_turn: string;
  phase: string;
  current_clue: any | null;
  guesses_remaining: number;
}

export interface DecryptoRoundPayload {
  round_number: number;
  public_clues: any;
  reveal_true_codes: any;
  actions: any;
  counters_after: any;
}

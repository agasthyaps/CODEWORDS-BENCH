export type GameType = "codenames" | "decrypto" | "hanabi";

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

export interface ScratchpadEntry {
  agent_id: string;
  addition: string;
  turn: number;
  timestamp: number;
}

// Hanabi types
export interface HanabiCard {
  color: string;
  number: number;
}

export interface HanabiCardKnowledge {
  known_color: string | null;
  known_number: number | null;
  possible_colors: string[];
  possible_numbers: number[];
}

export interface HanabiTurnPayload {
  turn_number: number;
  player_id: string;
  action: {
    action_type: "play" | "discard" | "hint";
    card_position?: number;
    target_player?: string;
    hint_type?: "color" | "number";
    hint_value?: string | number;
  };
  result: {
    success: boolean;
    message: string;
    card_played?: HanabiCard;
    card_discarded?: HanabiCard;
    was_playable?: boolean;
    positions_touched?: number[];
  };
  rationale: string;
  hint_tokens: number;
  fuse_tokens: number;
  score: number;
}

export interface HanabiInitPayload {
  game_type: "hanabi";
  config: {
    num_players: number;
    hand_size: number;
    max_hints: number;
    max_fuses: number;
    seed: number | null;
  };
  player_order: string[];
  episode_id: string;
}

// Leaderboard types
export interface CodenamesRanking {
  model: string;
  games: number;
  wins: number;
  win_rate: number;
  avg_cluer_surprise?: number | null;
  avg_cluer_bias?: number | null;
}

export interface DecryptoRanking {
  model: string;
  games: number;
  wins: number;
  win_rate: number;
  decode_accuracy: number | null;  // Teammate understanding
  intercept_accuracy: number | null;  // Opponent modeling (pure ToM)
  adversarial_score?: number | null;  // Composite adversarial score
  decode_brier?: number | null;
  intercept_brier?: number | null;
  decode_bias?: number | null;
  intercept_bias?: number | null;
}

export interface HanabiRanking {
  model: string;
  games: number;
  avg_score: number;
  score_pct: number;
  // Efficiency metrics (key research insight)
  efficiency: number;  // score/turn - measures true cooperative ToM
  avg_turns: number;
  turn_limit_pct: number;  // % games hitting turn limit
}

export interface OverallRanking {
  rank: number;
  model: string;
  games_played: number;
  overall_score: number;  // Efficiency-based composite
  codenames_score: number | null;
  decrypto_score: number | null;
  hanabi_score: number | null;  // Efficiency-based
  // Raw score composite (for comparison toggle)
  raw_overall_score: number;
  raw_hanabi_score: number | null;
  // Detailed metrics
  hanabi_efficiency: number | null;
  decrypto_decode: number | null;
  decrypto_intercept: number | null;
  decrypto_adversarial?: number | null;
  decrypto_win_rate?: number | null;
  codenames_cluer_surprise?: number | null;
  codenames_cluer_bias?: number | null;
  decrypto_decode_brier?: number | null;
  decrypto_intercept_brier?: number | null;
  decrypto_decode_bias?: number | null;
  decrypto_intercept_bias?: number | null;
}

export interface LeaderboardData {
  generated_at: string;
  total_episodes: {
    codenames: number;
    decrypto: number;
    hanabi: number;
  };
  overall_rankings: OverallRanking[];
  codenames_rankings: CodenamesRanking[];
  decrypto_rankings: DecryptoRanking[];
  hanabi_rankings: HanabiRanking[];
}

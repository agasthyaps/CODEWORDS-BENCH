import { useEffect, useMemo, useState } from "react";
import { openEventStream, startBatch } from "../api";
import ModelPicker from "../components/ModelPicker";
import { ModelInfo, TeamRoleConfig, TeamSelection } from "../types";

type Props = {
  models: ModelInfo[];
  defaultModel: string;
};

type SeedMode = "random" | "fixed" | "list";
type BatchGameType = "codenames" | "decrypto" | "both";

type GameResult = {
  game_index: number;
  seed: number;
  game_type: string;
  replay_id: string | null;
  status: string;
  error?: string | null;
};

const DEFAULT_TEAM: TeamRoleConfig = {
  cluer: "",
  guesser_1: "",
  guesser_2: "",
};

// Game-specific default options
const GAME_DEFAULTS = {
  codenames: {
    mode: "STANDARD",
    max_discussion_rounds: 3,
    max_turns: 50,
  },
  decrypto: {
    max_rounds: 8,
    max_discussion_turns_per_guesser: 2,
  },
  // Future games add their defaults here
} as const;

export default function BatchRunner({ models, defaultModel }: Props) {
  const baseTeam = useMemo(
    () => ({ ...DEFAULT_TEAM, cluer: defaultModel, guesser_1: defaultModel, guesser_2: defaultModel }),
    [defaultModel]
  );
  
  // Core config
  const [gameType, setGameType] = useState<BatchGameType>("codenames");
  const [red, setRed] = useState<TeamRoleConfig>(baseTeam);
  const [blue, setBlue] = useState<TeamRoleConfig>(baseTeam);
  
  // Seed config
  const [seedMode, setSeedMode] = useState<SeedMode>("random");
  const [count, setCount] = useState(5);
  const [fixedSeed, setFixedSeed] = useState<number>(42);
  const [seedListInput, setSeedListInput] = useState("42, 123, 456");
  
  // Game-specific options
  const [codenamesMode, setCodenamesMode] = useState<string>(GAME_DEFAULTS.codenames.mode);
  const [maxDiscussionRounds, setMaxDiscussionRounds] = useState<number>(GAME_DEFAULTS.codenames.max_discussion_rounds);
  const [maxTurns, setMaxTurns] = useState<number>(GAME_DEFAULTS.codenames.max_turns);
  const [maxRounds, setMaxRounds] = useState<number>(GAME_DEFAULTS.decrypto.max_rounds);
  const [maxDiscussionTurns, setMaxDiscussionTurns] = useState<number>(GAME_DEFAULTS.decrypto.max_discussion_turns_per_guesser);
  
  // UI state
  const [showAdvanced, setShowAdvanced] = useState(false);
  const [progress, setProgress] = useState({ completed: 0, total: 0 });
  const [status, setStatus] = useState<"idle" | "running" | "finished" | "error">("idle");
  const [results, setResults] = useState<GameResult[]>([]);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    setRed(baseTeam);
    setBlue(baseTeam);
  }, [baseTeam]);

  // Parse seed list from input
  const parsedSeedList = useMemo(() => {
    try {
      return seedListInput
        .split(/[,\s]+/)
        .map(s => parseInt(s.trim(), 10))
        .filter(n => !isNaN(n));
    } catch {
      return [];
    }
  }, [seedListInput]);

  // Calculate total games based on seed mode and game type
  const seedsCount = seedMode === "list" ? parsedSeedList.length : count;
  const totalGames = gameType === "both" ? seedsCount * 2 : seedsCount;

  function buildSelection(): TeamSelection {
    return { red, blue };
  }

  async function handleStart() {
    setProgress({ completed: 0, total: totalGames });
    setResults([]);
    setError(null);
    setStatus("running");
    
    const payload: any = {
      game_type: gameType,
      team_selection: buildSelection(),
      seed_mode: seedMode,
      count: seedMode !== "list" ? count : undefined,
      fixed_seed: seedMode === "fixed" ? fixedSeed : undefined,
      seed_list: seedMode === "list" ? parsedSeedList : undefined,
      // Codenames options
      codenames_mode: codenamesMode,
      max_discussion_rounds: maxDiscussionRounds,
      max_turns: maxTurns,
      // Decrypto options
      max_rounds: maxRounds,
      max_discussion_turns_per_guesser: maxDiscussionTurns,
    };
    
    try {
      const { job_id } = await startBatch(payload);
      const stream = openEventStream(`/batch/${job_id}/events`);
      
      stream.addEventListener("progress", (ev: MessageEvent) => {
        const data = JSON.parse(ev.data);
        setProgress({ completed: data.completed, total: data.total });
        if (data.last_result) {
          setResults(prev => [...prev, data.last_result]);
        }
      });
      
      stream.addEventListener("done", (ev: MessageEvent) => {
        const data = JSON.parse(ev.data);
        setResults(data.results || []);
        setStatus("finished");
        stream.close();
      });
      
      stream.addEventListener("job_error", (ev: MessageEvent) => {
        const data = JSON.parse(ev.data);
        setError(data.error || "Batch failed");
        setStatus("error");
        stream.close();
      });
      
      stream.addEventListener("error", () => {
        setStatus("error");
        stream.close();
      });
    } catch (e) {
      setError(String(e));
      setStatus("error");
    }
  }

  // Compute summary stats from results
  const summary = useMemo(() => {
    if (results.length === 0) return null;
    
    const completed = results.filter(r => r.status === "done").length;
    const failed = results.filter(r => r.status === "error").length;
    
    // For now, we don't have winner info in results - would need to fetch replays
    // This could be enhanced later
    return {
      total: results.length,
      completed,
      failed,
    };
  }, [results]);

  const progressPercent = progress.total > 0 
    ? Math.round((progress.completed / progress.total) * 100) 
    : 0;

  return (
    <div className="page">
      <h2>Batch Runner</h2>
      
      <div className="batch-layout">
        {/* Config Panel */}
        <div className="panel batch-config">
          <h3>Batch Configuration</h3>
          
          {/* Game Type */}
          <div className="form-row">
            <label>Game</label>
            <select 
              value={gameType} 
              onChange={(e) => setGameType(e.target.value as BatchGameType)}
              disabled={status === "running"}
            >
              <option value="codenames">Codenames</option>
              <option value="decrypto">Decrypto</option>
              <option value="both">Both (comparative)</option>
              {/* Future: <option value="hanabi">Hanabi</option> */}
            </select>
          </div>
          
          {/* Seed Mode */}
          <div className="form-row">
            <label>Seeds</label>
            <select 
              value={seedMode} 
              onChange={(e) => setSeedMode(e.target.value as SeedMode)}
              disabled={status === "running"}
            >
              <option value="random">Random (unique per game)</option>
              <option value="fixed">Fixed (same board each game)</option>
              <option value="list">Specific seeds</option>
            </select>
          </div>
          
          {/* Seed-specific inputs */}
          {seedMode === "random" && (
            <div className="form-row">
              <label>Games</label>
              <input
                type="number"
                value={count}
                min={1}
                max={100}
                onChange={(e) => setCount(Math.max(1, Math.min(100, Number(e.target.value))))}
                disabled={status === "running"}
              />
            </div>
          )}
          
          {seedMode === "fixed" && (
            <>
              <div className="form-row">
                <label>Seed</label>
                <input
                  type="number"
                  value={fixedSeed}
                  onChange={(e) => setFixedSeed(Number(e.target.value))}
                  disabled={status === "running"}
                />
              </div>
              <div className="form-row">
                <label>Games</label>
                <input
                  type="number"
                  value={count}
                  min={1}
                  max={100}
                  onChange={(e) => setCount(Math.max(1, Math.min(100, Number(e.target.value))))}
                  disabled={status === "running"}
                />
              </div>
            </>
          )}
          
          {seedMode === "list" && (
            <div className="form-row">
              <label>Seeds</label>
              <input
                type="text"
                value={seedListInput}
                onChange={(e) => setSeedListInput(e.target.value)}
                placeholder="42, 123, 456"
                disabled={status === "running"}
              />
              <span className="hint">{parsedSeedList.length} seeds</span>
            </div>
          )}
          
          {/* Advanced Options Toggle */}
          <div 
            className="advanced-toggle"
            onClick={() => setShowAdvanced(!showAdvanced)}
          >
            <span className={`toggle-arrow ${showAdvanced ? "open" : ""}`}>▸</span>
            <span>Game Options</span>
          </div>
          
          {/* Game-specific options */}
          {showAdvanced && (
            <div className="advanced-options">
              {(gameType === "codenames" || gameType === "both") && (
                <div className="options-group">
                  {gameType === "both" && <div className="options-label">Codenames</div>}
                  <div className="form-row-compact">
                    <label>Mode</label>
                    <select 
                      value={codenamesMode} 
                      onChange={(e) => setCodenamesMode(e.target.value)}
                      disabled={status === "running"}
                    >
                      <option value="STANDARD">Standard</option>
                      <option value="NO_ASSASSIN">No Assassin</option>
                      <option value="SINGLE_GUESSER">Single Guesser</option>
                    </select>
                  </div>
                  <div className="form-row-compact">
                    <label>Discussion rounds</label>
                    <input
                      type="number"
                      value={maxDiscussionRounds}
                      min={1}
                      max={10}
                      onChange={(e) => setMaxDiscussionRounds(Number(e.target.value))}
                      disabled={status === "running"}
                    />
                  </div>
                  <div className="form-row-compact">
                    <label>Max turns</label>
                    <input
                      type="number"
                      value={maxTurns}
                      min={10}
                      max={100}
                      onChange={(e) => setMaxTurns(Number(e.target.value))}
                      disabled={status === "running"}
                    />
                  </div>
                </div>
              )}
              
              {(gameType === "decrypto" || gameType === "both") && (
                <div className="options-group">
                  {gameType === "both" && <div className="options-label">Decrypto</div>}
                  <div className="form-row-compact">
                    <label>Max rounds</label>
                    <input
                      type="number"
                      value={maxRounds}
                      min={1}
                      max={16}
                      onChange={(e) => setMaxRounds(Number(e.target.value))}
                      disabled={status === "running"}
                    />
                  </div>
                  <div className="form-row-compact">
                    <label>Discussion turns</label>
                    <input
                      type="number"
                      value={maxDiscussionTurns}
                      min={1}
                      max={5}
                      onChange={(e) => setMaxDiscussionTurns(Number(e.target.value))}
                      disabled={status === "running"}
                    />
                  </div>
                </div>
              )}
              
              {/* Future: game-specific options for new games */}
            </div>
          )}
          
          {/* Action */}
          <div className="batch-action">
            <button 
              onClick={handleStart} 
              disabled={status === "running" || totalGames === 0}
            >
              {status === "running" ? "Running..." : `Start ${totalGames} Games`}
            </button>
            
            <div className={`status ${status}`}>
              <span className="status-dot" />
              {status === "running" 
                ? `${progress.completed}/${progress.total} (${progressPercent}%)`
                : status.charAt(0).toUpperCase() + status.slice(1)
              }
            </div>
          </div>
          
          {error && <div className="error-banner">{error}</div>}
        </div>
        
        {/* Team Pickers */}
        <div className="batch-teams">
          <ModelPicker 
            models={models} 
            value={red} 
            onChange={setRed} 
            label="Red Team" 
          />
          <ModelPicker 
            models={models} 
            value={blue} 
            onChange={setBlue} 
            label="Blue Team" 
          />
        </div>
      </div>
      
      {/* Results */}
      {(results.length > 0 || status === "finished") && (
        <div className="panel batch-results">
          <h3>Results</h3>
          
          {/* Summary */}
          {summary && (
            <div className="batch-summary">
              <div className="summary-stat">
                <span className="stat-value">{summary.total}</span>
                <span className="stat-label">Total</span>
              </div>
              <div className="summary-stat success">
                <span className="stat-value">{summary.completed}</span>
                <span className="stat-label">Completed</span>
              </div>
              {summary.failed > 0 && (
                <div className="summary-stat error">
                  <span className="stat-value">{summary.failed}</span>
                  <span className="stat-label">Failed</span>
                </div>
              )}
            </div>
          )}
          
          {/* Game list */}
          <div className="batch-games">
            {results.map((result, idx) => (
              <div 
                key={idx} 
                className={`batch-game ${result.status === "error" ? "error" : ""}`}
              >
                <span className="game-index">#{result.game_index + 1}</span>
                <span className={`game-type-badge ${result.game_type}`}>
                  {result.game_type === "codenames" ? "CN" : "DC"}
                </span>
                <span className="game-seed">seed: {result.seed}</span>
                <span className={`game-status ${result.status}`}>
                  {result.status === "done" ? "✓" : result.status === "error" ? "✗" : "..."}
                </span>
                {result.replay_id && (
                  <a 
                    href={`#replay/${result.game_type}/${result.replay_id}`}
                    className="game-replay"
                  >
                    View
                  </a>
                )}
                {result.error && (
                  <span className="game-error" title={result.error}>
                    {result.error.slice(0, 50)}
                  </span>
                )}
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}

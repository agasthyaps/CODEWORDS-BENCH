import { useEffect, useMemo, useState } from "react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import { fetchStats, openEventStream, startHanabi, estimateGameCost, CostEstimate } from "../api";
import HanabiBoard from "../components/HanabiBoard";
import ChatPanel from "../components/ChatPanel";
import ScratchpadPanel from "../components/ScratchpadPanel";
import { ModelInfo, HanabiCard, HanabiCardKnowledge, HanabiTurnPayload, ScratchpadEntry } from "../types";

type Props = {
  models: ModelInfo[];
  defaultModel: string;
};

type GameState = {
  playerOrder: string[];
  visibleHands: Record<string, HanabiCard[]>;
  myKnowledge: HanabiCardKnowledge[];
  currentPlayer: string;
  playedCards: Record<string, number>;
  discardPile: HanabiCard[];
  hintTokens: number;
  fuseTokens: number;
  score: number;
  deckRemaining: number;
};

const INITIAL_GAME_STATE: GameState = {
  playerOrder: [],
  visibleHands: {},
  myKnowledge: [],
  currentPlayer: "",
  playedCards: { red: 0, yellow: 0, green: 0, blue: 0, white: 0 },
  discardPile: [],
  hintTokens: 8,
  fuseTokens: 3,
  score: 0,
  deckRemaining: 50,
};

export default function HanabiViewer({ models, defaultModel }: Props) {
  const [playerModels, setPlayerModels] = useState<string[]>([
    defaultModel,
    defaultModel,
    defaultModel,
  ]);
  const [seed, setSeed] = useState<number | undefined>(undefined);
  const [eventDelay, setEventDelay] = useState(500);
  const [status, setStatus] = useState("idle");
  const [error, setError] = useState<string | null>(null);
  const [gameState, setGameState] = useState<GameState>(INITIAL_GAME_STATE);
  const [turnLog, setTurnLog] = useState<string[]>([]);
  const [scratchpadEntries, setScratchpadEntries] = useState<ScratchpadEntry[]>([]);
  const [finalScore, setFinalScore] = useState<number | null>(null);
  const [gameOverReason, setGameOverReason] = useState<string | null>(null);
  const [analysis, setAnalysis] = useState<string | null>(null);
  const [analysisLoading, setAnalysisLoading] = useState(false);
  const [replayId, setReplayId] = useState<string | null>(null);
  const [costEstimate, setCostEstimate] = useState<CostEstimate | null>(null);
  const [costLoading, setCostLoading] = useState(false);

  useEffect(() => {
    setPlayerModels([defaultModel, defaultModel, defaultModel]);
  }, [defaultModel]);

  const primaryModel = playerModels[0];

  useEffect(() => {
    if (!primaryModel) {
      setCostEstimate(null);
      return;
    }

    let cancelled = false;

    const estimateCost = async () => {
      setCostLoading(true);
      try {
        const estimate = await estimateGameCost(primaryModel, "hanabi");
        if (!cancelled) {
          setCostEstimate(estimate);
        }
      } catch (e) {
        if (!cancelled) {
          console.warn("Cost estimation failed:", e);
          setCostEstimate({
            estimated_cost_usd: 0,
            estimated_cost_display: "N/A",
            breakdown: {},
            confidence: "low",
            notes: ["Cost estimation unavailable"],
          });
        }
      } finally {
        if (!cancelled) {
          setCostLoading(false);
        }
      }
    };

    const timer = setTimeout(estimateCost, 300);
    return () => {
      cancelled = true;
      clearTimeout(timer);
    };
  }, [primaryModel]);

  const isRunning = status === "running" || status === "starting";

  async function handleStart() {
    setStatus("starting");
    setTurnLog([]);
    setError(null);
    setScratchpadEntries([]);
    setFinalScore(null);
    setGameOverReason(null);
    setAnalysis(null);
    setAnalysisLoading(false);
    setReplayId(null);
    setGameState(INITIAL_GAME_STATE);

    try {
      const { job_id } = await startHanabi({
        player_models: playerModels,
        seed,
        event_delay_ms: eventDelay,
      });

      const stream = openEventStream(`/hanabi/${job_id}/events`);
      
      stream.addEventListener("init", (ev: MessageEvent) => {
        const payload = JSON.parse(ev.data);
        setGameState(prev => ({
          ...prev,
          playerOrder: payload.player_order,
          visibleHands: payload.hands || {},
          playedCards: payload.played_cards || prev.playedCards,
          discardPile: payload.discard_pile || [],
          hintTokens: payload.hint_tokens ?? prev.hintTokens,
          fuseTokens: payload.fuse_tokens ?? prev.fuseTokens,
          deckRemaining: payload.deck_remaining ?? prev.deckRemaining,
          currentPlayer: payload.player_order?.[0] || "",
        }));
        setStatus("running");
      });

      stream.addEventListener("turn", (ev: MessageEvent) => {
        const payload = JSON.parse(ev.data);
        
        // Update full game state from turn result
        setGameState(prev => ({
          ...prev,
          hintTokens: payload.hint_tokens,
          fuseTokens: payload.fuse_tokens,
          score: payload.score,
          visibleHands: payload.hands || prev.visibleHands,
          playedCards: payload.played_cards || prev.playedCards,
          discardPile: payload.discard_pile || prev.discardPile,
          deckRemaining: payload.deck_remaining ?? prev.deckRemaining,
          currentPlayer: payload.current_player || prev.currentPlayer,
        }));
        
        // Build log entry
        let logEntry = `T${payload.turn_number} ${payload.player_id}: `;
        const action = payload.action;
        const result = payload.result;
        
        if (action.action_type === "play") {
          const card = result.card_played;
          const success = result.was_playable ? "✓" : "✗";
          logEntry += `PLAY pos ${action.card_position}`;
          if (card) {
            logEntry += ` → ${card.color[0].toUpperCase()}${card.number} ${success}`;
          }
        } else if (action.action_type === "discard") {
          const card = result.card_discarded;
          logEntry += `DISCARD pos ${action.card_position}`;
          if (card) {
            logEntry += ` → ${card.color[0].toUpperCase()}${card.number}`;
          }
        } else if (action.action_type === "hint") {
          logEntry += `HINT ${action.target_player} ${action.hint_type}=${action.hint_value}`;
          if (result.positions_touched) {
            logEntry += ` (pos ${result.positions_touched.join(",")})`;
          }
        }
        
        if (payload.rationale) {
          logEntry += ` | "${payload.rationale.slice(0, 60)}${payload.rationale.length > 60 ? '...' : ''}"`;
        }
        
        setTurnLog(prev => [...prev, logEntry]);
      });

      stream.addEventListener("scratchpad", (ev: MessageEvent) => {
        const payload = JSON.parse(ev.data);
        setScratchpadEntries(prev => [...prev, {
          agent_id: payload.agent_id,
          addition: payload.addition,
          turn: payload.turn,
          timestamp: Date.now(),
        }]);
      });

      stream.addEventListener("done", async (ev: MessageEvent) => {
        const payload = JSON.parse(ev.data);
        setStatus("finished");
        setFinalScore(payload.final_score);
        setGameOverReason(payload.game_over_reason);
        if (payload.replay_id) {
          setReplayId(payload.replay_id);
          setAnalysisLoading(true);
          // Poll for stats since stream closes before stats event arrives
          const start = Date.now();
          const poll = async () => {
            try {
              const stats = await fetchStats(payload.replay_id);
              if (stats?.analysis) {
                setAnalysis(stats.analysis);
                setAnalysisLoading(false);
                return;
              }
            } catch {
              // keep polling
            }
            if (Date.now() - start < 120000) {
              setTimeout(poll, 2000);
            } else {
              setAnalysisLoading(false);
            }
          };
          poll();
        }
        stream.close();
      });

      stream.addEventListener("stats", (ev: MessageEvent) => {
        const payload = JSON.parse(ev.data);
        setAnalysis(payload.analysis || null);
        setAnalysisLoading(false);
      });

      stream.addEventListener("job_error", (ev: MessageEvent) => {
        const payload = JSON.parse(ev.data);
        setError(payload.error || "Job failed");
        setStatus("error");
        stream.close();
      });

      stream.addEventListener("error", () => {
        setStatus("error");
        stream.close();
      });
    } catch (err: any) {
      setError(err.message);
      setStatus("error");
    }
  }

  const getScoreMessage = () => {
    if (finalScore === null) return null;
    if (finalScore === 25) return "Perfect score! 🎆";
    if (finalScore >= 21) return "Excellent! 🎇";
    if (finalScore >= 16) return "Good game! ✨";
    if (finalScore >= 11) return "Not bad";
    if (finalScore >= 6) return "Could be better";
    return "Better luck next time";
  };

  const getReasonMessage = () => {
    switch (gameOverReason) {
      case "perfect_score": return "All fireworks completed!";
      case "fuse_out": return "All fuse tokens lost";
      case "final_round_complete": return "Deck exhausted, final round complete";
      case "turn_limit": return "Turn limit reached";
      default: return gameOverReason;
    }
  };

  const modelNames = useMemo(() => {
    return playerModels.map(id => {
      const model = models.find(m => m.model_id === id);
      return model?.name || id;
    });
  }, [playerModels, models]);

  const costEstimateDisplay = costEstimate ?? {
    estimated_cost_usd: 0,
    estimated_cost_display: "N/A",
    breakdown: {},
    confidence: "low" as const,
    notes: ["Cost estimation unavailable"],
  };

  return (
    <div className="page">
      <h2>Hanabi</h2>

      {/* Settings Panel - Hidden when running */}
      {!isRunning && status !== "finished" && status !== "error" && (
        <div className="settings-panel">
          <div className="settings-content">
            <div className="settings-section">
              <h4 data-label="CONFIG">Game Settings</h4>
              <div className="form-row-compact">
                <label>Seed</label>
                <input
                  type="number"
                  placeholder="Random"
                  value={seed ?? ""}
                  onChange={(e) => setSeed(e.target.value ? Number(e.target.value) : undefined)}
                />
              </div>
              <div className="form-row-compact">
                <label>Event delay (ms)</label>
                <input
                  type="number"
                  value={eventDelay}
                  onChange={(e) => setEventDelay(Number(e.target.value))}
                />
              </div>
            </div>
            <div className="settings-section">
              <h4 data-label="AGENTS">Players (3 required)</h4>
              {[0, 1, 2].map(i => (
                <div className="form-row-compact" key={i}>
                  <label>Player {i + 1}</label>
                  <select
                    value={playerModels[i]}
                    onChange={(e) => {
                      const newModels = [...playerModels];
                      newModels[i] = e.target.value;
                      setPlayerModels(newModels);
                    }}
                  >
                    {models.map((m) => (
                      <option key={m.model_id} value={m.model_id}>{m.name}</option>
                    ))}
                  </select>
                </div>
              ))}
            </div>
          </div>
          <div className="settings-footer">
            <div className="settings-status-group">
              <div className={`status ${status}`}>
                <span className="status-dot" />
                {status.charAt(0).toUpperCase() + status.slice(1)}
              </div>
              {primaryModel && (
                <div className={`cost-estimate confidence-${costEstimateDisplay.confidence ?? 'low'}`}>
                  <span className="cost-value">
                    {costLoading ? "..." : costEstimateDisplay.estimated_cost_display}
                  </span>
                  <span className="cost-label">est. cost</span>
                  {costEstimateDisplay.confidence !== "high" && (
                    <span className="cost-confidence" title={costEstimateDisplay.notes?.join("; ")}>
                      ({costEstimateDisplay.confidence})
                    </span>
                  )}
                </div>
              )}
            </div>
            <button onClick={handleStart} disabled={!models.length}>
              Start Game
            </button>
          </div>
        </div>
      )}

      {/* Game Header - Running/Finished */}
      {(isRunning || status === "finished") && (
        <div className="game-header hanabi-header">
          <div className="game-header-teams">
            {modelNames.map((name, i) => (
              <span key={i} className="player-label">
                P{i + 1}: {name}
                {i < modelNames.length - 1 && <span className="separator"> | </span>}
              </span>
            ))}
          </div>
          
          <div className="game-header-scores">
            <div className="score-group">
              <span className="score-label">Score</span>
              <span className="score-value hanabi-score">{gameState.score}/25</span>
            </div>
            <div className="score-group">
              <span className="score-label">Hints</span>
              <span className="score-value">{gameState.hintTokens}/8</span>
            </div>
            <div className="score-group">
              <span className="score-label">Fuses</span>
              <span className="score-value fuse-count">{gameState.fuseTokens}/3</span>
            </div>
          </div>

          <div className={`status ${status}`} style={{ margin: 0 }}>
            <span className="status-dot" />
            {isRunning ? `Turn ${turnLog.length + 1}` : 'Finished'}
          </div>
        </div>
      )}

      {/* Game Over Banner */}
      {status === "finished" && finalScore !== null && (
        <div className={`win-banner ${finalScore >= 21 ? 'excellent' : finalScore >= 11 ? 'good' : 'poor'}`}>
          <span className="win-icon">
            {finalScore === 25 ? "🎆" : finalScore >= 16 ? "🎇" : "✨"}
          </span>
          <span className="win-text">
            Final Score: {finalScore}/25 - {getScoreMessage()}
          </span>
          <span className="reason-text">({getReasonMessage()})</span>
        </div>
      )}

      {/* Error State */}
      {status === "error" && (
        <div className="error-panel">
          <div className="error-panel-icon">⚠️</div>
          <h3>Game Error</h3>
          <p>The game encountered an error and could not continue.</p>
          {error && <div className="error-details">{error}</div>}
          <button onClick={() => setStatus("idle")} style={{ marginTop: 16 }}>
            Try Again
          </button>
        </div>
      )}

      {/* Game Area */}
      {status !== "error" && (isRunning || status === "finished") && (
        <>
          <div className="layout hanabi">
            <div className="left hanabi-log">
              <ChatPanel
                title="Turn Log"
                entries={turnLog}
                variant="hanabi"
                thinking={isRunning ? "Players thinking..." : undefined}
              />
            </div>
            <div className="center hanabi-center">
              <HanabiBoard
                playerOrder={gameState.playerOrder}
                visibleHands={gameState.visibleHands}
                myKnowledge={gameState.myKnowledge}
                currentPlayer={gameState.currentPlayer}
                playedCards={gameState.playedCards}
                discardPile={gameState.discardPile}
                hintTokens={gameState.hintTokens}
                fuseTokens={gameState.fuseTokens}
                score={gameState.score}
                deckRemaining={gameState.deckRemaining}
              />
            </div>
          </div>

          {/* Scratchpad Panel */}
          {(isRunning || scratchpadEntries.length > 0) && (
            <ScratchpadPanel entries={scratchpadEntries} isRunning={isRunning} />
          )}
        </>
      )}

      {/* Analysis Panel */}
      {status === "finished" && (analysisLoading || analysis) && (
        <div className="analysis-panel" style={{ marginTop: 24 }}>
          <h3>🧠 Opus Analysis</h3>
          {analysisLoading && !analysis && (
            <div className="analysis-loading">
              <span className="spinner" /> Analyzing game transcript...
            </div>
          )}
          {analysis && (
            <div className="analysis-content">
              <ReactMarkdown remarkPlugins={[remarkGfm]}>{analysis}</ReactMarkdown>
            </div>
          )}
        </div>
      )}

      {/* Finished state - allow restart */}
      {status === "finished" && (
        <div style={{ marginTop: 24, textAlign: 'center' }}>
          <button onClick={() => setStatus("idle")}>
            New Game
          </button>
        </div>
      )}
    </div>
  );
}

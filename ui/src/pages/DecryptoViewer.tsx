import { useEffect, useMemo, useState } from "react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import { fetchStats, openEventStream, startDecrypto } from "../api";
import DecryptoBoard from "../components/DecryptoBoard";
import ChatPanel from "../components/ChatPanel";
import { ModelInfo, TeamRoleConfig, TeamSelection } from "../types";

type Props = {
  models: ModelInfo[];
  defaultModel: string;
};

type Counters = {
  own_interceptions: number;
  own_miscommunications: number;
  opp_interceptions: number;
  opp_miscommunications: number;
};

type RoundResult = {
  round: number;
  redDecode: boolean;
  blueDecode: boolean;
  redIntercept: boolean;
  blueIntercept: boolean;
};

const DEFAULT_TEAM: TeamRoleConfig = {
  cluer: "",
  guesser_1: "",
  guesser_2: "",
};

export default function DecryptoViewer({ models, defaultModel }: Props) {
  const baseTeam = useMemo(
    () => ({ ...DEFAULT_TEAM, cluer: defaultModel, guesser_1: defaultModel, guesser_2: defaultModel }),
    [defaultModel]
  );
  const [red, setRed] = useState<TeamRoleConfig>(baseTeam);
  const [blue, setBlue] = useState<TeamRoleConfig>(baseTeam);
  const [seed, setSeed] = useState(0);
  const [eventDelay, setEventDelay] = useState(0);
  const [maxDiscussionTurns, setMaxDiscussionTurns] = useState(2);
  const [keys, setKeys] = useState<{ red: string[]; blue: string[] }>({ red: [], blue: [] });
  const [rounds, setRounds] = useState<any[]>([]);
  const [status, setStatus] = useState("idle");
  const [analysis, setAnalysis] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [redLog, setRedLog] = useState<string[]>([]);
  const [blueLog, setBlueLog] = useState<string[]>([]);
  const [analysisLoading, setAnalysisLoading] = useState(false);
  const [metricsOpen, setMetricsOpen] = useState(false);
  const [currentRound, setCurrentRound] = useState<{
    round: number;
    red: { code: number[]; clues: string[] };
    blue: { code: number[]; clues: string[] };
  } | null>(null);
  
  // Score tracking
  const [scores, setScores] = useState<{ red: Counters; blue: Counters } | null>(null);
  const [roundResults, setRoundResults] = useState<RoundResult[]>([]);
  const [winner, setWinner] = useState<string | null>(null);
  const [winReason, setWinReason] = useState<string | null>(null);

  useEffect(() => {
    setRed(baseTeam);
    setBlue(baseTeam);
  }, [baseTeam]);

  function buildSelection(): TeamSelection {
    return { red, blue };
  }

  const isRunning = status === "running" || status === "starting";

  async function handleStart() {
    setStatus("starting");
    setRounds([]);
    setAnalysis(null);
    setError(null);
    setRedLog([]);
    setBlueLog([]);
    setAnalysisLoading(false);
    setCurrentRound(null);
    setMetricsOpen(false);
    setScores(null);
    setRoundResults([]);
    setWinner(null);
    setWinReason(null);
    
    const { job_id } = await startDecrypto({
      team_selection: buildSelection(),
      seed,
      max_rounds: 8,
      max_discussion_turns_per_guesser: maxDiscussionTurns,
      event_delay_ms: eventDelay,
    });

    const stream = openEventStream(`/decrypto/${job_id}/events`);
    stream.addEventListener("init", (ev: MessageEvent) => {
      const payload = JSON.parse(ev.data);
      setKeys(payload.keys);
      setStatus("running");
    });
    stream.addEventListener("clue", (ev: MessageEvent) => {
      const payload = JSON.parse(ev.data);
      const entry = `Round ${payload.round}: ${payload.clues.join(" | ")} (code ${payload.code.join("-")})`;
      if (payload.team === "red") {
        setRedLog((prev) => [...prev, entry]);
      } else {
        setBlueLog((prev) => [...prev, entry]);
      }
      setCurrentRound((prev) => {
        const round = payload.round;
        const base = prev?.round === round
          ? prev
          : {
              round,
              red: { code: [], clues: [] },
              blue: { code: [], clues: [] },
            };
        const updated = { ...base };
        if (payload.team === "red") {
          updated.red = { code: payload.code || [], clues: payload.clues || [] };
        } else {
          updated.blue = { code: payload.code || [], clues: payload.clues || [] };
        }
        return updated;
      });
    });
    stream.addEventListener("action", (ev: MessageEvent) => {
      const payload = JSON.parse(ev.data);
      const action = payload.action;
      const guess = action?.consensus?.guess?.join("-") || "N/A";
      const line = `${payload.kind.toUpperCase()} consensus ${guess}`;
      const shares = (action?.share || []).map(
        (s: any) => `share (${s.agent_id}): ${s.message}`
      );
      if (payload.team === "red") {
        setRedLog((prev) => [...prev, line, ...shares]);
      } else {
        setBlueLog((prev) => [...prev, line, ...shares]);
      }
    });
    stream.addEventListener("discussion", (ev: MessageEvent) => {
      const payload = JSON.parse(ev.data);
      const kindLabel = payload.kind === "decode" ? "[DEC]" : "[INT]";
      const line = `${kindLabel} ${payload.agent_id}: ${payload.message}`;
      if (payload.team === "red") {
        setRedLog((prev) => [...prev, line]);
      } else {
        setBlueLog((prev) => [...prev, line]);
      }
    });
    stream.addEventListener("round", (ev: MessageEvent) => {
      const payload = JSON.parse(ev.data);
      setRounds((prev) => [...prev, payload]);
      
      // Update scores from counters_after
      if (payload.counters_after) {
        setScores(payload.counters_after);
      }
      
      // Track round results
      const finalGuesses = payload.final_guesses || [];
      const redDecode = finalGuesses.find((g: any) => g.team === "red" && g.kind === "decode");
      const blueDecode = finalGuesses.find((g: any) => g.team === "blue" && g.kind === "decode");
      const redIntercept = finalGuesses.find((g: any) => g.team === "red" && g.kind === "intercept");
      const blueIntercept = finalGuesses.find((g: any) => g.team === "blue" && g.kind === "intercept");
      
      setRoundResults((prev) => [...prev, {
        round: payload.round_number,
        redDecode: redDecode?.correct ?? false,
        blueDecode: blueDecode?.correct ?? false,
        redIntercept: redIntercept?.correct ?? false,
        blueIntercept: blueIntercept?.correct ?? false,
      }]);
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
      setAnalysisLoading(false);
      stream.close();
    });
    stream.addEventListener("done", (ev: MessageEvent) => {
      const payload = JSON.parse(ev.data);
      setStatus("finished");
      setWinner(payload.winner);
      setWinReason(payload.result_reason);
      setAnalysisLoading(true);
      if (payload.replay_id) {
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
    stream.addEventListener("error", () => {
      setStatus("error");
      stream.close();
    });
  }

  const fallbackRed = rounds.flatMap((round: any) => {
    const entries: string[] = [];
    const clue = round.public_clues?.red?.clues?.join(" | ");
    const code = round.reveal_true_codes?.red?.join("-");
    entries.push(`Round ${round.round_number}: ${clue} (code ${code})`);
    (round.actions || [])
      .filter((a: any) => a.team === "red")
      .forEach((action: any) => {
        const kind = action.kind?.toUpperCase();
        const guess = action.consensus?.guess?.join("-") || "N/A";
        entries.push(`${kind} consensus ${guess}`);
        (action.share || []).forEach((share: any) => {
          entries.push(`share (${share.agent_id}): ${share.message}`);
        });
      });
    return entries;
  });

  const fallbackBlue = rounds.flatMap((round: any) => {
    const entries: string[] = [];
    const clue = round.public_clues?.blue?.clues?.join(" | ");
    const code = round.reveal_true_codes?.blue?.join("-");
    entries.push(`Round ${round.round_number}: ${clue} (code ${code})`);
    (round.actions || [])
      .filter((a: any) => a.team === "blue")
      .forEach((action: any) => {
        const kind = action.kind?.toUpperCase();
        const guess = action.consensus?.guess?.join("-") || "N/A";
        entries.push(`${kind} consensus ${guess}`);
        (action.share || []).forEach((share: any) => {
          entries.push(`share (${share.agent_id}): ${share.message}`);
        });
      });
    return entries;
  });

  const redEntries = redLog.length ? redLog : fallbackRed;
  const blueEntries = blueLog.length ? blueLog : fallbackBlue;

  const detailedRoundResults = rounds.map((round: any) => {
    const trueCodes = round.true_codes || round.reveal_true_codes || {};
    const actions = round.final_guesses || round.actions || [];
    return {
      roundNumber: round.round_number,
      trueCodes,
      actions,
      clues: round.public_clues || {},
    };
  });

  const redModel = models.find(m => m.model_id === red.cluer);
  const blueModel = models.find(m => m.model_id === blue.cluer);

  const getWinMessage = () => {
    // Handle ties first (winner is null but we have a reason)
    if (winReason === "tie_interceptions") {
      return "It's a tie! Both teams reached 2 interceptions!";
    } else if (winReason === "tie_miscommunications") {
      return "It's a tie! Both teams had 2 miscommunications!";
    } else if (winReason === "survived" || winReason === "max_rounds") {
      return "Game ended — both teams survived all rounds!";
    }
    
    if (!winner) return null;
    const winnerName = winner === "red" ? "Red Team" : "Blue Team";
    const loserName = winner === "red" ? "Blue Team" : "Red Team";

    if (winReason === "interceptions") {
      return `${winnerName} wins by successful interceptions!`;
    } else if (winReason === "miscommunications") {
      return `${winnerName} wins — ${loserName} had too many miscommunications!`;
    }
    return `${winnerName} wins!`;
  };
  
  const isTieOrSurvived = winReason?.startsWith("tie_") || winReason === "survived" || winReason === "max_rounds";

  return (
    <div className="page">
      <h2>Decrypto</h2>
      
      {/* Settings Panel - Hidden when running */}
      {!isRunning && status !== "finished" && status !== "error" && (
        <div className="settings-panel">
          <div className="settings-content">
            <div className="settings-section">
              <h4>Game Settings</h4>
              <div className="form-row-compact">
                <label>Seed</label>
                <input
                  type="number"
                  value={seed}
                  onChange={(e) => setSeed(Number(e.target.value))}
                />
              </div>
              <div className="form-row-compact">
                <label>Event delay</label>
                <input
                  type="number"
                  value={eventDelay}
                  onChange={(e) => setEventDelay(Number(e.target.value))}
                />
              </div>
              <div className="form-row-compact">
                <label>Discussion</label>
                <input
                  type="number"
                  min={1}
                  max={4}
                  value={maxDiscussionTurns}
                  onChange={(e) => setMaxDiscussionTurns(Number(e.target.value))}
                />
              </div>
            </div>
            <div className="settings-section">
              <h4>Red Team</h4>
              <div className="form-row-compact">
                <label>Cluer</label>
                <select value={red.cluer} onChange={(e) => setRed({ ...red, cluer: e.target.value })}>
                  {models.map((m) => <option key={m.model_id} value={m.model_id}>{m.name}</option>)}
                </select>
              </div>
              <div className="form-row-compact">
                <label>Guesser 1</label>
                <select value={red.guesser_1} onChange={(e) => setRed({ ...red, guesser_1: e.target.value })}>
                  {models.map((m) => <option key={m.model_id} value={m.model_id}>{m.name}</option>)}
                </select>
              </div>
              <div className="form-row-compact">
                <label>Guesser 2</label>
                <select value={red.guesser_2 || red.guesser_1} onChange={(e) => setRed({ ...red, guesser_2: e.target.value })}>
                  {models.map((m) => <option key={m.model_id} value={m.model_id}>{m.name}</option>)}
                </select>
              </div>
            </div>
            <div className="settings-section">
              <h4>Blue Team</h4>
              <div className="form-row-compact">
                <label>Cluer</label>
                <select value={blue.cluer} onChange={(e) => setBlue({ ...blue, cluer: e.target.value })}>
                  {models.map((m) => <option key={m.model_id} value={m.model_id}>{m.name}</option>)}
                </select>
              </div>
              <div className="form-row-compact">
                <label>Guesser 1</label>
                <select value={blue.guesser_1} onChange={(e) => setBlue({ ...blue, guesser_1: e.target.value })}>
                  {models.map((m) => <option key={m.model_id} value={m.model_id}>{m.name}</option>)}
                </select>
              </div>
              <div className="form-row-compact">
                <label>Guesser 2</label>
                <select value={blue.guesser_2 || blue.guesser_1} onChange={(e) => setBlue({ ...blue, guesser_2: e.target.value })}>
                  {models.map((m) => <option key={m.model_id} value={m.model_id}>{m.name}</option>)}
                </select>
              </div>
            </div>
          </div>
          <div className="settings-footer">
            <div className={`status ${status}`}>
              <span className="status-dot" />
              {status.charAt(0).toUpperCase() + status.slice(1)}
            </div>
            <button onClick={handleStart} disabled={!models.length}>
              Start Game
            </button>
          </div>
        </div>
      )}

      {/* Game Header - Running/Finished */}
      {(isRunning || status === "finished") && (
        <div className="game-header">
          <div className="game-header-teams">
            <span className="team-label red">{redModel?.name}</span>
            <span className="vs">vs</span>
            <span className="team-label blue">{blueModel?.name}</span>
          </div>
          
          {/* Scores */}
          <div className="game-header-scores">
            <div className="score-group red">
              <span className="score-icon">🎯</span>
              <span className="score-value">{scores?.red?.own_interceptions ?? 0}</span>
              <span className="score-label">INT</span>
              <span className="score-icon">💀</span>
              <span className="score-value">{scores?.red?.own_miscommunications ?? 0}</span>
              <span className="score-label">MIS</span>
            </div>
            <div className="score-divider">|</div>
            <div className="score-group blue">
              <span className="score-icon">🎯</span>
              <span className="score-value">{scores?.blue?.own_interceptions ?? 0}</span>
              <span className="score-label">INT</span>
              <span className="score-icon">💀</span>
              <span className="score-value">{scores?.blue?.own_miscommunications ?? 0}</span>
              <span className="score-label">MIS</span>
            </div>
          </div>

          {/* Round Results */}
          {roundResults.length > 0 && (
            <div className="round-indicators">
              {roundResults.map((rr, idx) => (
                <div key={idx} className="round-indicator" title={`Round ${rr.round}`}>
                  <span className="round-num">R{rr.round}</span>
                  <span className={`result-icon ${rr.redDecode ? 'success' : 'fail'}`} title="Red Decode">
                    {rr.redDecode ? '✓' : '✗'}
                  </span>
                  <span className={`result-icon ${rr.redIntercept ? 'success' : ''}`} title="Red Intercept">
                    {rr.redIntercept ? '🎯' : '·'}
                  </span>
                  <span className="result-divider">/</span>
                  <span className={`result-icon ${rr.blueDecode ? 'success' : 'fail'}`} title="Blue Decode">
                    {rr.blueDecode ? '✓' : '✗'}
                  </span>
                  <span className={`result-icon ${rr.blueIntercept ? 'success' : ''}`} title="Blue Intercept">
                    {rr.blueIntercept ? '🎯' : '·'}
                  </span>
                </div>
              ))}
            </div>
          )}

          {/* Status */}
          <div className={`status ${status}`} style={{ margin: 0 }}>
            <span className="status-dot" />
            {isRunning ? `Round ${currentRound?.round || 1}` : 'Finished'}
          </div>
        </div>
      )}

      {/* Win/Tie Notification */}
      {status === "finished" && (winner || isTieOrSurvived) && (
        <div className={`win-banner ${winner || "tie"}`}>
          <span className="win-icon">
            {winner === "red" ? "🔴" : winner === "blue" ? "🔵" : "🤝"}
          </span>
          <span className="win-text">{getWinMessage()}</span>
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
      {status !== "error" && (keys.red.length > 0 || status === "finished") && (
        <>
          <div className="layout decrypto">
            <div className="left">
              <ChatPanel 
                title="Red Team Log" 
                entries={redEntries} 
                variant="decrypto" 
                teamColor="red"
                thinking={isRunning ? "Red team thinking..." : undefined}
              />
            </div>
            <div className="center">
              <DecryptoBoard 
                redKey={keys.red || []} 
                blueKey={keys.blue || []} 
                redCurrent={currentRound?.red}
                blueCurrent={currentRound?.blue}
                currentRound={currentRound?.round}
              />
            </div>
            <div className="right">
              <ChatPanel 
                title="Blue Team Log" 
                entries={blueEntries} 
                variant="decrypto" 
                teamColor="blue"
                thinking={isRunning ? "Blue team thinking..." : undefined}
              />
            </div>
          </div>

          {/* Analysis below the game board */}
          <div className="below-layout">
            {analysisLoading && (
              <div className="panel">
                <h3>Analysis</h3>
                <div className="loading-text">Analyzing game results...</div>
              </div>
            )}
            {analysis && (
              <div className="panel analysis-panel">
                <h3>Analysis</h3>
                <div className="analysis-content">
                  <ReactMarkdown remarkPlugins={[remarkGfm]}>{analysis}</ReactMarkdown>
                </div>
              </div>
            )}

            {/* Collapsible Metrics */}
            {detailedRoundResults.length > 0 && (
              <div className={`collapsible ${metricsOpen ? 'open' : ''}`}>
                <div className="collapsible-header" onClick={() => setMetricsOpen(!metricsOpen)}>
                  <h3>Detailed Round Results ({detailedRoundResults.length} rounds)</h3>
                  <span className="collapsible-toggle">▼</span>
                </div>
                <div className="collapsible-content">
                  {detailedRoundResults.map((rr) => (
                    <div key={`round-${rr.roundNumber}`} className="round-result">
                      <div className="round-num">Round {rr.roundNumber}</div>
                      <div>
                        <strong>Codes:</strong>{" "}
                        Red {rr.trueCodes.red?.join("-") || "—"} | Blue{" "}
                        {rr.trueCodes.blue?.join("-") || "—"}
                      </div>
                      <div>
                        <strong>Clues:</strong>{" "}
                        Red {(rr.clues.red?.clues || []).join(" | ")} | Blue{" "}
                        {(rr.clues.blue?.clues || []).join(" | ")}
                      </div>
                      <ul>
                        {rr.actions.map((a: any, idx: number) => (
                          <li key={`a-${rr.roundNumber}-${idx}`}>
                            {a.team?.toUpperCase?.()} {a.kind?.toUpperCase?.()}:{" "}
                            {(a.guess || a.consensus?.guess || []).join("-") || "N/A"}{" "}
                            {a.correct ? "✓" : "✗"}
                          </li>
                        ))}
                      </ul>
                    </div>
                  ))}
                </div>
              </div>
            )}
          </div>
        </>
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

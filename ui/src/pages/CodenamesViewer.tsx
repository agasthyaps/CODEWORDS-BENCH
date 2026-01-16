import { useEffect, useMemo, useState } from "react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import {
  fetchStats,
  openEventStream,
  startCodenames,
} from "../api";
import CodenamesBoard from "../components/CodenamesBoard";
import ChatPanel from "../components/ChatPanel";
import { ModelInfo, TeamRoleConfig, TeamSelection } from "../types";

type Props = {
  models: ModelInfo[];
  defaultModel: string;
};

const DEFAULT_TEAM: TeamRoleConfig = {
  cluer: "",
  guesser_1: "",
  guesser_2: "",
};

export default function CodenamesViewer({ models, defaultModel }: Props) {
  const baseTeam = useMemo(
    () => ({ ...DEFAULT_TEAM, cluer: defaultModel, guesser_1: defaultModel, guesser_2: defaultModel }),
    [defaultModel]
  );
  const [red, setRed] = useState<TeamRoleConfig>(baseTeam);
  const [blue, setBlue] = useState<TeamRoleConfig>(baseTeam);
  const [mode, setMode] = useState("STANDARD");
  const [seed, setSeed] = useState<number | undefined>(undefined);
  const [eventDelay, setEventDelay] = useState(0);
  const [boardWords, setBoardWords] = useState<string[]>([]);
  const [keyByWord, setKeyByWord] = useState<Record<string, string>>({});
  const [revealed, setRevealed] = useState<Record<string, string>>({});
  const [transcript, setTranscript] = useState<any[]>([]);
  const [status, setStatus] = useState<string>("idle");
  const [metrics, setMetrics] = useState<any | null>(null);
  const [analysis, setAnalysis] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [analysisLoading, setAnalysisLoading] = useState(false);
  const [metricsOpen, setMetricsOpen] = useState(false);
  const [winner, setWinner] = useState<string | null>(null);

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
    setMetrics(null);
    setAnalysis(null);
    setError(null);
    setAnalysisLoading(false);
    setTranscript([]);
    setRevealed({});
    setMetricsOpen(false);
    setWinner(null);
    
    const { job_id } = await startCodenames({
      team_selection: buildSelection(),
      mode,
      seed,
      max_discussion_rounds: 3,
      max_turns: 50,
      event_delay_ms: eventDelay,
    });

    const stream = openEventStream(`/codenames/${job_id}/events`);
    stream.addEventListener("init", (ev: MessageEvent) => {
      const payload = JSON.parse(ev.data);
      setBoardWords(payload.board.words || []);
      setKeyByWord(payload.board.key_by_word || {});
      setStatus("running");
    });
    stream.addEventListener("event", (ev: MessageEvent) => {
      const payload = JSON.parse(ev.data);
      setTranscript((prev) => [...prev, payload.event]);
      setRevealed(payload.revealed || {});
    });
    stream.addEventListener("done", async (ev: MessageEvent) => {
      const payload = JSON.parse(ev.data);
      setMetrics(payload.metrics || null);
      setWinner(payload.winner || null);
      setStatus("finished");
      if (payload.replay_id) {
        setAnalysisLoading(true);
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
  }

  const chatEntries = transcript.map((event) => {
    if (event.event_type === "discussion") {
      return `[${event.team}] ${event.agent_id}: ${event.content}`;
    }
    if (event.event_type === "clue") {
      return `[${event.team}] CLUE: ${event.word} (${event.number})`;
    }
    if (event.event_type === "guess") {
      return `[${event.team}] GUESS: ${event.word} (${event.result})`;
    }
    if (event.event_type === "pass") {
      return `[${event.team}] PASS`;
    }
    return JSON.stringify(event);
  });

  const redModel = models.find(m => m.model_id === red.cluer);
  const blueModel = models.find(m => m.model_id === blue.cluer);

  // Compute scores from revealed cards
  const redRevealed = Object.values(revealed).filter(v => v === "RED").length;
  const blueRevealed = Object.values(revealed).filter(v => v === "BLUE").length;
  const assassinHit = Object.values(revealed).includes("ASSASSIN");

  const getWinMessage = () => {
    if (!winner) return null;
    const winnerName = winner === "RED" ? "Red Team" : "Blue Team";
    
    if (assassinHit) {
      const loser = winner === "RED" ? "Blue" : "Red";
      return `${winnerName} wins — ${loser} hit the assassin!`;
    }
    return `${winnerName} wins!`;
  };

  return (
    <div className="page">
      <h2>Codenames</h2>

      {/* Settings Panel - Hidden when running */}
      {!isRunning && status !== "finished" && status !== "error" && (
        <div className="settings-panel">
          <div className="settings-content">
            <div className="settings-section">
              <h4>Game Settings</h4>
              <div className="form-row-compact">
                <label>Mode</label>
                <select value={mode} onChange={(e) => setMode(e.target.value)}>
                  <option value="STANDARD">Standard</option>
                  <option value="NO_ASSASSIN">No Assassin</option>
                  <option value="SINGLE_GUESSER">Single Guesser</option>
                </select>
              </div>
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
                <label>Event delay</label>
                <input
                  type="number"
                  value={eventDelay}
                  onChange={(e) => setEventDelay(Number(e.target.value))}
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
              <span className="score-value">{redRevealed}</span>
              <span className="score-label">/9</span>
            </div>
            <div className="score-divider">|</div>
            <div className="score-group blue">
              <span className="score-value">{blueRevealed}</span>
              <span className="score-label">/8</span>
            </div>
          </div>

          {/* Status */}
          <div className={`status ${status}`} style={{ margin: 0 }}>
            <span className="status-dot" />
            {isRunning ? 'Running' : 'Finished'}
          </div>
        </div>
      )}

      {/* Win Notification */}
      {status === "finished" && winner && (
        <div className={`win-banner ${winner.toLowerCase()}`}>
          <span className="win-icon">{winner === "RED" ? "🔴" : "🔵"}</span>
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
      {status !== "error" && (boardWords.length > 0 || status === "finished") && (
        <>
          <div className="layout">
            <div className="left">
              <CodenamesBoard words={boardWords} keyByWord={keyByWord} revealed={revealed} />
            </div>
            <div className="right">
              <ChatPanel 
                title="Transcript" 
                entries={chatEntries} 
                thinking={isRunning ? "Agents thinking..." : undefined}
              />
            </div>
          </div>

          {/* Analysis and Metrics below the game board */}
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
            {metrics && (
              <div className={`collapsible ${metricsOpen ? 'open' : ''}`}>
                <div className="collapsible-header" onClick={() => setMetricsOpen(!metricsOpen)}>
                  <h3>Detailed Metrics</h3>
                  <span className="collapsible-toggle">▼</span>
                </div>
                <div className="collapsible-content">
                  <pre style={{ margin: 0 }}>{JSON.stringify(metrics, null, 2)}</pre>
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

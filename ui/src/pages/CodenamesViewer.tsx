import { useEffect, useMemo, useState } from "react";
import {
  fetchStats,
  openEventStream,
  startCodenames,
} from "../api";
import CodenamesBoard from "../components/CodenamesBoard";
import ChatPanel from "../components/ChatPanel";
import ModelPicker from "../components/ModelPicker";
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

  useEffect(() => {
    setRed(baseTeam);
    setBlue(baseTeam);
  }, [baseTeam]);

  function buildSelection(): TeamSelection {
    return { red, blue };
  }

  async function handleStart() {
    setStatus("starting");
    setMetrics(null);
    setAnalysis(null);
    setError(null);
    setAnalysisLoading(false);
    setTranscript([]);
    setRevealed({});
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

  return (
    <div className="page">
      <h2>Codenames</h2>
      <div className="controls">
        <div className="panel">
          <div className="form-row">
            <label>Mode</label>
            <select value={mode} onChange={(e) => setMode(e.target.value)}>
              <option value="STANDARD">STANDARD</option>
              <option value="NO_ASSASSIN">NO_ASSASSIN</option>
              <option value="SINGLE_GUESSER">SINGLE_GUESSER</option>
            </select>
          </div>
          <div className="form-row">
            <label>Seed (optional)</label>
            <input
              type="number"
              value={seed ?? ""}
              onChange={(e) => setSeed(e.target.value ? Number(e.target.value) : undefined)}
            />
          </div>
          <div className="form-row">
            <label>Event delay (ms)</label>
            <input
              type="number"
              value={eventDelay}
              onChange={(e) => setEventDelay(Number(e.target.value))}
            />
          </div>
          <button onClick={handleStart} disabled={!models.length || status === "running"}>
            Start game
          </button>
          <div className="muted">Status: {status}</div>
          {error && <div className="error">{error}</div>}
        </div>
        <ModelPicker models={models} value={red} onChange={setRed} label="Red Team" />
        <ModelPicker models={models} value={blue} onChange={setBlue} label="Blue Team" />
      </div>

      <div className="layout">
        <div className="left">
          <CodenamesBoard words={boardWords} keyByWord={keyByWord} revealed={revealed} />
        </div>
        <div className="right">
          <ChatPanel title="Transcript" entries={chatEntries} />
          {metrics && (
            <div className="panel">
              <h3>Metrics</h3>
              <pre>{JSON.stringify(metrics, null, 2)}</pre>
            </div>
          )}
      {analysisLoading && (
        <div className="panel">
          <h3>Opus Analysis</h3>
          <div className="muted">Analyzing game results...</div>
        </div>
      )}
          {analysis && (
            <div className="panel">
              <h3>Opus Analysis</h3>
              <div className="analysis">{analysis}</div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}

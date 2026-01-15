import { useEffect, useMemo, useState } from "react";
import { fetchStats, openEventStream, startDecrypto } from "../api";
import DecryptoBoard from "../components/DecryptoBoard";
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
  const [currentRound, setCurrentRound] = useState<{
    round: number;
    red: { code: number[]; clues: string[] };
    blue: { code: number[]; clues: string[] };
  } | null>(null);

  useEffect(() => {
    setRed(baseTeam);
    setBlue(baseTeam);
  }, [baseTeam]);

  function buildSelection(): TeamSelection {
    return { red, blue };
  }

  async function handleStart() {
    setStatus("starting");
    setRounds([]);
    setAnalysis(null);
    setError(null);
    setRedLog([]);
    setBlueLog([]);
    setAnalysisLoading(false);
    setCurrentRound(null);
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
        (s: any) => `  share (${s.agent_id}): ${s.message}`
      );
      if (payload.team === "red") {
        setRedLog((prev) => [...prev, line, ...shares]);
      } else {
        setBlueLog((prev) => [...prev, line, ...shares]);
      }
    });
    stream.addEventListener("discussion", (ev: MessageEvent) => {
      const payload = JSON.parse(ev.data);
      const line = `DISCUSS ${payload.agent_id}: ${payload.message}`;
      if (payload.team === "red") {
        setRedLog((prev) => [...prev, line]);
      } else {
        setBlueLog((prev) => [...prev, line]);
      }
    });
    stream.addEventListener("round", (ev: MessageEvent) => {
      const payload = JSON.parse(ev.data);
      setRounds((prev) => [...prev, payload]);
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
        entries.push(`${kind} guess ${guess}`);
        (action.share || []).forEach((share: any) => {
          entries.push(`  share (${share.agent_id}): ${share.message}`);
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
        entries.push(`${kind} guess ${guess}`);
        (action.share || []).forEach((share: any) => {
          entries.push(`  share (${share.agent_id}): ${share.message}`);
        });
      });
    return entries;
  });

  const redEntries = redLog.length ? redLog : fallbackRed;
  const blueEntries = blueLog.length ? blueLog : fallbackBlue;

  const roundResults = rounds.map((round: any) => {
    const trueCodes = round.true_codes || round.reveal_true_codes || {};
    const actions = round.final_guesses || round.actions || [];
    return {
      roundNumber: round.round_number,
      trueCodes,
      actions,
      clues: round.public_clues || {},
    };
  });

  return (
    <div className="page">
      <h2>Decrypto</h2>
      <div className="controls">
        <div className="panel">
          <div className="form-row">
            <label>Seed</label>
            <input
              type="number"
              value={seed}
              onChange={(e) => setSeed(Number(e.target.value))}
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
          <div className="form-row">
            <label>Discussion turns per guesser</label>
            <input
              type="number"
              min={1}
              max={4}
              value={maxDiscussionTurns}
              onChange={(e) => setMaxDiscussionTurns(Number(e.target.value))}
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

      <div className="layout decrypto">
        <div className="left">
          <ChatPanel title="Red Log" entries={redEntries} />
        </div>
        <div className="center">
          <DecryptoBoard redKey={keys.red || []} blueKey={keys.blue || []} />
          {currentRound && (
            <div className="panel">
              <h3>Current Round</h3>
              <div className="muted">Round {currentRound.round}</div>
              <div>
                <strong>Red code:</strong> {currentRound.red.code.join("-") || "—"}
              </div>
              <div>
                <strong>Red clues:</strong> {currentRound.red.clues.join(" | ") || "—"}
              </div>
              <div>
                <strong>Blue code:</strong> {currentRound.blue.code.join("-") || "—"}
              </div>
              <div>
                <strong>Blue clues:</strong> {currentRound.blue.clues.join(" | ") || "—"}
              </div>
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
          {roundResults.length > 0 && (
            <div className="panel">
              <h3>Round Results</h3>
              {roundResults.map((rr) => (
                <div key={`round-${rr.roundNumber}`} className="round-result">
                  <div className="muted">Round {rr.roundNumber}</div>
                  <div>
                    <strong>True codes:</strong>{" "}
                    red {rr.trueCodes.red?.join("-") || "—"} | blue{" "}
                    {rr.trueCodes.blue?.join("-") || "—"}
                  </div>
                  <div>
                    <strong>Clues:</strong>{" "}
                    red {(rr.clues.red?.clues || []).join(" | ")} | blue{" "}
                    {(rr.clues.blue?.clues || []).join(" | ")}
                  </div>
                  <div>
                    <strong>Final guesses:</strong>
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
          )}
        </div>
        <div className="right">
          <ChatPanel title="Blue Log" entries={blueEntries} />
        </div>
      </div>
    </div>
  );
}

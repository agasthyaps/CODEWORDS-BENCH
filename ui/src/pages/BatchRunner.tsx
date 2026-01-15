import { useEffect, useMemo, useState } from "react";
import { openEventStream, startBatch } from "../api";
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

export default function BatchRunner({ models, defaultModel }: Props) {
  const baseTeam = useMemo(
    () => ({ ...DEFAULT_TEAM, cluer: defaultModel, guesser_1: defaultModel, guesser_2: defaultModel }),
    [defaultModel]
  );
  const [red, setRed] = useState<TeamRoleConfig>(baseTeam);
  const [blue, setBlue] = useState<TeamRoleConfig>(baseTeam);
  const [gameType, setGameType] = useState<"codenames" | "decrypto">("codenames");
  const [pinned, setPinned] = useState(true);
  const [count, setCount] = useState(5);
  const [modelPool, setModelPool] = useState<string[]>([]);
  const [progress, setProgress] = useState({ completed: 0, total: 0 });
  const [results, setResults] = useState<any[]>([]);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    setRed(baseTeam);
    setBlue(baseTeam);
  }, [baseTeam]);

  function togglePool(id: string) {
    setModelPool((prev) =>
      prev.includes(id) ? prev.filter((x) => x !== id) : [...prev, id]
    );
  }

  function buildSelection(): TeamSelection {
    return { red, blue };
  }

  async function handleStart() {
    setProgress({ completed: 0, total: count });
    setResults([]);
    setError(null);
    const payload: any = {
      game_type: gameType,
      count,
      pinned,
      team_selection: pinned ? buildSelection() : undefined,
      model_pool: pinned ? undefined : modelPool,
      codenames_mode: "STANDARD",
      max_discussion_rounds: 3,
      max_turns: 50,
      max_rounds: 8,
      event_delay_ms: 0,
    };
    const { job_id } = await startBatch(payload);
    const stream = openEventStream(`/batch/${job_id}/events`);
    stream.addEventListener("progress", (ev: MessageEvent) => {
      const payload = JSON.parse(ev.data);
      setProgress({ completed: payload.completed, total: payload.total });
    });
    stream.addEventListener("done", (ev: MessageEvent) => {
      const payload = JSON.parse(ev.data);
      setResults(payload.results || []);
      stream.close();
    });
    stream.addEventListener("job_error", (ev: MessageEvent) => {
      const payload = JSON.parse(ev.data);
      setError(payload.error || "Batch failed");
      stream.close();
    });
    stream.addEventListener("error", () => {
      stream.close();
    });
  }

  return (
    <div className="page">
      <h2>Batch Runner</h2>
      <div className="controls">
        <div className="panel">
          <div className="form-row">
            <label>Game type</label>
            <select value={gameType} onChange={(e) => setGameType(e.target.value as any)}>
              <option value="codenames">Codenames</option>
              <option value="decrypto">Decrypto</option>
            </select>
          </div>
          <div className="form-row">
            <label>Games</label>
            <input type="number" value={count} onChange={(e) => setCount(Number(e.target.value))} />
          </div>
          <div className="form-row">
            <label>Pinned teams</label>
            <input type="checkbox" checked={pinned} onChange={() => setPinned(!pinned)} />
          </div>
          {!pinned && (
            <div className="pool">
              {models.map((m) => (
                <label key={m.model_id} className="pool-item">
                  <input
                    type="checkbox"
                    checked={modelPool.includes(m.model_id)}
                    onChange={() => togglePool(m.model_id)}
                  />
                  {m.name}
                </label>
              ))}
            </div>
          )}
          <button onClick={handleStart}>Start batch</button>
          <div className="muted">
            Progress: {progress.completed} / {progress.total}
          </div>
          {error && <div className="error">{error}</div>}
        </div>
        {pinned && (
          <>
            <ModelPicker models={models} value={red} onChange={setRed} label="Red Team" />
            <ModelPicker models={models} value={blue} onChange={setBlue} label="Blue Team" />
          </>
        )}
      </div>
      {results.length > 0 && (
        <div className="panel">
          <h3>Batch Results</h3>
          <pre>{JSON.stringify(results, null, 2)}</pre>
        </div>
      )}
    </div>
  );
}

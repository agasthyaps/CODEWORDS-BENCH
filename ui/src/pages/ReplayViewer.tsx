import { useEffect, useMemo, useState } from "react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import { fetchReplay, fetchReplays, fetchStats } from "../api";
import CodenamesBoard from "../components/CodenamesBoard";
import ChatPanel from "../components/ChatPanel";
import DecryptoBoard from "../components/DecryptoBoard";
import { ReplaySummary } from "../types";

export default function ReplayViewer() {
  const [replays, setReplays] = useState<ReplaySummary[]>([]);
  const [selected, setSelected] = useState<ReplaySummary | null>(null);
  const [data, setData] = useState<any | null>(null);
  const [eventIndex, setEventIndex] = useState(0);
  const [analysis, setAnalysis] = useState<string | null>(null);
  const [analysisLoading, setAnalysisLoading] = useState(false);

  useEffect(() => {
    fetchReplays().then(setReplays).catch(() => setReplays([]));
  }, []);

  useEffect(() => {
    if (!selected) return;
    fetchReplay(selected.game_type, selected.replay_id).then((payload) => {
      setData(payload);
      setEventIndex(0);
      setAnalysis(null);
      setAnalysisLoading(true);
      const start = Date.now();
      const poll = async () => {
        try {
          const stats = await fetchStats(selected.replay_id);
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
    });
  }, [selected]);

  const codenames = selected?.game_type === "codenames" && data;
  const decrypto = selected?.game_type === "decrypto" && data;

  const transcript = codenames?.public_transcript || [];
  const maxIndex = transcript.length;

  const revealed = useMemo(() => {
    if (!codenames) return {};
    const output: Record<string, string> = {};
    transcript.slice(0, eventIndex).forEach((event: any) => {
      if (event.event_type === "guess") {
        output[event.word] = event.result;
      }
    });
    return output;
  }, [codenames, transcript, eventIndex]);

  const chatEntries = transcript.slice(0, eventIndex).map((event: any) => {
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

  const rounds = decrypto?.rounds || [];
  const roundIndex = Math.min(eventIndex, rounds.length);
  const redKey = decrypto?.keys?.red || [];
  const blueKey = decrypto?.keys?.blue || [];
  const redLog = rounds.slice(0, roundIndex).flatMap((round: any) => {
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
  const blueLog = rounds.slice(0, roundIndex).flatMap((round: any) => {
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

  return (
    <div className="page">
      <h2>Replay Viewer</h2>
      <div className="panel" style={{ marginBottom: 24 }}>
        <h3>Select Replay</h3>
        <div className="form-row">
          <label>Replay</label>
          <select
            value={selected?.replay_id || ""}
            onChange={(e) => {
              const next = replays.find((r) => r.replay_id === e.target.value) || null;
              setSelected(next);
            }}
          >
            <option value="">Choose a replay...</option>
            {replays.map((r) => (
              <option key={r.replay_id} value={r.replay_id}>
                {r.game_type.charAt(0).toUpperCase() + r.game_type.slice(1)} — {r.replay_id.slice(0, 12)}...
              </option>
            ))}
          </select>
        </div>
      </div>

      {codenames && (
        <>
          <div className="layout">
            <div className="left">
              <CodenamesBoard
                words={codenames.board.words || []}
                keyByWord={codenames.board.key_by_word || {}}
                revealed={revealed}
              />
              <div className="panel">
                <h3>Timeline</h3>
                <div className="form-row">
                  <label>Event</label>
                  <input
                    type="range"
                    min={0}
                    max={maxIndex}
                    value={eventIndex}
                    onChange={(e) => setEventIndex(Number(e.target.value))}
                  />
                  <span className="muted">{eventIndex} / {maxIndex}</span>
                </div>
              </div>
            </div>
            <div className="right">
              <ChatPanel title="Transcript" entries={chatEntries} />
            </div>
          </div>
          <div className="below-layout">
            {analysisLoading && (
              <div className="panel">
                <h3>Analysis</h3>
                <div className="loading-text">Loading analysis...</div>
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
          </div>
        </>
      )}

      {decrypto && (
        <>
          <div className="layout decrypto">
            <div className="left">
              <ChatPanel title="Red Team Log" entries={redLog} variant="decrypto" />
            </div>
            <div className="center">
              <DecryptoBoard redKey={redKey} blueKey={blueKey} />
              <div className="panel">
                <h3>Timeline</h3>
                <div className="form-row">
                  <label>Round</label>
                  <input
                    type="range"
                    min={0}
                    max={rounds.length}
                    value={eventIndex}
                    onChange={(e) => setEventIndex(Number(e.target.value))}
                  />
                  <span className="muted">{eventIndex} / {rounds.length}</span>
                </div>
              </div>
            </div>
            <div className="right">
              <ChatPanel title="Blue Team Log" entries={blueLog} variant="decrypto" />
            </div>
          </div>
          <div className="below-layout">
            {analysisLoading && (
              <div className="panel">
                <h3>Analysis</h3>
                <div className="loading-text">Loading analysis...</div>
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
          </div>
        </>
      )}

      {!selected && (
        <div className="panel">
          <div className="empty-state">Select a replay from the dropdown above to view it.</div>
        </div>
      )}
    </div>
  );
}

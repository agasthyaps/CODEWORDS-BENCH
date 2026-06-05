import { useEffect, useMemo, useState } from "react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import { fetchReplay, fetchReplays, fetchStats } from "../api";
import CodenamesBoard from "../components/CodenamesBoard";
import ChatPanel from "../components/ChatPanel";
import DecryptoBoard from "../components/DecryptoBoard";
import { BenchmarkPenalty, HarnessEvent, ReplayProtocolFields, ReplaySummary, RunManifest } from "../types";

type ReplayTab = "game" | "harness" | "scratchpads" | "manifest" | "metrics";

function formatValue(value: unknown) {
  if (value === null || value === undefined) return "N/A";
  if (typeof value === "object") return JSON.stringify(value);
  return String(value);
}

function compactEventContext(event: HarnessEvent) {
  const parts = [
    event.team,
    event.agent_id,
    event.role,
    event.round_number !== null && event.round_number !== undefined ? `round ${event.round_number}` : null,
    event.turn_number !== null && event.turn_number !== undefined ? `turn ${event.turn_number}` : null,
  ].filter(Boolean);
  return parts.join(" / ");
}

export default function ReplayViewer() {
  const [replays, setReplays] = useState<ReplaySummary[]>([]);
  const [selected, setSelected] = useState<ReplaySummary | null>(null);
  const [data, setData] = useState<any | null>(null);
  const [eventIndex, setEventIndex] = useState(0);
  const [analysis, setAnalysis] = useState<string | null>(null);
  const [analysisLoading, setAnalysisLoading] = useState(false);
  const [activeTab, setActiveTab] = useState<ReplayTab>("game");

  useEffect(() => {
    fetchReplays().then(setReplays).catch(() => setReplays([]));
  }, []);

  useEffect(() => {
    if (!selected) return;
    fetchReplay(selected.game_type, selected.replay_id).then((payload) => {
      setData(payload);
      setEventIndex(0);
      setActiveTab("game");
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

  const protocol = (data || null) as (ReplayProtocolFields & Record<string, any>) | null;
  const runManifest = ((protocol?.run_manifest ?? protocol?.metadata?.run_manifest) as RunManifest | null | undefined) ?? null;
  const condition = runManifest?.condition ?? (protocol?.metadata?.condition as RunManifest["condition"] | undefined) ?? null;
  const harnessEvents = (protocol?.harness_events || []) as HarnessEvent[];
  const penalties = (protocol?.benchmark_penalties || []) as BenchmarkPenalty[];
  const scratchpads = (protocol?.agent_scratchpads || {}) as Record<string, string>;
  const penaltyTotal = penalties.reduce((sum, penalty) => sum + (Number(penalty.points) || 0), 0);
  const penaltyCounts = penalties.reduce<Record<string, number>>((counts, penalty) => {
    counts[penalty.penalty_type] = (counts[penalty.penalty_type] || 0) + 1;
    return counts;
  }, {});
  const repairCount = harnessEvents.filter((event) => event.event_type.includes("repair")).length;
  const fallbackCount = harnessEvents.filter((event) => event.event_type.includes("fallback")).length;
  const invalidCount = harnessEvents.filter((event) => event.event_type.includes("invalid")).length;
  const acceptedActionCount = harnessEvents.filter((event) => event.event_type.includes("accepted")).length;
  const gameOutcome = (() => {
    if (!selected || !data) return {};
    if (selected.game_type === "codenames") {
      return {
        Winner: data.winner ?? "N/A",
        "Turn count": data.total_turns ?? data.turn_count ?? "N/A",
        "Transcript events": transcript.length,
      };
    }
    if (selected.game_type === "decrypto") {
      return {
        Winner: data.winner ?? "N/A",
        Rounds: rounds.length,
        "Red score": data.scores?.red ?? "N/A",
        "Blue score": data.scores?.blue ?? "N/A",
      };
    }
    return {
      "Final score": data.final_score ?? data.score ?? "N/A",
      "Turn count": data.turn_count ?? data.turns?.length ?? "N/A",
      "Game over": data.game_over_reason ?? data.termination_reason ?? "N/A",
    };
  })();

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
              if (!next) {
                setData(null);
                setEventIndex(0);
                setActiveTab("game");
                setAnalysis(null);
                setAnalysisLoading(false);
              }
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

      {data && (
        <div className="protocol-tabs" aria-label="Replay sections">
          {(["game", "harness", "scratchpads", "manifest", "metrics"] as ReplayTab[]).map((tab) => (
            <button
              key={tab}
              className={`protocol-tab ${activeTab === tab ? "active" : ""}`}
              type="button"
              onClick={() => setActiveTab(tab)}
            >
              {tab.charAt(0).toUpperCase() + tab.slice(1)}
            </button>
          ))}
        </div>
      )}

      {activeTab === "game" && codenames && (
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

      {activeTab === "game" && decrypto && (
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

      {activeTab === "game" && selected && data && !codenames && !decrypto && (
        <div className="panel">
          <div className="empty-state">This replay does not have a specialized game board yet. Use the protocol tabs to inspect saved artifacts.</div>
        </div>
      )}

      {activeTab === "harness" && data && (
        <div className="protocol-grid">
          <div className="protocol-panel condition-summary">
            <h3>Condition</h3>
            {condition ? (
              <div className="manifest-table">
                <div><span>Name</span><strong>{condition.name}</strong></div>
                <div><span>Scratchpad</span><strong>{condition.scratchpad_policy}</strong></div>
                <div><span>Repair</span><strong>{condition.repair_policy}</strong></div>
                <div><span>Invalid Actions</span><strong>{condition.invalid_action_policy}</strong></div>
                <div><span>Structured Output</span><strong>{condition.structured_output_policy}</strong></div>
              </div>
            ) : (
              <div className="empty-state">No condition metadata saved for this replay.</div>
            )}
          </div>

          <div className="protocol-panel">
            <h3>Penalties</h3>
            <div className="metric-row">
              <span>Total penalty points</span>
              <strong>{penaltyTotal}</strong>
            </div>
            {Object.keys(penaltyCounts).length > 0 && (
              <div className="penalty-breakdown">
                {Object.entries(penaltyCounts).map(([type, count]) => (
                  <span key={type}>{type}: {count}</span>
                ))}
              </div>
            )}
            <div className="penalty-list">
              {penalties.length === 0 ? (
                <div className="empty-state">No benchmark penalties recorded.</div>
              ) : (
                penalties.map((penalty, index) => (
                  <div className="penalty-item" key={`${penalty.penalty_type}:${index}`}>
                    <div className="penalty-item-header">
                      <strong>{penalty.penalty_type}</strong>
                      <span>{penalty.points} pts</span>
                    </div>
                    <div>{penalty.description}</div>
                    <div className="penalty-meta">
                      {[penalty.team, penalty.agent_id, penalty.round_number !== null && penalty.round_number !== undefined ? `round ${penalty.round_number}` : null, penalty.turn_number !== null && penalty.turn_number !== undefined ? `turn ${penalty.turn_number}` : null]
                        .filter(Boolean)
                        .join(" / ")}
                    </div>
                  </div>
                ))
              )}
            </div>
          </div>

          <div className="protocol-panel protocol-panel-wide">
            <h3>Harness Events</h3>
            <div className="harness-event-list">
              {harnessEvents.length === 0 ? (
                <div className="empty-state">No harness events recorded.</div>
              ) : (
                harnessEvents.slice(0, 250).map((event, index) => (
                  <details className="harness-event" key={`${event.event_type}:${index}`}>
                    <summary>
                      <span>{event.event_type}</span>
                      <small>{compactEventContext(event)}</small>
                    </summary>
                    <pre>{JSON.stringify(event.payload, null, 2)}</pre>
                  </details>
                ))
              )}
              {harnessEvents.length > 250 && (
                <div className="protocol-note">Showing the first 250 of {harnessEvents.length} events.</div>
              )}
            </div>
          </div>
        </div>
      )}

      {activeTab === "scratchpads" && data && (
        <div className="protocol-panel">
          <h3>Private Scratchpads</h3>
          <div className="scratchpad-map">
            {Object.keys(scratchpads).length === 0 ? (
              <div className="empty-state">No scratchpad content saved for this condition or replay.</div>
            ) : (
              Object.entries(scratchpads).map(([agentId, content]) => (
                <div className="scratchpad-entry" key={agentId}>
                  <h4>{agentId}</h4>
                  <pre>{content}</pre>
                </div>
              ))
            )}
          </div>
        </div>
      )}

      {activeTab === "manifest" && data && (
        <div className="protocol-panel">
          <h3>Run Manifest</h3>
          {runManifest ? (
            <>
              <div className="manifest-table">
                <div><span>Game Type</span><strong>{runManifest.game_type}</strong></div>
                <div><span>Rules Version</span><strong>{runManifest.game_rules_version}</strong></div>
                <div><span>Code SHA</span><strong>{runManifest.code_version || "unknown"}</strong></div>
                <div><span>Dirty Worktree</span><strong>{String(runManifest.dirty_worktree ?? "unknown")}</strong></div>
                <div><span>Seeds</span><strong>{runManifest.seed_schedule?.join(", ") || "N/A"}</strong></div>
                <div><span>Created</span><strong>{runManifest.created_at}</strong></div>
              </div>
              <h4>Prompt Hashes</h4>
              <div className="manifest-table">
                {Object.entries(runManifest.prompt_hashes || {}).map(([name, hash]) => (
                  <div key={name}><span>{name}</span><strong>{hash}</strong></div>
                ))}
              </div>
              <details className="manifest-raw">
                <summary>Raw manifest JSON</summary>
                <pre>{JSON.stringify(runManifest, null, 2)}</pre>
              </details>
            </>
          ) : (
            <div className="empty-state">No run manifest saved for this replay.</div>
          )}
        </div>
      )}

      {activeTab === "metrics" && data && (
        <div className="protocol-grid">
          <div className="protocol-panel">
            <h3>Game Outcome</h3>
            <div className="metric-list">
              {Object.entries(gameOutcome).map(([name, value]) => (
                <div className="metric-row" key={name}>
                  <span>{name}</span>
                  <strong>{formatValue(value)}</strong>
                </div>
              ))}
            </div>
          </div>
          <div className="protocol-panel">
            <h3>Benchmark Penalties</h3>
            <div className="metric-list">
              <div className="metric-row"><span>Total points</span><strong>{penaltyTotal}</strong></div>
              <div className="metric-row"><span>Penalty records</span><strong>{penalties.length}</strong></div>
              <div className="metric-row"><span>Invalid events</span><strong>{invalidCount}</strong></div>
              <div className="metric-row"><span>Repairs</span><strong>{repairCount}</strong></div>
              <div className="metric-row"><span>Fallbacks</span><strong>{fallbackCount}</strong></div>
            </div>
          </div>
          <div className="protocol-panel">
            <h3>Harness Dependence</h3>
            <div className="metric-list">
              <div className="metric-row"><span>Harness events</span><strong>{harnessEvents.length}</strong></div>
              <div className="metric-row"><span>Accepted actions</span><strong>{acceptedActionCount}</strong></div>
              <div className="metric-row"><span>Scratchpads saved</span><strong>{Object.keys(scratchpads).length}</strong></div>
            </div>
          </div>
        </div>
      )}

      {!selected && (
        <div className="panel">
          <div className="empty-state">Select a replay from the dropdown above to view it.</div>
        </div>
      )}
    </div>
  );
}

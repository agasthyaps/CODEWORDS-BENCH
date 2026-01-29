import { Fragment, useEffect, useMemo, useState } from "react";
import { fetchLeaderboard, refreshLeaderboard } from "../api";
import type {
  LeaderboardData,
  OverallRanking,
  CodenamesRanking,
  DecryptoRanking,
  HanabiRanking,
} from "../types";

type Props = {
  onNavigate: (view: "codenames" | "decrypto" | "hanabi" | "replay" | "batch" | "benchmark") => void;
};

export default function Home({ onNavigate }: Props) {
  const [leaderboard, setLeaderboard] = useState<LeaderboardData | null>(null);
  const [loading, setLoading] = useState(true);
  const [refreshing, setRefreshing] = useState(false);
  const [error, setError] = useState<string | null>(null);
  // Toggle between efficiency-based and raw score rankings
  const [useEfficiency, setUseEfficiency] = useState(true);

  useEffect(() => {
    fetchLeaderboard()
      .then((data) => {
        console.log("Fetched leaderboard:", data);
        setLeaderboard(data);
      })
      .catch((e) => {
        console.error("Fetch error:", e);
        setError(e.message);
      })
      .finally(() => setLoading(false));
  }, []);

  const handleRefresh = async () => {
    setRefreshing(true);
    setError(null);
    try {
      const result = await refreshLeaderboard();
      console.log("Refresh result:", result);
      // Fetch updated data
      const updated = await fetchLeaderboard();
      console.log("Updated leaderboard:", updated);
      setLeaderboard(updated);
    } catch (e: any) {
      console.error("Refresh error:", e);
      setError(e.message);
    } finally {
      setRefreshing(false);
    }
  };

  const totalGames = leaderboard
    ? leaderboard.total_episodes.codenames +
      leaderboard.total_episodes.decrypto +
      leaderboard.total_episodes.hanabi
    : 0;

  const formatDate = (iso: string) => {
    const d = new Date(iso);
    return d.toLocaleDateString(undefined, {
      month: "short",
      day: "numeric",
      hour: "2-digit",
      minute: "2-digit",
    });
  };

  const formatPct01 = (value?: number | null) =>
    value === null || value === undefined ? "—" : `${(value * 100).toFixed(0)}%`;

  const formatBrier = (value?: number | null) =>
    value === null || value === undefined ? "Brier —" : `Brier ${value.toFixed(3)}`;

  // Compute dynamic insights from leaderboard
  const bestCooperative = leaderboard?.hanabi_rankings[0];
  const bestAdversarial = leaderboard?.decrypto_rankings[0];
  const bestCollaborative = leaderboard?.codenames_rankings[0];

  const bestAdversarialDecode = bestAdversarial?.decode_accuracy ?? null;
  const bestAdversarialIntercept = bestAdversarial?.intercept_accuracy ?? null;
  const bestAdversarialComposite = bestAdversarial?.adversarial_score ?? null;

  const decodeCodenamesCorrelation = useMemo(() => {
    if (!leaderboard) return null;
    const pairs = leaderboard.overall_rankings
      .map((r) => [r.decrypto_decode, r.codenames_score] as const)
      .filter(([decode, codenames]) => decode !== null && codenames !== null)
      .map(([decode, codenames]) => [decode as number, codenames as number]);

    if (pairs.length < 2) return null;
    const xs = pairs.map(([x]) => x);
    const ys = pairs.map(([, y]) => y);
    const meanX = xs.reduce((a, b) => a + b, 0) / xs.length;
    const meanY = ys.reduce((a, b) => a + b, 0) / ys.length;
    const cov = xs.reduce((sum, x, i) => sum + (x - meanX) * (ys[i] - meanY), 0);
    const varX = xs.reduce((sum, x) => sum + (x - meanX) ** 2, 0);
    const varY = ys.reduce((sum, y) => sum + (y - meanY) ** 2, 0);
    const denom = Math.sqrt(varX * varY);
    if (denom === 0) return null;
    return { r: cov / denom, n: pairs.length };
  }, [leaderboard]);

  return (
    <div className="page home-page">
      {/* Hero Section */}
      <section className="home-hero">
        <div className="hero-content">
          <h1>MindGames</h1>
          <p className="tagline">Measuring how models think about each other</p>
          <p className="subtitle">
            Evaluating Theory of Mind through cooperative and competitive games—how
            well do language models reason about beliefs, knowledge, and intentions? (and a meta-experiment in semi-automated research)
          </p>
        </div>
        <div className="hero-meta">
          <span className="version-tag">v0.1.0</span>
        </div>
      </section>

      {/* Leaderboard Section */}
      <section className="leaderboard-section">
        <div className="leaderboard-header">
          <div className="leaderboard-title">
            <h2>Model Leaderboard</h2>
            {leaderboard && (
              <span className="leaderboard-meta">
                {totalGames} games analyzed
              </span>
            )}
          </div>
          <div className="leaderboard-controls">
            <div className="ranking-toggle" title="Toggle between efficiency-based and raw score rankings to see how rankings change">
              <button
                className={`toggle-btn ${useEfficiency ? "active" : ""}`}
                onClick={() => setUseEfficiency(true)}
              >
                Efficiency
              </button>
              <button
                className={`toggle-btn ${!useEfficiency ? "active" : ""}`}
                onClick={() => setUseEfficiency(false)}
              >
                Raw Score
              </button>
            </div>
            <button
              onClick={handleRefresh}
              disabled={refreshing || loading}
              className="refresh-btn"
            >
              {refreshing ? "Refreshing..." : "Refresh"}
            </button>
          </div>
        </div>

        {error && <div className="leaderboard-error">{error}</div>}

        {loading ? (
          <div className="leaderboard-loading">Loading leaderboard...</div>
        ) : leaderboard && leaderboard.overall_rankings.length > 0 ? (
          <LeaderboardTable rankings={leaderboard.overall_rankings} useEfficiency={useEfficiency} />
        ) : (
          <div className="leaderboard-empty">
            <p>No games recorded yet.</p>
            <p className="hint">Run some games to see the leaderboard!</p>
          </div>
        )}
      </section>

      {/* Capability Breakdown Section */}
      {leaderboard && leaderboard.overall_rankings.length > 0 && (
        <section className="capabilities-section">
          <CapabilityColumn
            icon="🤝"
            title="Cooperative ToM"
            subtitle="Hanabi"
            description="Convention establishment speed"
            rankings={leaderboard.hanabi_rankings}
            metric="efficiency"
            formatValue={(r: HanabiRanking) => `${(r.efficiency * 100).toFixed(0)}% eff`}
            secondaryValue={(r: HanabiRanking) => `${r.avg_score}/25 raw`}
            color="cooperative"
            onClick={() => onNavigate("hanabi")}
          />
          <CapabilityColumn
            icon="⚔️"
            title="Adversarial ToM"
            subtitle="Decrypto"
            description="Intercept skill penalized by miscommunications"
            rankings={leaderboard.decrypto_rankings}
            metric="adversarial_score"
            formatValue={(r: DecryptoRanking) =>
              r.adversarial_score === null || r.adversarial_score === undefined
                ? "—"
                : `${(r.adversarial_score * 100).toFixed(0)}% adv`
            }
            secondaryValue={(r: DecryptoRanking) =>
              `${formatPct01(r.intercept_accuracy)} int · ${formatPct01(r.decode_accuracy)} dec`
            }
            color="adversarial"
            onClick={() => onNavigate("decrypto")}
          />
          <CapabilityColumn
            icon="💬"
            title="Collaborative"
            subtitle="Codenames"
            description="Semantic coordination"
            rankings={leaderboard.codenames_rankings}
            metric="win_rate"
            formatValue={(r: CodenamesRanking) => `${(r.win_rate * 100).toFixed(0)}%`}
            secondaryValue={(r: CodenamesRanking) => formatBrier(r.avg_cluer_surprise)}
            color="collaborative"
            onClick={() => onNavigate("codenames")}
          />
        </section>
      )}

      {/* Stats Row */}
      {leaderboard && (
        <section className="stats-section">
          <div className="stat-card">
            <div className="stat-value">{leaderboard.total_episodes.codenames}</div>
            <div className="stat-label">Codenames</div>
          </div>
          <div className="stat-card">
            <div className="stat-value">{leaderboard.total_episodes.decrypto}</div>
            <div className="stat-label">Decrypto</div>
          </div>
          <div className="stat-card">
            <div className="stat-value">{leaderboard.total_episodes.hanabi}</div>
            <div className="stat-label">Hanabi</div>
          </div>
          <div className="stat-card">
            <div className="stat-value stat-date">
              {formatDate(leaderboard.generated_at)}
            </div>
            <div className="stat-label">Last Updated</div>
          </div>
        </section>
      )}

      {/* Research Insights Section */}
      {leaderboard && totalGames > 0 && (
        <section className="insights-section">
          <h2>Research Insights</h2>
          <div className="insights-grid">
            {/* The Efficiency Paradox */}
            <div className="insight-card insight-finding">
              <div className="insight-label">Key Finding</div>
              <h3>The Efficiency Paradox</h3>
              <p>
                Raw scores and efficiency are negatively correlated. Models with highest
                raw Hanabi scores have lowest efficiency—they succeed through persistence,
                not coordination.
              </p>
              <div className="insight-data">
                {bestCooperative && (
                  <>
                    <div className="data-point">
                      <span className="data-value">{(bestCooperative.efficiency * 100).toFixed(0)}%</span>
                      <span className="data-label">Best Efficiency</span>
                    </div>
                    <div className="data-point">
                      <span className="data-value">{bestCooperative.avg_turns.toFixed(0)}</span>
                      <span className="data-label">Avg Turns</span>
                    </div>
                  </>
                )}
              </div>
            </div>

            {/* Opponent Modeling Gap */}
            <div className="insight-card insight-chart">
              <div className="insight-label">Adversarial ToM</div>
              <h3>The Opponent Modeling Gap</h3>
              <p>
                Decode (understanding teammates) is much easier than intercept (modeling opponents).
                Adversarial score multiplies intercept by decode to penalize miscommunications.
              </p>
              {bestAdversarial && (
                <div className="decode-intercept-gap">
                  <div className="gap-bar">
                    <div className="gap-segment decode" style={{ width: `${(bestAdversarialDecode ?? 0) * 100}%` }}>
                      <span className="gap-label">Decode</span>
                      <span className="gap-value">{formatPct01(bestAdversarialDecode)}</span>
                    </div>
                  </div>
                  <div className="gap-bar">
                    <div className="gap-segment intercept" style={{ width: `${(bestAdversarialIntercept ?? 0) * 100}%` }}>
                      <span className="gap-label">Intercept</span>
                      <span className="gap-value">{formatPct01(bestAdversarialIntercept)}</span>
                    </div>
                  </div>
                  {bestAdversarialDecode !== null && bestAdversarialIntercept !== null && (
                    <div className="gap-note">
                      Gap: {((bestAdversarialDecode - bestAdversarialIntercept) * 100).toFixed(0)} pts
                    </div>
                  )}
                  {decodeCodenamesCorrelation && (
                    <div className="gap-note">
                      Decode↔Codenames correlation: r={decodeCodenamesCorrelation.r.toFixed(2)} (n={decodeCodenamesCorrelation.n})
                    </div>
                  )}
                </div>
              )}
            </div>

            {/* Specialization */}
            <div className="insight-card insight-chart">
              <div className="insight-label">Multi-Dimensional</div>
              <h3>ToM Is Not Monolithic</h3>
              <p>
                No model excels at all three ToM types. Different architectures and training
                choices favor different aspects of social cognition.
              </p>
              <div className="specialization-chart">
                <div className="spec-row">
                  <span className="spec-dimension cooperative">Cooperative</span>
                  <span className="spec-model">{bestCooperative?.model || "—"}</span>
                  <div
                    className="spec-bar"
                    style={{ width: bestCooperative ? `${bestCooperative.efficiency * 100 * 3}%` : "0%" }}
                  ></div>
                  <span className="spec-score">
                    {bestCooperative ? `${(bestCooperative.efficiency * 100).toFixed(0)}%` : "—"}
                  </span>
                </div>
                <div className="spec-row">
                  <span className="spec-dimension adversarial">Adversarial</span>
                  <span className="spec-model">{bestAdversarial?.model || "—"}</span>
                  <div
                    className="spec-bar adversarial"
                    style={{ width: bestAdversarialComposite !== null ? `${bestAdversarialComposite * 100 * 3}%` : "0%" }}
                  ></div>
                  <span className="spec-score">
                    {bestAdversarialComposite !== null ? `${(bestAdversarialComposite * 100).toFixed(0)}%` : "—"}
                  </span>
                </div>
                <div className="spec-row">
                  <span className="spec-dimension collaborative">Collaborative</span>
                  <span className="spec-model">{bestCollaborative?.model || "—"}</span>
                  <div
                    className="spec-bar collaborative"
                    style={{ width: bestCollaborative ? `${bestCollaborative.win_rate * 100}%` : "0%" }}
                  ></div>
                  <span className="spec-score">
                    {bestCollaborative ? `${(bestCollaborative.win_rate * 100).toFixed(0)}%` : "—"}
                  </span>
                </div>
              </div>
            </div>

            {/* Methodology Note */}
            <div className="insight-card insight-methodology">
              <div className="insight-label">Methodology</div>
              <h3>Minimal Scaffolding</h3>
              <p>
                All games use a minimal harness: LLMs receive rules, goal, scratchpad,
                and redacted game state. No strategy hints or specialized prompting.
                Results reflect latent ability under naturalistic play.
              </p>
              <div className="methodology-stats">
                <div className="method-stat">
                  <span className="method-value">{leaderboard.total_episodes.codenames}</span>
                  <span className="method-label">Codenames</span>
                </div>
                <div className="method-stat">
                  <span className="method-value">{leaderboard.total_episodes.decrypto}</span>
                  <span className="method-label">Decrypto</span>
                </div>
                <div className="method-stat">
                  <span className="method-value">{leaderboard.total_episodes.hanabi}</span>
                  <span className="method-label">Hanabi</span>
                </div>
              </div>
            </div>
          </div>
        </section>
      )}

      {/* Game Navigation Cards */}
      <section className="games-section">
        <h2>Explore Games</h2>
        <div className="game-cards">
          <GameCard
            icon="🎯"
            title="Codenames"
            description="Classic word association game. Teams compete to identify their agents using one-word clues."
            onClick={() => onNavigate("codenames")}
          />
          <GameCard
            icon="🔐"
            title="Decrypto"
            description="Encode secret messages while trying to intercept your opponents' communications."
            onClick={() => onNavigate("decrypto")}
          />
          <GameCard
            icon="🎆"
            title="Hanabi"
            description="Cooperative card game where players can see others' cards but not their own."
            onClick={() => onNavigate("hanabi")}
          />
          <GameCard
            icon="📼"
            title="Replays"
            description="Browse and analyze saved game episodes with full transcripts."
            onClick={() => onNavigate("replay")}
          />
          <GameCard
            icon="📊"
            title="Batch Runner"
            description="Run multiple games in sequence to gather statistics."
            onClick={() => onNavigate("batch")}
          />
          <GameCard
            icon="☁️"
            title="Cloud Benchmark"
            description="Run large-scale benchmarks with real-time monitoring."
            onClick={() => onNavigate("benchmark")}
          />
        </div>
      </section>
    </div>
  );
}

// Leaderboard Table Component
function LeaderboardTable({ rankings, useEfficiency }: { rankings: OverallRanking[]; useEfficiency: boolean }) {
  const [expandedModel, setExpandedModel] = useState<string | null>(null);

  const formatBias = (value?: number | null) => {
    if (value === null || value === undefined) return "Bias —";
    const label = value > 0.05 ? "over" : value < -0.05 ? "under" : "calibrated";
    const sign = value >= 0 ? "+" : "";
    return `Bias ${sign}${value.toFixed(2)} (${label})`;
  };

  const formatBrier = (value?: number | null) =>
    value === null || value === undefined ? "Brier —" : `Brier ${value.toFixed(3)}`;

  // Sort rankings based on selected metric
  const sortedRankings = [...rankings].sort((a, b) => {
    const aScore = useEfficiency ? a.overall_score : a.raw_overall_score;
    const bScore = useEfficiency ? b.overall_score : b.raw_overall_score;
    return bScore - aScore;
  });

  // Re-assign ranks based on current sort
  sortedRankings.forEach((r, i) => {
    r.rank = i + 1;
  });

  return (
    <div className="leaderboard-table-wrapper">
      <div className="score-explanation">
        <span className="score-explanation-label">Overall Score:</span>
        {useEfficiency
          ? " Average of Hanabi efficiency (score/turn), Decrypto adversarial score (intercept × decode), and Codenames win rate. Efficiency-based ranking rewards models that coordinate quickly, not just persistently."
          : " Average of Hanabi raw score (out of 25), Decrypto adversarial score (intercept × decode), and Codenames win rate. Raw scores can favor models that play more turns."
        }
      </div>
      <table className="leaderboard-table">
        <thead>
          <tr>
            <th className="col-rank">#</th>
            <th className="col-model">Model</th>
            <th className="col-score">
              <span className="tooltip-trigger" data-tooltip={useEfficiency
                ? "Efficiency-based composite: Hanabi efficiency + Decrypto (intercept × decode) + Codenames win rate"
                : "Raw score composite: Hanabi raw score + Decrypto (intercept × decode) + Codenames win rate"
              }>
                Overall
              </span>
            </th>
            <th className="col-dimension">
              <span className="tooltip-trigger" data-tooltip={useEfficiency
                ? "Hanabi Efficiency: Score per turn — measures convention establishment speed"
                : "Hanabi Raw Score: Average points out of 25"
              }>
                Cooperative
              </span>
            </th>
            <th className="col-dimension">
              <span className="tooltip-trigger" data-tooltip="Adversarial score = intercept × decode (penalizes miscommunications). Hover cells for breakdown.">
                Adversarial
              </span>
            </th>
            <th className="col-dimension">
              <span className="tooltip-trigger" data-tooltip="Codenames win rate — semantic coordination. Hover for cluer surprise (Brier, lower=better).">
                Collaborative
              </span>
            </th>
            <th className="col-games">
              <span className="tooltip-trigger" data-tooltip="Total games played across all game types">
                Games
              </span>
            </th>
          </tr>
        </thead>
        <tbody>
          {sortedRankings.map((r) => {
            const overallScore = useEfficiency ? r.overall_score : r.raw_overall_score;
            const hanabiScore = useEfficiency
              ? (r.hanabi_efficiency !== null ? r.hanabi_efficiency * 100 : null)
              : r.raw_hanabi_score;
            const isExpanded = expandedModel === r.model;

            return (
              <Fragment key={r.model}>
                <tr>
                  <td className={`rank rank-${r.rank}`}>{r.rank}</td>
                  <td className="model-name">
                    <button
                      type="button"
                      className={`row-expander ${isExpanded ? "open" : ""}`}
                      onClick={() => setExpandedModel(isExpanded ? null : r.model)}
                      aria-expanded={isExpanded}
                      aria-label={`Toggle details for ${r.model}`}
                    >
                      ▸
                    </button>
                    <span>{r.model}</span>
                  </td>
                  <td className="overall-score">
                    <span className="score-value">{overallScore.toFixed(1)}</span>
                    <div className="score-bar">
                      <div
                        className="score-bar-fill"
                        style={{ width: `${overallScore}%` }}
                      />
                    </div>
                  </td>
                  <td className="dimension-score cooperative">
                  <span
                    className="tooltip-cell"
                    data-tooltip={r.hanabi_efficiency !== null
                      ? `Efficiency: ${(r.hanabi_efficiency * 100).toFixed(1)}% | Raw: ${r.raw_hanabi_score?.toFixed(0) ?? "—"}%`
                      : "No Hanabi games played"
                    }
                  >
                    {hanabiScore !== null ? `${hanabiScore.toFixed(0)}%` : "—"}
                    {useEfficiency && r.hanabi_efficiency !== null && (
                      <span className="efficiency-indicator">⚡</span>
                    )}
                  </span>
                </td>
                <td className="dimension-score adversarial">
                  <span
                    className="tooltip-cell"
                  data-tooltip={r.decrypto_score !== null
                    ? `Adversarial: ${r.decrypto_score.toFixed(0)}% | Intercept: ${r.decrypto_intercept?.toFixed(0) ?? "—"}% | Decode: ${r.decrypto_decode?.toFixed(0) ?? "—"}% | Miscomm: ${r.decrypto_decode !== null ? (100 - r.decrypto_decode).toFixed(0) : "—"}% | Win: ${r.decrypto_win_rate?.toFixed(0) ?? "—"}%`
                    : "No Decrypto games played"
                  }
                >
                    {r.decrypto_score !== null ? `${r.decrypto_score.toFixed(0)}%` : "—"}
                  </span>
                </td>
                <td className="dimension-score collaborative">
                  <span
                    className="tooltip-cell"
                    data-tooltip={r.codenames_score !== null
                      ? `Win rate: ${r.codenames_score.toFixed(0)}% | Cluer surprise (Brier, lower=better): ${r.codenames_cluer_surprise?.toFixed(3) ?? "—"}`
                      : "No Codenames games played"}
                  >
                    {r.codenames_score !== null ? `${r.codenames_score.toFixed(0)}%` : "—"}
                  </span>
                </td>
                  <td className="games-count">{r.games_played}</td>
                </tr>
                {isExpanded && (
                  <tr className="leaderboard-detail">
                    <td colSpan={7}>
                      <div className="detail-grid">
                        <div className="detail-card">
                          <div className="detail-title">Codenames Calibration</div>
                          <div className="detail-metric">{formatBrier(r.codenames_cluer_surprise)}</div>
                          <div className="detail-sub">{formatBias(r.codenames_cluer_bias)}</div>
                        </div>
                        <div className="detail-card">
                          <div className="detail-title">Decrypto Decode Calibration</div>
                          <div className="detail-metric">{formatBrier(r.decrypto_decode_brier)}</div>
                          <div className="detail-sub">{formatBias(r.decrypto_decode_bias)}</div>
                        </div>
                        <div className="detail-card">
                          <div className="detail-title">Decrypto Intercept Calibration</div>
                          <div className="detail-metric">{formatBrier(r.decrypto_intercept_brier)}</div>
                          <div className="detail-sub">{formatBias(r.decrypto_intercept_bias)}</div>
                        </div>
                      </div>
                    </td>
                  </tr>
                )}
              </Fragment>
            );
          })}
        </tbody>
      </table>
    </div>
  );
}

// Capability Column Component
function CapabilityColumn<T extends { model: string; games: number }>({
  icon,
  title,
  subtitle,
  description,
  rankings,
  formatValue,
  secondaryValue,
  color,
  onClick,
}: {
  icon: string;
  title: string;
  subtitle: string;
  description: string;
  rankings: T[];
  metric: string;
  formatValue: (r: T) => string;
  secondaryValue?: (r: T) => string;
  color: "cooperative" | "adversarial" | "collaborative";
  onClick: () => void;
}) {
  const top5 = rankings.slice(0, 5);

  return (
    <div className={`capability-column ${color}`}>
      <div className="capability-header" onClick={onClick}>
        <span className="capability-icon">{icon}</span>
        <h3>{title}</h3>
        <p className="capability-subtitle">{subtitle}</p>
        <p className="capability-description">{description}</p>
      </div>
      <div className="capability-rankings">
        {top5.length > 0 ? (
          top5.map((r, i) => (
            <div key={r.model} className="capability-rank-item">
              <span className="rank-num">#{i + 1}</span>
              <span className="model">{r.model}</span>
              <div className="value-group">
                <span className="value">{formatValue(r)}</span>
                {secondaryValue && (
                  <span className="secondary-value">{secondaryValue(r)}</span>
                )}
              </div>
            </div>
          ))
        ) : (
          <div className="capability-empty">No data yet</div>
        )}
      </div>
      <button className="capability-play-btn" onClick={onClick}>
        Play {subtitle} →
      </button>
    </div>
  );
}

// Game Card Component
function GameCard({
  icon,
  title,
  description,
  onClick,
}: {
  icon: string;
  title: string;
  description: string;
  onClick: () => void;
}) {
  return (
    <div className="game-card" onClick={onClick}>
      <div className="game-card-icon">{icon}</div>
      <div className="game-card-content">
        <h3>{title}</h3>
        <p>{description}</p>
      </div>
    </div>
  );
}

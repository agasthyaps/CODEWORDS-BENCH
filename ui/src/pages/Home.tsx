import { useEffect, useState } from "react";
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

  // Compute dynamic insights from leaderboard
  const bestCooperative = leaderboard?.hanabi_rankings[0];
  const bestAdversarial = leaderboard?.decrypto_rankings[0];
  const bestCollaborative = leaderboard?.codenames_rankings[0];

  // Top 3 models by win rate for each game
  const topCodenames = leaderboard?.codenames_rankings.slice(0, 3) || [];
  const topDecrypto = leaderboard?.decrypto_rankings.slice(0, 3) || [];
  const topHanabi = leaderboard?.hanabi_rankings.slice(0, 3) || [];

  return (
    <div className="page home-page">
      {/* Hero Section */}
      <section className="home-hero">
        <div className="hero-content">
          <h1>MindGames</h1>
          <p className="tagline">Measuring how models think about each other</p>
          <p className="subtitle">
            Evaluating Theory of Mind through cooperative and competitive games—how
            well do language models reason about beliefs, knowledge, and intentions?
          </p>
        </div>
        <div className="hero-meta">
          <span className="version-tag">v1.0</span>
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
          <button
            onClick={handleRefresh}
            disabled={refreshing || loading}
            className="refresh-btn"
          >
            {refreshing ? "Refreshing..." : "Refresh"}
          </button>
        </div>

        {error && <div className="leaderboard-error">{error}</div>}

        {loading ? (
          <div className="leaderboard-loading">Loading leaderboard...</div>
        ) : leaderboard && leaderboard.overall_rankings.length > 0 ? (
          <LeaderboardTable rankings={leaderboard.overall_rankings} />
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
            description="What does my partner know?"
            rankings={leaderboard.hanabi_rankings}
            metric="avg_score"
            formatValue={(r: HanabiRanking) => `${r.avg_score}/25`}
            color="cooperative"
            onClick={() => onNavigate("hanabi")}
          />
          <CapabilityColumn
            icon="⚔️"
            title="Adversarial ToM"
            subtitle="Decrypto"
            description="What will my opponent infer?"
            rankings={leaderboard.decrypto_rankings}
            metric="win_rate"
            formatValue={(r: DecryptoRanking) => `${(r.win_rate * 100).toFixed(0)}%`}
            color="adversarial"
            onClick={() => onNavigate("decrypto")}
          />
          <CapabilityColumn
            icon="💬"
            title="Collaborative"
            subtitle="Codenames"
            description="What associations resonate?"
            rankings={leaderboard.codenames_rankings}
            metric="win_rate"
            formatValue={(r: CodenamesRanking) => `${(r.win_rate * 100).toFixed(0)}%`}
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
            {/* Key Finding - Static research insight */}
            <div className="insight-card insight-finding">
              <div className="insight-label">Key Finding</div>
              <h3>The ToM Paradox</h3>
              <p>
                Explicit Theory of Mind language does not correlate with success.
                Models with shorter, more direct reasoning consistently outperform
                those with verbose social deliberation.
              </p>
              <div className="insight-data">
                <div className="data-point">
                  <span className="data-value">-0.34</span>
                  <span className="data-label">Rationale Length vs Win Rate</span>
                </div>
                <div className="data-point">
                  <span className="data-value">+0.06</span>
                  <span className="data-label">ToM Density vs Win Rate</span>
                </div>
              </div>
            </div>

            {/* Best By Dimension - Dynamic */}
            <div className="insight-card insight-chart">
              <div className="insight-label">Model Specialization</div>
              <h3>Best By Dimension</h3>
              <div className="specialization-chart">
                <div className="spec-row">
                  <span className="spec-dimension cooperative">Cooperative</span>
                  <span className="spec-model">{bestCooperative?.model || "—"}</span>
                  <div
                    className="spec-bar"
                    style={{ width: bestCooperative ? `${bestCooperative.score_pct}%` : "0%" }}
                  ></div>
                  <span className="spec-score">
                    {bestCooperative ? `${bestCooperative.avg_score}/25` : "—"}
                  </span>
                </div>
                <div className="spec-row">
                  <span className="spec-dimension adversarial">Adversarial</span>
                  <span className="spec-model">{bestAdversarial?.model || "—"}</span>
                  <div
                    className="spec-bar adversarial"
                    style={{ width: bestAdversarial ? `${bestAdversarial.win_rate * 100}%` : "0%" }}
                  ></div>
                  <span className="spec-score">
                    {bestAdversarial ? `${(bestAdversarial.win_rate * 100).toFixed(0)}%` : "—"}
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

            {/* Top Performers - Codenames */}
            <div className="insight-card insight-top">
              <div className="insight-label">Codenames Leaders</div>
              <h3>Top Collaborative Performers</h3>
              <div className="top-list">
                {topCodenames.length > 0 ? (
                  topCodenames.map((r, i) => (
                    <div key={r.model} className="top-item">
                      <span className="top-rank">#{i + 1}</span>
                      <span className="top-model">{r.model}</span>
                      <div className="top-bar-container">
                        <div
                          className="top-bar collaborative"
                          style={{ width: `${r.win_rate * 100}%` }}
                        ></div>
                      </div>
                      <span className="top-score">{(r.win_rate * 100).toFixed(0)}%</span>
                    </div>
                  ))
                ) : (
                  <div className="top-empty">No data yet</div>
                )}
              </div>
            </div>

            {/* Top Performers - Decrypto */}
            <div className="insight-card insight-top">
              <div className="insight-label">Decrypto Leaders</div>
              <h3>Top Adversarial Performers</h3>
              <div className="top-list">
                {topDecrypto.length > 0 ? (
                  topDecrypto.map((r, i) => (
                    <div key={r.model} className="top-item">
                      <span className="top-rank">#{i + 1}</span>
                      <span className="top-model">{r.model}</span>
                      <div className="top-bar-container">
                        <div
                          className="top-bar adversarial"
                          style={{ width: `${r.win_rate * 100}%` }}
                        ></div>
                      </div>
                      <span className="top-score">{(r.win_rate * 100).toFixed(0)}%</span>
                    </div>
                  ))
                ) : (
                  <div className="top-empty">No data yet</div>
                )}
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
function LeaderboardTable({ rankings }: { rankings: OverallRanking[] }) {
  return (
    <div className="leaderboard-table-wrapper">
      <table className="leaderboard-table">
        <thead>
          <tr>
            <th className="col-rank">#</th>
            <th className="col-model">Model</th>
            <th className="col-score" title="Average performance across all game types (0-100)">
              Overall
            </th>
            <th
              className="col-dimension"
              title="Hanabi: Cooperative ToM — Can the model reason about what partners know and don't know? Score is avg points out of 25."
            >
              Cooperative
            </th>
            <th
              className="col-dimension"
              title="Decrypto: Adversarial ToM — Can the model craft clues teammates understand but opponents can't intercept? Score is win rate %."
            >
              Adversarial
            </th>
            <th
              className="col-dimension"
              title="Codenames: Collaborative Communication — Can the model find word associations that resonate with partners? Score is win rate %."
            >
              Collaborative
            </th>
            <th className="col-games" title="Total games played across all game types">
              Games
            </th>
          </tr>
        </thead>
        <tbody>
          {rankings.map((r) => (
            <tr key={r.model} title={`${r.model}: Overall ${r.overall_score.toFixed(1)}% across ${r.games_played} games`}>
              <td className={`rank rank-${r.rank}`}>{r.rank}</td>
              <td className="model-name">{r.model}</td>
              <td className="overall-score">
                <span className="score-value">{r.overall_score.toFixed(1)}</span>
                <div className="score-bar">
                  <div
                    className="score-bar-fill"
                    style={{ width: `${r.overall_score}%` }}
                  />
                </div>
              </td>
              <td
                className="dimension-score cooperative"
                title={r.hanabi_score !== null ? `Hanabi avg score: ${(r.hanabi_score * 0.25).toFixed(1)}/25` : "No Hanabi games played"}
              >
                {r.hanabi_score !== null ? `${r.hanabi_score.toFixed(0)}%` : "—"}
              </td>
              <td
                className="dimension-score adversarial"
                title={r.decrypto_score !== null ? `Decrypto win rate: ${r.decrypto_score.toFixed(0)}%` : "No Decrypto games played"}
              >
                {r.decrypto_score !== null ? `${r.decrypto_score.toFixed(0)}%` : "—"}
              </td>
              <td
                className="dimension-score collaborative"
                title={r.codenames_score !== null ? `Codenames win rate: ${r.codenames_score.toFixed(0)}%` : "No Codenames games played"}
              >
                {r.codenames_score !== null ? `${r.codenames_score.toFixed(0)}%` : "—"}
              </td>
              <td className="games-count">{r.games_played}</td>
            </tr>
          ))}
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
              <span className="value">{formatValue(r)}</span>
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

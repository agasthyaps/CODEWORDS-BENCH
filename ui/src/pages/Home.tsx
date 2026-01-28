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
      .then(setLeaderboard)
      .catch((e) => setError(e.message))
      .finally(() => setLoading(false));
  }, []);

  const handleRefresh = async () => {
    setRefreshing(true);
    setError(null);
    try {
      await refreshLeaderboard();
      // Wait a moment then fetch updated data
      await new Promise((r) => setTimeout(r, 1500));
      const updated = await fetchLeaderboard();
      setLeaderboard(updated);
    } catch (e: any) {
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

  return (
    <div className="page home-page">
      {/* Hero Section */}
      <section className="home-hero">
        <h1>MindGames</h1>
        <p className="tagline">Measuring how models think about each other</p>
        <p className="subtitle">
          A research platform for evaluating Theory of Mind and multi-agent
          coordination through cooperative and competitive games.
        </p>
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
            <th className="col-score">Overall</th>
            <th className="col-dimension">Cooperative</th>
            <th className="col-dimension">Adversarial</th>
            <th className="col-dimension">Collaborative</th>
            <th className="col-games">Games</th>
          </tr>
        </thead>
        <tbody>
          {rankings.map((r) => (
            <tr key={r.model}>
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
              <td className="dimension-score cooperative">
                {r.hanabi_score !== null ? `${r.hanabi_score.toFixed(0)}%` : "—"}
              </td>
              <td className="dimension-score adversarial">
                {r.decrypto_score !== null ? `${r.decrypto_score.toFixed(0)}%` : "—"}
              </td>
              <td className="dimension-score collaborative">
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

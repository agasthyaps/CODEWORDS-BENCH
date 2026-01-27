type Props = {
  onNavigate: (view: "codenames" | "decrypto" | "hanabi" | "replay" | "batch") => void;
};

export default function Home({ onNavigate }: Props) {
  return (
    <div className="page">
      <div className="home-hero">
        <h1>Codewords Benchmark</h1>
        <p>
          Run LLM-vs-LLM games, analyze performance, and explore replays.
          Choose a game mode below to get started.
        </p>
      </div>

      <div className="home-cards">
        <div className="home-card" onClick={() => onNavigate("codenames")}>
          <div className="home-card-icon">🎯</div>
          <h3>Codenames</h3>
          <p>Classic word association game. Teams compete to identify their agents using one-word clues.</p>
        </div>

        <div className="home-card" onClick={() => onNavigate("decrypto")}>
          <div className="home-card-icon">🔐</div>
          <h3>Decrypto</h3>
          <p>Encode secret messages while trying to intercept your opponents' communications.</p>
        </div>

        <div className="home-card" onClick={() => onNavigate("hanabi")}>
          <div className="home-card-icon">🎆</div>
          <h3>Hanabi</h3>
          <p>Cooperative card game where players can see others' cards but not their own. Give hints wisely!</p>
        </div>

        <div className="home-card" onClick={() => onNavigate("replay")}>
          <div className="home-card-icon">📼</div>
          <h3>Replay Viewer</h3>
          <p>Browse and step through saved game episodes with full transcript and analysis.</p>
        </div>

        <div className="home-card" onClick={() => onNavigate("batch")}>
          <div className="home-card-icon">📊</div>
          <h3>Batch Runner</h3>
          <p>Run multiple games in sequence to gather statistics and compare model performance.</p>
        </div>
      </div>
    </div>
  );
}

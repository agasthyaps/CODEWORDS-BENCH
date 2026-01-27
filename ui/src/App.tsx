import { useEffect, useMemo, useState } from "react";
import { fetchModels } from "./api";
import { ModelInfo } from "./types";
import { useTheme } from "./hooks/useTheme";
import Home from "./pages/Home";
import CodenamesViewer from "./pages/CodenamesViewer";
import DecryptoViewer from "./pages/DecryptoViewer";
import HanabiViewer from "./pages/HanabiViewer";
import ReplayViewer from "./pages/ReplayViewer";
import BatchRunner from "./pages/BatchRunner";

type View = "home" | "codenames" | "decrypto" | "hanabi" | "replay" | "batch";

export default function App() {
  const [view, setView] = useState<View>("home");
  const [models, setModels] = useState<ModelInfo[]>([]);
  const [error, setError] = useState<string | null>(null);
  const { theme, toggle } = useTheme();

  useEffect(() => {
    fetchModels()
      .then((data) => setModels(data))
      .catch((err) => setError(err.message));
  }, []);

  const firstModel = useMemo(() => models[0]?.model_id || "", [models]);

  return (
    <div className="app">
      <header className="topbar">
        <div className="brand">Codewords</div>
        <div className="nav-group">
          <nav className="nav">
            <button
              className={`nav-btn ${view === "home" ? "active" : ""}`}
              onClick={() => setView("home")}
            >
              Home
            </button>
            <button
              className={`nav-btn ${view === "codenames" ? "active" : ""}`}
              onClick={() => setView("codenames")}
            >
              Codenames
            </button>
            <button
              className={`nav-btn ${view === "decrypto" ? "active" : ""}`}
              onClick={() => setView("decrypto")}
            >
              Decrypto
            </button>
            <button
              className={`nav-btn ${view === "hanabi" ? "active" : ""}`}
              onClick={() => setView("hanabi")}
            >
              Hanabi
            </button>
            <button
              className={`nav-btn ${view === "replay" ? "active" : ""}`}
              onClick={() => setView("replay")}
            >
              Replay
            </button>
            <button
              className={`nav-btn ${view === "batch" ? "active" : ""}`}
              onClick={() => setView("batch")}
            >
              Batch
            </button>
          </nav>
          <button
            className="theme-toggle"
            onClick={toggle}
            aria-label="Toggle theme"
            title={`Switch to ${theme === 'light' ? 'dark' : 'light'} mode`}
          >
            {theme === 'light' ? '🌙' : '☀️'}
          </button>
        </div>
      </header>

      {error && <div className="error-banner">{error}</div>}

      {view === "home" && <Home onNavigate={setView} />}
      {view === "codenames" && (
        <CodenamesViewer models={models} defaultModel={firstModel} />
      )}
      {view === "decrypto" && (
        <DecryptoViewer models={models} defaultModel={firstModel} />
      )}
      {view === "hanabi" && (
        <HanabiViewer models={models} defaultModel={firstModel} />
      )}
      {view === "replay" && <ReplayViewer />}
      {view === "batch" && (
        <BatchRunner models={models} defaultModel={firstModel} />
      )}
    </div>
  );
}

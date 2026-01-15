import { useEffect, useMemo, useState } from "react";
import { fetchModels } from "./api";
import { ModelInfo } from "./types";
import Home from "./pages/Home";
import CodenamesViewer from "./pages/CodenamesViewer";
import DecryptoViewer from "./pages/DecryptoViewer";
import ReplayViewer from "./pages/ReplayViewer";
import BatchRunner from "./pages/BatchRunner";

type View = "home" | "codenames" | "decrypto" | "replay" | "batch";

export default function App() {
  const [view, setView] = useState<View>("home");
  const [models, setModels] = useState<ModelInfo[]>([]);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    fetchModels()
      .then((data) => setModels(data))
      .catch((err) => setError(err.message));
  }, []);

  const firstModel = useMemo(() => models[0]?.model_id || "", [models]);

  return (
    <div className="app">
      <header className="topbar">
        <div className="brand">Codewords UI</div>
        <nav className="nav">
          <button onClick={() => setView("home")}>Home</button>
          <button onClick={() => setView("codenames")}>Codenames</button>
          <button onClick={() => setView("decrypto")}>Decrypto</button>
          <button onClick={() => setView("replay")}>Replay</button>
          <button onClick={() => setView("batch")}>Batch</button>
        </nav>
      </header>

      {error && <div className="error">{error}</div>}

      {view === "home" && <Home />}
      {view === "codenames" && (
        <CodenamesViewer models={models} defaultModel={firstModel} />
      )}
      {view === "decrypto" && (
        <DecryptoViewer models={models} defaultModel={firstModel} />
      )}
      {view === "replay" && <ReplayViewer />}
      {view === "batch" && (
        <BatchRunner models={models} defaultModel={firstModel} />
      )}
    </div>
  );
}

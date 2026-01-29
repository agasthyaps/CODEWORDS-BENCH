import { useEffect, useMemo, useState } from "react";
import {
  fetchBenchmarkStatus,
  fetchBenchmarkFindings,
  fetchBenchmarkFinding,
  fetchExperiments,
  fetchRunningGames,
  peekGame,
  restartGame,
  startBenchmark,
  pauseBenchmark,
  cancelBenchmark,
  forceStopBenchmark,
  downloadBenchmarkResults,
  openEventStream,
  estimateBenchmarkCost,
  RunningGame,
  GamePeek,
  CostEstimate,
} from "../api";
import { ModelInfo } from "../types";

type Props = {
  models: ModelInfo[];
};

type BenchmarkStatus = {
  status: "idle" | "running" | "paused" | "complete" | "error" | "cancelled";
  experiment_name?: string;
  started_at?: string;
  codenames?: GameTypeProgress;
  decrypto?: GameTypeProgress;
  hanabi?: GameTypeProgress;
  findings_count: number;
  last_error?: string;
};

type GameTypeProgress = {
  total: number;
  completed: number;
  failed: number;
  running: number;
};

type FindingSummary = {
  finding_id: string;
  game_type: "codenames" | "decrypto" | "hanabi";
  batch_number: number;
  games_analyzed: number;
  timestamp: string;
  preview: string;
};

type FindingDetail = FindingSummary & {
  summary_metrics: Record<string, unknown>;
  analysis: string;
  model: string;
  usage?: Record<string, number>;
};

type Experiment = {
  experiment_name: string;
  status: string;
  started_at?: string;
  total_completed: number;
  total_failed: number;
  findings_count: number;
};

export default function BenchmarkMonitor({ models }: Props) {
  // Config state
  const [experimentName, setExperimentName] = useState("");
  const [selectedModels, setSelectedModels] = useState<string[]>([]);
  const [seedCount, setSeedCount] = useState(30);
  const [runCodenames, setRunCodenames] = useState(true);
  const [runDecrypto, setRunDecrypto] = useState(true);
  const [runHanabi, setRunHanabi] = useState(true);
  const [showAdvanced, setShowAdvanced] = useState(false);

  // Advanced options
  const [codenamesConcurrency, setCodenamesConcurrency] = useState(2);
  const [decryptoConcurrency, setDecryptoConcurrency] = useState(2);
  const [hanabiConcurrency, setHanabiConcurrency] = useState(1);
  const [analysisBatchSize, setAnalysisBatchSize] = useState(10);

  // Status state
  const [status, setStatus] = useState<BenchmarkStatus>({ status: "idle", findings_count: 0 });
  const [findings, setFindings] = useState<FindingSummary[]>([]);
  const [selectedFinding, setSelectedFinding] = useState<FindingDetail | null>(null);
  const [experiments, setExperiments] = useState<Experiment[]>([]);
  const [expandedExperiment, setExpandedExperiment] = useState<string | null>(null);
  const [expFindings, setExpFindings] = useState<FindingSummary[]>([]);
  const [expFindingsLoading, setExpFindingsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  // Running games state
  const [runningGames, setRunningGames] = useState<RunningGame[]>([]);
  const [selectedGamePeek, setSelectedGamePeek] = useState<GamePeek | null>(null);
  const [peekLoading, setPeekLoading] = useState(false);

  // Cost estimation
  const [costEstimate, setCostEstimate] = useState<CostEstimate | null>(null);
  const [costLoading, setCostLoading] = useState(false);

  // Load findings when an experiment is expanded
  useEffect(() => {
    if (expandedExperiment) {
      setExpFindingsLoading(true);
      fetchBenchmarkFindings(expandedExperiment)
        .then(setExpFindings)
        .catch(() => setExpFindings([]))
        .finally(() => setExpFindingsLoading(false));
    } else {
      setExpFindings([]);
    }
  }, [expandedExperiment]);

  // Polling status
  useEffect(() => {
    const loadStatus = async () => {
      try {
        const data = await fetchBenchmarkStatus();
        setStatus(data);
      } catch (e) {
        console.error("Failed to load status", e);
      }
    };

    loadStatus();
    const interval = setInterval(loadStatus, 5000);
    return () => clearInterval(interval);
  }, []);

  // Load experiments on mount
  useEffect(() => {
    fetchExperiments()
      .then(setExperiments)
      .catch(console.error);
  }, []);

  // Poll findings while running or when status changes
  useEffect(() => {
    if (status.status === "idle") {
      return;
    }

    // Initial load
    fetchBenchmarkFindings()
      .then(setFindings)
      .catch(console.error);

    // Poll findings every 10 seconds while running
    if (status.status === "running") {
      const interval = setInterval(() => {
        fetchBenchmarkFindings()
          .then(setFindings)
          .catch(console.error);
      }, 10000);
      return () => clearInterval(interval);
    }
  }, [status.status]);

  // Poll running games while benchmark is running
  useEffect(() => {
    if (status.status !== "running") {
      setRunningGames([]);
      return;
    }

    // Initial load
    fetchRunningGames()
      .then(setRunningGames)
      .catch(console.error);

    // Poll every 3 seconds
    const interval = setInterval(() => {
      fetchRunningGames()
        .then(setRunningGames)
        .catch(console.error);
    }, 3000);
    return () => clearInterval(interval);
  }, [status.status]);

  // Generate experiment name
  useEffect(() => {
    if (!experimentName) {
      const ts = new Date().toISOString().slice(0, 16).replace("T", "_").replace(":", "");
      setExperimentName(`benchmark_${ts}`);
    }
  }, []);

  // Handle model selection
  const toggleModel = (modelId: string) => {
    setSelectedModels((prev) =>
      prev.includes(modelId)
        ? prev.filter((m) => m !== modelId)
        : [...prev, modelId]
    );
  };

  // Calculate total games
  const totalGames = useMemo(() => {
    if (selectedModels.length < 2) return 0;

    // Matchups: for n models, round-robin pairs = n*(n-1)/2
    // Each pair generates 4 configs for Codenames/Decrypto
    const numPairs = (selectedModels.length * (selectedModels.length - 1)) / 2;

    let total = 0;
    if (runCodenames) {
      // 8 matchups per pair (4 configs × 2 directions) × seeds
      total += numPairs * 8 * seedCount;
    }
    if (runDecrypto) {
      // 4 matchups per pair × seeds
      total += numPairs * 4 * seedCount;
    }
    if (runHanabi) {
      // Each model × seeds
      total += selectedModels.length * seedCount;
    }
    return total;
  }, [selectedModels, seedCount, runCodenames, runDecrypto, runHanabi]);

  // Estimate cost when config changes
  useEffect(() => {
    if (selectedModels.length === 0 || totalGames === 0) {
      setCostEstimate(null);
      return;
    }

    const gameTypes = [
      ...(runCodenames ? ["codenames"] : []),
      ...(runDecrypto ? ["decrypto"] : []),
      ...(runHanabi ? ["hanabi"] : []),
    ];

    const estimateCost = async () => {
      setCostLoading(true);
      try {
        // Use a simplified estimation: per-model, per-game-type, scaled by seed count
        const estimate = await estimateBenchmarkCost(selectedModels, gameTypes, seedCount);
        setCostEstimate(estimate);
      } catch (e) {
        console.warn("Cost estimation failed:", e);
        setCostEstimate(null);
      } finally {
        setCostLoading(false);
      }
    };

    // Debounce the estimation
    const timer = setTimeout(estimateCost, 300);
    return () => clearTimeout(timer);
  }, [selectedModels, seedCount, runCodenames, runDecrypto, runHanabi, totalGames]);

  const handleStart = async () => {
    if (selectedModels.length < 2) {
      setError("Select at least 2 models for competitive games");
      return;
    }
    if (!experimentName.trim()) {
      setError("Enter an experiment name");
      return;
    }

    setError(null);

    try {
      await startBenchmark({
        experiment_name: experimentName,
        model_ids: selectedModels,
        seed_count: seedCount,
        run_codenames: runCodenames,
        run_decrypto: runDecrypto,
        run_hanabi: runHanabi,
        codenames_concurrency: codenamesConcurrency,
        decrypto_concurrency: decryptoConcurrency,
        hanabi_concurrency: hanabiConcurrency,
        interim_analysis_batch_size: analysisBatchSize,
      });

      // Refresh status
      const newStatus = await fetchBenchmarkStatus();
      setStatus(newStatus);
    } catch (e) {
      setError(String(e));
    }
  };

  const handlePause = async () => {
    try {
      await pauseBenchmark();
      const newStatus = await fetchBenchmarkStatus();
      setStatus(newStatus);
    } catch (e) {
      setError(String(e));
    }
  };

  const handleCancel = async () => {
    if (!confirm("Are you sure you want to cancel the benchmark? This cannot be undone.")) {
      return;
    }
    try {
      await cancelBenchmark();
      const newStatus = await fetchBenchmarkStatus();
      setStatus(newStatus);
    } catch (e) {
      setError(String(e));
    }
  };

  const handleDownload = () => {
    if (status.experiment_name) {
      downloadBenchmarkResults(status.experiment_name);
    }
  };

  const handleResume = async () => {
    if (!status.experiment_name) return;

    setError(null);
    try {
      // Resume by calling start with the same experiment name
      // The backend will load existing state and continue from where it left off
      await startBenchmark({
        experiment_name: status.experiment_name,
        model_ids: selectedModels.length >= 2 ? selectedModels : [], // Will use existing config
        seed_count: seedCount,
        run_codenames: runCodenames,
        run_decrypto: runDecrypto,
        run_hanabi: runHanabi,
        codenames_concurrency: codenamesConcurrency,
        decrypto_concurrency: decryptoConcurrency,
        hanabi_concurrency: hanabiConcurrency,
        interim_analysis_batch_size: analysisBatchSize,
      });

      const newStatus = await fetchBenchmarkStatus();
      setStatus(newStatus);
    } catch (e) {
      setError(String(e));
    }
  };

  const loadFindingDetail = async (findingId: string, experimentName?: string) => {
    try {
      const detail = await fetchBenchmarkFinding(findingId, experimentName);
      setSelectedFinding(detail);
    } catch (e) {
      console.error("Failed to load finding", e);
    }
  };

  const handlePeekGame = async (gameId: string) => {
    setPeekLoading(true);
    try {
      const peek = await peekGame(gameId);
      setSelectedGamePeek(peek);
    } catch (e) {
      setError(`Failed to peek at game: ${e}`);
    } finally {
      setPeekLoading(false);
    }
  };

  const handleRestartGame = async (gameId: string) => {
    if (!confirm(`Restart game "${gameId}"? This will cancel the current execution and re-queue it.`)) {
      return;
    }
    try {
      await restartGame(gameId);
      // Refresh running games
      const games = await fetchRunningGames();
      setRunningGames(games);
      setSelectedGamePeek(null);
    } catch (e) {
      setError(`Failed to restart game: ${e}`);
    }
  };

  const formatDuration = (seconds: number): string => {
    const mins = Math.floor(seconds / 60);
    const secs = Math.floor(seconds % 60);
    if (mins >= 60) {
      const hrs = Math.floor(mins / 60);
      const remainMins = mins % 60;
      return `${hrs}h ${remainMins}m`;
    }
    return `${mins}m ${secs}s`;
  };

  const isRunning = status.status === "running";
  const canStart = !isRunning && selectedModels.length >= 2 && experimentName.trim();

  return (
    <div className="page benchmark-monitor">
      <h2>Cloud Benchmark</h2>

      <div className="benchmark-layout">
        {/* Config Panel */}
        <div className="panel benchmark-config">
          <h3>Configuration</h3>

          <div className="form-row">
            <label>Experiment Name</label>
            <input
              type="text"
              value={experimentName}
              onChange={(e) => setExperimentName(e.target.value)}
              disabled={isRunning}
              placeholder="my_benchmark"
            />
          </div>

          <div className="form-row">
            <label>Seeds</label>
            <input
              type="number"
              value={seedCount}
              min={1}
              max={100}
              onChange={(e) => setSeedCount(Math.max(1, Math.min(100, Number(e.target.value))))}
              disabled={isRunning}
            />
          </div>

          <div className="form-row">
            <label>Games</label>
            <div className="checkbox-group">
              <label className="checkbox-label">
                <input
                  type="checkbox"
                  checked={runCodenames}
                  onChange={(e) => setRunCodenames(e.target.checked)}
                  disabled={isRunning}
                />
                Codenames
              </label>
              <label className="checkbox-label">
                <input
                  type="checkbox"
                  checked={runDecrypto}
                  onChange={(e) => setRunDecrypto(e.target.checked)}
                  disabled={isRunning}
                />
                Decrypto
              </label>
              <label className="checkbox-label">
                <input
                  type="checkbox"
                  checked={runHanabi}
                  onChange={(e) => setRunHanabi(e.target.checked)}
                  disabled={isRunning}
                />
                Hanabi
              </label>
            </div>
          </div>

          <div className="form-section">
            <label>Models ({selectedModels.length} selected)</label>
            <div className="model-grid">
              {models.map((m) => (
                <label
                  key={m.model_id}
                  className={`model-chip ${selectedModels.includes(m.model_id) ? "selected" : ""}`}
                >
                  <input
                    type="checkbox"
                    checked={selectedModels.includes(m.model_id)}
                    onChange={() => toggleModel(m.model_id)}
                    disabled={isRunning}
                  />
                  {m.name || m.model_id.split("/").pop()}
                </label>
              ))}
            </div>
          </div>

          {/* Advanced Toggle */}
          <div
            className="advanced-toggle"
            onClick={() => setShowAdvanced(!showAdvanced)}
          >
            <span className={`toggle-arrow ${showAdvanced ? "open" : ""}`}>▸</span>
            <span>Advanced Options</span>
          </div>

          {showAdvanced && (
            <div className="advanced-options">
              <div className="form-row-compact">
                <label>Codenames Concurrency</label>
                <input
                  type="number"
                  value={codenamesConcurrency}
                  min={1}
                  max={5}
                  onChange={(e) => setCodenamesConcurrency(Number(e.target.value))}
                  disabled={isRunning}
                />
              </div>
              <div className="form-row-compact">
                <label>Decrypto Concurrency</label>
                <input
                  type="number"
                  value={decryptoConcurrency}
                  min={1}
                  max={5}
                  onChange={(e) => setDecryptoConcurrency(Number(e.target.value))}
                  disabled={isRunning}
                />
              </div>
              <div className="form-row-compact">
                <label>Hanabi Concurrency</label>
                <input
                  type="number"
                  value={hanabiConcurrency}
                  min={1}
                  max={3}
                  onChange={(e) => setHanabiConcurrency(Number(e.target.value))}
                  disabled={isRunning}
                />
              </div>
              <div className="form-row-compact">
                <label>Analysis Batch Size</label>
                <input
                  type="number"
                  value={analysisBatchSize}
                  min={5}
                  max={50}
                  onChange={(e) => setAnalysisBatchSize(Number(e.target.value))}
                  disabled={isRunning}
                />
              </div>
            </div>
          )}

          <div className="benchmark-action">
            {isRunning ? (
              <div className="action-buttons">
                <button onClick={handlePause} className="pause-btn">
                  Pause
                </button>
                <button onClick={handleCancel} className="cancel-btn">
                  Cancel
                </button>
              </div>
            ) : (
              <>
                <button onClick={handleStart} disabled={!canStart}>
                  Start Benchmark ({totalGames} games)
                </button>
                {/* Cost Estimate */}
                {costEstimate && (
                  <div className={`cost-estimate confidence-${costEstimate.confidence}`}>
                    <span className="cost-value">
                      {costLoading ? "..." : costEstimate.estimated_cost_display}
                    </span>
                    <span className="cost-label">est. cost</span>
                    {costEstimate.confidence !== "high" && (
                      <span className="cost-confidence" title={costEstimate.notes.join("; ")}>
                        ({costEstimate.confidence})
                      </span>
                    )}
                  </div>
                )}
              </>
            )}
          </div>

          {error && <div className="error-banner">{error}</div>}
        </div>

        {/* Progress Panel */}
        <div className="panel benchmark-progress">
          <h3>
            Progress
            <span className={`status-badge ${status.status}`}>
              {status.status}
            </span>
          </h3>

          {status.experiment_name && (
            <div className="experiment-name">{status.experiment_name}</div>
          )}

          {status.codenames && (
            <ProgressBar
              label="Codenames"
              progress={status.codenames}
            />
          )}

          {status.decrypto && (
            <ProgressBar
              label="Decrypto"
              progress={status.decrypto}
            />
          )}

          {status.hanabi && (
            <ProgressBar
              label="Hanabi"
              progress={status.hanabi}
            />
          )}

          {/* Findings count indicator */}
          {status.status !== "idle" && (
            <div className="findings-indicator">
              <span className="findings-label">Analysis Findings:</span>
              <span className="findings-value">{status.findings_count}</span>
              {status.status === "running" && status.findings_count === 0 && (
                <span className="findings-hint">(generated every {analysisBatchSize} games)</span>
              )}
            </div>
          )}

          {status.last_error && (
            <div className="error-banner">
              Last error: {status.last_error}
            </div>
          )}

          {status.experiment_name && (status.status === "paused" || status.status === "cancelled") && (
            <button onClick={handleResume} className="resume-btn">
              Continue Benchmark
            </button>
          )}

          {status.experiment_name && (status.status === "complete" || status.status === "paused" || status.status === "cancelled" || status.status === "error") && (
            <button onClick={handleDownload} className="download-btn">
              Download Results
            </button>
          )}

          {/* Running Games Section */}
          {isRunning && runningGames.length > 0 && (
            <div className="running-games-section">
              <h4>Running Games ({runningGames.length})</h4>
              <div className="running-games-list">
                {runningGames.map((game) => (
                  <div key={game.game_id} className="running-game-item">
                    <div className="running-game-header">
                      <span className={`game-type-badge ${game.game_type}`}>
                        {game.game_type === "codenames" ? "CN" : game.game_type === "decrypto" ? "DC" : "HB"}
                      </span>
                      <span className="running-game-id" title={game.game_id}>
                        {game.game_id.length > 20 ? game.game_id.slice(0, 20) + "..." : game.game_id}
                      </span>
                      <span className={`running-game-duration ${game.duration_seconds > 300 ? "warning" : ""} ${game.duration_seconds > 600 ? "danger" : ""}`}>
                        {formatDuration(game.duration_seconds)}
                      </span>
                    </div>
                    <div className="running-game-actions">
                      <button
                        className="peek-btn"
                        onClick={() => handlePeekGame(game.game_id)}
                        disabled={peekLoading}
                      >
                        Peek
                      </button>
                      <button
                        className="restart-btn"
                        onClick={() => handleRestartGame(game.game_id)}
                      >
                        Restart
                      </button>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          )}
        </div>

        {/* Findings Panel */}
        <div className="panel benchmark-findings">
          <h3>
            Findings
            {findings.length > 0 && (
              <span className="findings-count">{findings.length}</span>
            )}
          </h3>

          <div className="findings-list">
            {findings.length === 0 ? (
              status.findings_count > 0 ? (
                <div className="findings-mismatch-warning">
                  ⚠️ {status.findings_count} finding(s) recorded but not loaded.
                  Try refreshing the page.
                </div>
              ) : (
                <div className="no-findings">
                  Analysis findings will appear here after every {analysisBatchSize} games
                </div>
              )
            ) : (
              findings.map((f) => (
                <div
                  key={f.finding_id}
                  className={`finding-item ${selectedFinding?.finding_id === f.finding_id ? "selected" : ""}`}
                  onClick={() => loadFindingDetail(f.finding_id)}
                >
                  <div className="finding-header">
                    <span className={`game-type-badge ${f.game_type}`}>
                      {f.game_type === "codenames" ? "CN" : f.game_type === "decrypto" ? "DC" : "HB"}
                    </span>
                    <span className="finding-batch">Batch {f.batch_number}</span>
                    <span className="finding-count">{f.games_analyzed} games</span>
                  </div>
                  <div className="finding-preview">{f.preview}</div>
                </div>
              ))
            )}
          </div>
        </div>
      </div>

      {/* Finding Detail Modal */}
      {selectedFinding && (
        <div className="finding-modal-backdrop" onClick={() => setSelectedFinding(null)}>
          <div className="finding-modal" onClick={(e) => e.stopPropagation()}>
            <div className="modal-header">
              <h3>
                {selectedFinding.game_type} Analysis - Batch {selectedFinding.batch_number}
              </h3>
              <button className="close-btn" onClick={() => setSelectedFinding(null)}>
                ×
              </button>
            </div>
            <div className="modal-body">
              <div className="finding-metrics">
                <pre>{JSON.stringify(selectedFinding.summary_metrics, null, 2)}</pre>
              </div>
              <div className="finding-analysis">
                {selectedFinding.analysis}
              </div>
            </div>
            <div className="modal-footer">
              <span className="finding-meta">
                Model: {selectedFinding.model}
                {selectedFinding.usage && (
                  <> | Tokens: {selectedFinding.usage.input_tokens}+{selectedFinding.usage.output_tokens}</>
                )}
              </span>
            </div>
          </div>
        </div>
      )}

      {/* Game Peek Modal */}
      {selectedGamePeek && (
        <div className="finding-modal-backdrop" onClick={() => setSelectedGamePeek(null)}>
          <div className="finding-modal game-peek-modal" onClick={(e) => e.stopPropagation()}>
            <div className="modal-header">
              <h3>
                <span className={`game-type-badge ${selectedGamePeek.game_type}`}>
                  {selectedGamePeek.game_type}
                </span>
                {selectedGamePeek.game_id}
              </h3>
              <button className="close-btn" onClick={() => setSelectedGamePeek(null)}>
                ×
              </button>
            </div>
            <div className="modal-body">
              {selectedGamePeek.stale_warning && (
                <div className="stale-warning">
                  ⚠️ Game running for &gt;5 minutes without updates. Consider restarting if stuck.
                </div>
              )}

              <div className="peek-info-row">
                <span><strong>Duration:</strong> {formatDuration(selectedGamePeek.duration_seconds)}</span>
                <span><strong>Turn:</strong> {selectedGamePeek.current_turn ?? "N/A"}</span>
                <span><strong>Seed:</strong> {runningGames.find(g => g.game_id === selectedGamePeek.game_id)?.seed ?? "N/A"}</span>
              </div>

              {/* Show models involved */}
              {(() => {
                const game = runningGames.find(g => g.game_id === selectedGamePeek.game_id);
                if (!game?.models) return null;
                return (
                  <div className="peek-section">
                    <h4>Models</h4>
                    <div className="models-list">
                      {Object.entries(game.models).map(([role, model]) => (
                        <div key={role} className="model-item">
                          <span className="model-role">{role}:</span>
                          <span className="model-id">{model}</span>
                        </div>
                      ))}
                    </div>
                  </div>
                );
              })()}

              {selectedGamePeek.recent_transcript.length > 0 && (
                <div className="peek-section">
                  <h4>Recent Events ({selectedGamePeek.recent_transcript.length})</h4>
                  <div className="transcript-preview">
                    {selectedGamePeek.recent_transcript.slice(-10).map((event, i) => (
                      <pre key={i} className="transcript-event">
                        {JSON.stringify(event, null, 2)}
                      </pre>
                    ))}
                  </div>
                </div>
              )}

              {Object.keys(selectedGamePeek.agent_scratchpads).length > 0 && (
                <div className="peek-section">
                  <h4>Agent Scratchpads</h4>
                  {Object.entries(selectedGamePeek.agent_scratchpads).map(([agentId, content]) => (
                    <div key={agentId} className="scratchpad-preview">
                      <strong>{agentId}:</strong>
                      <pre>{content.slice(-500)}{content.length > 500 ? "..." : ""}</pre>
                    </div>
                  ))}
                </div>
              )}

              {selectedGamePeek.recent_transcript.length === 0 &&
               Object.keys(selectedGamePeek.agent_scratchpads).length === 0 && (
                <div className="no-peek-data">
                  <p>Live state streaming is not yet implemented for cloud benchmark games.</p>
                  <p>The game is running - use <strong>Restart</strong> if it appears stuck (running &gt; 10 min).</p>
                </div>
              )}
            </div>
            <div className="modal-footer">
              <button
                className="restart-btn danger"
                onClick={() => handleRestartGame(selectedGamePeek.game_id)}
              >
                Restart This Game
              </button>
              <button
                className="refresh-btn"
                onClick={() => handlePeekGame(selectedGamePeek.game_id)}
                disabled={peekLoading}
              >
                {peekLoading ? "Loading..." : "Refresh"}
              </button>
            </div>
          </div>
        </div>
      )}

      {/* Previous Experiments */}
      {experiments.length > 0 && (
        <div className="panel previous-experiments">
          <h3>Previous Experiments</h3>
          <div className="experiments-list">
            {experiments.map((exp) => (
              <div
                key={exp.experiment_name}
                className={`experiment-item ${expandedExperiment === exp.experiment_name ? "expanded" : ""}`}
              >
                <div
                  className="experiment-header"
                  onClick={() => setExpandedExperiment(
                    expandedExperiment === exp.experiment_name ? null : exp.experiment_name
                  )}
                >
                  <span className="exp-expand">{expandedExperiment === exp.experiment_name ? "▼" : "▶"}</span>
                  <span className="exp-name">{exp.experiment_name}</span>
                  <span className={`exp-status ${exp.status}`}>{exp.status}</span>
                  <span className="exp-stats">
                    {exp.total_completed} completed, {exp.total_failed} failed
                  </span>
                  {exp.findings_count > 0 && (
                    <span className="exp-findings">{exp.findings_count} findings</span>
                  )}
                </div>

                {expandedExperiment === exp.experiment_name && (
                  <div className="experiment-details">
                    <div className="exp-detail-row">
                      <span className="exp-detail-label">Started:</span>
                      <span>{exp.started_at ? new Date(exp.started_at).toLocaleString() : "Unknown"}</span>
                    </div>
                    <div className="exp-detail-row">
                      <span className="exp-detail-label">Games:</span>
                      <span>{exp.total_completed} completed, {exp.total_failed} failed</span>
                    </div>

                    <div className="exp-actions-row">
                      {(exp.status === "paused" || exp.status === "cancelled") && (
                        <button
                          className="exp-action-btn resume"
                          onClick={(e) => {
                            e.stopPropagation();
                            setExperimentName(exp.experiment_name);
                            startBenchmark({
                              experiment_name: exp.experiment_name,
                              model_ids: selectedModels.length >= 2 ? selectedModels : [],
                              seed_count: seedCount,
                              run_codenames: runCodenames,
                              run_decrypto: runDecrypto,
                              run_hanabi: runHanabi,
                            }).then(() => {
                              fetchBenchmarkStatus().then(setStatus);
                              fetchExperiments().then(setExperiments);
                            }).catch((e) => setError(String(e)));
                          }}
                          disabled={isRunning}
                        >
                          Resume
                        </button>
                      )}
                      {exp.status === "running" && (
                        <button
                          className="exp-action-btn stop"
                          onClick={(e) => {
                            e.stopPropagation();
                            if (!confirm(`Force stop "${exp.experiment_name}"? Use this if the experiment appears stuck.`)) {
                              return;
                            }
                            forceStopBenchmark(exp.experiment_name)
                              .then(() => {
                                fetchBenchmarkStatus().then(setStatus);
                                fetchExperiments().then(setExperiments);
                              })
                              .catch((e) => setError(String(e)));
                          }}
                        >
                          Force Stop
                        </button>
                      )}
                      <button
                        className="exp-action-btn download"
                        onClick={(e) => {
                          e.stopPropagation();
                          downloadBenchmarkResults(exp.experiment_name);
                        }}
                      >
                        Download Results
                      </button>
                    </div>

                    <div className="exp-findings-section">
                      <div className="exp-findings-header">
                        Analysis Findings {expFindings.length > 0 && `(${expFindings.length})`}
                      </div>
                      {expFindingsLoading ? (
                        <div className="exp-findings-loading">Loading findings...</div>
                      ) : expFindings.length === 0 ? (
                        <div className="exp-findings-empty">No analysis findings yet</div>
                      ) : (
                        <div className="exp-findings-list">
                          {expFindings.map((f) => (
                            <div
                              key={f.finding_id}
                              className="exp-finding-item"
                              onClick={(e) => {
                                e.stopPropagation();
                                loadFindingDetail(f.finding_id, exp.experiment_name);
                              }}
                            >
                              <div className="exp-finding-header-row">
                                <span className={`finding-type ${f.game_type}`}>
                                  {f.game_type === "codenames" ? "CN" : f.game_type === "decrypto" ? "DC" : "HB"}
                                </span>
                                <span className="finding-batch">Batch {f.batch_number}</span>
                                <span className="finding-games">{f.games_analyzed} games</span>
                              </div>
                              <div className="exp-finding-preview">{f.preview}</div>
                            </div>
                          ))}
                        </div>
                      )}
                    </div>
                  </div>
                )}
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}

function ProgressBar({ label, progress }: { label: string; progress: GameTypeProgress }) {
  const percent = progress.total > 0
    ? Math.round(((progress.completed + progress.failed) / progress.total) * 100)
    : 0;

  return (
    <div className="progress-section">
      <div className="progress-header">
        <span className="progress-label">{label}</span>
        <span className="progress-stats">
          {progress.completed}/{progress.total}
          {progress.failed > 0 && <span className="failed"> ({progress.failed} failed)</span>}
          {progress.running > 0 && <span className="running"> ({progress.running} running)</span>}
        </span>
      </div>
      <div className="progress-bar">
        <div
          className="progress-fill completed"
          style={{ width: `${(progress.completed / progress.total) * 100}%` }}
        />
        <div
          className="progress-fill failed"
          style={{ width: `${(progress.failed / progress.total) * 100}%` }}
        />
      </div>
      <div className="progress-percent">{percent}%</div>
    </div>
  );
}

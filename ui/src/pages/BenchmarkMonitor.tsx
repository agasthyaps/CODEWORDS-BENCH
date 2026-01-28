import { useEffect, useMemo, useState } from "react";
import {
  fetchBenchmarkStatus,
  fetchBenchmarkFindings,
  fetchBenchmarkFinding,
  fetchExperiments,
  startBenchmark,
  pauseBenchmark,
  cancelBenchmark,
  downloadBenchmarkResults,
  openEventStream,
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
  const [error, setError] = useState<string | null>(null);

  // Load findings when an experiment is expanded
  useEffect(() => {
    if (expandedExperiment) {
      fetchBenchmarkFindings(expandedExperiment)
        .then(setExpFindings)
        .catch(() => setExpFindings([]));
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
              <button onClick={handleStart} disabled={!canStart}>
                Start Benchmark ({totalGames} games)
              </button>
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
              <div className="no-findings">
                Analysis findings will appear here after every {analysisBatchSize} games
              </div>
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
                      {(exp.status === "paused" || exp.status === "cancelled" || exp.status === "running") && (
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
                          {exp.status === "running" ? "Reconnect" : "Resume"}
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

                    {expFindings.length > 0 && (
                      <div className="exp-findings-section">
                        <div className="exp-findings-header">Analysis Findings ({expFindings.length})</div>
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
                              <span className={`finding-type ${f.game_type}`}>
                                {f.game_type === "codenames" ? "CN" : f.game_type === "decrypto" ? "DC" : "HB"}
                              </span>
                              <span className="finding-batch">Batch {f.batch_number}</span>
                              <span className="finding-games">{f.games_analyzed} games</span>
                              <span className="finding-preview">{f.preview.slice(0, 80)}...</span>
                            </div>
                          ))}
                        </div>
                      </div>
                    )}
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

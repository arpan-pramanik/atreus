"use client";

import { useEffect, useState, useRef } from "react";

interface HardwareInfo {
  cpu: {
    model: string;
    threads: number;
    ram_total_mb: number;
    ram_used_mb: number;
    ram_util_pct: number;
  };
  gpu: {
    model: string;
    cuda_available: boolean;
    vram_total_mb: number;
    vram_used_mb: number;
    vram_util_pct: number;
    gpu_util_pct: number;
    device_ordinal: number;
  };
}

interface TrajectoryPoint {
  sample: number;
  cumulative_acc: number;
  window_acc: number;
  f1: number;
  drift: number;
}

interface ComponentMod {
  sample_index: number;
  event: string;
  components_affected: number;
  adaptation_latency_ms: number;
  device: string;
}

interface FeatureRank {
  rank: number;
  feature_identifier?: string;
  feature_name: string;
  feature_index: number;
  active: boolean;
}

interface LogEntry {
  timestamp: string;
  message: string;
  level: string;
  details?: Record<string, unknown>;
}

interface TelemetrySnapshot {
  running: boolean;
  dataset_name: string;
  model_name: string;
  detector_name: string;
  stage: string;
  device: string;
  current_sample: number;
  total_samples: number;
  accuracy: number;
  window_accuracy: number;
  f1_score: number;
  drift_count: number;
  adaptation_count: number;
  components_modified: number;
  adaptation_time_ms: number;
  throughput: number;
  logs: LogEntry[];
  trajectory: TrajectoryPoint[];
  feature_rankings: FeatureRank[];
  component_modifications: ComponentMod[];
}

export default function ControlPanel() {
  const [hardware, setHardware] = useState<HardwareInfo | null>(null);
  const [telemetry, setTelemetry] = useState<TelemetrySnapshot | null>(null);
  const [datasets, setDatasets] = useState<Array<{ id: string; name: string; type: string }>>([
    { id: "sea", name: "SEA Concepts Benchmark (10,000 Samples)", type: "Synthetic Abrupt" },
    { id: "agrawal", name: "Agrawal Stream Benchmark (10,000 Samples)", type: "Synthetic Subspace" },
    { id: "electricity", name: "NSW Electricity Market (45,312 Samples)", type: "Real Recurring" },
    { id: "covertype", name: "Forest Covertype Stream (30,000 Samples)", type: "Real Multi-Class" },
    { id: "airlines", name: "Airlines Flight Delay Stream (15,000 Samples)", type: "Real Shift" },
    { id: "hf_adult", name: "Hugging Face: Adult Census Income (32,561 Samples)", type: "Hugging Face" },
    { id: "hf_bank", name: "Hugging Face: Bank Marketing Stream (10,578 Samples)", type: "Hugging Face" },
    { id: "hf_credit", name: "Hugging Face: Credit Default Risk (13,272 Samples)", type: "Hugging Face" },
  ]);

  const [selectedDataset, setSelectedDataset] = useState("sea");
  const [selectedModel, setSelectedModel] = useState("proposed");
  const [selectedDetector, setSelectedDetector] = useState("adwin");
  const [warmup, setWarmup] = useState(500);
  const [windowSize, setWindowSize] = useState(200);

  const [showAddDataset, setShowAddDataset] = useState(false);
  const [hfInput, setHfInput] = useState("");
  const [uploadStatus, setUploadStatus] = useState("");
  const [exportNotice, setExportNotice] = useState("");
  const fileInputRef = useRef<HTMLInputElement>(null);
  const logConsoleRef = useRef<HTMLDivElement>(null);

  // Poll hardware
  const fetchHardware = async () => {
    try {
      const res = await fetch("/api/hardware");
      if (res.ok) {
        const data = await res.json();
        setHardware(data);
      }
    } catch {}
  };

  // Poll telemetry
  const fetchTelemetry = async () => {
    try {
      const res = await fetch("/api/status");
      if (res.ok) {
        const data: TelemetrySnapshot = await res.json();
        setTelemetry(data);
      }
    } catch {}
  };

  // Fetch datasets list
  const fetchDatasets = async () => {
    try {
      const res = await fetch("/api/datasets");
      if (res.ok) {
        const data = await res.json();
        if (data.datasets) setDatasets(data.datasets);
      }
    } catch {}
  };

  useEffect(() => {
    fetchHardware();
    fetchTelemetry();
    fetchDatasets();

    const hwInterval = setInterval(fetchHardware, 4000);
    const telInterval = setInterval(fetchTelemetry, 700);

    return () => {
      clearInterval(hwInterval);
      clearInterval(telInterval);
    };
  }, []);

  // Auto-scroll logs
  useEffect(() => {
    if (logConsoleRef.current) {
      logConsoleRef.current.scrollTop = logConsoleRef.current.scrollHeight;
    }
  }, [telemetry?.logs]);

  const handleStart = async () => {
    try {
      const res = await fetch("/api/run", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          dataset: selectedDataset,
          model: selectedModel,
          detector: selectedDetector,
          warmup: warmup,
          window: windowSize,
        }),
      });
      if (!res.ok) {
        const err = await res.json();
        alert("Execution error: " + (err.message || "Failed to start"));
      } else {
        fetchTelemetry();
      }
    } catch (e) {
      alert("Network error: " + e);
    }
  };

  const handleStop = async () => {
    try {
      await fetch("/api/stop", { method: "POST" });
      fetchTelemetry();
    } catch {}
  };

  const handleUploadHf = async () => {
    if (!hfInput.trim()) return alert("Enter a Hugging Face dataset identifier");
    setUploadStatus("Importing from Hugging Face Hub...");
    try {
      const res = await fetch("/api/upload_dataset", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ hf_id: hfInput.trim() }),
      });
      const data = await res.json();
      if (res.ok) {
        setUploadStatus(`Success! Imported ${data.rows.toLocaleString()} instances. Selected and ready.`);
        await fetchDatasets();
        setSelectedDataset("custom");
      } else {
        setUploadStatus("Error: " + data.message);
      }
    } catch (e) {
      setUploadStatus("Error: " + e);
    }
  };

  const handleUploadCsv = async () => {
    const file = fileInputRef.current?.files?.[0];
    if (!file) return alert("Select a CSV file first");

    setUploadStatus("Reading CSV file...");
    const reader = new FileReader();
    reader.onload = async (e) => {
      const content = e.target?.result as string;
      try {
        const res = await fetch("/api/upload_dataset", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ csv_content: content }),
        });
        const data = await res.json();
        if (res.ok) {
          setUploadStatus(`Success! Loaded ${data.rows.toLocaleString()} instances (${data.columns.length} columns). Proceed to run.`);
          await fetchDatasets();
          setSelectedDataset("custom");
        } else {
          setUploadStatus("Error: " + data.message);
        }
      } catch (err) {
        setUploadStatus("Error: " + err);
      }
    };
    reader.readAsText(file);
  };

  const triggerExport = async (format: "json" | "summary_csv" | "trajectory_csv" | "modifications_csv" | "logs") => {
    try {
      const res = await fetch(`/api/export?format=${format}`);
      if (res.ok) {
        const blob = await res.blob();
        const disposition = res.headers.get("Content-Disposition");
        let filename = `export_${format}_${Date.now()}`;
        if (disposition && disposition.includes("filename=")) {
          filename = disposition.split("filename=")[1].replace(/"/g, "").trim();
        } else {
          if (format === "json") filename += ".json";
          else if (format.includes("csv")) filename += ".csv";
          else if (format === "logs") filename += ".txt";
        }
        const url = URL.createObjectURL(blob);
        const a = document.createElement("a");
        a.href = url;
        a.download = filename;
        a.click();
        URL.revokeObjectURL(url);
        setExportNotice(`DOWNLOADED: ${filename}`);
        setTimeout(() => setExportNotice(""), 4000);
        return;
      }
    } catch (err) {
      console.warn("API export failed, using client-side generator:", err);
    }

    // Client-side fallback if API is not yet reloaded or offline
    if (!telemetry) return;
    let content = "";
    let mimeType = "text/plain";
    let filename = `export_${Date.now()}`;
    const dName = telemetry.dataset_name || "stream";
    const mName = telemetry.model_name || "model";

    if (format === "json") {
      content = JSON.stringify(telemetry, null, 2);
      mimeType = "application/json";
      filename = `full_telemetry_${dName}_${mName}_${Date.now()}.json`;
    } else if (format === "summary_csv") {
      content = `timestamp,dataset,model,detector,device,current_sample,total_samples,cumulative_accuracy_pct,window_accuracy_pct,f1_score_pct,drift_signals,adaptations_count,throughput\n${new Date().toISOString()},${dName},${mName},${telemetry.detector_name},${telemetry.device},${telemetry.current_sample},${telemetry.total_samples},${telemetry.accuracy},${telemetry.window_accuracy},${telemetry.f1_score},${telemetry.drift_count},${telemetry.adaptation_count},${telemetry.throughput}\n`;
      mimeType = "text/csv";
      filename = `summary_${dName}_${mName}_${Date.now()}.csv`;
    } else if (format === "trajectory_csv") {
      content = "sample,cumulative_accuracy_pct,window_accuracy_pct,f1_score_pct,drift_signal\n" +
        (telemetry.trajectory || []).map(p => `${p.sample},${p.cumulative_acc},${p.window_acc},${p.f1},${p.drift}`).join("\n");
      mimeType = "text/csv";
      filename = `trajectory_${dName}_${mName}_${Date.now()}.csv`;
    } else if (format === "modifications_csv") {
      content = "sample_index,event,components_affected,adaptation_latency_ms,device\n" +
        (telemetry.component_modifications || []).map(m => `${m.sample_index},"${m.event}",${m.components_affected},${m.adaptation_latency_ms},"${m.device}"`).join("\n");
      mimeType = "text/csv";
      filename = `audit_${dName}_${mName}_${Date.now()}.csv`;
    } else if (format === "logs") {
      content = (telemetry.logs || []).map(l => `[${l.timestamp}] [${l.level}] ${l.message}`).join("\n");
      mimeType = "text/plain";
      filename = `logs_${dName}_${mName}_${Date.now()}.txt`;
    }

    const blob = new Blob([content], { type: mimeType });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = filename;
    a.click();
    URL.revokeObjectURL(url);
    setExportNotice(`DOWNLOADED: ${filename}`);
    setTimeout(() => setExportNotice(""), 4000);
  };

  // Trajectory SVG calculations
  const trajectory = telemetry?.trajectory || [];
  const totalSamples = telemetry?.total_samples || 1;
  const maxSample = Math.max(totalSamples, trajectory[trajectory.length - 1]?.sample || 1);
  const svgWidth = 800;
  const svgHeight = 160;
  const padX = 45;
  const padTop = 15;
  const padBottom = 20;
  const plotW = Math.max(10, svgWidth - padX - 15);
  const plotH = svgHeight - padTop - padBottom;

  const getX = (s: number) => padX + (s / maxSample) * plotW;
  const getY = (acc: number) => padTop + plotH - (acc / 100) * plotH;

  let dCum = "";
  let dWin = "";
  const driftLines: number[] = [];

  trajectory.forEach((pt, i) => {
    const x = getX(pt.sample);
    const yC = getY(pt.cumulative_acc);
    const yW = getY(pt.window_acc);

    if (i === 0) {
      dCum += `M ${x} ${yC}`;
      dWin += `M ${x} ${yW}`;
    } else {
      dCum += ` L ${x} ${yC}`;
      dWin += ` L ${x} ${yW}`;
    }

    if (pt.drift === 1) {
      driftLines.push(x);
    }
  });

  const stages = [
    { id: "IDLE", label: "1. STANDBY" },
    { id: "DATA_LOAD", label: "2. DATA INGESTION" },
    { id: "WARMUP_TRAINING", label: "3. WARMUP TRAINING" },
    { id: "STREAMING_PREQUENTIAL", label: "4. STREAM SCORING" },
    { id: "DRIFT_ADAPTING", label: "5. DRIFT & ADAPTATION" },
    { id: "COMPLETED", label: "6. COMPLETED" },
  ];

  const currentStage = telemetry?.stage || "IDLE";

  return (
    <div className="max-w-[1500px] mx-auto flex flex-col gap-3 text-[12px] bg-white text-black p-3">
      {/* Top Hardware Telemetry */}
      <div className="border border-black p-2.5 bg-white">
        <div className="font-bold uppercase tracking-wide border-b border-black pb-1 mb-2 flex justify-between items-center text-[11px]">
          <span>HARDWARE & COMPUTE ENGINE TELEMETRY</span>
          <span className="font-bold">ACTIVE BINDING: {telemetry?.device || "STANDBY"}</span>
        </div>
        <div className="grid grid-cols-4 gap-2.5">
          <div className="border border-black p-2">
            <div className="text-[10px] uppercase text-neutral-600">CPU ARCHITECTURE</div>
            <div className="text-[13px] font-bold mt-0.5">{hardware?.cpu?.model || "AMD Ryzen 9 9955HX"}</div>
            <div className="text-[10px] text-neutral-600">{hardware?.cpu?.threads || 32} Execution Threads</div>
          </div>
          <div className="border border-black p-2">
            <div className="text-[10px] uppercase text-neutral-600">SYSTEM MEMORY (RAM)</div>
            <div className="text-[13px] font-bold mt-0.5">
              {hardware ? `${(hardware.cpu.ram_used_mb / 1024).toFixed(1)} / ${(hardware.cpu.ram_total_mb / 1024).toFixed(1)} GB` : "Loading..."}
            </div>
            <div className="text-[10px] text-neutral-600">Util: {hardware?.cpu?.ram_util_pct || 0}%</div>
          </div>
          <div className="border border-black p-2">
            <div className="text-[10px] uppercase text-neutral-600">GPU ACCELERATOR</div>
            <div className="text-[13px] font-bold mt-0.5">{hardware?.gpu?.model || "NVIDIA GeForce RTX 5070"}</div>
            <div className="text-[10px] text-neutral-600">
              CUDA: {hardware?.gpu?.cuda_available ? "READY (DEVICE 0)" : "INITIALIZING"}
            </div>
          </div>
          <div className="border border-black p-2">
            <div className="text-[10px] uppercase text-neutral-600">GPU VRAM & LOAD</div>
            <div className="text-[13px] font-bold mt-0.5">
              {hardware ? `${hardware.gpu.vram_used_mb} / ${hardware.gpu.vram_total_mb} MiB` : "0 / 8151 MiB"}
            </div>
            <div className="text-[10px] text-neutral-600">Core Load: {hardware?.gpu?.gpu_util_pct || 0}%</div>
          </div>
        </div>
      </div>

      {/* Pipeline Stage Bar */}
      <div className="flex border border-black text-[10px] uppercase">
        {stages.map((st) => {
          const isActive = currentStage === st.id;
          return (
            <div
              key={st.id}
              className={`flex-1 p-1.5 text-center border-r last:border-r-0 border-black ${
                isActive ? "bg-black text-white font-bold" : "bg-white text-black"
              }`}
            >
              {st.label}
            </div>
          );
        })}
      </div>

      {/* Main Grid: Config & Primary Metrics */}
      <div className="grid grid-cols-2 gap-3">
        {/* Configuration */}
        <div className="border border-black p-2.5 bg-white">
          <div className="font-bold uppercase tracking-wide border-b border-black pb-1 mb-2 text-[11px]">
            STREAM & MODEL CONFIGURATION
          </div>
          <div className="flex flex-col gap-2">
            <div>
              <label className="block font-bold uppercase text-[10px] mb-1">Streaming Dataset</label>
              <select
                className="w-full border border-black p-1.5 bg-white text-[12px] outline-none"
                value={selectedDataset}
                onChange={(e) => setSelectedDataset(e.target.value)}
              >
                {datasets.map((d) => (
                  <option key={d.id} value={d.id}>
                    {d.name} [{d.type}]
                  </option>
                ))}
              </select>
            </div>

            <div>
              <button
                type="button"
                className="w-full border border-black p-1.5 font-bold uppercase hover:bg-black hover:text-white"
                onClick={() => setShowAddDataset(!showAddDataset)}
              >
                {showAddDataset ? "− HIDE DATASET IMPORT" : "+ ADD DATASET (CSV / HUGGING FACE)"}
              </button>
            </div>

            {/* Add Dataset Drawer */}
            {showAddDataset && (
              <div className="border border-dashed border-black p-2.5 my-1 flex flex-col gap-2">
                <div className="font-bold text-[10px] uppercase">OPTION A: IMPORT FROM HUGGING FACE HUB</div>
                <div className="flex gap-1.5">
                  <input
                    type="text"
                    className="flex-1 border border-black p-1.5 bg-white text-[11px] outline-none"
                    placeholder="e.g. scikit-learn/adult-census-income"
                    value={hfInput}
                    onChange={(e) => setHfInput(e.target.value)}
                  />
                  <button
                    type="button"
                    className="border border-black px-3 font-bold uppercase hover:bg-black hover:text-white"
                    onClick={handleUploadHf}
                  >
                    FETCH & PROCEED
                  </button>
                </div>

                <div className="font-bold text-[10px] uppercase mt-1">OPTION B: UPLOAD LOCAL CSV</div>
                <input
                  type="file"
                  ref={fileInputRef}
                  accept=".csv"
                  className="w-full border border-black p-1 bg-white text-[11px]"
                />
                <button
                  type="button"
                  className="w-full border border-black p-1.5 font-bold uppercase hover:bg-black hover:text-white"
                  onClick={handleUploadCsv}
                >
                  LOAD CSV & PROCEED
                </button>

                {uploadStatus && (
                  <div className="font-bold text-[11px] mt-1 border-t border-black pt-1">{uploadStatus}</div>
                )}
              </div>
            )}

            <div className="grid grid-cols-2 gap-2">
              <div>
                <label className="block font-bold uppercase text-[10px] mb-1">Model Architecture</label>
                <select
                  className="w-full border border-black p-1.5 bg-white text-[12px] outline-none"
                  value={selectedModel}
                  onChange={(e) => setSelectedModel(e.target.value)}
                >
                  <option value="proposed">Proposed Selective Adapt (RF + SVM + Online Scaler)</option>
                  <option value="boosted_forest">Boosted ARF-DWM Forest (Shadow Trees)</option>
                  <option value="dual_memory">Self-Healing Dual-Memory Meta-Ensemble</option>
                  <option value="xgboost_cuda">Hardware-Accelerated Adaptive XGBoost (RTX 5070 CUDA)</option>
                  <option value="full_retrain">Full Retraining (Baseline)</option>
                  <option value="selective">Selective Adaptive Ensemble (Standard)</option>
                  <option value="static">Static Streaming Model (No Adapt)</option>
                </select>
              </div>
              <div>
                <label className="block font-bold uppercase text-[10px] mb-1">Drift Detector</label>
                <select
                  className="w-full border border-black p-1.5 bg-white text-[12px] outline-none"
                  value={selectedDetector}
                  onChange={(e) => setSelectedDetector(e.target.value)}
                >
                  <option value="adwin">ADWIN (Adaptive Sliding Window)</option>
                  <option value="ddm">DDM (Drift Detection Method)</option>
                  <option value="eddm">EDDM (Early Drift Detection Method)</option>
                  <option value="page_hinkley">Page-Hinkley Cumulative Sum</option>
                </select>
              </div>
            </div>

            <div className="grid grid-cols-2 gap-2">
              <div>
                <label className="block font-bold uppercase text-[10px] mb-1">Warmup Training Samples</label>
                <input
                  type="number"
                  className="w-full border border-black p-1.5 bg-white text-[12px] outline-none"
                  value={warmup}
                  onChange={(e) => setWarmup(parseInt(e.target.value) || 200)}
                />
              </div>
              <div>
                <label className="block font-bold uppercase text-[10px] mb-1">Rolling Metrics Window</label>
                <input
                  type="number"
                  className="w-full border border-black p-1.5 bg-white text-[12px] outline-none"
                  value={windowSize}
                  onChange={(e) => setWindowSize(parseInt(e.target.value) || 100)}
                />
              </div>
            </div>

            <div className="flex gap-2 mt-2">
              <button
                type="button"
                className="flex-1 bg-black text-white p-2 font-bold uppercase hover:bg-neutral-800"
                onClick={handleStart}
              >
                START STREAMING EXECUTION
              </button>
              <button
                type="button"
                className="w-28 border border-black p-2 font-bold uppercase hover:bg-black hover:text-white"
                onClick={handleStop}
              >
                HALT STREAM
              </button>
              <button
                type="button"
                className="w-28 border border-black p-2 font-bold uppercase hover:bg-black hover:text-white"
                onClick={() => triggerExport("json")}
              >
                EXPORT (JSON)
              </button>
            </div>
          </div>
        </div>

        {/* Live Metrics Panel */}
        <div className="border border-black p-2.5 bg-white flex flex-col justify-between">
          <div>
            <div className="font-bold uppercase tracking-wide border-b border-black pb-1 mb-2 flex justify-between items-center text-[11px]">
              <span>STREAMING PERFORMANCE & ADAPTATION METRICS</span>
              <div className="flex gap-2 items-center">
                <button
                  type="button"
                  onClick={() => triggerExport("summary_csv")}
                  className="border border-black px-1.5 py-0.5 text-[10px] uppercase font-bold hover:bg-black hover:text-white"
                >
                  EXPORT METRICS (CSV)
                </button>
                <span className="font-bold">STATUS: {telemetry?.stage || "IDLE"}</span>
              </div>
            </div>
            <div className="grid grid-cols-3 gap-2 mb-2">
              <div className="border border-black p-2">
                <div className="text-[10px] uppercase text-neutral-600 border-b border-neutral-300 pb-0.5 mb-1">
                  CUMULATIVE ACCURACY
                </div>
                <div className="text-[20px] font-bold">{(telemetry?.accuracy || 0).toFixed(2)}%</div>
                <div className="text-[10px] text-neutral-600">Overall Stream Accuracy</div>
              </div>
              <div className="border border-black p-2">
                <div className="text-[10px] uppercase text-neutral-600 border-b border-neutral-300 pb-0.5 mb-1">
                  WINDOWED ACCURACY
                </div>
                <div className="text-[20px] font-bold">{(telemetry?.window_accuracy || 0).toFixed(2)}%</div>
                <div className="text-[10px] text-neutral-600">Last Rolling Window</div>
              </div>
              <div className="border border-black p-2">
                <div className="text-[10px] uppercase text-neutral-600 border-b border-neutral-300 pb-0.5 mb-1">
                  F1-SCORE
                </div>
                <div className="text-[20px] font-bold">{(telemetry?.f1_score || 0).toFixed(2)}%</div>
                <div className="text-[10px] text-neutral-600">Precision / Recall Balance</div>
              </div>
            </div>

            <div className="grid grid-cols-4 gap-2">
              <div className="border border-black p-2">
                <div className="text-[10px] uppercase text-neutral-600 border-b border-neutral-300 pb-0.5 mb-1">
                  PROCESSED SAMPLES
                </div>
                <div className="text-[14px] font-bold">
                  {(telemetry?.current_sample || 0).toLocaleString()} / {(telemetry?.total_samples || 0).toLocaleString()}
                </div>
                <div className="text-[10px] text-neutral-600">
                  {telemetry?.total_samples
                    ? `${((telemetry.current_sample / telemetry.total_samples) * 100).toFixed(1)}% Done`
                    : "0.0%"}
                </div>
              </div>
              <div className="border border-black p-2">
                <div className="text-[10px] uppercase text-neutral-600 border-b border-neutral-300 pb-0.5 mb-1">
                  DRIFT SIGNALS
                </div>
                <div className="text-[16px] font-bold">{telemetry?.drift_count || 0}</div>
                <div className="text-[10px] text-neutral-600">Detected Shifts</div>
              </div>
              <div className="border border-black p-2">
                <div className="text-[10px] uppercase text-neutral-600 border-b border-neutral-300 pb-0.5 mb-1">
                  ADAPTATIONS
                </div>
                <div className="text-[16px] font-bold">{telemetry?.adaptation_count || 0}</div>
                <div className="text-[10px] text-neutral-600">{telemetry?.components_modified || 0} Mod. Trees</div>
              </div>
              <div className="border border-black p-2">
                <div className="text-[10px] uppercase text-neutral-600 border-b border-neutral-300 pb-0.5 mb-1">
                  THROUGHPUT
                </div>
                <div className="text-[16px] font-bold">{telemetry?.throughput || 0} smp/s</div>
                <div className="text-[10px] text-neutral-600">
                  Latency: {(telemetry?.adaptation_time_ms || 0).toFixed(1)}ms
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* Output & Telemetry Export Center */}
      <div className="border border-black p-2.5 bg-white">
        <div className="font-bold uppercase tracking-wide border-b border-black pb-1 mb-2 flex justify-between items-center text-[11px]">
          <span>OUTPUT & TELEMETRY EXPORT CENTER</span>
          {exportNotice ? (
            <span className="bg-black text-white px-2 py-0.5 text-[10px] font-bold tracking-wider">
              {exportNotice}
            </span>
          ) : (
            <span className="text-[10px] text-neutral-500">FORMATS: CSV / JSON / TXT</span>
          )}
        </div>
        <div className="grid grid-cols-5 gap-2">
          <button
            type="button"
            onClick={() => triggerExport("summary_csv")}
            className="border border-black p-2 text-center hover:bg-black hover:text-white uppercase font-bold text-[11px] flex flex-col items-center justify-center gap-0.5"
          >
            <span>EXPORT SUMMARY (CSV)</span>
            <span className="text-[9px] font-normal text-neutral-500 hover:text-neutral-300">Accuracy, F1, Latency & Stats</span>
          </button>
          <button
            type="button"
            onClick={() => triggerExport("trajectory_csv")}
            className="border border-black p-2 text-center hover:bg-black hover:text-white uppercase font-bold text-[11px] flex flex-col items-center justify-center gap-0.5"
          >
            <span>EXPORT TRAJECTORY (CSV)</span>
            <span className="text-[9px] font-normal text-neutral-500 hover:text-neutral-300">Sample-by-Sample Time Series</span>
          </button>
          <button
            type="button"
            onClick={() => triggerExport("modifications_csv")}
            className="border border-black p-2 text-center hover:bg-black hover:text-white uppercase font-bold text-[11px] flex flex-col items-center justify-center gap-0.5"
          >
            <span>EXPORT AUDIT LOG (CSV)</span>
            <span className="text-[9px] font-normal text-neutral-500 hover:text-neutral-300">Component Adaptation Events</span>
          </button>
          <button
            type="button"
            onClick={() => triggerExport("logs")}
            className="border border-black p-2 text-center hover:bg-black hover:text-white uppercase font-bold text-[11px] flex flex-col items-center justify-center gap-0.5"
          >
            <span>EXPORT LOGS (TXT)</span>
            <span className="text-[9px] font-normal text-neutral-500 hover:text-neutral-300">Raw Console Execution Trace</span>
          </button>
          <button
            type="button"
            onClick={() => triggerExport("json")}
            className="border border-black p-2 text-center hover:bg-black hover:text-white uppercase font-bold text-[11px] flex flex-col items-center justify-center gap-0.5"
          >
            <span>EXPORT FULL RUN (JSON)</span>
            <span className="text-[9px] font-normal text-neutral-500 hover:text-neutral-300">Complete Telemetry & Hardware</span>
          </button>
        </div>
      </div>

      {/* Live Pure SVG Stream Trajectory Chart */}
      <div className="border border-black p-2.5 bg-white">
        <div className="font-bold uppercase tracking-wide border-b border-black pb-1 mb-2 flex justify-between items-center text-[11px]">
          <span>LIVE STREAM ACCURACY & DRIFT TRAJECTORY (SVG)</span>
          <div className="flex gap-2 items-center">
            <span className="text-[10px]">— SOLID: CUMULATIVE | ··· DOTTED: WINDOWED | ╎ DASH: DRIFT SIGNAL</span>
            <button
              type="button"
              className="border border-black px-2 py-0.5 uppercase text-[10px] font-bold hover:bg-black hover:text-white"
              onClick={() => triggerExport("trajectory_csv")}
            >
              EXPORT TRAJECTORY (CSV)
            </button>
            <button
              type="button"
              className="border border-black px-2 py-0.5 uppercase text-[10px] font-bold hover:bg-black hover:text-white"
              onClick={() => triggerExport("json")}
            >
              EXPORT (JSON)
            </button>
          </div>
        </div>
        <div className="w-full h-40 border border-black bg-white relative">
          <svg width="100%" height="160" viewBox={`0 0 ${svgWidth} ${svgHeight}`} preserveAspectRatio="none">
            {/* Grid Ticks */}
            <line x1={padX} y1={20} x2={svgWidth} y2={20} stroke="#eeeeee" strokeWidth="1" />
            <text x="5" y="24" fontSize="9" fill="#888888">
              100%
            </text>
            <line x1={padX} y1={55} x2={svgWidth} y2={55} stroke="#eeeeee" strokeWidth="1" />
            <text x="5" y="59" fontSize="9" fill="#888888">
              75%
            </text>
            <line x1={padX} y1={90} x2={svgWidth} y2={90} stroke="#eeeeee" strokeWidth="1" />
            <text x="5" y="94" fontSize="9" fill="#888888">
              50%
            </text>
            <line x1={padX} y1={125} x2={svgWidth} y2={125} stroke="#eeeeee" strokeWidth="1" />
            <text x="5" y="129" fontSize="9" fill="#888888">
              25%
            </text>
            <line x1={padX} y1={145} x2={svgWidth} y2={145} stroke="#000000" strokeWidth="1" />

            {/* Drift Vertical Markers */}
            {driftLines.map((dx, i) => (
              <line
                key={i}
                x1={dx}
                y1={padTop}
                x2={dx}
                y2={svgHeight - padBottom}
                stroke="#000000"
                strokeWidth="1"
                strokeDasharray="2,2"
              />
            ))}

            {/* Cumulative Accuracy Path */}
            {dCum && <path d={dCum} fill="none" stroke="#000000" strokeWidth="2" />}
            {/* Windowed Accuracy Path */}
            {dWin && <path d={dWin} fill="none" stroke="#000000" strokeWidth="1" strokeDasharray="3,3" />}
          </svg>
        </div>
      </div>

      {/* Modifications & Feature Selection Tables */}
      <div className="grid grid-cols-2 gap-3">
        {/* Component Modifications Table */}
        <div className="border border-black p-2.5 bg-white">
          <div className="font-bold uppercase tracking-wide border-b border-black pb-1 mb-2 flex justify-between items-center text-[11px]">
            <span>MODEL COMPONENT MODIFICATIONS & REPLACEMENTS LOG</span>
            <button
              type="button"
              className="border border-black px-1.5 py-0.5 text-[10px] uppercase font-bold hover:bg-black hover:text-white"
              onClick={() => triggerExport("modifications_csv")}
            >
              EXPORT AUDIT (CSV)
            </button>
          </div>
          <div className="max-h-44 overflow-y-auto border border-black">
            <table className="w-full text-left border-collapse text-[11px]">
              <thead>
                <tr className="bg-neutral-100 border-b border-black">
                  <th className="p-1 border-r border-black">Sample #</th>
                  <th className="p-1 border-r border-black">Event Trigger</th>
                  <th className="p-1 border-r border-black">Components Modified</th>
                  <th className="p-1 border-r border-black">Latency</th>
                  <th className="p-1">Device Executed</th>
                </tr>
              </thead>
              <tbody>
                {telemetry?.component_modifications && telemetry.component_modifications.length > 0 ? (
                  telemetry.component_modifications
                    .slice(-15)
                    .reverse()
                    .map((m, idx) => (
                      <tr key={idx} className="border-b border-black last:border-b-0">
                        <td className="p-1 border-r border-black font-bold">#{m.sample_index}</td>
                        <td className="p-1 border-r border-black">{m.event}</td>
                        <td className="p-1 border-r border-black">{m.components_affected} sub-components</td>
                        <td className="p-1 border-r border-black">{m.adaptation_latency_ms}ms</td>
                        <td className="p-1">{m.device}</td>
                      </tr>
                    ))
                ) : (
                  <tr>
                    <td colSpan={5} className="p-2 text-center text-neutral-500">
                      No adaptation events recorded yet.
                    </td>
                  </tr>
                )}
              </tbody>
            </table>
          </div>
        </div>

        {/* Feature Rankings Table */}
        <div className="border border-black p-2.5 bg-white">
          <div className="font-bold uppercase tracking-wide border-b border-black pb-1 mb-2 text-[11px]">
            ONLINE FEATURE SELECTION & RELEVANCE TRACKER
          </div>
          <div className="max-h-44 overflow-y-auto border border-black">
            <table className="w-full text-left border-collapse text-[11px]">
              <thead>
                <tr className="bg-neutral-100 border-b border-black">
                  <th className="p-1 border-r border-black">Rank</th>
                  <th className="p-1 border-r border-black">Feature Identifier</th>
                  <th className="p-1 border-r border-black">Index</th>
                  <th className="p-1">Active In Ensemble</th>
                </tr>
              </thead>
              <tbody>
                {telemetry?.feature_rankings && telemetry.feature_rankings.length > 0 ? (
                  telemetry.feature_rankings.map((f, idx) => (
                    <tr key={idx} className="border-b border-black last:border-b-0">
                      <td className="p-1 border-r border-black font-bold">{f.rank}</td>
                      <td className="p-1 border-r border-black">{f.feature_name}</td>
                      <td className="p-1 border-r border-black">{f.feature_index}</td>
                      <td className="p-1 font-bold">YES (Retained)</td>
                    </tr>
                  ))
                ) : (
                  <tr>
                    <td colSpan={4} className="p-2 text-center text-neutral-500">
                      Feature tracking initializes during warmup.
                    </td>
                  </tr>
                )}
              </tbody>
            </table>
          </div>
        </div>
      </div>

      {/* Real-time Process Console */}
      <div className="border border-black p-2.5 bg-white">
        <div className="font-bold uppercase tracking-wide border-b border-black pb-1 mb-2 flex justify-between items-center text-[11px]">
          <span>REAL-TIME PROCESS EXECUTION CONSOLE</span>
          <div className="flex gap-2 items-center">
            <button
              type="button"
              className="border border-black px-1.5 py-0.5 text-[10px] uppercase font-bold hover:bg-black hover:text-white"
              onClick={() => triggerExport("logs")}
            >
              EXPORT LOGS (TXT)
            </button>
            <span className="text-[10px] text-neutral-500">LOGSTREAM</span>
          </div>
        </div>
        <div
          ref={logConsoleRef}
          className="h-44 overflow-y-auto border border-black p-2 text-[11px] font-mono bg-white flex flex-col gap-0.5"
        >
          {telemetry?.logs && telemetry.logs.length > 0 ? (
            telemetry.logs.map((l, idx) => {
              let cls = "text-black";
              if (l.level === "WARNING") cls = "font-bold underline";
              if (l.level === "ERROR") cls = "bg-black text-white px-1 font-bold";
              return (
                <div key={idx} className={cls}>
                  [{l.timestamp}] [{l.level}] {l.message}
                </div>
              );
            })
          ) : (
            <div className="text-neutral-400">Awaiting stream execution start...</div>
          )}
        </div>
      </div>
    </div>
  );
}

"""Web Application Server for Concept Drift Adaptive Streaming Framework.

Zero-dependency Python standard library HTTP server providing:
- Black & white, high-density telemetry interface
- Real-time hardware monitoring (AMD Ryzen 9 9955HX 32 threads, NVIDIA GeForce RTX 5070 CUDA)
- Custom dataset upload and Hugging Face dataset import
- Granular streaming logs, metrics, component modifications, and feature rankings
"""

import os
import sys

# Ensure root directory is always on sys.path
BASE_DIR = os.path.abspath(os.path.dirname(__file__))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

import json
import time
import threading
import subprocess
import csv
import io
import re
from urllib.parse import urlparse, parse_qs
from http.server import ThreadingHTTPServer, BaseHTTPRequestHandler
from typing import Dict, Any, Optional, List, Tuple
import numpy as np
import pandas as pd

# Project modules
from src.datasets.synthetic import generate_sea_stream, generate_agrawal_stream
from src.datasets.loader import load_electricity_stream, load_covertype_stream, load_airlines_stream
from src.datasets.hf_loader import load_hf_adult_income_stream, load_hf_bank_marketing_stream, load_hf_credit_default_stream
from src.detectors.adwin import ADWINDetector
from src.detectors.ddm import DDMDetector
from src.detectors.eddm import EDDMDetector
from src.detectors.page_hinkley import PageHinkleyDetector
from src.models.static_model import StaticStreamingModel
from src.models.full_retrain import FullRetrainingModel
from src.models.selective_adaptive import SelectiveAdaptiveEnsemble
from src.models.framework_pipeline import SelfHealingConceptDriftFramework
from src.models.boosted_adaptive_forest import BoostedAdaptiveForest
from src.models.self_healing_meta_ensemble import SelfHealingMetaEnsemble
from src.models.xgboost_adaptive import GPUAcceleratedAdaptiveXGBoost
from src.models.adaptive_svm import AdaptiveOnlineSVM

MAX_CONTENT_LENGTH = 50 * 1024 * 1024  # 50MB
RUN_LOCK = threading.Lock()

VALID_DATASETS = {
    "sea", "agrawal", "electricity", "covertype", "airlines",
    "hf_adult", "hf_bank", "hf_credit", "custom",
}

VALID_MODELS = {
    "proposed", "boosted_forest", "dual_memory", "xgboost_cuda",
    "full_retrain", "adaptive_svm", "selective", "static",
}


class TelemetryEngine:
    """Thread-safe engine capturing granular streaming stats, logs, modifications, and device activity."""

    def __init__(self):
        self.lock = threading.Lock()
        self.reset()

    def reset(self):
        with self.lock:
            self.running = False
            self.should_stop = False
            self.dataset_name = ""
            self.model_name = ""
            self.detector_name = ""
            self.stage = "IDLE"
            self.device = "CPU (32 Threads)"
            self.current_sample = 0
            self.total_samples = 0
            self.accuracy = 0.0
            self.window_accuracy = 0.0
            self.f1_score = 0.0
            self.drift_count = 0
            self.adaptation_count = 0
            self.components_modified = 0
            self.adaptation_time_ms = 0.0
            self.throughput = 0.0
            self.start_time = 0.0
            self.logs: List[Dict[str, Any]] = []
            self.trajectory: List[Dict[str, Any]] = []
            self.feature_rankings: List[Dict[str, Any]] = []
            self.component_modifications: List[Dict[str, Any]] = []

    def log(self, message: str, level: str = "INFO", details: Optional[Dict[str, Any]] = None):
        t_str = time.strftime("%H:%M:%S") + f".{int((time.time() % 1) * 1000):03d}"
        entry = {
            "timestamp": t_str,
            "message": message,
            "level": level,
            "details": details or {},
        }
        with self.lock:
            self.logs.append(entry)
            if len(self.logs) > 10000:
                self.logs.pop(0)

    def get_snapshot(self, full: bool = False) -> Dict[str, Any]:
        with self.lock:
            return {
                "running": self.running,
                "dataset_name": self.dataset_name,
                "model_name": self.model_name,
                "detector_name": self.detector_name,
                "stage": self.stage,
                "device": self.device,
                "current_sample": self.current_sample,
                "total_samples": self.total_samples,
                "accuracy": round(self.accuracy * 100, 2),
                "window_accuracy": round(self.window_accuracy * 100, 2),
                "f1_score": round(self.f1_score * 100, 2),
                "drift_count": self.drift_count,
                "adaptation_count": self.adaptation_count,
                "components_modified": self.components_modified,
                "adaptation_time_ms": round(self.adaptation_time_ms, 2),
                "throughput": round(self.throughput, 1),
                "logs": list(self.logs) if full else list(self.logs[-100:]),
                "trajectory": list(self.trajectory) if full else list(self.trajectory[-60:]),
                "feature_rankings": list(self.feature_rankings),
                "component_modifications": list(self.component_modifications) if full else list(self.component_modifications[-30:]),
            }


TELEMETRY = TelemetryEngine()
STREAM_THREAD: Optional[threading.Thread] = None


CACHED_HARDWARE: Optional[Dict[str, Any]] = None
LAST_HW_CHECK = 0.0


def get_hardware_info() -> Dict[str, Any]:
    """Query genuine CPU, RAM, and NVIDIA GPU stats with caching, reporting 'Unavailable' on missing metrics."""
    global CACHED_HARDWARE, LAST_HW_CHECK
    now = time.time()
    if CACHED_HARDWARE is not None and (now - LAST_HW_CHECK) < 4.0:
        return CACHED_HARDWARE

    cpu_count = os.cpu_count() or 1
    cpu_model = "Generic CPU"
    try:
        with open("/proc/cpuinfo", "r") as f:
            for line in f:
                if "model name" in line:
                    cpu_model = line.split(":", 1)[1].strip()
                    break
    except Exception:
        import platform
        cpu_model = platform.processor() or "CPU"

    mem_total_mb = 0
    mem_avail_mb = 0
    try:
        with open("/proc/meminfo", "r") as f:
            for line in f:
                if line.startswith("MemTotal:"):
                    mem_total_mb = int(line.split()[1]) // 1024
                elif line.startswith("MemAvailable:"):
                    mem_avail_mb = int(line.split()[1]) // 1024
    except Exception:
        pass

    mem_used_mb = max(0, mem_total_mb - mem_avail_mb) if mem_total_mb > 0 else 0

    gpu_name = "Unavailable"
    gpu_total_mb = 0
    gpu_used_mb = 0
    gpu_util = 0
    cuda_available = False

    try:
        res = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total,memory.used,utilization.gpu", "--format=csv,noheader,nounits"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=5,
        )
        if res.returncode == 0 and res.stdout.strip():
            parts = [p.strip() for p in res.stdout.strip().split(",")]
            if len(parts) >= 4:
                gpu_name = parts[0]
                gpu_total_mb = int(parts[1])
                gpu_used_mb = int(parts[2])
                gpu_util = int(parts[3])
                cuda_available = True
    except Exception:
        pass

    CACHED_HARDWARE = {
        "cpu": {
            "model": f"{cpu_model} ({cpu_count} Threads)",
            "threads": cpu_count,
            "ram_total_mb": mem_total_mb,
            "ram_used_mb": mem_used_mb,
            "ram_util_pct": round((mem_used_mb / max(1, mem_total_mb)) * 100, 1) if mem_total_mb > 0 else 0.0,
        },
        "gpu": {
            "model": gpu_name,
            "cuda_available": cuda_available,
            "vram_total_mb": gpu_total_mb,
            "vram_used_mb": gpu_used_mb,
            "vram_util_pct": round((gpu_used_mb / max(1, gpu_total_mb)) * 100, 1) if gpu_total_mb > 0 else 0.0,
            "gpu_util_pct": gpu_util,
            "device_ordinal": 0 if cuda_available else None,
        },
    }
    LAST_HW_CHECK = now
    return CACHED_HARDWARE



def load_dataset_by_key(key: str) -> Tuple[np.ndarray, np.ndarray, str]:
    """Resolve and load requested streaming dataset."""
    if key == "sea":
        X, y, _ = generate_sea_stream(n_samples=5000, noise_ratio=0.05, seed=42)
        return X, y, "SEA Concepts Benchmark (5k Stream)"
    elif key == "agrawal":
        X, y, _ = generate_agrawal_stream(n_samples=5000, noise_ratio=0.05, seed=42)
        return X, y, "Agrawal Benchmark (5k Stream)"
    elif key == "electricity":
        return load_electricity_stream(max_samples=10000)
    elif key == "covertype":
        return load_covertype_stream(max_samples=10000)
    elif key == "airlines":
        return load_airlines_stream(max_samples=2000)
    elif key == "hf_adult":
        return load_hf_adult_income_stream(max_samples=10000)
    elif key == "hf_bank":
        return load_hf_bank_marketing_stream(max_samples=10000)
    elif key == "hf_credit":
        return load_hf_credit_default_stream(max_samples=10000)
    elif key == "custom":
        p = "data/custom_stream.csv"
        if not os.path.exists(p):
            raise ValueError("No custom dataset uploaded yet.")
        df = pd.read_csv(p)
        target_col = df.columns[-1]
        feature_cols = [c for c in df.columns if c != target_col]
        X_df = df[feature_cols].copy()
        for c in X_df.columns:
            if X_df[c].dtype == object or str(X_df[c].dtype).startswith("category"):
                X_df[c] = pd.factorize(X_df[c])[0]
            X_df[c] = pd.to_numeric(X_df[c], errors="coerce").fillna(0.0)
        X = X_df.values.astype(np.float32)
        
        y_raw = df[target_col].values
        if df[target_col].dtype == object or str(df[target_col].dtype).startswith("category"):
            y = pd.factorize(y_raw)[0].astype(int)
        else:
            y_num = pd.to_numeric(df[target_col], errors="coerce").fillna(0).values
            y = (y_num > np.median(y_num)).astype(int) if len(np.unique(y_num)) > 2 else (y_num == y_num.max()).astype(int)
        return X, y, "Custom Added Dataset"
    else:
        raise ValueError(f"Unknown dataset key: {key}")


def build_detector(name: str):
    name = name.lower()
    if "eddm" in name:
        return EDDMDetector(warm_start=30)
    elif "ddm" in name:
        return DDMDetector(warm_start=30)
    elif "page" in name:
        return PageHinkleyDetector(threshold=25.0)
    return ADWINDetector(delta=0.005)


def build_model(model_key: str, detector_name: str):
    if model_key not in VALID_MODELS:
        raise ValueError(f"Unknown model key: {model_key}")

    det = build_detector(detector_name)
    if model_key == "proposed":
        return SelfHealingConceptDriftFramework(n_trees=25, drift_detector=det, name="Proposed Selective Adapt")
    elif model_key == "boosted_forest":
        d_type = "eddm" if "eddm" in detector_name.lower() else ("ddm" if "ddm" in detector_name.lower() else "adwin")
        return BoostedAdaptiveForest(n_estimators=25, detector_type=d_type, name="Boosted ARF-DWM Forest")
    elif model_key == "dual_memory":
        return SelfHealingMetaEnsemble(n_tree_components=20, name="Self-Healing Dual-Memory Meta-Ensemble")
    elif model_key == "xgboost_cuda":
        hw = get_hardware_info()
        dev = "cuda" if hw["gpu"]["cuda_available"] else "cpu"
        threads = hw["cpu"]["threads"]
        return GPUAcceleratedAdaptiveXGBoost(n_estimators=30, drift_detector=det, device=dev, n_jobs=threads, name=f"Adaptive XGBoost ({dev.upper()})")
    elif model_key == "full_retrain":
        return FullRetrainingModel(drift_detector=det, name="Full Retraining")
    elif model_key == "adaptive_svm":
        return AdaptiveOnlineSVM(drift_detector=det, name="Adaptive Online SVM (Incremental Margin)")
    elif model_key == "selective":
        return SelectiveAdaptiveEnsemble(n_estimators=25, replacement_ratio=0.3, drift_detector=det, name="Selective Adaptive Ensemble")
    elif model_key == "static":
        return StaticStreamingModel(name="Static Baseline")
    else:
        raise ValueError(f"Unknown model key: {model_key}")


def run_streaming_pipeline(dataset_key: str, model_key: str, detector_name: str, warmup: int = 500, window_size: int = 200):
    """Execute streaming pipeline with live step-by-step telemetry."""
    TELEMETRY.reset()
    TELEMETRY.running = True
    TELEMETRY.dataset_name = dataset_key
    TELEMETRY.model_name = model_key
    TELEMETRY.detector_name = detector_name
    TELEMETRY.start_time = time.time()

    TELEMETRY.log(f"Initializing streaming pipeline: dataset={dataset_key}, model={model_key}, detector={detector_name}")

    try:
        # Step 1: Load Data
        TELEMETRY.stage = "DATA_LOAD"
        TELEMETRY.log("Loading dataset instances...")
        X, y, full_dataset_name = load_dataset_by_key(dataset_key)
        TELEMETRY.total_samples = len(X)
        TELEMETRY.log(f"Loaded {len(X):,} samples with {X.shape[1]} features. Positive class ratio: {np.mean(y):.2%}")

        # Step 2: Build Model & Device Allocation
        TELEMETRY.stage = "MODEL_INIT"
        model = build_model(model_key, detector_name)
        hw = get_hardware_info()
        if ("cuda" in model_key or "xgboost" in model_key) and hw["gpu"]["cuda_available"]:
            TELEMETRY.device = f"CUDA:0 ({hw['gpu']['model']})"
            TELEMETRY.log(f"Bound execution to {hw['gpu']['model']} via CUDA runtime", details={"device": "cuda:0"})
        else:
            cpu_name = hw["cpu"]["model"]
            cpu_threads = hw["cpu"]["threads"]
            TELEMETRY.device = f"CPU ({cpu_threads} Threads - {cpu_name})"
            TELEMETRY.log(f"Bound execution to CPU threadpool ({cpu_threads} threads)", details={"threads": cpu_threads})

        # Step 3: Warmup Training (Training Data -> ML Model)
        TELEMETRY.stage = "WARMUP_TRAINING"
        warmup_samples = min(warmup, len(X) // 4)
        TELEMETRY.log(f"Phase 1: Warmup training on initial {warmup_samples} samples...")
        t0 = time.perf_counter()
        model.fit_initial(X[:warmup_samples], y[:warmup_samples])
        warmup_dur = time.perf_counter() - t0
        TELEMETRY.log(f"Warmup fit complete in {warmup_dur*1000:.1f}ms. Initial model established.")

        # Compute initial feature importance if available
        if hasattr(model, "selected_feature_indices") and model.selected_feature_indices is not None:
            rankings = []
            for rank, idx in enumerate(model.selected_feature_indices):
                rankings.append({"feature_index": int(idx), "feature_name": f"Feature_{idx}", "rank": rank + 1, "active": True})
            TELEMETRY.feature_rankings = rankings
            TELEMETRY.log(f"Online Feature Selection active: retained {len(rankings)} / {X.shape[1]} informative features.")

        # Step 4: Stream Scoring (Test-Then-Train)
        TELEMETRY.stage = "STREAMING_PREQUENTIAL"
        correct = 0
        total_streamed = 0
        rolling_correct: List[int] = []
        tp, fp, fn, tn = 0, 0, 0, 0
        drift_in_current_window = False
        stream_start_t = time.perf_counter()

        for idx in range(warmup_samples, len(X)):
            if TELEMETRY.should_stop:
                TELEMETRY.log("Streaming manually halted by operator.")
                break

            x_t = X[idx]
            y_t = int(y[idx])

            # A. Predict
            pred = model.predict_one(x_t)
            is_correct = 1 if pred == y_t else 0
            correct += is_correct
            total_streamed += 1

            rolling_correct.append(is_correct)
            if len(rolling_correct) > window_size:
                rolling_correct.pop(0)

            # Confusion matrix metrics
            if pred == 1 and y_t == 1:
                tp += 1
            elif pred == 1 and y_t == 0:
                fp += 1
            elif pred == 0 and y_t == 1:
                fn += 1
            else:
                tn += 1

            # B. Update model & monitor for concept drift
            t_adapt_start = time.perf_counter()
            drift_occurred = model.update_sample(x_t, y_t, sample_idx=idx)
            adapt_time = (time.perf_counter() - t_adapt_start) * 1000.0

            if drift_occurred:
                drift_in_current_window = True
                TELEMETRY.stage = "DRIFT_ADAPTING"
                TELEMETRY.drift_count += 1
                TELEMETRY.adaptation_count += 1
                TELEMETRY.adaptation_time_ms += adapt_time

                # Record modification details per event (actual event delta, not cumulative sum)
                n_mod = 1
                if hasattr(model, "last_components_updated") and model.last_components_updated > 0:
                    n_mod = model.last_components_updated
                elif hasattr(model, "replacement_ratio"):
                    n_mod = max(1, int(np.ceil(getattr(model, "n_estimators", 20) * getattr(model, "replacement_ratio", 0.3))))
                TELEMETRY.components_modified = n_mod

                mod_info = {
                    "sample_index": idx,
                    "event": "Drift Triggered -> Selective Adaptation",
                    "components_affected": n_mod,
                    "adaptation_latency_ms": round(adapt_time, 2),
                    "device": TELEMETRY.device,
                }
                TELEMETRY.component_modifications.append(mod_info)
                TELEMETRY.log(
                    f"Drift detected at sample #{idx}! Adapted model components ({adapt_time:.1f}ms)",
                    level="WARNING",
                    details=mod_info,
                )

                # Re-sync feature rankings if model reselected features
                if hasattr(model, "selected_feature_indices") and model.selected_feature_indices is not None:
                    rankings = []
                    for rank, feat_idx in enumerate(model.selected_feature_indices):
                        rankings.append({"feature_index": int(feat_idx), "feature_name": f"Feature_{feat_idx}", "rank": rank + 1, "active": True})
                    TELEMETRY.feature_rankings = rankings

            # Periodically update telemetry metrics
            if total_streamed % 20 == 0 or idx == len(X) - 1:
                elapsed = time.perf_counter() - stream_start_t
                acc = correct / total_streamed
                w_acc = np.mean(rolling_correct) if rolling_correct else acc
                
                # Weighted F1 aligned with offline evaluation
                n_pos = tp + fn
                n_neg = tn + fp
                n_tot = n_pos + n_neg
                prec_1 = tp / max(1, (tp + fp))
                rec_1 = tp / max(1, (tp + fn))
                f1_1 = (2 * prec_1 * rec_1) / max(1e-6, (prec_1 + rec_1))
                prec_0 = tn / max(1, (tn + fn))
                rec_0 = tn / max(1, (tn + fp))
                f1_0 = (2 * prec_0 * rec_0) / max(1e-6, (prec_0 + rec_0))
                f1_weighted = ((n_pos * f1_1) + (n_neg * f1_0)) / max(1, n_tot)

                TELEMETRY.current_sample = idx + 1
                TELEMETRY.accuracy = acc
                TELEMETRY.window_accuracy = float(w_acc)
                TELEMETRY.f1_score = f1_weighted
                TELEMETRY.throughput = total_streamed / max(1e-4, elapsed)

                # Append trajectory point with drift occurrence in window
                if total_streamed % 100 == 0 or idx == len(X) - 1:
                    TELEMETRY.trajectory.append({
                        "sample": idx + 1,
                        "cumulative_acc": round(acc * 100, 2),
                        "window_acc": round(float(w_acc) * 100, 2),
                        "f1": round(f1_weighted * 100, 2),
                        "drift": 1 if drift_in_current_window else 0,
                    })
                    drift_in_current_window = False

            # Slight yield for non-blocking UI responsiveness
            if total_streamed % 400 == 0:
                time.sleep(0.001)

        TELEMETRY.stage = "COMPLETED"
        TELEMETRY.log(f"Stream execution concluded. Final Accuracy: {TELEMETRY.accuracy*100:.2f}%, F1: {TELEMETRY.f1_score*100:.2f}%")

    except Exception as e:
        TELEMETRY.stage = "ERROR"
        TELEMETRY.log(f"Pipeline error: {str(e)}", level="ERROR")
    finally:
        TELEMETRY.running = False


class AppRequestHandler(BaseHTTPRequestHandler):
    """HTTP Request Handler."""

    def log_message(self, format, *args):
        # Silence default terminal request logs to keep shell clean
        return

    def _get_allowed_origin(self) -> str:
        origin = self.headers.get("Origin", "")
        return origin if origin in ("http://localhost:3000", "http://127.0.0.1:3000") else "http://localhost:3000"

    def _send_json(self, data: Any, status: int = 200):
        body = json.dumps(data).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Access-Control-Allow-Origin", self._get_allowed_origin())
        self.end_headers()
        self.wfile.write(body)

    def _send_file(self, body: bytes, content_type: str, filename: str, status: int = 200):
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Disposition", f'attachment; filename="{filename}"')
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Access-Control-Allow-Origin", self._get_allowed_origin())
        self.end_headers()
        self.wfile.write(body)

    def do_OPTIONS(self):
        self.send_response(200)
        self.send_header("Access-Control-Allow-Origin", self._get_allowed_origin())
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()

    def do_GET(self):
        parsed = urlparse(self.path)
        path = parsed.path

        if path == "/" or path == "/index.html":
            self._send_json({
                "status": "online",
                "service": "Atreus Streaming Concept Drift ML Backend",
                "frontend": "http://localhost:3000",
                "api_docs": ["/api/hardware", "/api/status", "/api/datasets", "/api/run", "/api/stop", "/api/export"]
            })
        elif path == "/api/hardware":
            self._send_json(get_hardware_info())
        elif path == "/api/status":
            query_params = parse_qs(parsed.query)
            full = query_params.get("full", ["false"])[0].lower() in ("true", "1")
            self._send_json(TELEMETRY.get_snapshot(full=full))
        elif path == "/api/export":
            query_params = parse_qs(parsed.query)
            fmt = query_params.get("format", ["json"])[0].lower()
            snap = TELEMETRY.get_snapshot(full=True)
            hw = get_hardware_info()
            d_name = snap["dataset_name"] or "stream"
            m_name = snap["model_name"] or "model"
            ts = int(time.time())

            if fmt == "json":
                export_data = {
                    "metadata": {
                        "exported_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                        "timestamp": ts,
                        "hardware": hw,
                    },
                    "telemetry": snap,
                }
                body = json.dumps(export_data, indent=2).encode("utf-8")
                filename = f"export_{d_name}_{m_name}_{ts}.json"
                self._send_file(body, "application/json", filename)
            elif fmt in ("csv", "summary_csv", "summary"):
                out = io.StringIO()
                writer = csv.writer(out)
                writer.writerow([
                    "timestamp", "dataset", "model", "detector", "device", "stage",
                    "current_sample", "total_samples", "cumulative_accuracy_pct",
                    "window_accuracy_pct", "f1_score_pct", "drift_signals",
                    "adaptations_count", "components_modified", "adaptation_time_ms", "throughput_smp_s"
                ])
                writer.writerow([
                    time.strftime("%Y-%m-%d %H:%M:%S"),
                    snap["dataset_name"],
                    snap["model_name"],
                    snap["detector_name"],
                    snap["device"],
                    snap["stage"],
                    snap["current_sample"],
                    snap["total_samples"],
                    snap["accuracy"],
                    snap["window_accuracy"],
                    snap["f1_score"],
                    snap["drift_count"],
                    snap["adaptation_count"],
                    snap["components_modified"],
                    snap["adaptation_time_ms"],
                    snap["throughput"]
                ])
                body = out.getvalue().encode("utf-8")
                filename = f"summary_{d_name}_{m_name}_{ts}.csv"
                self._send_file(body, "text/csv", filename)
            elif fmt in ("trajectory_csv", "trajectory"):
                out = io.StringIO()
                writer = csv.writer(out)
                writer.writerow(["sample", "cumulative_accuracy_pct", "window_accuracy_pct", "f1_score_pct", "drift_signal"])
                for pt in snap["trajectory"]:
                    writer.writerow([
                        pt.get("sample", 0),
                        pt.get("cumulative_acc", 0.0),
                        pt.get("window_acc", 0.0),
                        pt.get("f1", 0.0),
                        pt.get("drift", 0)
                    ])
                body = out.getvalue().encode("utf-8")
                filename = f"trajectory_{d_name}_{m_name}_{ts}.csv"
                self._send_file(body, "text/csv", filename)
            elif fmt in ("modifications_csv", "audit", "modifications"):
                out = io.StringIO()
                writer = csv.writer(out)
                writer.writerow(["sample_index", "event", "components_affected", "adaptation_latency_ms", "device"])
                for m in snap["component_modifications"]:
                    writer.writerow([
                        m.get("sample_index", 0),
                        m.get("event", ""),
                        m.get("components_affected", 0),
                        m.get("adaptation_latency_ms", 0.0),
                        m.get("device", "")
                    ])
                body = out.getvalue().encode("utf-8")
                filename = f"audit_{d_name}_{m_name}_{ts}.csv"
                self._send_file(body, "text/csv", filename)
            elif fmt in ("logs", "txt", "log"):
                lines = []
                for l in snap["logs"]:
                    lines.append(f"[{l.get('timestamp','')}] [{l.get('level','')}] {l.get('message','')}")
                body = "\n".join(lines).encode("utf-8")
                filename = f"logs_{d_name}_{m_name}_{ts}.txt"
                self._send_file(body, "text/plain; charset=utf-8", filename)
            else:
                self.send_error(400, f"Unsupported export format: {fmt}")
        elif path == "/api/datasets":
            datasets_list = [
                {"id": "sea", "name": "SEA Concepts Benchmark (5k)", "features": 3, "type": "Synthetic Abrupt"},
                {"id": "agrawal", "name": "Agrawal Stream Benchmark (5k)", "features": 9, "type": "Synthetic Subspace"},
                {"id": "electricity", "name": "NSW Electricity Market (10k)", "features": 8, "type": "Real-World Recurring"},
                {"id": "covertype", "name": "Forest Covertype Stream (10k)", "features": 54, "type": "Real-World Multi-Class"},
                {"id": "airlines", "name": "Airlines Flight Delay Stream (2k)", "features": 7, "type": "Real-World Delay"},
                {"id": "hf_adult", "name": "Hugging Face: Adult Census Income (10k)", "features": 14, "type": "Hugging Face"},
                {"id": "hf_bank", "name": "Hugging Face: Bank Marketing Stream (10k)", "features": 7, "type": "Hugging Face"},
                {"id": "hf_credit", "name": "Hugging Face: Credit Default Risk (10k)", "features": 21, "type": "Hugging Face"},
            ]
            if os.path.exists("data/custom_stream.csv"):
                datasets_list.append({"id": "custom", "name": "Custom Added Dataset (data/custom_stream.csv)", "features": "Auto", "type": "User Added"})
            self._send_json({"datasets": datasets_list})
        else:
            self.send_error(404, "Path not found")

    def do_POST(self):
        global STREAM_THREAD
        parsed = urlparse(self.path)
        path = parsed.path

        content_length = int(self.headers.get("Content-Length", 0))
        if content_length > MAX_CONTENT_LENGTH:
            self.send_error(413, "Payload Too Large (Maximum 50MB permitted)")
            return

        post_data = self.rfile.read(content_length).decode("utf-8") if content_length > 0 else "{}"
        try:
            payload = json.loads(post_data)
        except Exception:
            payload = {}

        if path == "/api/run":
            with RUN_LOCK:
                if TELEMETRY.running:
                    self._send_json({"status": "error", "message": "Pipeline already running"}, status=400)
                    return

                dataset_key = str(payload.get("dataset", "sea")).lower()
                model_key = str(payload.get("model", "proposed")).lower()
                detector_name = str(payload.get("detector", "adwin")).lower()

                if dataset_key not in VALID_DATASETS:
                    self._send_json({"status": "error", "message": f"Unknown dataset key: {dataset_key}"}, status=400)
                    return
                if model_key not in VALID_MODELS:
                    self._send_json({"status": "error", "message": f"Unknown model key: {model_key}"}, status=400)
                    return

                try:
                    warmup = max(10, min(10000, int(payload.get("warmup", 500))))
                    window = max(10, min(5000, int(payload.get("window", 200))))
                except (ValueError, TypeError):
                    self._send_json({"status": "error", "message": "Invalid warmup or window parameter"}, status=400)
                    return

                TELEMETRY.running = True
                TELEMETRY.should_stop = False
                STREAM_THREAD = threading.Thread(
                    target=run_streaming_pipeline,
                    args=(dataset_key, model_key, detector_name, warmup, window),
                    daemon=True,
                )
                STREAM_THREAD.start()
                self._send_json({"status": "started", "dataset": dataset_key, "model": model_key, "detector": detector_name})

        elif path == "/api/stop":
            TELEMETRY.should_stop = True
            self._send_json({"status": "stopping"})

        elif path == "/api/upload_dataset":
            csv_content = payload.get("csv_content", "")
            hf_id = payload.get("hf_id", "")

            os.makedirs("data", exist_ok=True)
            target_path = "data/custom_stream.csv"

            if hf_id:
                if not re.match(r"^[a-zA-Z0-9_\-\.]+\/[a-zA-Z0-9_\-\.]+$", hf_id):
                    self._send_json({"status": "error", "message": "Invalid Hugging Face ID format. Expected 'namespace/dataset_name'."}, status=400)
                    return
                try:
                    from datasets import load_dataset
                    ds = load_dataset(hf_id, split="train")
                    df = ds.to_pandas()
                    df.to_csv(target_path, index=False)
                    self._send_json({"status": "success", "rows": len(df), "columns": list(df.columns)})
                except Exception as e:
                    self._send_json({"status": "error", "message": str(e)}, status=400)
            elif csv_content:
                try:
                    with open(target_path, "w") as f:
                        f.write(csv_content)
                    df = pd.read_csv(target_path)
                    self._send_json({"status": "success", "rows": len(df), "columns": list(df.columns)})
                except Exception as e:
                    self._send_json({"status": "error", "message": str(e)}, status=400)
            else:
                self._send_json({"status": "error", "message": "Provide either csv_content or hf_id"}, status=400)
        else:
            self.send_error(404, "Unknown POST endpoint")


def start_server(port: int = 8000):
    server_address = ("127.0.0.1", port)
    httpd = ThreadingHTTPServer(server_address, AppRequestHandler)
    print(f"Starting server on http://127.0.0.1:{port}")
    httpd.serve_forever()


if __name__ == "__main__":
    port = 8000
    if len(sys.argv) > 1:
        port = int(sys.argv[1])
    start_server(port)

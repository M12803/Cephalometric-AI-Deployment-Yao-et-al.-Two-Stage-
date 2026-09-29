"""
api_server.py
============================================================
FastAPI Production Server for Cephalometric Landmark Localization.
Supports both Single-Stage (HRNet-W48) and Two-Stage (Yao et al. Coarse-to-Fine) Modes.

Features:
- RESTful API for single image cephalometric landmark prediction.
- Auto device selection (CUDA / Apple Silicon MPS / CPU).
- CORS enabled for seamless Web & Mobile frontend integration.
- Outputs structured JSON (coordinates, confidence, anatomical symbols, offsets).
- Generates high-resolution annotated image with cephalometric planes.
- Built-in interactive Web UI at root (http://localhost:8000).

Run server:
    python3 api_server.py
    # or: uvicorn api_server:app --host 0.0.0.0 --port 8000 --reload
============================================================
"""

from __future__ import annotations

import base64
import io
import os
from pathlib import Path
from typing import Dict, List, Optional

import cv2
import numpy as np
import torch
import uvicorn
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel

from data_loader.create_training_data import CANONICAL_29_SYMBOLS
from predict import CephalometricPredictor

# Initialize FastAPI App
app = FastAPI(
    title="Cephalometric AI Landmark Detection API",
    description="High-Precision 29-Landmark Localization & Cephalometric Analysis Powered by Yao et al. Two-Stage Architecture & HRNet-W48.",
    version="2.0.0",
)

_cors_origins = [
    origin.strip()
    for origin in os.getenv("CORS_ALLOW_ORIGINS", "*").split(",")
    if origin.strip()
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=_cors_origins or ["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Anatomical Descriptions for all 29 canonical landmarks
LANDMARK_METADATA = {
    "A": {"name": "Subspinale (A-point)", "region": "Maxilla", "type": "Hard Tissue"},
    "ANS": {"name": "Anterior Nasal Spine", "region": "Maxilla", "type": "Hard Tissue"},
    "B": {"name": "Supramentale (B-point)", "region": "Mandible", "type": "Hard Tissue"},
    "Me": {"name": "Menton", "region": "Mandible", "type": "Hard Tissue"},
    "N": {"name": "Nasion", "region": "Cranial Base", "type": "Hard Tissue"},
    "Or": {"name": "Orbitale", "region": "Craniofacial", "type": "Hard Tissue"},
    "Pog": {"name": "Pogonion", "region": "Mandible", "type": "Hard Tissue"},
    "PNS": {"name": "Posterior Nasal Spine", "region": "Maxilla", "type": "Hard Tissue"},
    "Pn": {"name": "Pronasale (Nose Tip)", "region": "Profile", "type": "Soft Tissue"},
    "R": {"name": "R-point", "region": "Cranial", "type": "Hard Tissue"},
    "S": {"name": "Sella Turcica", "region": "Cranial Base", "type": "Hard Tissue"},
    "Ar": {"name": "Articulare", "region": "Mandibular Condyle", "type": "Hard Tissue"},
    "Co": {"name": "Condylion", "region": "Mandibular Condyle", "type": "Hard Tissue"},
    "Gn": {"name": "Gnathion", "region": "Mandible", "type": "Hard Tissue"},
    "Go": {"name": "Gonion", "region": "Mandibular Angle", "type": "Hard Tissue"},
    "L1": {"name": "Lower Incisor Tip", "region": "Dentition", "type": "Dental"},
    "L1A": {"name": "Lower Incisor Apex", "region": "Dentition", "type": "Dental"},
    "Li": {"name": "Labrale Inferius (Lower Lip)", "region": "Profile", "type": "Soft Tissue"},
    "Ls": {"name": "Labrale Superius (Upper Lip)", "region": "Profile", "type": "Soft Tissue"},
    "P": {"name": "Porion", "region": "Temporal Bone", "type": "Hard Tissue"},
    "PMA": {"name": "Posterior Maxillary Point", "region": "Maxilla", "type": "Hard Tissue"},
    "Po": {"name": "Porion (External Acoustic Meatus)", "region": "Temporal Bone", "type": "Hard Tissue"},
    "Pos": {"name": "Pogonion Soft Tissue", "region": "Profile", "type": "Soft Tissue"},
    "Prn": {"name": "Pronasale Soft Tissue", "region": "Profile", "type": "Soft Tissue"},
    "Sn": {"name": "Subnasale", "region": "Profile", "type": "Soft Tissue"},
    "U1": {"name": "Upper Incisor Tip", "region": "Dentition", "type": "Dental"},
    "U1A": {"name": "Upper Incisor Apex", "region": "Dentition", "type": "Dental"},
    "Ba": {"name": "Basion", "region": "Cranial Base", "type": "Hard Tissue"},
    "S_T": {"name": "Sella Soft Tissue", "region": "Cranial", "type": "Soft Tissue"},
}

DEFAULT_CHECKPOINTS = [
    "saved_models/best_two_stage_cephnet.pth",
    "saved_models/best_hrnet_w48_clean.pth",
    "saved_models/best_hrnet_w48.pth",
]

CHECKPOINT_PATH = os.getenv("CHECKPOINT_PATH")
if not CHECKPOINT_PATH:
    for cp in DEFAULT_CHECKPOINTS:
        if Path(cp).exists():
            CHECKPOINT_PATH = cp
            break
    if not CHECKPOINT_PATH:
        CHECKPOINT_PATH = DEFAULT_CHECKPOINTS[0]

predictor: Optional[CephalometricPredictor] = None


def get_predictor() -> CephalometricPredictor:
    global predictor
    if predictor is None:
        if not Path(CHECKPOINT_PATH).exists():
            raise RuntimeError(f"Model checkpoint not found at: {CHECKPOINT_PATH}")
        print(f"[INFO] Initializing CephalometricPredictor from {CHECKPOINT_PATH}...")
        predictor = CephalometricPredictor(checkpoint_path=CHECKPOINT_PATH, mode="two-stage")
        print(f"[INFO] Predictor ready on device: {predictor.device} (Mode: {predictor.mode})")
    return predictor


@app.on_event("startup")
def startup_event():
    try:
        get_predictor()
    except Exception as e:
        print(f"[WARN] Startup preload deferred: {e}")


# ----------------------------------------------------------------------------
# Pydantic Schemas
# ----------------------------------------------------------------------------
class LandmarkItem(BaseModel):
    symbol: str
    name: str
    region: str
    tissue_type: str
    x_px: float
    y_px: float
    coarse_x_px: Optional[float] = None
    coarse_y_px: Optional[float] = None
    offset_dx_px: Optional[float] = None
    offset_dy_px: Optional[float] = None
    refinement_delta_mm: Optional[float] = None
    confidence: float


class PredictionResponse(BaseModel):
    success: bool
    mode: str
    architecture: str
    image_width: int
    image_height: int
    pixel_size_mm: float
    execution_time_seconds: float
    stage1_time_seconds: float
    stage2_time_seconds: float
    total_landmarks: int
    landmarks: List[LandmarkItem]
    annotated_image_base64: Optional[str] = None


# ----------------------------------------------------------------------------
# API Endpoints
# ----------------------------------------------------------------------------
@app.get("/health")
def health_check():
    pred = get_predictor()
    return {
        "status": "healthy",
        "model": "TwoStageCephNet (Yao et al.) / HRNet-W48",
        "device": str(pred.device),
        "active_mode": pred.mode,
        "checkpoint": str(pred.ckpt_path),
        "landmarks_supported": len(CANONICAL_29_SYMBOLS),
    }


@app.get("/landmarks")
def list_landmarks():
    """Return canonical 29 landmarks dictionary and their anatomical metadata."""
    return {
        "count": len(LANDMARK_METADATA),
        "landmarks": LANDMARK_METADATA,
    }


@app.post("/predict", response_model=PredictionResponse)
async def predict_xray(
    file: UploadFile = File(..., description="Lateral Cephalometric X-ray image (PNG/JPG/BMP)"),
    mode: str = Form("two-stage", description="Inference mode: 'two-stage' (Yao et al. Coarse-to-Fine) or 'global' (HRNet-W48)"),
    pixel_size_mm: float = Form(0.1, description="Physical resolution in mm/pixel (default: 0.1)"),
    return_image: bool = Form(True, description="Whether to include base64 annotated image in response"),
):
    """
    Predict 29 anatomical landmarks on an uploaded Cephalometric X-ray radiograph.
    """
    try:
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img is None:
            raise HTTPException(status_code=400, detail="Invalid image file format.")

        orig_h, orig_w = img.shape[:2]
        pred_engine = get_predictor()

        # Run prediction
        res = pred_engine.predict(
            image=img,
            pixel_size=pixel_size_mm,
            mode=mode,
            draw_overlay=return_image,
        )

        landmark_items = []
        for lm in res["landmarks"]:
            sym = lm["symbol"]
            meta = LANDMARK_METADATA.get(sym, {"name": sym, "region": "Unknown", "type": "Unknown"})
            landmark_items.append(
                LandmarkItem(
                    symbol=sym,
                    name=meta["name"],
                    region=meta["region"],
                    tissue_type=meta["type"],
                    x_px=round(lm["x"], 2),
                    y_px=round(lm["y"], 2),
                    coarse_x_px=round(lm.get("coarse_x", lm["x"]), 2),
                    coarse_y_px=round(lm.get("coarse_y", lm["y"]), 2),
                    offset_dx_px=round(lm.get("offset_dx_px", 0.0), 2),
                    offset_dy_px=round(lm.get("offset_dy_px", 0.0), 2),
                    refinement_delta_mm=round(lm.get("refinement_delta_mm", 0.0), 3),
                    confidence=round(lm["confidence"], 4),
                )
            )

        b64_image = None
        if return_image and "annotated_image" in res and res["annotated_image"] is not None:
            _, buffer = cv2.imencode(".jpg", res["annotated_image"], [cv2.IMWRITE_JPEG_QUALITY, 90])
            b64_image = f"data:image/jpeg;base64,{base64.b64encode(buffer).decode('utf-8')}"

        return PredictionResponse(
            success=True,
            mode=res.get("mode", mode),
            architecture=res.get("architecture", "Yao et al. Two-Stage Coarse-to-Fine"),
            image_width=orig_w,
            image_height=orig_h,
            pixel_size_mm=pixel_size_mm,
            execution_time_seconds=res.get("execution_time_seconds", 0.0),
            stage1_time_seconds=res.get("stage1_time_seconds", 0.0),
            stage2_time_seconds=res.get("stage2_time_seconds", 0.0),
            total_landmarks=len(landmark_items),
            landmarks=landmark_items,
            annotated_image_base64=b64_image,
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/", response_class=HTMLResponse)
def interactive_web_ui():
    """
    Embedded interactive Web UI for testing directly in any web browser.
    """
    return """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Cephalometric AI Landmark Analyzer - Two-Stage (Yao et al.)</title>
    <style>
        :root {
            --primary: #2563eb;
            --primary-dark: #1d4ed8;
            --bg: #0f172a;
            --card-bg: #1e293b;
            --text: #f8fafc;
            --text-muted: #94a3b8;
            --border: #334155;
            --accent: #10b981;
        }
        * { box-sizing: border-box; margin: 0; padding: 0; }
        body {
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
            background: var(--bg);
            color: var(--text);
            padding: 24px;
            min-height: 100vh;
        }
        .container { max-width: 1440px; margin: 0 auto; }
        header {
            text-align: center;
            margin-bottom: 24px;
            padding-bottom: 16px;
            border-bottom: 1px solid var(--border);
        }
        header h1 { font-size: 28px; margin-bottom: 8px; color: #60a5fa; }
        header p { color: var(--text-muted); font-size: 14px; }
        .tagline {
            display: inline-block;
            background: rgba(37,99,235,0.2);
            color: #93c5fd;
            padding: 4px 12px;
            border-radius: 20px;
            font-size: 12px;
            margin-top: 6px;
            border: 1px solid rgba(147,197,253,0.3);
        }
        .grid { display: grid; grid-template-columns: 1.1fr 0.9fr; gap: 24px; }
        @media (max-width: 960px) { .grid { grid-template-columns: 1fr; } }
        .card {
            background: var(--card-bg);
            border: 1px solid var(--border);
            border-radius: 12px;
            padding: 20px;
        }
        .card h2 { font-size: 18px; margin-bottom: 14px; color: var(--text); display: flex; align-items: center; justify-content: space-between; }
        .upload-zone {
            border: 2px dashed var(--border);
            border-radius: 8px;
            padding: 30px 20px;
            text-align: center;
            cursor: pointer;
            transition: all 0.2s;
            margin-bottom: 16px;
        }
        .upload-zone:hover { border-color: var(--primary); background: rgba(37,99,235,0.05); }
        .upload-zone input { display: none; }
        .controls { display: grid; grid-template-columns: 1fr 1fr; gap: 12px; margin-bottom: 14px; }
        .control-group label { display: block; font-size: 12px; color: var(--text-muted); margin-bottom: 4px; }
        .control-group select, .control-group input {
            width: 100%;
            background: #0f172a;
            border: 1px solid var(--border);
            color: var(--text);
            padding: 8px 10px;
            border-radius: 6px;
            font-size: 13px;
        }
        .btn {
            background: var(--primary);
            color: white;
            border: none;
            padding: 10px 20px;
            border-radius: 6px;
            cursor: pointer;
            font-size: 14px;
            font-weight: 600;
            width: 100%;
            transition: background 0.2s;
        }
        .btn:hover { background: var(--primary-dark); }
        .btn:disabled { opacity: 0.5; cursor: not-allowed; }
        .image-preview {
            width: 100%;
            max-height: 580px;
            object-fit: contain;
            background: #000;
            border-radius: 8px;
            border: 1px solid var(--border);
            margin-top: 14px;
            display: none;
        }
        .table-wrapper {
            max-height: 520px;
            overflow-y: auto;
            border: 1px solid var(--border);
            border-radius: 8px;
        }
        table { width: 100%; border-collapse: collapse; font-size: 12px; }
        th, td { padding: 8px 10px; text-align: left; border-bottom: 1px solid var(--border); }
        th { background: #0f172a; position: sticky; top: 0; color: #93c5fd; }
        tr:hover { background: rgba(255,255,255,0.03); }
        .badge {
            display: inline-block;
            padding: 2px 6px;
            border-radius: 4px;
            font-size: 11px;
            font-weight: bold;
        }
        .badge-hard { background: #1e3a8a; color: #93c5fd; }
        .badge-soft { background: #064e3b; color: #6ee7b7; }
        .badge-dental { background: #581c87; color: #d8b4fe; }
        .badge-mode { background: #065f46; color: #a7f3d0; padding: 2px 8px; border-radius: 12px; font-size: 11px; }
        .spinner {
            display: inline-block;
            width: 16px;
            height: 16px;
            border: 2px solid rgba(255,255,255,0.3);
            border-top-color: white;
            border-radius: 50%;
            animation: spin 0.8s linear infinite;
            margin-right: 8px;
            vertical-align: middle;
        }
        @keyframes spin { to { transform: rotate(360deg); } }
        .status-bar {
            margin-top: 12px;
            font-size: 13px;
            color: var(--text-muted);
            display: flex;
            justify-content: space-between;
        }
        .action-btns {
            display: flex;
            gap: 8px;
            margin-top: 12px;
        }
        .action-btns a {
            flex: 1;
            text-align: center;
            padding: 8px;
            background: #334155;
            color: white;
            text-decoration: none;
            border-radius: 6px;
            font-size: 12px;
        }
        .action-btns a:hover { background: #475569; }
    </style>
</head>
<body>
    <div class="container">
        <header>
            <h1>Cephalometric AI Landmark Analyzer</h1>
            <p>Clinical-Grade 29-Landmark Localization with Cephalometric Planes</p>
            <div class="tagline">Yao et al. Coarse-to-Fine Two-Stage Architecture + HRNet-W48</div>
        </header>

        <div class="grid">
            <!-- Left Column: Upload & Image Viewer -->
            <div class="card">
                <h2>
                    <span>Radiograph Viewer</span>
                    <span id="timingBadge" class="badge-mode" style="display:none">0.35s</span>
                </h2>

                <div class="upload-zone" id="uploadZone" onclick="document.getElementById('fileInput').click()">
                    <input type="file" id="fileInput" accept="image/*" onchange="handleFileSelect(event)">
                    <div style="font-size: 32px; margin-bottom: 8px;">🔬</div>
                    <div style="font-size: 14px; font-weight: 600;" id="uploadPrompt">Drag & drop Cephalometric X-ray, or click to browse</div>
                    <div style="font-size: 12px; color: var(--text-muted); margin-top: 4px;">PNG, JPG, BMP formats supported</div>
                </div>

                <div class="controls">
                    <div class="control-group">
                        <label for="modeSelect">Inference Architecture</label>
                        <select id="modeSelect">
                            <option value="two-stage" selected>Two-Stage (Yao et al. Coarse-to-Fine)</option>
                            <option value="global">Single-Stage (HRNet-W48 Global)</option>
                        </select>
                    </div>
                    <div class="control-group">
                        <label for="pixelSize">Pixel Resolution (mm/px)</label>
                        <input type="number" id="pixelSize" value="0.1" step="0.01" min="0.01" max="1.0">
                    </div>
                </div>

                <button class="btn" id="analyzeBtn" onclick="runInference()" disabled>
                    <span id="btnText">Analyze Radiograph</span>
                </button>

                <div class="status-bar" id="statusBar">Ready for upload</div>

                <img id="previewImg" class="image-preview" alt="Annotated Cephalogram Preview">

                <div class="action-btns" id="actionBtns" style="display:none">
                    <a id="downloadJsonBtn" href="#" download="cephalometric_landmarks.json">Download JSON</a>
                    <a id="downloadImgBtn" href="#" download="cephalometric_annotated.jpg">Download Image</a>
                </div>
            </div>

            <!-- Right Column: Landmark Data Table -->
            <div class="card">
                <h2>
                    <span>Detected Landmarks</span>
                    <span id="landmarkCount" style="font-size: 13px; color: var(--text-muted);">0 / 29</span>
                </h2>

                <div class="table-wrapper">
                    <table id="landmarksTable">
                        <thead>
                            <tr>
                                <th>#</th>
                                <th>Sym</th>
                                <th>Anatomical Name</th>
                                <th>Region</th>
                                <th>X (px)</th>
                                <th>Y (px)</th>
                                <th>Conf</th>
                            </tr>
                        </thead>
                        <tbody id="tableBody">
                            <tr>
                                <td colspan="7" style="text-align: center; color: var(--text-muted); padding: 40px 0;">
                                    Upload an X-ray radiograph to detect cephalometric landmarks
                                </td>
                            </tr>
                        </tbody>
                    </table>
                </div>
            </div>
        </div>
    </div>

    <script>
        let selectedFile = null;
        let lastResponseData = null;

        const uploadZone = document.getElementById('uploadZone');
        const fileInput = document.getElementById('fileInput');
        const analyzeBtn = document.getElementById('analyzeBtn');
        const btnText = document.getElementById('btnText');
        const previewImg = document.getElementById('previewImg');
        const tableBody = document.getElementById('tableBody');
        const statusBar = document.getElementById('statusBar');
        const timingBadge = document.getElementById('timingBadge');
        const landmarkCount = document.getElementById('landmarkCount');
        const actionBtns = document.getElementById('actionBtns');

        ['dragenter', 'dragover'].forEach(name => {
            uploadZone.addEventListener(name, (e) => {
                e.preventDefault();
                uploadZone.style.borderColor = '#2563eb';
                uploadZone.style.background = 'rgba(37,99,235,0.1)';
            });
        });

        ['dragleave', 'drop'].forEach(name => {
            uploadZone.addEventListener(name, (e) => {
                e.preventDefault();
                uploadZone.style.borderColor = '#334155';
                uploadZone.style.background = 'transparent';
            });
        });

        uploadZone.addEventListener('drop', (e) => {
            if (e.dataTransfer.files.length > 0) {
                fileInput.files = e.dataTransfer.files;
                handleFileSelect({ target: fileInput });
            }
        });

        function handleFileSelect(event) {
            const file = event.target.files[0];
            if (!file) return;
            selectedFile = file;
            document.getElementById('uploadPrompt').innerText = file.name + ' (' + (file.size / 1024 / 1024).toFixed(2) + ' MB)';
            analyzeBtn.disabled = false;
            statusBar.innerText = 'Image loaded. Click Analyze to run model.';
        }

        async function runInference() {
            if (!selectedFile) return;

            const mode = document.getElementById('modeSelect').value;
            const pixelSize = document.getElementById('pixelSize').value || 0.1;

            analyzeBtn.disabled = true;
            btnText.innerHTML = '<span class="spinner"></span>Analyzing...';
            statusBar.innerText = 'Running ' + (mode === 'two-stage' ? 'Two-Stage (Yao et al.)' : 'HRNet-W48') + ' inference...';

            const formData = new FormData();
            formData.append('file', selectedFile);
            formData.append('mode', mode);
            formData.append('pixel_size_mm', pixelSize);
            formData.append('return_image', 'true');

            try {
                const resp = await fetch('/predict', {
                    method: 'POST',
                    body: formData
                });

                if (!resp.ok) {
                    const err = await resp.json();
                    throw new Error(err.detail || 'Prediction failed');
                }

                const data = await resp.json();
                lastResponseData = data;
                renderResults(data);

            } catch (err) {
                statusBar.innerText = 'Error: ' + err.message;
                statusBar.style.color = '#ef4444';
            } finally {
                analyzeBtn.disabled = false;
                btnText.innerText = 'Analyze Radiograph';
            }
        }

        function renderResults(data) {
            timingBadge.style.display = 'inline-block';
            timingBadge.innerText = data.execution_time_seconds.toFixed(3) + 's (' + data.mode + ')';
            statusBar.innerText = 'Analysis complete! ' + data.total_landmarks + ' landmarks detected in ' + data.execution_time_seconds.toFixed(3) + 's';
            statusBar.style.color = '#10b981';
            landmarkCount.innerText = data.landmarks.length + ' / 29';

            if (data.annotated_image_base64) {
                previewImg.src = data.annotated_image_base64;
                previewImg.style.display = 'block';

                const downloadImg = document.getElementById('downloadImgBtn');
                downloadImg.href = data.annotated_image_base64;
            }

            const jsonStr = 'data:text/json;charset=utf-8,' + encodeURIComponent(JSON.stringify(data, null, 2));
            document.getElementById('downloadJsonBtn').href = jsonStr;
            actionBtns.style.display = 'flex';

            tableBody.innerHTML = '';
            data.landmarks.forEach((lm, idx) => {
                const tr = document.createElement('tr');
                let badgeClass = 'badge-hard';
                if (lm.tissue_type === 'Soft Tissue') badgeClass = 'badge-soft';
                if (lm.tissue_type === 'Dental') badgeClass = 'badge-dental';

                tr.innerHTML = `
                    <td>${idx + 1}</td>
                    <td><strong>${lm.symbol}</strong></td>
                    <td>${lm.name}</td>
                    <td><span class="badge ${badgeClass}">${lm.region}</span></td>
                    <td>${lm.x_px.toFixed(1)}</td>
                    <td>${lm.y_px.toFixed(1)}</td>
                    <td>${(lm.confidence * 100).toFixed(1)}%</td>
                `;
                tableBody.appendChild(tr);
            });
        }
    </script>
</body>
</html>
    """


def main():
    port = int(os.getenv("PORT", 8000))
    host = os.getenv("HOST", "0.0.0.0")
    print(f"[INFO] Starting Cephalometric AI API Server on http://{host}:{port} ...")
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    main()

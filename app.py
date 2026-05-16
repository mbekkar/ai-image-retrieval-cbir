"""
CBIR Web App — Flask Interface
================================
Upload a query image and get similar images displayed in the browser.

Author : Mounir Bekkar
"""

import os
import base64
import io
from pathlib import Path
from flask import Flask, request, jsonify, render_template_string
from werkzeug.utils import secure_filename
import cv2
import numpy as np

from cbir_engine import load_index, find_similar, build_index, INDEX_FILE

# ── CONFIG ────────────────────────────────────────────────────────────────────
UPLOAD_FOLDER    = "uploads"
ALLOWED_EXT      = {"jpg", "jpeg", "png", "webp", "bmp"}
MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16 MB
DATASET_DIR      = os.environ.get("CBIR_DATASET", "./dataset")

app = Flask(__name__)
app.config["UPLOAD_FOLDER"]        = UPLOAD_FOLDER
app.config["MAX_CONTENT_LENGTH"]   = MAX_CONTENT_LENGTH
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Load index at startup
_index = {}
def get_index():
    global _index
    if not _index:
        try:
            _index = load_index()
        except FileNotFoundError:
            print("[WARN] No index found. Build one with: python search.py index --dir ./dataset")
    return _index


# ── HTML TEMPLATE ─────────────────────────────────────────────────────────────
HTML = """
<!DOCTYPE html>
<html lang="fr">
<head>
  <meta charset="UTF-8"/>
  <meta name="viewport" content="width=device-width, initial-scale=1"/>
  <title>CBIR — Image Search</title>
  <style>
    * { box-sizing: border-box; margin: 0; padding: 0; }
    body { font-family: 'Segoe UI', sans-serif; background: #0a0f1e; color: #e2e8f0; min-height: 100vh; }
    nav { background: rgba(15,20,40,.95); border-bottom: 1px solid rgba(99,179,237,.15); padding: 1rem 2rem; display: flex; align-items: center; gap: 12px; }
    nav h1 { font-size: 1.1rem; font-weight: 700; color: #63b3ed; letter-spacing: .05em; }
    nav span { color: #475569; font-size: .8rem; font-family: monospace; }
    .hero { background: linear-gradient(135deg, rgba(99,179,237,.06), rgba(167,139,250,.04)); border-bottom: 1px solid rgba(255,255,255,.05); padding: 3rem 2rem; text-align: center; }
    .hero h2 { font-size: 2rem; font-weight: 800; margin-bottom: .5rem; background: linear-gradient(90deg,#63b3ed,#a78bfa); -webkit-background-clip: text; -webkit-text-fill-color: transparent; }
    .hero p { color: #64748b; font-size: .9rem; }
    .container { max-width: 1100px; margin: 2rem auto; padding: 0 1.5rem; }
    .upload-zone { border: 2px dashed rgba(99,179,237,.35); border-radius: 16px; padding: 3rem; text-align: center; cursor: pointer; transition: all .3s; background: rgba(99,179,237,.03); margin-bottom: 2rem; }
    .upload-zone:hover, .upload-zone.drag { border-color: #63b3ed; background: rgba(99,179,237,.08); }
    .upload-zone .icon { font-size: 3rem; margin-bottom: 1rem; }
    .upload-zone p { color: #94a3b8; font-size: .9rem; margin-bottom: 1rem; }
    .upload-zone input[type=file] { display: none; }
    .btn { padding: 10px 24px; border-radius: 10px; font-weight: 600; font-size: .85rem; cursor: pointer; border: none; transition: all .2s; }
    .btn-primary { background: linear-gradient(90deg,#3b82f6,#8b5cf6); color: #fff; }
    .btn-primary:hover { transform: scale(1.03); opacity: .9; }
    .top-k { display: flex; align-items: center; gap: 12px; margin-bottom: 1rem; justify-content: center; }
    .top-k label { color: #94a3b8; font-size: .85rem; }
    .top-k input { background: rgba(255,255,255,.05); border: 1px solid rgba(255,255,255,.1); border-radius: 8px; padding: 6px 12px; color: #fff; width: 70px; text-align: center; font-size: .9rem; }
    #preview { display: none; margin-bottom: 2rem; text-align: center; }
    #preview img { max-width: 280px; border-radius: 12px; border: 2px solid rgba(99,179,237,.3); }
    #preview p { color: #64748b; font-size: .8rem; margin-top: 8px; font-family: monospace; }
    .results-title { font-size: 1.1rem; font-weight: 700; color: #fff; margin-bottom: 1rem; }
    .results-title span { color: #63b3ed; }
    .results-grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(200px, 1fr)); gap: 16px; }
    .result-card { background: rgba(15,20,40,.7); border: 1px solid rgba(255,255,255,.08); border-radius: 12px; overflow: hidden; transition: transform .2s; }
    .result-card:hover { transform: translateY(-4px); border-color: rgba(99,179,237,.3); }
    .result-card img { width: 100%; height: 160px; object-fit: cover; display: block; }
    .result-card .info { padding: 10px 12px; }
    .result-card .rank { font-family: monospace; font-size: .7rem; color: #a78bfa; margin-bottom: 4px; }
    .score-bar { height: 4px; border-radius: 2px; background: rgba(255,255,255,.08); margin-bottom: 6px; overflow: hidden; }
    .score-fill { height: 100%; border-radius: 2px; }
    .score-text { font-family: monospace; font-size: .75rem; display: flex; justify-content: space-between; }
    .filename { font-size: .7rem; color: #64748b; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; margin-top: 4px; }
    .loader { display: none; text-align: center; padding: 2rem; }
    .spinner { width: 40px; height: 40px; border: 3px solid rgba(99,179,237,.2); border-top-color: #63b3ed; border-radius: 50%; animation: spin .8s linear infinite; margin: 0 auto 1rem; }
    @keyframes spin { to { transform: rotate(360deg); } }
    .status { padding: 12px 16px; border-radius: 10px; margin-bottom: 1.5rem; font-size: .85rem; font-family: monospace; }
    .status.info { background: rgba(99,179,237,.08); border: 1px solid rgba(99,179,237,.2); color: #63b3ed; }
    .status.error { background: rgba(244,114,182,.08); border: 1px solid rgba(244,114,182,.2); color: #f472b6; }
    .empty { text-align: center; padding: 3rem; color: #475569; }
    .badge { display: inline-block; padding: 2px 8px; border-radius: 999px; font-size: .7rem; font-family: monospace; }
    .badge-green { background: rgba(52,211,153,.12); color: #34d399; border: 1px solid rgba(52,211,153,.25); }
    .badge-blue  { background: rgba(99,179,237,.12); color: #63b3ed;  border: 1px solid rgba(99,179,237,.25); }
    .badge-red   { background: rgba(251,146,60,.12);  color: #fb923c;  border: 1px solid rgba(251,146,60,.25); }
  </style>
</head>
<body>

<nav>
  <h1>🔍 CBIR</h1>
  <span>Content-Based Image Retrieval · HSV + ORB · Cosine Similarity</span>
  <span style="margin-left:auto" id="index-status">Chargement...</span>
</nav>

<div class="hero">
  <h2>Recherche d'Images par Contenu</h2>
  <p>Uploadez une image — le système trouve les images les plus similaires visuellement.</p>
</div>

<div class="container">
  <!-- Status bar -->
  <div id="status-bar"></div>

  <!-- Upload zone -->
  <div class="upload-zone" id="drop-zone" onclick="document.getElementById('file-input').click()">
    <div class="icon">🖼️</div>
    <p>Glissez-déposez une image ici ou cliquez pour choisir</p>
    <p style="font-size:.75rem;color:#475569">JPG, PNG, WebP · Max 16 MB</p>
    <input type="file" id="file-input" accept="image/*"/>
  </div>

  <!-- Preview + options -->
  <div id="preview">
    <img id="preview-img" src="" alt="Query image"/>
    <p id="preview-name"></p>
    <div class="top-k" style="margin-top:12px">
      <label>Nombre de résultats :</label>
      <input type="number" id="top-k" value="6" min="1" max="20"/>
    </div>
    <button class="btn btn-primary" id="search-btn" onclick="doSearch()">🔍 Rechercher</button>
  </div>

  <!-- Loader -->
  <div class="loader" id="loader">
    <div class="spinner"></div>
    <p style="color:#64748b;font-size:.85rem">Extraction des features et recherche en cours...</p>
  </div>

  <!-- Results -->
  <div id="results-section" style="display:none">
    <p class="results-title">
      Résultats similaires pour <span id="results-query"></span>
      <span id="results-count" style="color:#64748b;font-size:.8rem;margin-left:8px"></span>
    </p>
    <div class="results-grid" id="results-grid"></div>
  </div>
</div>

<script>
let uploadedFile = null;

// ── Drag & drop ──────────────────────────────────────────────────────────────
const dropZone  = document.getElementById('drop-zone');
const fileInput = document.getElementById('file-input');

dropZone.addEventListener('dragover',  e => { e.preventDefault(); dropZone.classList.add('drag'); });
dropZone.addEventListener('dragleave', ()  => dropZone.classList.remove('drag'));
dropZone.addEventListener('drop', e => {
  e.preventDefault(); dropZone.classList.remove('drag');
  const file = e.dataTransfer.files[0];
  if (file) handleFile(file);
});
fileInput.addEventListener('change', () => { if (fileInput.files[0]) handleFile(fileInput.files[0]); });

function handleFile(file) {
  uploadedFile = file;
  const reader = new FileReader();
  reader.onload = e => {
    document.getElementById('preview-img').src = e.target.result;
    document.getElementById('preview-name').textContent = file.name + ' · ' + (file.size / 1024).toFixed(1) + ' KB';
    document.getElementById('preview').style.display = 'block';
    document.getElementById('results-section').style.display = 'none';
    document.getElementById('status-bar').innerHTML = '';
  };
  reader.readAsDataURL(file);
}

// ── Search ───────────────────────────────────────────────────────────────────
async function doSearch() {
  if (!uploadedFile) { showStatus('Veuillez choisir une image.', 'error'); return; }

  const topK = parseInt(document.getElementById('top-k').value) || 6;
  const formData = new FormData();
  formData.append('image', uploadedFile);
  formData.append('top_k', topK);

  document.getElementById('loader').style.display = 'block';
  document.getElementById('results-section').style.display = 'none';
  document.getElementById('search-btn').disabled = true;
  document.getElementById('status-bar').innerHTML = '';

  try {
    const resp = await fetch('/search', { method: 'POST', body: formData });
    const data = await resp.json();

    if (!resp.ok || data.error) {
      showStatus(data.error || 'Erreur serveur.', 'error');
      return;
    }

    renderResults(data.results, uploadedFile.name);

  } catch (err) {
    showStatus('Erreur réseau : ' + err.message, 'error');
  } finally {
    document.getElementById('loader').style.display = 'none';
    document.getElementById('search-btn').disabled = false;
  }
}

function renderResults(results, queryName) {
  const grid = document.getElementById('results-grid');
  grid.innerHTML = '';

  document.getElementById('results-query').textContent = queryName;
  document.getElementById('results-count').textContent = `(${results.length} résultat${results.length > 1 ? 's' : ''})`;
  document.getElementById('results-section').style.display = 'block';

  if (results.length === 0) {
    grid.innerHTML = '<div class="empty">Aucun résultat trouvé.<br><small>Vérifiez que le dataset est bien indexé.</small></div>';
    return;
  }

  results.forEach((r, i) => {
    const score   = r.score;
    const color   = score > 0.85 ? '#34d399' : score > 0.65 ? '#63b3ed' : score > 0.45 ? '#a78bfa' : '#fb923c';
    const label   = score > 0.85 ? 'Très similaire' : score > 0.65 ? 'Similaire' : score > 0.45 ? 'Modéré' : 'Faible';
    const badgeC  = score > 0.85 ? 'badge-green' : score > 0.65 ? 'badge-blue' : 'badge-red';
    const filename = r.path.replace(/\\/g, '/').split('/').pop();

    const card = document.createElement('div');
    card.className = 'result-card';
    card.innerHTML = `
      <img src="/image?path=${encodeURIComponent(r.path)}" alt="${filename}" onerror="this.src='data:image/svg+xml,<svg xmlns=\\'http://www.w3.org/2000/svg\\'><rect width=\\'200\\' height=\\'160\\' fill=\\'%230a1020\\'/><text x=\\'50%\\' y=\\'50%\\' fill=\\'%23334155\\' font-size=\\'12\\' text-anchor=\\'middle\\' dy=\\'.3em\\'>No preview</text></svg>'"/>
      <div class="info">
        <div class="rank">#${i+1} — <span class="badge ${badgeC}">${label}</span></div>
        <div class="score-bar"><div class="score-fill" style="width:${score*100}%;background:${color}"></div></div>
        <div class="score-text"><span style="color:${color};font-weight:600">${(score*100).toFixed(1)}%</span><span style="color:#475569">cosine</span></div>
        <div class="filename" title="${r.path}">${filename}</div>
      </div>`;
    grid.appendChild(card);
  });
}

function showStatus(msg, type='info') {
  document.getElementById('status-bar').innerHTML = `<div class="status ${type}">${msg}</div>`;
}

// ── Index status ──────────────────────────────────────────────────────────────
fetch('/status')
  .then(r => r.json())
  .then(d => {
    const el = document.getElementById('index-status');
    if (d.indexed > 0) {
      el.innerHTML = `<span style="color:#34d399">✓ ${d.indexed} images indexées</span>`;
    } else {
      el.innerHTML = `<span style="color:#f472b6">⚠ Index vide</span>`;
      showStatus('Index vide. Lancez : python search.py index --dir ./dataset', 'error');
    }
  })
  .catch(() => {});
</script>
</body>
</html>
"""


# ── ROUTES ────────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return render_template_string(HTML)


@app.route("/status")
def status():
    idx = get_index()
    return jsonify({ "indexed": len(idx) })


@app.route("/search", methods=["POST"])
def search():
    if "image" not in request.files:
        return jsonify({ "error": "No image provided." }), 400

    file  = request.files["image"]
    top_k = int(request.form.get("top_k", 6))

    if file.filename == "":
        return jsonify({ "error": "Empty filename." }), 400

    ext = Path(file.filename).suffix.lower().lstrip(".")
    if ext not in ALLOWED_EXT:
        return jsonify({ "error": f"Extension non supportée: .{ext}" }), 400

    # Save query image temporarily
    filename = secure_filename(file.filename)
    tmp_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
    file.save(tmp_path)

    idx = get_index()
    if not idx:
        return jsonify({ "error": "Index vide. Lancez: python search.py index --dir ./dataset" }), 500

    try:
        results = find_similar(tmp_path, idx, top_k=top_k, exclude_self=False)
        return jsonify({
            "results": [{ "path": path, "score": score } for path, score in results]
        })
    except Exception as e:
        return jsonify({ "error": str(e) }), 500


@app.route("/image")
def serve_image():
    """Serve an image from the indexed dataset."""
    from flask import send_file, abort
    import urllib.parse

    path = request.args.get("path", "")
    path = urllib.parse.unquote(path)

    if not os.path.exists(path):
        abort(404)

    # Security: only serve from DATASET_DIR and UPLOAD_FOLDER
    abs_path    = os.path.abspath(path)
    abs_dataset = os.path.abspath(DATASET_DIR)
    abs_upload  = os.path.abspath(UPLOAD_FOLDER)

    if not (abs_path.startswith(abs_dataset) or abs_path.startswith(abs_upload)):
        abort(403)

    return send_file(abs_path)


@app.route("/rebuild", methods=["POST"])
def rebuild():
    """Rebuild the index from the dataset directory."""
    global _index
    if not os.path.isdir(DATASET_DIR):
        return jsonify({ "error": f"Dataset directory not found: {DATASET_DIR}" }), 400
    try:
        _index = build_index(DATASET_DIR)
        return jsonify({ "indexed": len(_index) })
    except Exception as e:
        return jsonify({ "error": str(e) }), 500


# ── ENTRY POINT ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="CBIR Web App")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=5000)
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    print(f"\n[CBIR] Starting web app at http://{args.host}:{args.port}")
    print(f"[CBIR] Dataset directory: {DATASET_DIR}")
    print(f"[CBIR] Make sure to build the index first:\n       python search.py index --dir {DATASET_DIR}\n")

    app.run(host=args.host, port=args.port, debug=args.debug)

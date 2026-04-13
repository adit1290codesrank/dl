import os
import json
import struct
import subprocess
import tempfile
from pathlib import Path

from flask import Flask, request, render_template_string, jsonify
from PIL import Image
import numpy as np
import pillow_avif

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 8 * 1024 * 1024  # 8 MB upload limit

PREDICT_BINARY = "cifar_app.exe"
WEIGHTS_FILE   = "cifar10_weights.bin"

CIFAR_CLASSES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck"
]

CLASS_EMOJIS = {
    "airplane":    "✈️",
    "automobile":  "🚗",
    "bird":        "🐦",
    "cat":         "🐱",
    "deer":        "🦌",
    "dog":         "🐶",
    "frog":        "🐸",
    "horse":       "🐴",
    "ship":        "🚢",
    "truck":       "🚛",
}

HTML = """
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>CIFAR-10 Classifier</title>
<style>
  * { box-sizing: border-box; margin: 0; padding: 0; }

  body {
    font-family: 'Segoe UI', system-ui, sans-serif;
    background: #0f0f1a;
    color: #e0e0f0;
    min-height: 100vh;
    display: flex;
    flex-direction: column;
    align-items: center;
    padding: 40px 20px;
  }

  h1 {
    font-size: 2rem;
    font-weight: 700;
    background: linear-gradient(135deg, #7c6aff, #a855f7, #ec4899);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: 8px;
  }

  .subtitle {
    color: #888;
    font-size: 0.95rem;
    margin-bottom: 40px;
  }

  .card {
    background: #1a1a2e;
    border: 1px solid #2a2a45;
    border-radius: 16px;
    padding: 32px;
    width: 100%;
    max-width: 560px;
    box-shadow: 0 8px 32px rgba(0,0,0,0.4);
  }

  /* ---- Drop zone ---- */
  .drop-zone {
    border: 2px dashed #3a3a5e;
    border-radius: 12px;
    padding: 48px 24px;
    text-align: center;
    cursor: pointer;
    transition: border-color 0.2s, background 0.2s;
    position: relative;
  }
  .drop-zone:hover, .drop-zone.drag-over {
    border-color: #7c6aff;
    background: #1f1f38;
  }
  .drop-zone input[type=file] {
    position: absolute; inset: 0; opacity: 0; cursor: pointer;
  }
  .drop-icon { font-size: 3rem; display: block; margin-bottom: 12px; }
  .drop-label { color: #aaa; font-size: 0.95rem; }
  .drop-label span { color: #7c6aff; }

  /* ---- Preview ---- */
  #preview-wrap {
    display: none;
    margin-top: 20px;
    text-align: center;
  }
  #preview {
    width: 160px;
    height: 160px;
    object-fit: contain;
    border-radius: 10px;
    border: 1px solid #2a2a45;
    image-rendering: pixelated;
    background: #111;
  }

  /* ---- Button ---- */
  #classify-btn {
    display: none;
    width: 100%;
    margin-top: 20px;
    padding: 14px;
    background: linear-gradient(135deg, #7c6aff, #a855f7);
    color: white;
    font-size: 1rem;
    font-weight: 600;
    border: none;
    border-radius: 10px;
    cursor: pointer;
    transition: opacity 0.2s, transform 0.1s;
  }
  #classify-btn:hover { opacity: 0.9; }
  #classify-btn:active { transform: scale(0.98); }
  #classify-btn:disabled { opacity: 0.5; cursor: not-allowed; }

  /* ---- Results ---- */
  #results { display: none; margin-top: 28px; }

  .top-pred {
    background: linear-gradient(135deg, #2a1f4e, #1f1535);
    border: 1px solid #4a3a8a;
    border-radius: 12px;
    padding: 20px 24px;
    display: flex;
    align-items: center;
    gap: 16px;
    margin-bottom: 20px;
  }
  .top-emoji { font-size: 3rem; }
  .top-label { font-size: 1.5rem; font-weight: 700; text-transform: capitalize; }
  .top-conf  { color: #a78bfa; font-size: 1rem; margin-top: 2px; }

  .bar-list { display: flex; flex-direction: column; gap: 8px; }

  .bar-row {
    display: grid;
    grid-template-columns: 100px 1fr 52px;
    align-items: center;
    gap: 10px;
  }
  .bar-name {
    font-size: 0.82rem;
    text-transform: capitalize;
    color: #bbb;
    text-align: right;
  }
  .bar-track {
    background: #22223a;
    border-radius: 4px;
    height: 10px;
    overflow: hidden;
  }
  .bar-fill {
    height: 100%;
    border-radius: 4px;
    background: linear-gradient(90deg, #7c6aff, #a855f7);
    transition: width 0.6s ease;
  }
  .bar-fill.top { background: linear-gradient(90deg, #a855f7, #ec4899); }
  .bar-pct {
    font-size: 0.78rem;
    color: #888;
    text-align: right;
  }

  /* ---- Spinner ---- */
  .spinner {
    display: inline-block;
    width: 18px; height: 18px;
    border: 2px solid rgba(255,255,255,0.3);
    border-top-color: white;
    border-radius: 50%;
    animation: spin 0.7s linear infinite;
    vertical-align: middle;
    margin-right: 8px;
  }
  @keyframes spin { to { transform: rotate(360deg); } }

  .error-box {
    background: #2d1515;
    border: 1px solid #8b2222;
    color: #ff8888;
    border-radius: 10px;
    padding: 14px 18px;
    margin-top: 16px;
    font-size: 0.9rem;
  }
</style>
</head>
<body>

<h1>CIFAR-10 Classifier</h1>
<p class="subtitle">Powered by your custom CUDA neural network</p>

<div class="card">
  <div class="drop-zone" id="drop-zone">
    <input type="file" id="file-input" accept="image/*">
    <span class="drop-icon">🖼️</span>
    <p class="drop-label">Drop an image here or <span>click to browse</span></p>
    <p class="drop-label" style="font-size:0.8rem; margin-top:6px; color:#555;">
      jpg, png, webp, bmp — any size
    </p>
  </div>

  <div id="preview-wrap">
    <img id="preview" alt="Preview">
    <p style="color:#555; font-size:0.78rem; margin-top:6px;">Resized to 32×32 for the model</p>
  </div>

  <button id="classify-btn">Classify Image</button>

  <div id="results">
    <div class="top-pred" id="top-pred"></div>
    <div class="bar-list" id="bar-list"></div>
  </div>
</div>

<script>
  const dropZone    = document.getElementById('drop-zone');
  const fileInput   = document.getElementById('file-input');
  const preview     = document.getElementById('preview');
  const previewWrap = document.getElementById('preview-wrap');
  const classifyBtn = document.getElementById('classify-btn');
  const resultsDiv  = document.getElementById('results');
  const topPred     = document.getElementById('top-pred');
  const barList     = document.getElementById('bar-list');

  let selectedFile = null;

  // Drag-and-drop styling
  dropZone.addEventListener('dragover', e => { e.preventDefault(); dropZone.classList.add('drag-over'); });
  dropZone.addEventListener('dragleave', () => dropZone.classList.remove('drag-over'));
  dropZone.addEventListener('drop', e => {
    e.preventDefault();
    dropZone.classList.remove('drag-over');
    if (e.dataTransfer.files[0]) handleFile(e.dataTransfer.files[0]);
  });

  fileInput.addEventListener('change', () => {
    if (fileInput.files[0]) handleFile(fileInput.files[0]);
  });

  function handleFile(file) {
    selectedFile = file;
    const url = URL.createObjectURL(file);
    preview.src = url;
    previewWrap.style.display = 'block';
    classifyBtn.style.display = 'block';
    resultsDiv.style.display  = 'none';
  }

  classifyBtn.addEventListener('click', async () => {
    if (!selectedFile) return;

    classifyBtn.disabled = true;
    classifyBtn.innerHTML = '<span class="spinner"></span>Running inference…';
    resultsDiv.style.display = 'none';

    const formData = new FormData();
    formData.append('image', selectedFile);

    try {
      const res  = await fetch('/classify', { method: 'POST', body: formData });
      const data = await res.json();

      if (data.error) {
        showError(data.error);
      } else {
        showResults(data);
      }
    } catch (err) {
      showError('Network error: ' + err.message);
    } finally {
      classifyBtn.disabled = false;
      classifyBtn.innerHTML = 'Classify Image';
    }
  });

  const emojis = {
    airplane:"✈️", automobile:"🚗", bird:"🐦", cat:"🐱",
    deer:"🦌", dog:"🐶", frog:"🐸", horse:"🐴", ship:"🚢", truck:"🚛"
  };

  function showResults(data) {
    // Sort by probability descending
    const sorted = Object.entries(data.probabilities)
      .sort((a, b) => b[1] - a[1]);

    const [topClass, topProb] = sorted[0];

    topPred.innerHTML = `
      <span class="top-emoji">${emojis[topClass] || '❓'}</span>
      <div>
        <div class="top-label">${topClass}</div>
        <div class="top-conf">${(topProb * 100).toFixed(1)}% confidence</div>
      </div>
    `;

    barList.innerHTML = sorted.map(([cls, prob], i) => `
      <div class="bar-row">
        <span class="bar-name">${cls}</span>
        <div class="bar-track">
          <div class="bar-fill ${i === 0 ? 'top' : ''}" style="width:${(prob*100).toFixed(1)}%"></div>
        </div>
        <span class="bar-pct">${(prob*100).toFixed(1)}%</span>
      </div>
    `).join('');

    resultsDiv.style.display = 'block';
  }

  function showError(msg) {
    resultsDiv.innerHTML = `<div class="error-box">⚠️ ${msg}</div>`;
    resultsDiv.style.display = 'block';
  }
</script>
</body>
</html>
"""


def preprocess_image(pil_img: Image.Image) -> bytes:
    """
    Resize to 32×32, convert to HWC float32, normalize to [0, 1].
    Returns raw bytes suitable for writing to a .bin file and passing to ./predict.
    """
    img = pil_img.convert("RGB").resize((32, 32), Image.BILINEAR)
    arr = np.array(img, dtype=np.float32) / 255.0   # shape: (32, 32, 3), HWC
    return arr.tobytes()


def run_predict(bin_path: str) -> dict:
    """
    Calls the ./predict binary and parses its JSON output.
    Returns a dict of {class_name: probability}.
    """
    if not Path(PREDICT_BINARY).exists():
        raise FileNotFoundError(
            f"Predict binary '{PREDICT_BINARY}' not found. "
            "Run: make predict"
        )
    if not Path(WEIGHTS_FILE).exists():
        raise FileNotFoundError(
            f"Weights file '{WEIGHTS_FILE}' not found. "
            "Train the model first with: make cifar && ./cifar"
        )

    binary_path = str(Path(PREDICT_BINARY).resolve())
    result = subprocess.run(
        [binary_path, bin_path],
        capture_output=True,
        text=True,
        timeout=30,
        cwd=str(Path(__file__).parent),
    )

    if result.returncode != 0:
        raise RuntimeError(f"predict binary failed:\n{result.stderr.strip()}")

    # Parse JSON line from stdout (ignore any other log lines)
    for line in result.stdout.splitlines():
        line = line.strip()
        if line.startswith("{"):
            return json.loads(line)

    raise RuntimeError(f"No JSON output from predict binary. stdout:\n{result.stdout}")


@app.route("/")
def index():
    return render_template_string(HTML)


@app.route("/classify", methods=["POST"])
def classify():
    if "image" not in request.files:
        return jsonify({"error": "No image uploaded."}), 400

    file = request.files["image"]
    if file.filename == "":
        return jsonify({"error": "Empty filename."}), 400

    try:
        file.stream.seek(0)
        img = Image.open(file.stream)
    except Exception as e:
        return jsonify({"error": f"Could not open image: {e}"}), 400

    raw_bytes = preprocess_image(img)

    # Write to a temp file so the C++ binary can read it
    with tempfile.NamedTemporaryFile(suffix=".bin", delete=False) as tmp:
        tmp.write(raw_bytes)
        tmp_path = tmp.name

    try:
        probs = run_predict(tmp_path)
        top_class = max(probs, key=probs.get)
        return jsonify({
            "top_class":     top_class,
            "top_prob":      probs[top_class],
            "probabilities": probs,
        })
    except (FileNotFoundError, RuntimeError) as e:
        return jsonify({"error": str(e)}), 500
    finally:
        os.unlink(tmp_path)


if __name__ == "__main__":
    print("CIFAR-10 Classifier running at http://localhost:5000")
    app.run(host="0.0.0.0", port=5000, debug=False)
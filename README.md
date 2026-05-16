# 🔍 CBIR — Content-Based Image Retrieval

> A Python system that finds visually similar images using color histograms (HSV) and texture descriptors (ORB), with a web interface built with Flask.

**Author:** Mounir Bekkar · Master Informatique · Université Lumière Lyon 2  
**GitHub:** [github.com/mbekkar](https://github.com/mbekkar)

---

## 📸 Features

| Feature | Details |
|---------|---------|
| **Color features** | HSV histogram — 8×8×8 bins = 512-dim vector |
| **Texture features** | ORB descriptors — mean of keypoints = 32-dim vector |
| **Similarity metric** | Cosine similarity (scikit-learn) |
| **Total feature size** | 544-dimensional vector per image |
| **CLI** | Index, search, and evaluate from the terminal |
| **Web app** | Flask interface with drag & drop upload |
| **Tests** | 20+ unit & integration tests with pytest |

---

## 🗂️ Project Structure

```
cbir/
├── cbir_engine.py       # Core: feature extraction, indexing, search
├── search.py            # CLI: index / search / eval commands
├── app.py               # Flask web interface
├── generate_dataset.py  # Generate a synthetic test dataset
├── tests.py             # Unit & integration tests
├── requirements.txt     # Python dependencies
└── README.md
```

---

## 🚀 Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Generate a test dataset (or use your own images)

```bash
python generate_dataset.py
# Creates ./dataset/ with 50 synthetic images (5 color categories × 10 images)
```

Or use your own dataset — just put images in a folder:
```
my_dataset/
├── dogs/
│   ├── dog_001.jpg
│   ├── dog_002.jpg
├── cats/
│   ├── cat_001.jpg
```

### 3. Build the index

```bash
python search.py index --dir ./dataset
# Extracts HSV + ORB features for every image
# Saves index to cbir_index.pkl
```

### 4. Search from the terminal

```bash
python search.py search --query ./dataset/red/red_001.jpg --top 5
```

Output:
```
#1     0.9821  ./dataset/red/red_002.jpg
       [████████████████████] Très similaire
#2     0.9745  ./dataset/red/red_003.jpg
       [███████████████████ ] Très similaire
#3     0.8123  ./dataset/red/red_005.jpg
       [████████████████    ] Similaire
```

### 5. Launch the web interface

```bash
python app.py
# Open http://127.0.0.1:5000 in your browser
```

---

## 🌐 Web Interface

The Flask web app lets you:
- **Drag & drop** or click to upload any image
- **Choose** how many results to display (1–20)
- **View** matched images with similarity scores and color-coded labels
- **See** index status in the navbar

![CBIR Web Interface](https://via.placeholder.com/800x400/0a0f1e/63b3ed?text=CBIR+Web+Interface)

---

## 📟 CLI Reference

```bash
# Build index from a dataset folder
python search.py index --dir ./dataset [--save cbir_index.pkl]

# Search for similar images
python search.py search --query ./my_photo.jpg [--top 5] [--index cbir_index.pkl]

# Evaluate Precision@k (requires filename format: label_NNN.jpg)
python search.py eval [--k 5] [--index cbir_index.pkl]

# Launch web app
python app.py [--host 127.0.0.1] [--port 5000] [--debug]
```

---

## 🧪 Running Tests

```bash
pytest tests.py -v
```

Expected output:
```
test_shape .................................................... PASSED
test_normalized ............................................... PASSED
test_different_images_differ .................................. PASSED
test_build_count .............................................. PASSED
test_scores_descending ........................................ PASSED
test_same_color_ranks_first ................................... PASSED
test_full_pipeline ............................................ PASSED
... (20+ tests)
```

---

## 🧠 How It Works

```
Query Image
    │
    ▼
┌─────────────────────────────────────┐
│  Feature Extraction (544-dim)       │
│  ┌────────────────┐  ┌───────────┐  │
│  │ HSV Histogram  │  │ ORB Mean  │  │
│  │   (512-dim)    │  │ (32-dim)  │  │
│  └────────────────┘  └───────────┘  │
│           └──────────┬──────────┘   │
│                   concat            │
└─────────────────────────────────────┘
    │
    ▼
Cosine Similarity vs all indexed images
    │
    ▼
Top-K most similar images (ranked)
```

### Feature details

**HSV Color Histogram:**
- Converts image to HSV color space
- Computes 3D histogram (8 × 8 × 8 bins)
- Normalizes to [0, 1]
- 512-dimensional vector per image

**ORB Texture Descriptor:**
- Detects up to 100 keypoints
- Computes BRIEF binary descriptor for each
- Returns the **mean** descriptor → 32-dim vector
- Returns zeros if no keypoints (uniform image)

**Cosine Similarity:**
```
similarity(A, B) = (A · B) / (||A|| × ||B||)
```
- Range: [0.0, 1.0]
- 1.0 = identical features
- 0.0 = completely different

---

## 📊 Precision@k Evaluation

If your dataset images are named with their label prefix (`cat_001.jpg`, `dog_002.jpg`), you can evaluate automatically:

```bash
python search.py eval --k 5
# → Mean Precision@5 = 0.8400 (84.0%)
```

This tests every image in the index as a query and measures how many of the top-5 results share the same label.

---

## 🔧 Using Your Own Dataset

Any folder of images works:

```bash
python search.py index --dir /path/to/my/photos
python search.py search --query /path/to/query.jpg --top 10
```

Supported formats: `.jpg` `.jpeg` `.png` `.bmp` `.webp`

---

## 🚀 Possible Improvements

- [ ] Deep features using a pre-trained CNN (ResNet, EfficientNet)
- [ ] FAISS index for fast nearest-neighbor search on large datasets
- [ ] Spatial verification with RANSAC for geometric matching
- [ ] mAP (mean Average Precision) evaluation metric
- [ ] Docker containerization

---

## 📄 License

MIT License — free to use and modify.

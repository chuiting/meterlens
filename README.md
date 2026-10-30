# MeterLens - Automated Meter Reading using Pattern Recognition

![Python](https://img.shields.io/badge/python-3.11-blue) ![TensorFlow](https://img.shields.io/badge/TensorFlow-2.16-orange) ![YOLOv8](https://img.shields.io/badge/YOLOv8-ultralytics-purple) ![FastAPI](https://img.shields.io/badge/FastAPI-backend-009688)

MeterLens is an automated meter reading system built with pattern recognition techniques. It reads the 5-digit value from a photo of a utility meter and returns the reading with a confidence score and a decision: **APPROVED** if it can be accepted automatically, or **FLAGGED** for human review, with the reasons. It is trained on the UFPR-AMR dataset of 2,000 meter photos, using a YOLOv8 detector to locate the counter and an ensemble of CNNs and an SVM to read the digits.

This project was developed for the Practice Module of the NUS-ISS Graduate Certificate in Pattern Recognition Systems.

**Demo:** [Web App](https://meterlens-776165799141.asia-southeast1.run.app/app/) | [Backend Portal](https://meterlens-776165799141.asia-southeast1.run.app)

> **Note:** The app demo runs on Google Cloud Run within the free usage tier, so the server sleeps when idle. Expect the first load to take about a minute while it starts up. To keep hosting costs down, the demo may also be switched off at times. If the app is not accessible, run it locally (see [Getting started](#3-getting-started)) or contact the author.

---

## 1. Features

- **End-to-end reading:** finds the counter in a full photo, reads all five digits, and scores its own confidence.
- **Ensemble of models:** two CNNs trained from scratch plus a HOG + SVM classifier, combined by confidence-weighted fusion.
- **Image-quality gate:** flags unusually dark or overexposed photos before recognition, so the user can retake them.
- **Human in the loop:** low-confidence readings are flagged for review, and every confirmation or correction is logged.
- **Two web apps:** a phone-style app for meter readers and an Inspector portal that shows every stage of the pipeline for one photo.

## 2. How it works

```
photo
  → YOLOv8 counter detection          (fine-tuned from COCO)
  → image-quality gate                (PCA + KMeans distance)
  → digit recognition                 (CNN ×2 + HOG/SVM)
  → confidence-weighted fusion        (per digit)
  → decision layer                    (confidence threshold + quality flag)
  → APPROVED / FLAGGED
```

| Phase | Stage | Method | Script |
|---|---|---|---|
| 1 | Data preparation | Parse annotations; build detection and recognition sets; augmentation | `src/parse.py`, `src/prepare_*.py`, `src/augment.py` |
| 2 | Counter detection | YOLOv8n fine-tuned from COCO (transfer learning) | `src/train_yolo.py` |
| 3 | Digit recognition | Multi-task CNN trained from scratch, one softmax head per digit | `src/train_cnn.py` |
| 4 | Digit recognition | HOG features + SVM | `src/train_svm.py` |
| 5 | Digit recognition | HOG features + Random Forest (comparison model) | `src/train_rf.py` |
| 6 | Fusion | Per-digit probabilities combined, weighted by validation accuracy | `src/fusion.py` |
| 7 | Image-quality gate | PCA + KMeans distance on the counter crop | `src/sensing.py` |
| 8 | Decision | Confidence threshold tuned on validation, plus the quality flag | `src/decision.py` |
| 9 | Evaluation | Full test-set evaluation and failure analysis | `src/evaluate.py` |
| 10 | Web app | FastAPI backend, phone app and Inspector portal | `api/`, `web/` |

---

## 3. Getting started

### 3.1 Prerequisites

- Python 3.11
- About 4 GB of free disk space (TensorFlow, PyTorch and the models)
- The two CNN weight files (`cnn_gray.h5`, `cnn_rgb.h5`, 245 MB each). They are too large for GitHub and are hosted in a private Hugging Face repository. Please contact the author for an access key. Every other model is included in `models/`.

### 3.2 Installation

```bash
git clone https://github.com/chuiting/prs-meterlens.git
cd prs-meterlens
python3.11 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 3.3 Download the models and dataset

Both Hugging Face repositories are private. Replace `<access key>` with the key from the author (see [Prerequisites](#31-prerequisites)):

```bash
# CNN models -> models/  (required to run the app)
hf download monsterlum/meterlens-weights cnn_gray.h5 cnn_rgb.h5 --local-dir models --token <access key>

# UFPR-AMR photos -> data/raw/  (optional, only for retraining and the Inspector's dataset view)
hf download monsterlum/meterlens-ufpr-amr --repo-type dataset --local-dir data/raw --token <access key>
```

### 3.4 Run the app

```bash
uvicorn api.main:app --host 0.0.0.0 --port 8000
```

| URL | What it is |
|---|---|
| `http://localhost:8000` | Inspector portal |
| `http://localhost:8000/app` | Phone app (from a phone on the same Wi-Fi, use your computer's IP) |
| `http://localhost:8000/docs` | Interactive API documentation |

### 3.5 Use the API

```bash
curl -F "file=@web/samples/meter1212.jpg" http://localhost:8000/predict
```

The response contains the reading, the confidence score, the decision, and the reasons for any flag.

| Endpoint | Purpose |
|---|---|
| `POST /predict` | Photo → reading, confidence, decision and reasons |
| `POST /predict/check`, `POST /predict/recognize` | The same in two steps, so the app can ask before recognising a flagged photo |
| `GET/POST /history`, `PATCH/DELETE /history/{id}` | Saved readings and feedback |
| `GET /health` | Loaded models, fusion weights, threshold |

---

## 4. Reproducing the results

To retrain everything from scratch, download the UFPR-AMR dataset into `data/raw/` (see [Download the models and dataset](#33-download-the-models-and-dataset)) and run the scripts in phase order:

```
python src/parse.py                  # results/data/dataset_details.csv
python src/prepare_yolo.py           # data/yolo/
python src/prepare_recognition.py    # data/recognition/
python src/augment.py                # results/data/augment_demo.jpg (preview only)
python src/train_yolo.py             # models/yolo.pt, results/yolo/            (GPU recommended; ~1 h on a laptop)
python src/train_cnn.py --start-from +digit_permutation   # On Colab, see below; models/cnn_gray.h5, results/cnn/
python src/train_svm.py              # models/svm*.pkl, results/svm/            (~5 min, builds the detector-crop cache)
python src/train_rf.py               # models/rf*.pkl, results/rf/
python src/fusion.py                 # results/fusion/, writes fusion weights into config.yaml
python src/sensing.py                # models/pca.pkl, models/kmeans.pkl, results/sensing/, writes the threshold into config.yaml
python src/decision.py               # results/decision/, writes the threshold into config.yaml
python src/evaluate.py               # results/evaluation/, plus every Models-page figure (~4 min)
uvicorn api.main:app --host 0.0.0.0 --port 8000     # Inspector at http://localhost:8000, phone app at /app
```

The second CNN, `models/cnn_rgb.h5`, is an earlier training run kept as a fusion member; its weights and probability files are provided rather than rebuilt.

### 4.1 Training the CNN on Colab

The CNN is too large to train on an 8 GB laptop, so it is trained on Colab:

1. Zip `src/`, `config.yaml`, `requirements.txt` and `data/recognition/` (without `digits/`) and upload it to Colab.
2. Run `notebooks/03_train_cnn_colab.ipynb`. It unzips the files and runs `train_cnn.py --start-from +digit_permutation`, which takes about 35 minutes on a Colab GPU. The first five ablation steps are reused from `results/cnn/cnn_ablation.json`; drop `--start-from` to retrain all seven.
3. Copy `models/cnn_gray.h5` and `results/cnn/cnn_*` back before the runtime is recycled.

The notebook keeps the log of the run that produced the deployed model.

### 4.2 Notebooks

One notebook per phase. Each reads the saved results from the scripts to show that phase's metrics, figures and failure cases; nothing is retrained.

| Notebook | Phase |
|---|---|
| `01_data_preparation.ipynb` | Data preparation |
| `02_yolo_detection.ipynb` | Counter detection (includes a "try it yourself" cell) |
| `03_cnn.ipynb` | Multi-task CNN |
| `03_train_cnn_colab.ipynb` | Colab launcher that trains the CNN (not a results notebook) |
| `04_svm.ipynb` | HOG + SVM |
| `05_random_forest.ipynb` | HOG + Random Forest |
| `06_fusion.ipynb` | Fusion |
| `07_intelligent_sensing.ipynb` | Image-quality gate (includes a "try it yourself" cell) |
| `08_decision_layer.ipynb` | Decision layer |
| `09_evaluation.ipynb` | Evaluation |

---

## 5. Hosting

The live demo runs on Google Cloud Run as a single Docker container (`Dockerfile`), so the system can be tried in a browser without installing anything. Saved readings are not kept between restarts.

---

## 6. Project structure

```
prs-meterlens/
├── config.yaml        # Paths, hyperparameters, thresholds
├── requirements.txt
├── Dockerfile         # Cloud Run container
├── src/               # One script per phase (see section 2), plus:
│   ├── pipeline.py      # Inference pipeline used by evaluation and the web app
│   └── trace.py         # Stage-by-stage trace for the Inspector
├── api/               # FastAPI backend
├── web/
│   ├── app/             # Phone app, served at /app
│   ├── inspector/       # Inspector portal, served at /
│   └── samples/         # Sample photos
├── notebooks/         # One notebook per phase (see section 4.2)
├── models/            # Trained models (CNN weights downloaded separately)
├── results/           # Metrics and figures, one folder per phase
├── data/              # Dataset and training sets (not in git)
└── storage/           # Web-app history (not in git)
```

## 7. Tech stack

- **Detection:** Ultralytics YOLOv8
- **Recognition:** TensorFlow / Keras, scikit-learn, scikit-image (HOG)
- **Image processing:** OpenCV, Pillow
- **Backend:** FastAPI, Uvicorn, SQLite
- **Frontend:** plain HTML, CSS and JavaScript (no build step)
- **Deployment:** Docker, Google Cloud Run, Hugging Face Hub (model hosting)

## 8. Dataset and acknowledgements

This project uses the [UFPR-AMR dataset](https://web.inf.ufpr.br/vri/databases/ufpr-amr/): 2,000 meter photos, split 800 / 400 / 800 for training, validation and testing. The images are not included in this repository; request them from the dataset authors.

> R. Laroca, V. Barroso, M. A. Diniz, G. R. Gonçalves, W. R. Schwartz, D. Menotti. *Convolutional Neural Networks for Automatic Meter Reading.* Journal of Electronic Imaging, 28(1), 2019.

# 🏀 Basketball Player Tracker

Tracks players across a basketball game video using Roboflow models + SAM2 segmentation. Produces:

- **Player masks** with team colors and `#NUMBER Name` labels
- **Top-down court minimap** showing real-time player positions
- **Shot detection** (jump shots, layups, dunks, made/miss)

---

## Setup

### 1. Clone the repo

```bash
git clone https://github.com/YOUR_USERNAME/basketball-tracker.git
cd basketball-tracker
```

### 2. Install SAM2

```bash
git clone https://github.com/Gy920/segment-anything-2-real-time.git
cd segment-anything-2-real-time
pip install -e . -q
python setup.py build_ext --inplace
cd checkpoints && bash download_ckpts.sh
cd ../..
```

### 3. Install Python dependencies

```bash
pip install -r requirements.txt
pip install git+https://github.com/roboflow/sports.git@feat/basketball
```

### 4. Set up API keys

Copy `.env.example` to `.env` and fill in your keys (both are free):

```bash
cp .env.example .env
```

- **Roboflow API key** → [roboflow.com](https://roboflow.com)
- **Hugging Face token** → [huggingface.co](https://huggingface.co)

### 5. Add your video

Place your game clip inside a `source/` folder:

```
source/
└── game.mp4
```

### 6. Configure teams & rosters

Open `config.py` and update:

- `SOURCE_VIDEO_FILENAME` — your video filename
- `TEAM_NAMES` / `TEAM_COLORS` — your two teams
- `TEAM_ROSTERS` — jersey numbers → player last names

> **Tip:** Run with `--mode track` first, then inspect the team clustering output to confirm cluster 0 and 1 are mapped to the correct teams.

---

## Usage

```bash
# Run all three pipelines
python main.py --video source/game.mp4

# Run a specific pipeline
python main.py --video source/game.mp4 --mode track   # player masks + labels
python main.py --video source/game.mp4 --mode map     # court minimap
python main.py --video source/game.mp4 --mode shots   # shot detection
```

### Output files

| File | Description |
|------|-------------|
| `game-result-compressed.mp4` | Annotated video with player masks and labels |
| `game-map-compressed.mp4` | Top-down court minimap |
| Shot events printed to stdout | Frame index + event type |

---

## Models Used

| Model | Source | Purpose |
|-------|--------|---------|
| RF-DETR (basketball-player-detection) | Roboflow | Detects players, jersey numbers, ball, shot events |
| SAM2 Real-Time | Meta / Gy920 fork | Segments and tracks players frame-by-frame |
| SmolVLM2 (basketball-jersey-numbers-ocr) | Roboflow | Reads jersey numbers from crops |
| Court keypoint model | Roboflow | Detects court landmarks for homography |

---

## Requirements

- Python 3.10+
- CUDA-capable GPU (recommended: 16GB+ VRAM)
- ffmpeg installed on your system

# 🐉 BDH Interpretability Suite

**The Definitive Explorer for Baby Dragon Hatchling Architecture**

An interactive visualization and interpretability platform for the BDH (Baby Dragon Hatchling) post-transformer architecture. Built for the KRITI 2026 AI Interpretability Challenge.

## 🎯 What This Project Does

This suite makes BDH's unique properties **visible and explorable**:

- **~95% Sparse Activations**: Watch neurons selectively fire (vs transformers where ~95% activate)
- **Monosemantic Synapses**: Individual synapses encode specific concepts (currencies, countries, languages)
- **Hebbian Learning**: Memory forms during inference without backpropagation
- **Model Merging**: Combine separately trained specialists into a unified polyglot

## 🏗️ Architecture

```
bdh-interpretability/
├── training/                 # Model training pipeline
│   ├── bdh.py               # BDH architecture (from official repo)
│   ├── train.py             # Training script with extraction hooks
│   ├── download_europarl.py # Dataset downloader
│   └── configs/             # Training configurations
├── extraction/              # Activation extraction pipeline
│   ├── hooks.py             # PyTorch hooks for capturing internals
│   ├── extractor.py         # Main extraction service
│   └── exporters.py         # JSON export for frontend playback
├── analysis/                # Interpretability analysis
│   ├── monosemanticity.py   # Concept probing and synapse discovery
│   ├── sparsity.py          # Sparsity measurement
│   ├── topology.py          # Graph extraction from weights
│   └── merge.py             # Model merging utilities
├── backend/                 # FastAPI server
│   ├── main.py              # API entry point
│   ├── routes/              # API endpoints
│   └── services/            # Business logic
├── frontend/                # React + TypeScript + Vite
│   ├── src/
│   │   ├── components/      # Reusable UI components
│   │   ├── features/        # Feature-specific modules
│   │   │   ├── architecture/    # Interactive architecture diagram
│   │   │   ├── sparsity/        # Sparse brain comparator
│   │   │   ├── topology/        # Graph brain explorer
│   │   │   ├── hebbian/         # Learning animator
│   │   │   └── monosemanticity/ # Concept dashboard
│   │   ├── stores/          # State management
│   │   └── utils/           # Helpers
│   └── public/
│       └── playback/        # Pre-computed JSON for offline mode
└── scripts/                 # Utility scripts
    └── generate_playback.py # Generate JSON playback data
```

## 🚀 Quick Start

### Prerequisites

- Python 3.10+
- Node.js 18+
- CUDA-capable GPU (for training) or Google Colab Pro
- ~10GB disk space for datasets

### 1. Setup Python Environment

```bash
cd bdh-interpretability
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Download Europarl Dataset

```bash
python training/download_europarl.py --languages en-fr en-pt
```

### 3. Train Models (or use pre-trained)

```bash
# Train French specialist
python training/train.py --config training/configs/french.yaml

# Train Portuguese specialist
python training/train.py --config training/configs/portuguese.yaml

# Merge models
python analysis/merge.py --model1 checkpoints/french.pt --model2 checkpoints/portuguese.pt
```

### 4. Generate Playback Data

```bash
python scripts/generate_playback.py --model checkpoints/french.pt --output frontend/public/playback/
```

### 5. Start Backend

```bash
# Run from the project root (NOT from backend/)
uvicorn backend.main:app --reload --port 8000
```

### 6. Start Frontend

```bash
cd frontend
npm install
npm run dev
```

Visit `http://localhost:5173` to explore!

## 📊 Features

### Path A: Visualization

| Component                | Description                                   | Status |
| ------------------------ | --------------------------------------------- | ------ |
| Interactive Architecture | Animated BDH diagram with live data flow      | ✅     |
| Sparse Brain             | Side-by-side BDH vs Transformer comparison    | ✅     |
| Graph Brain              | 3D topology explorer with merge visualization | ✅     |
| Hebbian Animator         | Watch memory form token-by-token              | ✅     |

### Path B: Interpretability

| Component                 | Description                             | Status |
| ------------------------- | --------------------------------------- | ------ |
| Monosemanticity Dashboard | Discover concept-specific synapses      | ✅     |
| Synapse Genealogy         | Track synapse origins post-merge        | ✅     |
| Comparative Study         | Quantitative BDH vs Transformer metrics | ✅     |

### Path C: Frontier

| Component       | Description                             | Status |
| --------------- | --------------------------------------- | ------ |
| Model Merging   | Combine French + Portuguese specialists | ✅     |
| Benchmark Suite | BDH-specific evaluation tasks           | ✅     |

## 🎮 Playback Mode

The frontend works without a running backend by loading pre-computed JSON data. This enables:

- Smooth 60fps animations with 32k+ neurons
- Offline demos and presentations
- Fast loading for judges reviewing submissions

## 📚 Key Insights Demonstrated

1. **Sparsity is Architectural**: BDH achieves ~95% sparsity through ReLU after expansion, not regularization
2. **Monosemanticity is Real**: Individual synapses consistently encode specific semantic concepts
3. **Merging Works**: Separately trained models combine without fine-tuning
4. **Hebbian Learning**: Memory forms during inference via synaptic co-activation

## 🔗 Links

- [BDH Paper](https://arxiv.org/abs/2509.26507)
- [Official BDH Repository](https://github.com/pathwaycom/bdh)
- [KRITI 2026 Challenge](https://kriti.org)
- [Live Demo](https://huggingface.co/spaces/YOUR_TEAM/bdh-explorer)
- [Demo Video](https://youtube.com/watch?v=YOUR_VIDEO)

## 👥 Team

- [Your Name] - [Role]
- [Teammate] - [Role]

## 📄 License

MIT License - See LICENSE file

## 🙏 Acknowledgments

- Pathway Research for the BDH architecture
- Andrej Karpathy for nanoGPT inspiration
- The Distill.pub team for visualization philosophy

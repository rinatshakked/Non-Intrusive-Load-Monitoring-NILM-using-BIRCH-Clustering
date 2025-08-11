# Non-Intrusive Load Monitoring (NILM) using BIRCH Clustering

This project implements a **semi-supervised Non-Intrusive Load Monitoring (NILM)** pipeline for smart grids.
Using only **active (P)** and **reactive (Q)** power measurements from a single-phase smart meter, the system detects
ON/OFF events of electrical appliances, extracts features, and clusters them (with **BIRCH**) to identify devices in real time.

> Final B.Sc. project — Tel Aviv University, Energy Conversion Lab.  
> Authors: **Rinat Feldhuhn** & **Yuval Mandelbaum** · Advisor: **Dr. Yuval Beck**

---

The algorithm:
1. Detects power events from the aggregated signal.
2. Extracts active/reactive power deltas as unique appliance signatures.
3. Clusters events using the **BIRCH** algorithm.
4. Learns connected appliances over time and provides real-time usage statistics, including estimated energy consumption and cost per device.

The system was developed as part of our final B.Sc. project at Tel Aviv University, using data from a SATEC smart meter in the Energy Conversion Lab.


## ✨ Key Features
- **Unsupervised**: no labeled per-appliance data required.
- **Real-time Event detection** from the aggregated power signal.
- **Feature extraction**: ΔP and ΔQ signatures per event.
- **Clustering with BIRCH** to form appliance groups.
- **Interactive labeling** on first sighting of a new appliance (learns over time).
- **Usage analytics**: estimated energy (kWh) & cost per device and total.

---

## 🧰 Tech Stack
- Python 3.9+
- NumPy, scikit-learn, Matplotlib

---

## 📦 Installation
```bash
# (optional) create & activate a virtual environment
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS/Linux:
# source .venv/bin/activate

# install dependencies
pip install -r requirements.txt
```

---

## ▶️ Usage
1. Place your measurement files in the project root:
   - `KW.csv`  – Active power (kW)
   - `KVAR.csv` – Reactive power (kVAr)
2. Run:
   ```bash
   python main.py
   ```
3. Follow the on-screen prompts to **label** newly detected appliances when the system forms a new cluster.

> The script will also generate plots for the event detection process and the clustering space (ΔP vs. ΔQ).

---

## 📂 Project Structure
```
├── main.py                 # NILM pipeline with event detection, feature extraction, BIRCH clustering & CLI
├── README.md               # This file
├── requirements.txt        # Python dependencies
├── .gitignore              # Ignore caches, envs, large local data, plots
├── KW.csv                  # (user-provided) active power measurements - excluded from Git
└── KVAR.csv                # (user-provided) reactive power measurements - excluded from Git
```

---

## ⚙️ Configuration
You can tweak a few core parameters inside `main.py`:
- `startval` (default: 30): bootstrap window (samples) for the first clustering fit
- `jump_val` (default: 21): window advance per iteration (trade-off: latency vs. resolution)
- `tskip` (default: 5): guard samples to reframe events near window edges
- `Birch(..., threshold=0.1, branching_factor=50)`: cluster granularity & CF-tree shape

---

## ✅ Requirements
See `requirements.txt`. Tested with:
- numpy ≥ 1.24
- scikit-learn ≥ 1.2
- matplotlib ≥ 3.7

---

## 🔎 Notes & Limitations
- Designed for **low-rate sampling (~1 Hz)** NILM scenarios.
- Assumes **no two events occur within 1 second**.
- Best performance after a short **learning period** while appliances are toggled at least once.
- Current version targets appliances with relatively **stable signatures**; future work can add richer features (e.g., harmonics) and advanced models for variable loads.

---

## 🚀 Roadmap / Next Steps

- [ ] **Code refactoring** — split `main.py` into separate modules (event detection, feature extraction, clustering, analytics, CLI).
- [ ] **Unit tests** for event detection and clustering.
- [ ] **Config file support** — move parameters (`startval`, `jump_val`, etc.) to YAML/JSON for easier tuning.
- [ ] **Data input abstraction** — support direct smart meter streaming, not just CSV.
- [ ] **Advanced features** — add harmonic analysis, appliance state tracking, and variable-load disaggregation.
- [ ] **UI improvements** — replace CLI prompts with GUI or web dashboard.
- [ ] **Deployment** — package as Python module or Docker image for easier use.

---

## 📚 Background
The approach follows classic NILM ideas (event-based disaggregation) and uses **BIRCH** for incremental, memory-aware clustering suitable for data streams.

For additional context, see the accompanying report: *Final Project - NILM - Rinat and Yuval (PDF)*.

---

## 📝 License
For now this repository is **private** and provided **for academic/educational use** by the authors.  
If you plan to open-source it later, consider adding an OSI license (e.g., MIT/BSD-3-Clause) or keep “All rights reserved”.

---

## 🙌 Credits
- **Rinat Feldhuhn** & **Yuval Mandelbaum** — Authors
- **Dr. Yuval Beck** — Advisor, Tel Aviv University

# Training Guide (Beginner Friendly)

Think of this project like **teaching a student to recognize cow behaviors** from sensor readings (movement, temperature, location, etc.).

## What is in the `models/` folder?

| File | What it is | Keep? |
|------|------------|-------|
| `behavior_master.joblib` | **Main brain** — XGBoost fusion model (API uses this) | ✅ Yes |
| `behavior_master_features.json` | List of sensor columns the main model expects | ✅ Yes |
| `behavior_master_label_encoder.json` | Maps behavior numbers → class IDs | ✅ Yes |
| `behavior_deep.pth` | **Backup brain** — deep learning (CNN+LSTM) | ✅ Yes |
| `behavior_deep_meta.json` | Settings for the deep model (feature count, etc.) | ✅ Yes |
| `report_master.txt` | How well the main model did (scores per class) | ✅ Yes |
| `confusion_matrix_master.png` | Picture: which behaviors get confused | ✅ Yes |
| `feature_importance_master.png` | Which sensors matter most | ✅ Yes |
| `report_deep.txt` / `confusion_matrix_deep.png` | Same for deep model | ✅ After DL training |
| `report_ensemble.txt` | Combined master + deep score | ✅ After ensemble eval |

**Removed (old experiments):** `behavior_cbt.joblib`, `behavior_immu.joblib`, etc. — one model per sensor. The project now uses **one master model** instead.

---

## Simple metrics (what to look at)

- **Accuracy** — % of correct guesses. Easy to read but **misleading** if some behaviors are rare.
- **F1 macro** — fair average across *all* behaviors. **Use this as your main goal.**
- **Precision** — when the model says "Walking", how often is it right?
- **Recall** — of all real "Walking" moments, how many did we catch?

Open `models/report_master.txt` after training and read the table at the bottom.

---

## Step-by-step: train on your RTX 3050 Ti (4 GB)

### 1. Install dependencies (once)

```bash
pip install -r requirements.txt
```

**GPU (RTX 3050 Ti):** `requirements.txt` installs CPU-only PyTorch by default. For CUDA:

```bash
pip uninstall torch torchvision torchaudio -y
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
```

Check GPU works:

```bash
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

Should print `True` and your GPU name. If `False`, training still runs on CPU (slower).

### 2. Put sensor data in the folder configured in `app/ml/sensor_configs.py`

### 3. Train the **main** model (use ALL sessions)

```bash
python train_master_model.py --max-sessions 0 --top-features 50
```

- Master XGBoost uses **CPU** by default (your sensor tables are on CPU — avoids warnings).
- Your **RTX 3050 Ti** is used by the **deep** model: `train_dl_model.py`.
- Optional: `--xgb-gpu` only if you want XGBoost on CUDA (not required).

- `--max-sessions 0` → use **all** cow/day recordings (best accuracy).
- Merges rare noisy classes (3, 5, 7) into one bucket so the model learns easier.
- Keeps only the **top 50** most useful sensor columns.

**Time:** can take 10–60+ minutes depending on data size.

### 4. Train the **deep** backup model

```bash
python train_dl_model.py --max-sessions 50 --batch-size 16 --epochs 30
```

Uses your GPU with batch size 16 (safe for 4 GB VRAM).

### 5. Check results (no coding)

1. Open `models/confusion_matrix_master.png` — dark diagonal = good.
2. Open `models/report_master.txt` — look at **F1 macro**.
3. If F1 macro is still low, rare classes may need cleaner labels in your CSV files.

### 6. Optional: combine both models

```bash
python evaluate_ensemble.py --max-sessions 0
```

Reads `models/report_ensemble.txt` for the combined score.

---

## What we changed in code (for your checklist)

| Improvement | What it means in plain English |
|-------------|-------------------------------|
| Deleted old per-sensor `.joblib` files | Less clutter; only one main model |
| `--max-sessions 0` | Train on full dataset, not a small sample |
| Merge classes 3, 5, 7 | Group rare/hard behaviors so the model stops failing on them |
| Class weights | Pay more attention to rare behaviors during training |
| Session-aware split | Test on **new days/cows**, not random rows (honest score) |
| Top 50 features | Drop weak sensor columns → less noise |
| Macro F1 + confusion matrix | Reports that match real-world fairness |
| Ensemble script | Average master + deep predictions offline |

---

## Realistic expectations

- **85–90% accuracy** on common behaviors is achievable with full data + feature selection.
- **90%+ F1 macro on every rare class** is hard without more labels or merging classes further.
- If scores stay near 60%, the problem is often **labels** (wrong behavior in CSV) or **too little data** for rare classes—not only the algorithm.

---

## Quick troubleshooting

| Problem | Try |
|---------|-----|
| CUDA out of memory | `python train_dl_model.py --batch-size 8` |
| Training very slow | Normal on laptop; use `--max-sessions 20` for a quick test first |
| API says model missing | Run `train_master_model.py` first |
| Master works, deep fails | Run `train_dl_model.py` after master finishes |

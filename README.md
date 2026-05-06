# WM-811K Wafer Map Defect Classification

Classifying semiconductor wafer defect patterns from the [WM-811K dataset](https://www.kaggle.com/datasets/qingyi/wm811k-wafer-map) using a range of approaches — from a KL-divergence baseline to CNNs and a Vision Transformer.

---

## Dataset

| Item | Detail |
|---|---|
| Source file | `Data/LSWMD.pkl` |
| Total wafers | 811,457 |
| Labeled wafers | 172,950 (after dropping unlabeled / `unknown`) |
| Classes | 9 (8 defect types + `none`) |
| Image encoding | 0 = background · 1 = normal die · 2 = defective die |

### Class distribution (labeled set)

| Class | Count |
|---|---|
| none | 147,431 |
| Edge-Ring | 9,680 |
| Edge-Loc | 5,189 |
| Center | 4,294 |
| Loc | 3,593 |
| Scratch | 1,193 |
| Random | 866 |
| Donut | 555 |
| Near-full | 149 |

### Train / Val / Test split

Stratified 60 / 20 / 20 split (`random_state=42`), identical across all model notebooks.

| Split | Samples |
|---|---|
| Train | 103,770 |
| Val | 34,590 |
| Test | 34,590 |

### Class balancing (training only)

Every class is brought to exactly **20,000 samples** before each training epoch:

- `none` (~88k in train) → randomly subsampled to 20,000 without replacement
- All minority classes → augmented via random flips (none / h-flip / v-flip / hv-flip) to reach 20,000

Final training set: **180,000 samples** (9 × 20,000), perfectly balanced.

---

## Notebooks

### `EDA.ipynb` — Exploratory Data Analysis
- Raw column inspection and label extraction
- Wafer map dimension distributions
- Sample wafer maps for every defect type
- 3×3 average-heatmap grid for all 9 classes

---

### `model1.ipynb` — Simple CNN  ✦ best result

**Architecture** (1,192,745 parameters):

```
Input 1×64×64
→ Conv(32, k=6, p=1) + BN + ReLU + MaxPool → 32×30×30
→ Conv(64, k=6, p=1) + BN + ReLU + MaxPool → 64×13×13
→ Conv(128, k=6, p=1) + BN + ReLU + MaxPool + Dropout2d(0.1) → 128×5×5
→ Flatten → 3200
→ FC(256) + BN + ReLU + Dropout(0.5)
→ FC(9)
```

| Metric | Value |
|---|---|
| Test accuracy | **96.05 %** |
| Balanced accuracy | 88.27 % |
| Cohen's kappa | 0.860 |
| Epochs / LR | 15 / 1e-3 (Adam) |

---

### `model2.ipynb` — CNN variant

Same architecture family as Model 1 with different hyperparameters.

| Metric | Value |
|---|---|
| Test accuracy | **92.17 %** |

---

### `model3.ipynb` — Vision Transformer (ViT)

**Architecture** (548,361 parameters):

```
Input 1×64×64
→ Patch Embed: Conv(k=8, s=8) + BN → 64 patches × 128
→ Positional Embedding + Dropout(0.15)
→ 4× Transformer Block:
     LayerNorm → MHA(4 heads, attn_drop=0.10) + Dropout(0.20) + DropPath
     LayerNorm → MLP(256) + Dropout(0.20) + DropPath  (rate 0→0.20)
→ LayerNorm → Global Average Pool → 128
→ BN + Dropout(0.40) → FC(9)
```

| Metric | Value |
|---|---|
| Test accuracy | **91.67 %** |
| Balanced accuracy | 85.03 % |
| Cohen's kappa | 0.738 |
| Epochs / LR | 15 / 3e-4 (AdamW, cosine schedule) |

---

### `model5.ipynb` — KL-divergence baseline (no training)

Builds one **average heatmap per class** from the training set, treats each as a probability distribution, and classifies test wafers by minimum KL-divergence to the 9 prototypes. No gradient descent involved.

| Metric | Value |
|---|---|
| Test accuracy | **34.76 %** |

---

### `model6.ipynb` — CNN overfitted on 9 prototypes

Trains the same CNN architecture as Model 1 (Dropout and BatchNorm removed) to **overfit** on only the 9 class-average heatmaps. Acts as a sanity-check / ablation showing that memorising prototypes generalises poorly.

| Metric | Value |
|---|---|
| Test accuracy | **8.02 %** |

---

## Results summary

| Model | Approach | Test Acc | Balanced Acc |
|---|---|---|---|
| Model 1 | Simple CNN | **96.05 %** | **88.27 %** |
| Model 2 | CNN variant | 92.17 % | — |
| Model 3 | Vision Transformer | 91.67 % | 85.03 % |
| Model 5 | KL-divergence baseline | 34.76 % | — |
| Model 6 | CNN on 9 prototypes | 8.02 % | — |

---

## Setup

```bash
# Activate the shared virtual environment
source /synology/datasets/proj_pnc/QA/zhora/HW/venv_for_hw/bin/activate

# Key dependencies
# torch 2.5.1+cu121  (GTX 1080 Ti — Pascal sm_61)
# torchvision, numpy, pandas, scipy, scikit-learn, matplotlib, seaborn
```

Run any notebook with Jupyter Lab / Jupyter Notebook from the project directory:

```bash
cd /synology/datasets/proj_pnc/QA/zhora/HW/WM-811K_wafer_map/WM-811K_wafer_map
jupyter lab
```

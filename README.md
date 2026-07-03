# Stock Market Index Prediction using Deep Transformer Model

Implementation based on the paper:
**"Stock market index prediction using deep Transformer model"**
Wang, Chen, Zhang, Zhang — *Expert Systems With Applications*, 208 (2022) 118128.

This project implements:
- A **Transformer encoder–decoder** model (Section 4 of the paper) for next-day
  closing-price prediction.
- Baseline models: **CNN, RNN, LSTM** (Section 3) for comparison.
- **Moving-window** dataset construction, train/test split (80/20), and
  normalization (Section 5.1).
- Evaluation metrics: **MAE, MSE, MAPE** (Section 5.3).
- A simple **trading strategy backtest**: Return, Volatility, Max Drawdown,
  Sharpe Ratio, and Net Value curves (Section 5.3–5.5), compared against
  Buy & Hold.

Indices supported (same as the paper): `CSI300`, `SP500`, `HSI`, `N225`.

---

## 1. Project structure

```
stock_transformer_project/
├── data_utils.py     # download, normalize, moving-window dataset
├── model.py           # Transformer + CNN/RNN/LSTM models
├── train.py            # training loop + MAE/MSE/MAPE metrics
├── backtest.py          # trading strategy, net value, Sharpe, drawdown
├── main.py                # runs the full pipeline end-to-end
├── requirements.txt
└── README.md
```

---

## 2. Setup (run these steps)

### Step 1 — Install Python 3.8+ (if not already installed)
Check with:
```bash
python3 --version
```

### Step 2 — Create a virtual environment (recommended)
```bash
python3 -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
```

### Step 3 — Install dependencies
```bash
pip install -r requirements.txt
```
This installs: `torch`, `numpy`, `pandas`, `matplotlib`, `yfinance`.

> Note: `yfinance` needs an active internet connection since it downloads
> real stock index data from Yahoo Finance.

---

## 3. Run the project

### Quick demo run (S&P 500, fewer epochs, faster)
```bash
python main.py --index SP500 --epochs 200
```

### Full paper-style run (1000 epochs, as in the paper — will take longer)
```bash
python main.py --index CSI300 --epochs 1000
```

### Run for all four indices
```bash
python main.py --index CSI300 --epochs 300
python main.py --index SP500  --epochs 300
python main.py --index HSI    --epochs 300
python main.py --index N225   --epochs 300
```

### Only train the Transformer (skip baselines, faster)
```bash
python main.py --index SP500 --epochs 300 --models transformer
```

### All CLI options
```bash
python main.py --help
```
| Flag | Meaning | Default |
|---|---|---|
| `--index` | Which index: CSI300 / SP500 / HSI / N225 | SP500 |
| `--T` | Encoder input window length | 9 |
| `--N` | Decoder input length | 2 |
| `--epochs` | Training epochs | 300 |
| `--batch_size` | Mini-batch size | 16 |
| `--lr` | Learning rate (Adam) | 0.0001 |
| `--out_dir` | Output folder for plots/metrics | results |
| `--models` | Comma-separated models to train | transformer,lstm,rnn,cnn |

---

## 4. Outputs

After running, check the `results/` folder for:
- `{INDEX}_metrics.json` — MAE / MSE / MAPE for each model (like Table 2 in the paper)
- `{INDEX}_prediction.png` — predicted vs real price curves on the test set (like Fig. 7/8)
- `{INDEX}_networth.png` — net value curves of each model vs Buy & Hold (like Fig. 9)

Console output also prints backtest stats (Return %, Sharpe ratio, Max Drawdown %)
for every model — this matches Table 4 in the paper.

---


- GPU is optional — training auto-detects CUDA (`train_model` picks `cuda` if
  available, else falls back to CPU).
- To match the paper's exact architecture more closely (multiple random seeds,
  10 independent runs for the Mann-Whitney U test in Table 3/5), simply loop
  `main.run(...)` multiple times with different seeds and aggregate the metrics.

"""
main.py
Run the full experiment for one stock index:
  1. Download & preprocess data
  2. Train Transformer, CNN, RNN, LSTM
  3. Compute MAE / MSE / MAPE (Table 2 style)
  4. Run trading-strategy backtest (Table 4 style): Return, Volatility, MaxDrawdown, Sharpe
  5. Save plots: predicted vs real curves, net value curves

Usage:
    python main.py --index SP500 --epochs 300
    python main.py --index CSI300 --epochs 1000 --T 9

Note: reduce --epochs for a quick run (paper uses 1000, which can take a
while on CPU). 200-300 epochs already gives a reasonable demo.
"""

import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from data_utils import prepare_dataset, denormalize
from train import train_model, evaluate, set_seed
from backtest import backtest_summary, buy_and_hold_returns, net_value


def run(index_name, T=9, N=2, epochs=300, batch_size=16, lr=1e-4,
        out_dir="results", models_to_run=("transformer", "lstm", "rnn", "cnn")):
    os.makedirs(out_dir, exist_ok=True)
    set_seed(42)

    print(f"\n=== Preparing dataset for {index_name} ===")
    data = prepare_dataset(index_name, T=T)
    X_train, y_train = data["X_train"], data["y_train"]
    X_test, y_test = data["X_test"], data["y_test"]
    mu, sigma = data["mu"], data["sigma"]

    results = {}
    preds_test_denorm = {}

    for model_name in models_to_run:
        print(f"\n--- Training {model_name.upper()} on {index_name} ---")
        model, loss_hist, y_pred_train, y_pred_test = train_model(
            model_name, X_train, y_train, X_test, y_test,
            T=T, N=N, epochs=epochs, batch_size=batch_size, lr=lr,
            verbose_every=max(1, epochs // 5),
        )
        metrics = evaluate(y_pred_test, y_test)
        results[model_name] = {"metrics": metrics, "loss_history": loss_hist}
        preds_test_denorm[model_name] = denormalize(y_pred_test, mu, sigma)
        print(f"{model_name.upper()} test metrics: {metrics}")

    # ---- Backtest (trading strategy) ----
    y_test_denorm = denormalize(y_test, mu, sigma)
    # y_true_prev = actual price at t (i.e. last value of each input window)
    y_true_prev = denormalize(X_test[:, -1], mu, sigma)

    backtest_results = {}
    for model_name, y_pred in preds_test_denorm.items():
        bt = backtest_summary(y_pred, y_test_denorm, y_true_prev)
        backtest_results[model_name] = bt
        print(f"{model_name.upper()} backtest: Return={bt['Return(%)']:.2f}%, "
              f"Sharpe={bt['Sharpe']:.4f}, MaxDD={bt['MaxDrawdown(%)']:.2f}%")

    # Buy & hold benchmark
    bh_returns = buy_and_hold_returns(y_test_denorm, y_true_prev)
    bh_nv = net_value(bh_returns)
    backtest_results["B&H"] = {
        "Return(%)": (bh_nv[-1] - 1) * 100,
        "net_value": bh_nv,
    }

    # ---- Save metrics summary ----
    summary = {m: results[m]["metrics"] for m in results}
    with open(os.path.join(out_dir, f"{index_name}_metrics.json"), "w") as f:
        json.dump(summary, f, indent=2)

    # ---- Plot: predicted vs real (test set) ----
    plt.figure(figsize=(10, 5))
    plt.plot(y_test_denorm, label="Real data", color="black", linewidth=1.5)
    for model_name, y_pred in preds_test_denorm.items():
        plt.plot(y_pred, label=model_name.upper(), alpha=0.8)
    plt.title(f"Test set prediction ({index_name})")
    plt.xlabel("Time step")
    plt.ylabel("Closing Price")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"{index_name}_prediction.png"), dpi=150)
    plt.close()

    # ---- Plot: net value curves ----
    plt.figure(figsize=(10, 5))
    plt.plot(backtest_results["B&H"]["net_value"], label="B&H", alpha=0.8)
    for model_name in models_to_run:
        plt.plot(backtest_results[model_name]["net_value"], label=model_name.upper(), alpha=0.9)
    plt.title(f"Net value curves ({index_name})")
    plt.xlabel("Time step")
    plt.ylabel("Net Value")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"{index_name}_networth.png"), dpi=150)
    plt.close()

    print(f"\nSaved results to '{out_dir}/'")
    return summary, backtest_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--index", type=str, default="SP500",
                         choices=["CSI300", "SP500", "HSI", "N225"])
    parser.add_argument("--T", type=int, default=9, help="Encoder input window length")
    parser.add_argument("--N", type=int, default=2, help="Decoder input length")
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--out_dir", type=str, default="results")
    parser.add_argument("--models", type=str, default="transformer,lstm,rnn,cnn",
                         help="Comma-separated list of models to train")
    args = parser.parse_args()

    models_to_run = tuple(m.strip() for m in args.models.split(","))

    run(args.index, T=args.T, N=args.N, epochs=args.epochs,
        batch_size=args.batch_size, lr=args.lr, out_dir=args.out_dir,
        models_to_run=models_to_run)

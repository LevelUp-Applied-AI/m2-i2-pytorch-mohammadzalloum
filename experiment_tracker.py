import argparse
import itertools
import json
import random
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn


# ----------------------------
# Reproducibility
# ----------------------------
def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


# ----------------------------
# Model
# ----------------------------
class HousingModel(nn.Module):
    def __init__(self, input_size: int, hidden_size: int = 32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1)
        )

    def forward(self, x):
        return self.net(x)


# ----------------------------
# Data utilities
# ----------------------------
def load_data(csv_path: str, target_col: str):
    df = pd.read_csv(csv_path)

    if target_col not in df.columns:
        raise ValueError(
            f"Target column '{target_col}' not found.\n"
            f"Available columns: {list(df.columns)}"
        )

    # Keep numeric columns only
    numeric_df = df.select_dtypes(include=[np.number]).copy()

    if target_col not in numeric_df.columns:
        raise ValueError(
            f"Target column '{target_col}' is not numeric. "
            f"Make sure your target column contains numbers."
        )

    # Drop rows with missing values in numeric columns
    numeric_df = numeric_df.dropna()

    feature_cols = [col for col in numeric_df.columns if col != target_col]
    if not feature_cols:
        raise ValueError("No numeric feature columns found after preprocessing.")

    X = numeric_df[feature_cols].to_numpy(dtype=np.float32)
    y = numeric_df[target_col].to_numpy(dtype=np.float32).reshape(-1, 1)

    return X, y, feature_cols


def fixed_train_test_split(X, y, seed=42, train_ratio=0.8):
    set_seed(seed)
    indices = np.random.permutation(len(X))
    split_idx = int(train_ratio * len(X))

    train_idx = indices[:split_idx]
    test_idx = indices[split_idx:]

    X_train = X[train_idx]
    X_test = X[test_idx]
    y_train = y[train_idx]
    y_test = y[test_idx]

    return X_train, X_test, y_train, y_test


def standardize_train_test(X_train, X_test):
    mean = X_train.mean(axis=0)
    std = X_train.std(axis=0)
    std[std == 0] = 1.0

    X_train_scaled = (X_train - mean) / std
    X_test_scaled = (X_test - mean) / std

    return X_train_scaled, X_test_scaled


# ----------------------------
# Metrics
# ----------------------------
def mean_absolute_error(y_true, y_pred):
    return float(np.mean(np.abs(y_true - y_pred)))


def r_squared(y_true, y_pred):
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)

    if ss_tot == 0:
        return 0.0

    return float(1 - (ss_res / ss_tot))


# ----------------------------
# Training
# ----------------------------
def train_one_experiment(
    X_train,
    y_train,
    X_test,
    y_test,
    input_size,
    learning_rate,
    hidden_size,
    num_epochs,
    seed=42,
):
    set_seed(seed)

    model = HousingModel(input_size=input_size, hidden_size=hidden_size)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32)

    X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test, dtype=torch.float32)

    start_time = time.time()

    for _ in range(num_epochs):
        model.train()
        optimizer.zero_grad()

        train_preds = model(X_train_tensor)
        train_loss = criterion(train_preds, y_train_tensor)

        train_loss.backward()
        optimizer.step()

    training_time = time.time() - start_time

    model.eval()
    with torch.no_grad():
        final_train_preds = model(X_train_tensor)
        final_test_preds = model(X_test_tensor)

        final_train_loss = criterion(final_train_preds, y_train_tensor).item()
        final_test_loss = criterion(final_test_preds, y_test_tensor).item()

        y_test_pred_np = final_test_preds.numpy().flatten()
        y_test_true_np = y_test_tensor.numpy().flatten()

    test_mae = mean_absolute_error(y_test_true_np, y_test_pred_np)
    test_r2 = r_squared(y_test_true_np, y_test_pred_np)

    return {
        "learning_rate": float(learning_rate),
        "hidden_size": int(hidden_size),
        "num_epochs": int(num_epochs),
        "final_train_loss": float(final_train_loss),
        "final_test_loss": float(final_test_loss),
        "test_mae": float(test_mae),
        "test_r2": float(test_r2),
        "training_time_seconds": float(training_time),
    }


# ----------------------------
# Reporting
# ----------------------------
def save_experiments_json(results, output_path="experiments.json"):
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)


def print_leaderboard(results, top_k=10):
    ranked = sorted(results, key=lambda x: x["test_mae"])[:top_k]

    print("\nTop Configurations by Test MAE")
    print("-" * 78)
    print(
        f"{'Rank':<6}{'LR':<12}{'Hidden':<10}{'Epochs':<10}"
        f"{'Test MAE':<16}{'Test R²':<12}{'Time (s)':<10}"
    )
    print("-" * 78)

    for i, row in enumerate(ranked, start=1):
        print(
            f"{i:<6}"
            f"{row['learning_rate']:<12.4g}"
            f"{row['hidden_size']:<10}"
            f"{row['num_epochs']:<10}"
            f"{row['test_mae']:<16.2f}"
            f"{row['test_r2']:<12.4f}"
            f"{row['training_time_seconds']:<10.2f}"
        )


def save_summary_plot(results, output_path="experiment_summary.png"):
    df = pd.DataFrame(results)

    plt.figure(figsize=(10, 6))

    # Plot one series per hidden size
    for hidden_size in sorted(df["hidden_size"].unique()):
        subset = df[df["hidden_size"] == hidden_size].copy()
        subset = subset.sort_values("learning_rate")

        plt.plot(
            subset["learning_rate"],
            subset["test_mae"],
            marker="o",
            linestyle="-",
            label=f"hidden_size={hidden_size}"
        )

        # annotate points with epochs
        for _, row in subset.iterrows():
            plt.annotate(
                f"{int(row['num_epochs'])}",
                (row["learning_rate"], row["test_mae"]),
                textcoords="offset points",
                xytext=(4, 4),
                fontsize=8
            )

    plt.xscale("log")
    plt.xlabel("Learning Rate (log scale)")
    plt.ylabel("Test MAE")
    plt.title("Experiment Summary: Test MAE vs Learning Rate")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def print_analysis(results):
    ranked = sorted(results, key=lambda x: x["test_mae"])
    best = ranked[0]

    print("\nBest Configuration")
    print("-" * 40)
    print(json.dumps(best, indent=2))

    print("\nShort Analysis")
    print("-" * 40)

    if best["test_mae"] < 10000:
        print(
            f"Great: target achieved. Best test MAE = {best['test_mae']:.2f} JOD "
            f"with lr={best['learning_rate']}, hidden_size={best['hidden_size']}, "
            f"epochs={best['num_epochs']}."
        )
    else:
        print(
            f"Target not reached yet. Best test MAE = {best['test_mae']:.2f} JOD "
            f"with lr={best['learning_rate']}, hidden_size={best['hidden_size']}, "
            f"epochs={best['num_epochs']}."
        )
        print("What to try next:")
        print("- Add feature scaling if not already present")
        print("- Try more learning rates around the best one")
        print("- Try larger/smaller hidden sizes around the best one")
        print("- Add early stopping")
        print("- Try another activation or a second hidden layer")


# ----------------------------
# Main
# ----------------------------
def main():
    parser = argparse.ArgumentParser(description="Run experiment tracker for housing model.")
    parser.add_argument("--csv", type=str, required=True, help="Path to CSV dataset")
    parser.add_argument("--target", type=str, required=True, help="Target column name")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    csv_path = Path(args.csv)
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV file not found: {csv_path}")

    # 1) Load data
    X, y, feature_cols = load_data(str(csv_path), args.target)
    print(f"Loaded dataset: {len(X)} rows, {len(feature_cols)} numeric features")

    # 2) Fixed split once for all experiments
    X_train, X_test, y_train, y_test = fixed_train_test_split(
        X, y, seed=args.seed, train_ratio=0.8
    )

    # 3) Standardize based on train only
    X_train, X_test = standardize_train_test(X_train, X_test)

    input_size = X_train.shape[1]

    # 4) Hyperparameter grid (48 configs)
    learning_rates = [0.0005, 0.001, 0.005, 0.01]
    hidden_sizes = [16, 32, 64, 128]
    epoch_options = [100, 200, 300]

    configs = list(itertools.product(learning_rates, hidden_sizes, epoch_options))
    print(f"Running {len(configs)} experiments...")

    results = []

    for idx, (lr, hidden, epochs) in enumerate(configs, start=1):
        print(
            f"[{idx}/{len(configs)}] "
            f"lr={lr}, hidden_size={hidden}, epochs={epochs}"
        )

        metrics = train_one_experiment(
            X_train=X_train,
            y_train=y_train,
            X_test=X_test,
            y_test=y_test,
            input_size=input_size,
            learning_rate=lr,
            hidden_size=hidden,
            num_epochs=epochs,
            seed=args.seed,
        )

        results.append(metrics)

    # 5) Save logs
    save_experiments_json(results, "experiments.json")

    # 6) Print leaderboard
    print_leaderboard(results, top_k=10)

    # 7) Save visualization
    save_summary_plot(results, "experiment_summary.png")

    # 8) Analysis summary
    print_analysis(results)

    print("\nDone.")
    print("Generated:")
    print("- experiments.json")
    print("- experiment_summary.png")


if __name__ == "__main__":
    main()
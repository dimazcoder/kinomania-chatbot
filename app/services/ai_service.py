from datetime import datetime

import pandas as pd
from matplotlib import pyplot as plt

from app.core.models.mt5_model import MT5Trainer
from app.helpers.os_helper import prep_path, save_json_to_disc, save_plot_to_disc

log_path = 'logs/mt5'
model_path = 'models/mt5'

def train_model():
    model = MT5Trainer(
        model_path=prep_path(model_path),
        log_path=prep_path(log_path)
    )

    train_results = model.train(
        dataset_path = prep_path('datasets/train_plain.json')
    )

    save_train_results(train_results)


def load_model():
    model = MT5Trainer(
        model_path=prep_path(model_path),
        log_path=prep_path(log_path)
    )

    model.load_model()
    return model


def generate(model, input_text: str):
    return model.generate(
        input_text=input_text
    )


def save_train_results(results):
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    save_json_to_disc(
        results, f"{log_path}/{timestamp}_train_log.json"
    )

    plot_results(
        results.get("log_history", []), timestamp
    )

    print({
        "val_metrics": results.get("val_metrics", {}),
        "test_metrics": results.get("test_metrics", {})
    })


def plot_results(log_history, timestamp):
    if not log_history:
        raise Exception("log_history is empty")

    history = {}
    for entry in log_history:
        epoch = entry.get("epoch")
        if epoch is None:
            continue
        if epoch not in history:
            history[epoch] = {}
        history[epoch].update(entry)

    df = pd.DataFrame([
        {"epoch": epoch, "loss": entry.get("loss"), "eval_loss": entry.get("eval_loss")}
        for epoch, entry in history.items()
    ])

    df = df.sort_values("epoch")

    if not df.empty:
        if df["loss"].notna().any():
            plt.plot(df["epoch"], df["loss"], label="Train Loss")
        if df["eval_loss"].notna().any():
            plt.plot(df["epoch"], df["eval_loss"], label="Eval Loss")

        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("Training Loss vs Eval Loss")
        plt.legend()
        plt.grid()
        save_plot_to_disc(
            plt, f"{log_path}/{timestamp}_train_plt.png"
        )
    else:
        print("Нет данных для отображения.")



import json
import os

from app.core.config import config


def prep_path(path, makedirs=False):
    os_path = os.path.join(config.static_path, *path.split("/"))

    if makedirs:
        os.makedirs(os.path.dirname(os_path), exist_ok=True)

    return os_path


def save_json_to_disc(json_data, file_path):
    file_path = prep_path(file_path, makedirs=True)
    with open(file_path, "w", encoding='utf-8') as f:
        json.dump(json_data, f, ensure_ascii=False, indent=4)


def save_df_to_disc(df, file_path):
    file_path = prep_path(file_path, makedirs=True)
    df.to_json(file_path, orient="records", indent=4)


def save_plot_to_disc(plt, file_path):
    file_path = prep_path(file_path, makedirs=True)
    plt.savefig(file_path, dpi=300)

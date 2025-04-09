from collections import Counter
import json

from app.helpers.os_helper import prep_path

dataset_path = prep_path('datasets/train_plain.json')

with open(dataset_path, "r", encoding='utf-8') as f:
    data = json.load(f)

key_counter = Counter()
intent_counter = Counter()
total = len(data)

for item in data:
    output = item["output"]
    parts = [part.strip() for part in output.split(";") if part.strip()]
    for part in parts:
        if ":" not in part:
            continue
        key, value = [p.strip() for p in part.split(":", 1)]
        key_counter[key] += 1
        if key == "intent":
            intent_counter[value] += 1

key_stats = [
    (key, count, round(count / total * 100, 2))
    for key, count in key_counter.items()
]
key_stats.sort(key=lambda x: x[2], reverse=True)

print("\nOutput key statistics:")
for key, count, percent in key_stats:
    print(f"{key}: {percent}%")

# Статистика по intent
print("\nIntent statistics:")
for intent, count in intent_counter.most_common():
    percent = round(count / total * 100, 2)
    print(f"{intent}: {percent}%")

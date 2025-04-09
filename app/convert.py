import json

from app.helpers.os_helper import prep_path, save_json_to_disc

dataset_path = prep_path('datasets/train.json')

with open(dataset_path, "r", encoding='utf-8') as f:
    raw_data = json.load(f)

converted = []
for item in raw_data:
    input_text = item["input"]
    output_data = item["output"]
    intent = output_data.get("intent", "")
    entities = output_data.get("entities", {})

    # Преобразуем сущности в строку вида key: value
    entity_parts = []
    for key, value in entities.items():
        if isinstance(value, list):
            entity_parts.append(f"{key}: {', '.join(value)}")
        else:
            entity_parts.append(f"{key}: {value}")

    # Финальная строка
    output_str = f"intent: {intent}; " + "; ".join(entity_parts)
    output_str = output_str.strip().rstrip(";").strip()

    converted.append({
        "input": input_text,
        "output": output_str
    })

save_json_to_disc(
    converted,
    'datasets/train_plain.json'
)
# # Выведем результат или сохраним
# with open('converted_dataset.json', 'w', encoding='utf-8') as f:
#     json.dump(converted, f, ensure_ascii=False, indent=2)

print("Готово! Сохранено в converted_dataset.json")


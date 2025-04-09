from typing import Any


def process_text(text: str) -> dict[str, Any]:
    cleaned_text = preprocess_text(text)

    intent = define_intent(cleaned_text)

    entities = extract_entities(cleaned_text)

    return {"intent": intent, "entities": entities}

def preprocess_text(text: str) -> str:
    return text.lower()

def define_intent(text: str) -> str:
    return text

def extract_entities(text: str) -> dict[str, Any]:
    entities = {}
    return entities

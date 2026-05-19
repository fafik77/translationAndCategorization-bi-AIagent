import os
import json

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.multioutput import MultiOutputClassifier

from langchain_ollama import ChatOllama


# =========================================================
# CONFIG
# =========================================================

MODEL_NAME = "translategemma"
TEMPERATURE = 0

DATA_FOLDER = "Dictionary"
WORD_DATA_FILE = "word_data.json"


# =========================================================
# LOAD DATA
# =========================================================

word_data_path = os.path.join(DATA_FOLDER, WORD_DATA_FILE)

with open(word_data_path, "r", encoding="utf-8") as f:
    word_data = json.load(f)


# =========================================================
# PREPARE TRAINING DATA FOR CATEGORY CLASSIFIER
# =========================================================

X_category_train = []
y_category_train = []

for polish_word, data in word_data.items():
    translation = data["translation"].lower()
    categories = data["categories"]

    X_category_train.append(polish_word.lower())
    y_category_train.append(categories)

    X_category_train.append(translation)
    y_category_train.append(categories)


# =========================================================
# TRAIN CATEGORY MLP CLASSIFIER
# =========================================================

category_binarizer = MultiLabelBinarizer()
Y_category_train = category_binarizer.fit_transform(y_category_train)

category_classifier = Pipeline([
    (
        "tfidf",
        TfidfVectorizer(
            analyzer="char",
            ngram_range=(2, 4)
        )
    ),
    (
        "mlp",
        MultiOutputClassifier(
            MLPClassifier(
                hidden_layer_sizes=(64, 32),
                activation="relu",
                max_iter=1000,
                random_state=42
            )
        )
    )
])

category_classifier.fit(X_category_train, Y_category_train)


# =========================================================
# PREPARE TRAINING DATA FOR LANGUAGE CLASSIFIER
# =========================================================

X_language_train = []
y_language_train = []

for polish_word, data in word_data.items():
    X_language_train.append(polish_word.lower())
    y_language_train.append("pl")

    X_language_train.append(data["translation"].lower())
    y_language_train.append("en")


# =========================================================
# TRAIN LANGUAGE MLP CLASSIFIER
# =========================================================

language_classifier = Pipeline([
    (
        "tfidf",
        TfidfVectorizer(
            analyzer="char",
            ngram_range=(2, 4)
        )
    ),
    (
        "mlp",
        MLPClassifier(
            hidden_layer_sizes=(32,),
            activation="relu",
            max_iter=1000,
            random_state=42
        )
    )
])

language_classifier.fit(X_language_train, y_language_train)


# =========================================================
# LLM
# =========================================================

llm = ChatOllama(
    model=MODEL_NAME,
    temperature=TEMPERATURE
)


# =========================================================
# FUNCTIONS
# =========================================================

def detect_language(word):
    word = word.lower().strip()
    return language_classifier.predict([word])[0]


def translate_word(word, language):
    word = word.lower().strip()

    if language == "pl":
        if word in word_data:
            return word_data[word]["translation"]

    if language == "en":
        for polish_word, data in word_data.items():
            if data["translation"].lower() == word:
                return polish_word

    if language == "pl":
        prompt = f"""
Translate this Polish word to English.
Return only the translation.
Do not add explanation.

Word: {word}
"""
    else:
        prompt = f"""
Translate this English word to Polish.
Return only the translation.
Do not add explanation.

Word: {word}
"""

    response = llm.invoke(prompt)
    return response.content.strip()


def predict_categories(word):
    word = word.lower().strip()

    prediction = category_classifier.predict([word])
    categories = category_binarizer.inverse_transform(prediction)

    if categories and len(categories[0]) > 0:
        return list(categories[0])

    return ["other"]


def process_word(word):
    word = word.lower().strip()

    detected_language = detect_language(word)

    if detected_language == "pl":
        target_language = "en"
    else:
        target_language = "pl"

    translation = translate_word(word, detected_language)
    categories = predict_categories(word)

    return {
        "input_word": word,
        "detected_language": detected_language,
        "translation": translation,
        "target_language": target_language,
        "categories": categories
    }


# =========================================================
# MAIN LOOP
# =========================================================

def main():
    print("\n=== Translation And Categorization AI Project ===")
    print("Type 'exit' or 'quit' to stop.\n")

    while True:
        try:
            user_input = input("User >> ").strip()

            if user_input.lower() in ["exit", "quit"]:
                break

            if not user_input:
                continue

            result = process_word(user_input)

            print("\nAI:")
            print(json.dumps(result, ensure_ascii=False, indent=2))
            print()

        except KeyboardInterrupt:
            break

        except Exception as e:
            print(f"\nError: {e}\n")


if __name__ == "__main__":
    main()
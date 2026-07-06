# ml_experiments.py
"""
Simple ML experiments for the Mood Machine lab.

This file uses a "real" machine learning library (scikit-learn)
to train a tiny text classifier on the same SAMPLE_POSTS and
TRUE_LABELS that you use with the rule based model.
"""

from typing import List, Tuple

import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay

from dataset import SAMPLE_POSTS, TRUE_LABELS


TEST_POSTS = [
    "Oh great, my laptop died again.",
    "I passed the test, but barely.",
    "That movie was fire.",
    "I have class at noon.",
    "I am not excited about this.",
]

TEST_LABELS = [
    "negative",
    "mixed",
    "positive",
    "neutral",
    "negative",
]


def train_ml_model(
    texts: List[str],
    labels: List[str],
) -> Tuple[CountVectorizer, LogisticRegression]:
    """
    Train a simple text classifier using bag of words features
    and logistic regression.
    """
    if len(texts) != len(labels):
        raise ValueError(
            "texts and labels must be the same length. "
            "Check SAMPLE_POSTS and TRUE_LABELS in dataset.py."
        )

    if not texts:
        raise ValueError("No training data provided. Add examples in dataset.py.")

    vectorizer = CountVectorizer()
    X = vectorizer.fit_transform(texts)

    model = LogisticRegression(max_iter=1000)
    model.fit(X, labels)

    return vectorizer, model


def evaluate_on_dataset(
    texts: List[str],
    labels: List[str],
    vectorizer: CountVectorizer,
    model: LogisticRegression,
    title: str,
) -> float:
    """
    Evaluate the trained model on a labeled dataset.
    """
    if len(texts) != len(labels):
        raise ValueError("texts and labels must be the same length.")

    X = vectorizer.transform(texts)
    preds = model.predict(X)

    print(f"=== {title} ===")
    for text, true_label, pred_label in zip(texts, labels, preds):
        print(f'"{text}" -> predicted={pred_label}, true={true_label}')

    accuracy = accuracy_score(labels, preds)
    print(f"\nAccuracy: {accuracy:.2f}")
    return accuracy


def display_confusion_matrix(
    texts: List[str],
    labels: List[str],
    vectorizer: CountVectorizer,
    model: LogisticRegression,
    title: str,
    filename: str,
) -> None:
    """
    Save a confusion matrix image.

    Rows are true labels.
    Columns are predicted labels.
    """
    label_order = ["positive", "negative", "neutral", "mixed"]

    X = vectorizer.transform(texts)
    preds = model.predict(X)

    matrix = confusion_matrix(labels, preds, labels=label_order)

    display = ConfusionMatrixDisplay(
        confusion_matrix=matrix,
        display_labels=label_order,
    )

    fig, ax = plt.subplots(figsize=(6, 6))
    display.plot(ax=ax, cmap="Blues", colorbar=False)

    plt.title(title)
    plt.tight_layout()
    plt.savefig(filename)
    plt.close(fig)

    print(f"Saved confusion matrix to {filename}")


def predict_single_text(
    text: str,
    vectorizer: CountVectorizer,
    model: LogisticRegression,
) -> str:
    """
    Predict the mood label for a single text string using
    the trained ML model.
    """
    X = vectorizer.transform([text])
    pred = model.predict(X)[0]
    return pred


def run_interactive_loop(
    vectorizer: CountVectorizer,
    model: LogisticRegression,
) -> None:
    """
    Let the user type their own sentences and see the ML model's
    predicted mood label.
    """
    print("\n=== Interactive Mood Machine (ML model) ===")
    print("Type a sentence to analyze its mood.")
    print("Type 'quit' or press Enter on an empty line to exit.\n")

    while True:
        user_input = input("You: ").strip()
        if user_input == "" or user_input.lower() == "quit":
            print("Goodbye from the ML Mood Machine.")
            break

        label = predict_single_text(user_input, vectorizer, model)
        print(f"ML model: {label}")


if __name__ == "__main__":
    print("Training an ML model on SAMPLE_POSTS and TRUE_LABELS from dataset.py...")
    print("Make sure you have added enough labeled examples before running this.\n")

    vectorizer, model = train_ml_model(SAMPLE_POSTS, TRUE_LABELS)

    evaluate_on_dataset(
        SAMPLE_POSTS,
        TRUE_LABELS,
        vectorizer,
        model,
        "ML Model Evaluation on Training Dataset",
    )

    display_confusion_matrix(
        SAMPLE_POSTS,
        TRUE_LABELS,
        vectorizer,
        model,
        "Training Confusion Matrix",
        "training_confusion_matrix.png",
    )

    print("\n=== ML Model Evaluation on New Test Posts ===")

    evaluate_on_dataset(
        TEST_POSTS,
        TEST_LABELS,
        vectorizer,
        model,
        "ML Model Evaluation on Test Dataset",
    )

    display_confusion_matrix(
        TEST_POSTS,
        TEST_LABELS,
        vectorizer,
        model,
        "Test Confusion Matrix",
        "test_confusion_matrix.png",
    )

    run_interactive_loop(vectorizer, model)

    print("\nTip: Compare these predictions with the rule based model")
    print("by running `python3 main.py`. Notice where they fail in")
    print("similar ways and where they fail in different ways.")
# The Mood Machine

The Mood Machine is a sentiment analysis project that explores two different approaches to artificial intelligence:

* **A rule-based classifier** that follows handcrafted rules.
* **A machine learning classifier** that learns from labeled examples.

The project classifies short text into one of four sentiment categories:

* Positive
* Negative
* Neutral
* Mixed

Rather than focusing only on building an accurate classifier, this project explores how AI systems make decisions, where they fail, and how data influences model behavior.

---

# Features

## Rule-Based Mood Analyzer

* Text preprocessing (lowercasing, punctuation removal, tokenization)
* Positive and negative word scoring
* Negation handling (for example, "not happy")
* Mixed sentiment detection
* Interactive command-line mood prediction

## Machine Learning Mood Analyzer

* Bag-of-Words representation using **CountVectorizer**
* Logistic Regression classifier using **scikit-learn**
* Training on labeled examples
* Interactive prediction mode
* Evaluation on both training data and a separate test dataset

---

# Project Structure

```text
├── dataset.py
├── mood_analyzer.py
├── main.py
├── ml_experiments.py
├── model_card.md
├── requirements.txt
└── README.md
```

---

# What I Implemented

During this project I:

* Expanded the labeled dataset with 10 additional realistic social-media posts.
* Added examples containing:

  * Slang
  * Emojis
  * Sarcasm
  * Mixed emotions
  * Ambiguous language
* Implemented the rule-based scoring algorithm.
* Added punctuation removal during preprocessing.
* Implemented negation handling for phrases such as "not happy."
* Improved sentiment detection by recognizing mixed positive and negative signals.
* Compared the rule-based model with a machine learning classifier.
* Created a separate test set to evaluate the ML model on unseen examples.
* Documented the system using a complete model card.

---

# Evaluation

### Rule-Based Model

The rule-based classifier performed well on simple, explicit sentiment but struggled with:

* Sarcasm
* Context-dependent language
* Unseen slang
* Emojis
* Indirect emotional expressions

Example:

```text
Sure, because waiting in traffic for an hour is exactly how I wanted to spend my morning.
```

Humans recognize this as sarcastic, while the rule-based model cannot infer the speaker's intent.

### Machine Learning Model

The machine learning model achieved:

* **100% accuracy** on the training dataset.
* Approximately **60% accuracy** on a separate set of unseen test examples.

This demonstrated the importance of evaluating models on data they have not previously seen.

---

# What I Learned

This project helped me understand several important AI concepts:

* The difference between rule-based AI and machine learning.
* How preprocessing affects model performance.
* Why data quality is just as important as model design.
* How overfitting can produce misleadingly high accuracy.
* Why training accuracy should not be confused with real-world performance.
* How small changes to a dataset can significantly change model behavior.
* The importance of documenting limitations through a model card.

---

# Technologies Used

* Python
* scikit-learn
* CountVectorizer
* Logistic Regression
* Git & GitHub

---

# Running the Project

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the rule-based model:

```bash
python3 main.py
```

Run the machine learning model:

```bash
python3 ml_experiments.py
```

---

# Future Improvements

Possible future enhancements include:

* Larger and more diverse training datasets.
* Better handling of sarcasm and context.
* Improved emoji and slang recognition.
* TF-IDF or word embeddings instead of Bag-of-Words.
* Separate training, validation, and testing datasets.
* Experimenting with transformer-based language models.

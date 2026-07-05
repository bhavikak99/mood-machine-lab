# Model Card: Mood Machine

This model card is for the Mood Machine project, which includes **two** versions of a mood classifier:

1. A **rule based model** implemented in `mood_analyzer.py`
2. A **machine learning model** implemented in `ml_experiments.py` using scikit-learn

---

## 1. Model Overview

**Model type:**

I implemented and evaluated both the rule based model and the machine learning model, then compared how they performed on the same dataset.

**Intended purpose:**

The Mood Machine classifies short text messages into one of four sentiment labels:

* Positive
* Negative
* Neutral
* Mixed

It is designed for short social media style posts and simple mood detection.

**How it works (brief):**

The rule-based model preprocesses text by converting it to lowercase, removing punctuation and splitting it into tokens. It assigns positive and negative scores based on predefined word lists, handles simple negation (such as "not happy") and predicts a label based on the final score.

The machine learning model uses CountVectorizer to convert text into numerical features and trains a Logistic Regression classifier using the labeled examples in `SAMPLE_POSTS` and `TRUE_LABELS`.

---

## 2. Data

**Dataset description:**

The starter dataset contained 6 labeled posts. I expanded it by adding 10 additional posts containing realistic social-media language, including sarcasm, slang, emojis, sports and mixed emotions.

**Labeling process:**

Each new post received one matching label. Some examples were straightforward, while others required judgment because they expressed multiple emotions or depended on context.

Examples that were more difficult to label included:

* "I'm exhausted, but at least it's finally the weekend."
* "The concert got canceled, but now I can finally catch up on sleep."

These could reasonably be interpreted differently by different people.

**Important characteristics of the dataset:**

* Includes slang (for example, "Lowkey", "No cap")
* Includes emojis (💀 😭 🎉 😩)
* Includes sarcasm
* Includes mixed emotions
* Includes short and ambiguous posts

**Possible issues with the dataset:**

The dataset is very small and may not represent the wide variety of language people actually use. Some labels are subjective and the dataset contains mostly English informal language.

---

## 3. How the Rule-Based Model Works

**Your scoring rules:**

The model:

* Converts text to lowercase.
* Removes punctuation.
* Splits text into individual words.
* Adds one point for positive words.
* Subtracts one point for negative words.
* Handles simple negation by reversing the sentiment of words following "not", "no", or "never".
* Detects mixed sentiment when both positive and negative signals appear in the same sentence.

**Strengths of this approach:**

* Easy to understand and explain.
* Predictable behavior.
* Performs well on simple positive and negative sentences.
* Easy to modify by changing rules.

**Weaknesses of this approach:**

The model struggles with:

* Sarcasm
* Context
* New slang
* Unseen vocabulary
* Complex emotional expressions

For example:

"Sure, because waiting in traffic for an hour is exactly how I wanted to spend my morning."

Humans recognize this as sarcastic and negative, but the model cannot understand the speaker's intent.

---

## 4. How the ML Model Works

**Features used:**

The ML model uses a Bag-of-Words representation created with CountVectorizer.

**Training data:**

The model trains using the `SAMPLE_POSTS` and `TRUE_LABELS` dataset.

**Training behavior:**

The model achieved 100% accuracy when evaluated on the same dataset it was trained on. However, when evaluated on a separate set of unseen test posts, the accuracy dropped to about 60%.

This demonstrated that training accuracy alone does not accurately measure real-world performance.

**Strengths and weaknesses:**

Strengths:

* Learns patterns automatically.
* Correctly classified several examples the rule-based model struggled with.
* Required fewer manually written rules.

Weaknesses:

* Performance depends heavily on the training data.
* Incorrectly classified unfamiliar slang such as "That movie was fire."
* Can overfit when evaluated on its training data.

---

## 5. Evaluation

**How you evaluated the model:**

The rule-based model was evaluated using the labeled examples in `dataset.py`.

The machine learning model was evaluated twice:

* On the training dataset.
* On a separate set of manually created test posts.

**Examples of correct predictions:**

"I love this class so much"

Correctly predicted as positive because it contains a strong positive word.

"Today was a terrible day"

Correctly predicted as negative because of the word "terrible."

"I am not happy about this"

Initially incorrect, but after adding negation handling, it was correctly classified as negative.

**Examples of incorrect predictions:**

"Sure, because waiting in traffic for an hour is exactly how I wanted to spend my morning."

The rule-based model could not recognize sarcasm because it only evaluates words rather than intent.

"That movie was fire."

The ML model predicted negative because it had not learned that "fire" can be positive slang.

"I have class at noon."

The ML model incorrectly predicted negative because it associated "class" with emotional examples from the training data.

---

## 6. Limitations

The biggest limitations are:

* Very small dataset.
* Limited vocabulary.
* Cannot reliably detect sarcasm.
* Difficult to interpret mixed emotions.
* Performance depends heavily on manually chosen words or labeled examples.
* Machine learning evaluation on the training dataset overestimates performance.

---

## 7. Ethical Considerations

Mood detection systems should not be used to make important decisions about people without human review.

Possible concerns include:

* Misclassifying messages expressing distress.
* Misinterpreting slang or dialect from different communities.
* Incorrectly identifying emotions because of sarcasm or context.
* Privacy concerns if personal messages are analyzed without permission.

---

## 8. Ideas for Improvement

Possible future improvements include:

* Collect a much larger and more diverse dataset.
* Create a dedicated training and testing split.
* Improve emoji and slang handling.
* Add better support for sarcasm and context.
* Experiment with TF-IDF instead of CountVectorizer.
* Compare additional machine learning models.
* Explore transformer-based language models for more accurate sentiment analysis.

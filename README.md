# Text Sentiment Classification with Logistic Regression

This project builds a sentiment analysis model to classify Amazon product reviews as positive or negative, using TF-IDF vectorization and logistic regression.

---

## Dataset

**Source:** [Kaggle - Amazon Reviews](https://www.kaggle.com/datasets/kritanjalijain/amazon-reviews)

- Contains over 30 million labeled reviews.
- Labels:
  - `1` = negative  
  - `2` = positive  
  - Remapped to binary `0` and `1` for classification.

---

## Data Inspection

- Key columns: `label`, `title`, `text`
- Balanced class distribution
- Minimal missing values
- Some reviews are duplicates or have low word count

### Cleaning Steps

- Dropped reviews under 3 words
- Removed duplicate rows based on combined `text` and `title`
- Mapped sentiment labels to `0` (negative) and `1` (positive)

---

## Model Pipeline

Built using scikit-learn:

### TF-IDF Vectorization

TF-IDF (Term Frequency-Inverse Document Frequency) transforms raw text into numerical features that reflect the importance of terms in context.

**Vectorizer parameters:**
- `max_features=10000`
- `ngram_range=(1, 2)`
- `min_df=10`
- `max_df=0.85`
- `stop_words='english'`

### Logistic Regression Classifier

Logistic regression is used for its efficiency and strong performance on high-dimensional sparse data like TF-IDF. No hyperparameter tuning was applied for this baseline.

---

## Evaluation

Model performance on 400,000 review samples:

- **Accuracy:** 88%
- **Precision, Recall, F1-score:** 88% (balanced across both classes)

Logistic regression offers a strong baseline with fast training and interpretable outputs. Further improvements could involve hyperparameter tuning or testing alternative classifiers.

---

## Saving Model

```python
import joblib
joblib.dump(model, "sentiment_model.pkl")
```

---

# Text Sentiment Classification with Logistic Regression

This project builds a sentiment analysis model to classify Amazon product reviews as positive or negative, using TF-IDF vectorization and logistic regression. A [dashboard](https://reviewsentiment.streamlit.app/) using the model is deployed using Streamlit Cloud.

---

## Files Overview

- **reviews-sentiment.ipynb**  
  Jupyter notebook that walks through the process of creating and evaluating the classification model.

- **sentiment_app.py**  
  Streamlit app script for running the sentiment analysis dashboard, including review input, batch CSV upload, and visualizations.

- **model_utils.py**  
  Utility functions for model prediction, data processing, and visualization.

- **sentiment_model.pkl**  
  Pre-trained logistic regression sentiment classification model saved with joblib. Created in reviews-sentiment.ipynb.

- **sample_with_time.csv**  
  Sample test dataset with synthetic timestamps for demonstration and testing in the app.

- **requirements.txt**  
  Python dependencies needed to run the app and notebook.

- **about.md**  
  Markdown file containing app description, usage details, dataset information, and limitations. Loaded in the about section of Streamlit.



## Dataset

**Source:** [Kaggle - Amazon Reviews](https://www.kaggle.com/datasets/kritanjalijain/amazon-reviews)

- Contains over 30 million labeled reviews.
- Labels:
  - `1` = negative  
  - `2` = positive  
  - Remapped to binary `0` and `1` for classification.

  The Amazon Reviews Polarity dataset was created by Xiang Zhang et al. and used as a benchmark in the paper: Character-level Convolutional Networks for Text Classification, NIPS 2015.

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

## Limitations
- The training dataset excludes neutral reviews (e.g., 3-star ratings), so the model cannot identify neutral or mixed sentiments.

- Predictions are restricted to positive or negative classes, limiting detection of subtle sentiment variations.

- The model relies on TF-IDF features derived from Amazon product reviews, which may not fully capture nuanced sentiment or generalize effectively to other domains or types of text.

## Usage
- The model is available for use at [reviewsentiment.streamlit.app].
- To run locally, install dependencies via pip install -r requirements.txt and launch the app with streamlit run 'sentiment_app.py'.
- To recreate or modify the model, run 'reviews-sentiment.ipynb' and run all cells sequentially.
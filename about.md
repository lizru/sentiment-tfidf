## About
This app performs text sentiment classification on customer reviews using a logistic regression model with TF-IDF features.

---

### How It Works

- The model classifies reviews as positive or negative based on their text.
- It uses TF-IDF vectorization to convert text training data into numerical features.
    - TF-IDF (Term Frequency-Inverse Document Frequency) is a method that scores words based on how important they are in the dataset by balancing how often they appear in one review versus across all reviews.
- Logistic regression classifies sentiment based on the TF-IDF features by estimating probabilities of the text being positive/negative.

---

### Dataset & Sample Data

The preloaded sample data is the test dataset, with synthetically generated timestamps added for demonstration purposes.
- **Train/test data source**: [Amazon Review Polarity Dataset](https://www.kaggle.com/datasets/kritanjalijain/amazon-reviews)
- **Dataset Citation**: The Amazon Reviews Polarity dataset was created by Xiang Zhang et al. and used as a benchmark in the paper: *Character-level Convolutional Networks for Text Classification, NIPS 2015*.
- Reviews with 1 and 2 stars are labeled Negative (0), 4 and 5 stars labeled Positive (1).

---

### Model Details

- **Vectorizer**: TF-IDF with bigrams, max 10,000 features, English stop words removed.
- **Classifier**: Logistic Regression.
- **Performance**: ~88% accuracy on test data.

### Limitations
- The training dataset excludes neutral (3-star) reviews, so the model has been trained to distinguish positive and negative sentiments as a binary.
- Because predictions are limited to positive or negative classes, subtle, neutral, or mixed sentiments may not be captured.
- The model uses TF-IDF features, which capture word importance based on the training data (Amazon product reviews) rather than true semantic meaning. It may miss nuanced sentiment and may not generalize well to different domains.

---

For more information, see the [project repository](https://github.com/lizru/sentiment-tfidf).

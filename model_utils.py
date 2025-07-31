import joblib
import numpy as np
import pandas as pd
import seaborn as sns
sns.set_style("darkgrid")
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer


model = joblib.load('sentiment_model.pkl')




def validate_and_predict(model, texts, return_proba=True):
    """
    Takes a string or list of strings, cleans input, and returns either binary predicitons or probabilities. 0 is negative sentiment, 1 is positive.
        Params:
            model: trained pipeline
            texts: str or list of str
            return_proba: boolean indicating whether the return includes probabilities. 
                          Probabilities will be returned in [(Chance negative), (Chance positive)] form.
        Returns an array of either predictions & probabilities or only predictions.
    """

    if isinstance(texts, (int, float)):
        texts = str(texts)

    if isinstance(texts, str):
        texts = [texts]

    elif not hasattr(texts, '__iter__'):
        raise ValueError("Input must be a string or list of strings.")
    
    texts = [t.strip() for t in texts]

    if return_proba:
        return model.predict(texts), model.predict_proba(texts)
    else:
        return model.predict(texts)
    

def add_preds_to_df(df, preds):
    """Attaches the model's predictions to the original input."""
    df = df.copy()
    df['preds'] = preds
    return df

def add_probs_to_df(df, probs):
    """Attaches the model's probabilities to the original input."""
    df = df.copy()
    df['probs'] = probs
    return df

def get_class_distribution(preds):
    """
    Gets the percent of positive and negative reviews.
    Param: preds, the list of class predictions
    Returns percentages as rounded integers.
    """
    total = len(preds)
    percent_pos = (np.sum(preds == 1) / total) * 100
    percent_neg = (np.sum(preds == 0) / total) * 100
    return round(percent_pos), round(percent_neg)


def get_average_confidence(probs):
    """
    Gets the average confidence for all preds.
    Returns as a percentage with 1 decimal.
    """
    # uses the max to find which class was predicted, averages all
    avg_confidence = np.mean(np.max(probs, axis=1))
    percent_avg_confidence = round((avg_confidence * 100), 1)
    return percent_avg_confidence



def create_prob_kde(preds, probs, bw):
    """Creates a KDE of the probabilites with adustable bandwidth."""

    # df of predicitons and sliced positive probs
    df = pd.DataFrame({
        'prediction': preds,
        'positive_prob': probs[:,1]
    })

    # filter positive predictions among both classes
    pos_probs_pos_pred = df.loc[df['prediction'] == 1, 'positive_prob']
    pos_probs_neg_pred = df.loc[df['prediction'] == 0, 'positive_prob']

    fig, ax = plt.subplots()
    sns.kdeplot(pos_probs_pos_pred, fill=True, bw_adjust=bw, label='Predicted Positive', ax=ax)
    sns.kdeplot(pos_probs_neg_pred, fill=True, bw_adjust=bw, label='Predicted Negative', ax=ax)
    ax.set_xlabel('Positive Class Probility')
    ax.set_ylabel('Density')
    ax.set_title('Model Confidence by Predicted Class')
    ax.legend()
    return fig

def create_prob_hist(preds, probs, bins=30):
    """Creates a histogram of the probabilities with adjustable bin size."""
    # df of predicitons and sliced positive probs
    df = pd.DataFrame({
        'prediction': preds,
        'positive_prob': probs[:,1]
    })

    # filter positive predictions among both classes
    pos_probs_pos_pred = df.loc[df['prediction'] == 1, 'positive_prob']
    pos_probs_neg_pred = df.loc[df['prediction'] == 0, 'positive_prob']
    fig, ax = plt.subplots()
    ax.hist(pos_probs_pos_pred, bins=bins, alpha=0.6, label='Predicted Positive', color='blue', density=True)
    ax.hist(pos_probs_neg_pred, bins=bins, alpha=0.6, label='Predicted Negative', color='red', density=True)
    ax.set_xlabel('Positive Class Probability')
    ax.set_ylabel('Density')
    ax.set_title('Model Confidence by Predicted Class')
    ax.legend()
    return fig



def create_class_bar(preds):
    """Creates a bar chart showing the class distribution of predictions."""
    counts = {
        'positive': (preds == 1).sum(),
        'negative': (preds == 0).sum()
    }
    fig, ax = plt.subplots()
    ax.bar(counts.keys(), counts.values(), color=['blue', 'red'])
    ax.set_ylabel('Count')
    ax.set_title('Count for Each Class')
    # loops thru index/value and places count value above the corresponding bar
    for i, v in enumerate(counts.values()):
        ax.text(i, v + max(counts.values())*0.01, str(v), ha='center', va='bottom')

    return fig

def get_tfidf(texts, n=10, max_features = 10000):
    """Creates a list of top scoring TF-IDF words. Returns a series of top words and scores."""
    vectorizer = TfidfVectorizer(stop_words='english', max_features=max_features)
    X = vectorizer.fit_transform(texts)

    # averages across matrix
    mean_tfidf = X.mean(axis=0).A1 
    terms = vectorizer.get_feature_names_out()
    tfidf_scores = pd.Series(mean_tfidf, index=terms)

    # returns the series descending, first n entries
    return tfidf_scores.sort_values(ascending=False).head(n)



def get_top_negative_words(df, preds):
    """
    Uses TF-IDF to find words associated with low sentiment.
    Params:
        df: input dataframe
        preds: the model's predictions
    """
    df = df.copy()
    df = add_preds_to_df(df, preds)
    neg_df = df[df['preds'] == 0]
    top_terms = get_tfidf(neg_df['text'])

    fig, ax = plt.subplots()
    # series plotting
    top_terms.plot.barh(ax=ax)
    ax.set_xlabel("Average TF-IDF Score")
    ax.set_title("Highlighted Terms in Negative Reviews")

    return fig


def filter_5th_percent(df, preds, probs):
    """
    Finds the reviews & data in the top and bottom 5th percentiles of model's probabilites.
    Params:
        df: input df
        preds: model predictions
        probs: model probabilities
    Returns:
        strongest_positive_df: top 5% in positive scores
        strongest_negative_df: bottom 5% of positive scores
    """
    df = add_preds_to_df(df, preds)
    df = add_probs_to_df(df, probs)

    # filter positive probabilites
    df['positive_prob'] = probs[:, 1]

    lower_thresh = df['positive_prob'].quantile(0.05)
    upper_thresh = df['positive_prob'].quantile(0.95)

    strongest_positive_df = df[df['positive_prob'] >= upper_thresh] 
    strongest_negative_df = df[df['positive_prob'] <= lower_thresh]
    

    return strongest_positive_df, strongest_negative_df
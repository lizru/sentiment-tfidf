import joblib
import numpy as np
import pandas as pd
import seaborn as sns
sns.set_style("darkgrid")
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
import streamlit as st
import plotly.graph_objects as go
from scipy.stats import gaussian_kde
from collections import defaultdict


BLUE = '#4A90E2'
RED = '#D46A6A'


@st.cache_resource
def get_model():
    """Returns sentiment model."""
    model = joblib.load('sentiment_model.pkl')
    return model

@st.cache_resource
def get_preloaded_data(file="test.csv"):
    """Returns dataset."""
    df = pd.read_csv(file)
    return df




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
    
    texts = [str(t).strip() for t in texts if str(t).strip() != '']

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
    """Attaches the model's positive probabilities to the original input."""
    df = df.copy()
    df['positive_prob'] = probs[:, 1]
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
    """Returns a KDE of the probabilities with adjustable bandwidth."""
    # df of predictions and sliced positive probs
    df = pd.DataFrame({'prediction': preds, 'positive_prob': probs[:,1]})

    # filter positive predictions among both classes
    pos_probs_pos_pred = df.loc[df['prediction'] == 1, 'positive_prob'].values
    pos_probs_neg_pred = df.loc[df['prediction'] == 0, 'positive_prob'].values

    # generate evaluation points
    x_eval = np.linspace(0, 1, 200)

    # KDEs with bandwidth adjustment
    kde_pos = gaussian_kde(pos_probs_pos_pred)
    kde_pos.set_bandwidth(bw / kde_pos.factor)
    kde_neg = gaussian_kde(pos_probs_neg_pred)
    kde_neg.set_bandwidth(bw / kde_neg.factor)

    y_pos = kde_pos(x_eval)
    y_neg = kde_neg(x_eval)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x_eval, y=y_pos, mode='lines', fill='tozeroy', name='Predicted Positive', line_color=BLUE))
    fig.add_trace(go.Scatter(x=x_eval, y=y_neg, mode='lines', fill='tozeroy', name='Predicted Negative', line_color=RED))

    fig.update_layout(xaxis_title=' ', yaxis_title='Density', title=' ', legend=dict(title='Prediction'))

    return fig



def create_prob_hist(preds, probs, bins=30):
    """Returns a histogram of the probabilities with adjustable bin size."""
    # df of predictions and sliced positive probs
    df = pd.DataFrame({'prediction': preds, 'positive_prob': probs[:,1]})

    # filter positive predictions among both classes
    pos_probs_pos_pred = df.loc[df['prediction'] == 1, 'positive_prob']
    pos_probs_neg_pred = df.loc[df['prediction'] == 0, 'positive_prob']

    fig = go.Figure()
    fig.add_trace(go.Histogram(x=pos_probs_pos_pred, nbinsx=bins, opacity=0.6, name='Predicted Positive', marker_color=BLUE, histnorm='probability density'))
    fig.add_trace(go.Histogram(x=pos_probs_neg_pred, nbinsx=bins, opacity=0.6, name='Predicted Negative', marker_color=RED, histnorm='probability density'))

    fig.update_layout(barmode='overlay', xaxis_title='Positive Class Probability', yaxis_title='Density', title=' ', legend=dict(title='Prediction'))

    return fig


def create_class_bar(preds):
    """Creates a bar chart showing the class distribution of predictions."""
    counts = {
        'positive': int((preds == 1).sum()),
        'negative': int((preds == 0).sum())
    }

    fig = go.Figure(go.Bar(
        x=list(counts.keys()),
        y=list(counts.values()),
        marker_color=[BLUE, RED],
        text=list(counts.values()),
        textposition='outside'
    ))

    fig.update_layout(
        yaxis_title='Count',
        # extra space for labels
        yaxis=dict(showticklabels=False, range=[0, max(counts.values()) * 1.2]),
        margin=dict(t=60)
    )

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
        df: full input dataframe, positives included
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

    strongest_positive_df = df[df['positive_prob'] >= upper_thresh].reset_index(drop=True)
    strongest_negative_df = df[df['positive_prob'] <= lower_thresh].reset_index(drop=True)
    

    return strongest_positive_df, strongest_negative_df



def explain_predictions(model, texts, top_n=10):
    """
    Aggregates top contributions across texts.
    Returns top_n words with highest absolute total contribution.
    """
    vectorizer = model.named_steps['tfidf']
    clf = model.named_steps['clf']
    coefs = clf.coef_[0]
    feature_names = vectorizer.get_feature_names_out()

    agg_contribs = defaultdict(float)

    # loops thru all texts and aggregates tf-idf * clf coef to get full weight
    for text in texts:
        X = vectorizer.transform([text])
        nonzero_indices = X.nonzero()[1]
        for i in nonzero_indices:
            agg_contribs[feature_names[i]] += X[0, i] * coefs[i]

    # sort by absolute value to find top contributers
    top_words = sorted(agg_contribs.items(), key=lambda x: abs(x[1]), reverse=True)[:top_n]

    return top_words




def plot_explanation(model, text, top_n=10):
    """Returns a Plotly figure of top contributions."""
    contributions = explain_predictions(model, text, top_n=10)
    if not contributions:
        return None

    words, scores = zip(*contributions)
    # blue for positive scores, red for negative
    colors = [BLUE if s > 0 else RED for s in scores]

    fig = go.Figure(go.Bar(x=scores, y=words, orientation='h', marker_color=colors))
                           
    fig.update_layout(
        xaxis=dict(showgrid=True, gridcolor='lightgrey', zeroline=False),
        yaxis=dict(showgrid=True, gridcolor='lightgrey', zeroline=False, autorange='reversed'),
        plot_bgcolor='white'
    )
    return fig
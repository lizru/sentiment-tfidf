import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import model_utils as mu





def display_dash(df):
    """Displays full dashboard from valid DataFrame input."""
    st.subheader("Model Predictions and Evaluations")

    preds, probs = mu.validate_and_predict(mu.get_model(), df['text'])
   

    # show three summary nums
    sum1, sum2, sum3 = st.columns(3)
    with sum1:
        st.header(f"{len(preds)}")
        st.write("Reviews Analyzed")
    with sum2:
        pos_preds = preds[preds==1]
        st.header(f"{((len(pos_preds)/len(preds))*100):.1f}%")
        st.write("Classified Positive")

    with sum3:
        avg_con = float(mu.get_average_confidence(probs))
        st.header(f"{avg_con:.1f}%")
        st.write("Average Confidence")


    # top positive & negative reviews shown in collapsable dfs, two columns
    pos_col, neg_col = st.columns(2)
    strong_pos, strong_neg = mu.filter_5th_percent(df, preds, probs)

    with pos_col:
        with st.expander("Strongest Negative Reviews"):
            st.dataframe(strong_neg[['text', 'positive_prob']].sort_values('positive_prob'))
    with neg_col:
        with st.expander("Strongest Positive Reviews"):
            st.dataframe(strong_pos[['text', 'positive_prob']].sort_values('positive_prob', ascending=False))


    # shows either the kde or the histogram
    prediction_viz = st.radio("Confidence of prediction display: ", ["Histogram", "KDE"])
    if prediction_viz == "Histogram":
        st.pyplot(mu.create_prob_hist(preds, probs, bins=30))

    if prediction_viz == "KDE":
        bw = st.slider("Bandwidth Adjustment (KDE only)", 0.1, 2.0, 0.5, 0.1)
        st.pyplot(mu.create_prob_kde(preds, probs, bw))


    # shows the class balance and negative TF-IDF terms
    display1, display2 = st.columns(2)
    
    with display1:
        st.pyplot(mu.create_class_bar(preds))

    with display2:
        st.pyplot(mu.get_top_negative_words(df, preds))



    








# sidebar navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Analyze Reviews", "About"])

if page == "Analyze Reviews":
    st.title("Reviews Sentiment Dashboard")

    # select type of input
    mode = st.radio("Select input method:", ["Preloaded Sample Data", "Single Review", "Upload CSV"])


    # single reviews as a text box
    if mode == "Single Review":
        text_input = st.text_area("Enter a customer review: ")
        if st.button("Analyze") and text_input:
            pred, prob = mu.validate_and_predict(mu.get_model(), text_input)
            pred_label = ('Positive' if pred[0] == 1 else 'Negative')
            confidence = prob[0][pred[0]]
            st.write(f"**Prediction**: {pred_label} Confidence: {confidence:.2f}")


    # if mode is the preloaded dataset (Amazon test set), display full dash
    if mode == "Preloaded Sample Data":
        df = mu.get_preloaded_data("sample.csv")
        if df.shape[1] == 1:
            df.columns = ['text']
        elif 'text' not in df.columns:
            st.error("Uploaded CSV must contain a column named 'text'.")
        else:
            df = df[['text']]
        display_dash(df)

    # if mode is the uploaded csv, verify input & display full dash
    if mode == "Upload CSV":
        uploaded_file = st.file_uploader("Upload a CSV file", type=['csv'])
        if uploaded_file is not None:
            df = pd.read_csv(uploaded_file)
            # if one col name text, else raise error, delete other cols
            if df.shape[1] == 1:
                df.columns = ['text']
            elif 'text' not in df.columns:
                st.error("Uploaded CSV must contain a column named 'text'.")
            else:
                df = df[['text']]
            display_dash(df)



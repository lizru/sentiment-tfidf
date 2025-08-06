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
        amount_pos = (len(pos_preds)/len(preds))
        st.header(f"{(amount_pos*100):.1f}%")
        st.write("Classified Positive")

    with sum3:
        avg_con = float(mu.get_average_confidence(probs))
        st.header(f"{avg_con:.1f}%")
        st.write("Average Confidence")


   





    # shows either the kde or the histogram
    prediction_viz = st.radio("Confidence of prediction display: ", ["Histogram", "KDE"])
    st.subheader("Prediction Confidence Distribution")

    if prediction_viz == "Histogram":
        st.plotly_chart(mu.create_prob_hist(preds, probs, bins=30))

    if prediction_viz == "KDE":
        bw = st.slider("Bandwidth Adjustment (KDE only)", 0.1, 2.0, 0.5, 0.1)
        st.plotly_chart(mu.create_prob_kde(preds, probs, bw))



    st.markdown("---")


    # top positive & negative reviews shown in collapsable dfs, two columns
    pos_col, neg_col = st.columns(2)
    strong_pos, strong_neg = mu.filter_5th_percent(df, preds, probs)

    with pos_col:
        with st.expander("Show Strongest Negative Reviews"):
            st.dataframe(strong_neg[['text', 'positive_prob']].sort_values('positive_prob'))
    with neg_col:
        with st.expander("Show Strongest Positive Reviews"):
            st.dataframe(strong_pos[['text', 'positive_prob']].sort_values('positive_prob', ascending=False))




    st.markdown("---")




    # shows the class balance and negative TF-IDF terms
    display1, col_divide, display2 = st.columns([4, .05, 4])

    col_divide.markdown(
        """
        <style>
        .divider {
            border-left: 1px solid #ddd;
            height: 100%;
            margin: 0 auto;
        }
        </style>
        <div class="divider"></div>
        """,
        unsafe_allow_html=True
    )
 
    with display2:
        st.subheader("Prediction Class Distribution")
        if amount_pos > .75:
            balance_cat = 'mostly positive'
        if amount_pos < .25:
            balance_cat = 'mostly negative'
        else:
            balance_cat = 'mixed'
        st.write(f"*Reviews are {balance_cat}.*")
        st.plotly_chart(mu.create_class_bar(preds))

    with display1:
        st.subheader("Key Terms Driving Predictions")
        st.write("*Blue indicates positive influence; red indicates negative influence*")
        st.plotly_chart(mu.plot_explanation(mu.get_model(), df['text'], top_n=10))
        



    








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



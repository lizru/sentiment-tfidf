import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import model_utils as mu



# adjusts to fit whole page
#st.set_page_config(layout="wide")



def display_dash(df):
    """Displays full dashboard from valid DataFrame input."""

    st.write("---")
    st.subheader("Predictions Overview")
    
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


   
    # shows time-series plot
    st.subheader("Sentiment Over Time")
    st.write("*Smoothed to reduce daily volatility; values reflect the average sentiment over the past 3 days.*")
    st.plotly_chart(mu.plot_time_series(df, probs), use_container_width=True)
    st.caption("Scores above 0.5 indicate positive sentiment; below 0.5 indicate negative.")
    
    


    
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




    # class distribution/terms in cols
    display1, extra, display2 = st.columns([3,.5,3])


    with display1:  
        # shows bar chart of class
        st.subheader("Class Distribution")
        st.plotly_chart(mu.create_class_bar(preds))
        if amount_pos > .75:
            balance_cat = 'mostly positive'
        if amount_pos < .25:
            balance_cat = 'mostly negative'
        else:
            balance_cat = 'mixed'
        st.caption(f"*The sentiment of reviews is {balance_cat}.*")

    with extra:
        st.write("")
    # shows the top prediction drivers
    with display2:
        st.subheader("Key Prediction Drivers")
        st.plotly_chart(mu.plot_explanation(mu.get_model(), df['text'], top_n=10))
        st.caption("*Blue indicates positive influence; red indicates negative influence.*")
        
    st.write("---")

    # shows either the kde or the histogram
    st.subheader("Distribution of Model Prediction Probabilities")
    prediction_viz = st.radio("Select view: ", ["Histogram", "KDE"])
    if prediction_viz == "Histogram":
        st.plotly_chart(mu.create_prob_hist(preds, probs, bins=30))

    if prediction_viz == "KDE":
        bw = st.slider("Bandwidth Adjustment (KDE only)", 0.1, 2.0, 0.5, 0.1)
        st.plotly_chart(mu.create_prob_kde(preds, probs, bw))

    








# sidebar navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Analyze Reviews", "About"])

if page == "Analyze Reviews":
    st.title("Customer Review Sentiment Analysis")

    # select type of input
    mode = st.radio("Select input method:", ["Preloaded Sample Data", "Single Review", "Upload CSV"])


    # single reviews as a text box
    if mode == "Single Review":
        text_input = st.text_area("Enter a customer review: ")
        if st.button("Analyze") and text_input:
            pred, prob = mu.validate_and_predict(mu.get_model(), text_input)
            pred_label = ('Positive' if pred[0] == 1 else 'Negative')
            confidence = prob[0][pred[0]]

            # unsure flag
            if confidence < .6:
                st.caption("*Low confidence: the model is uncertain about this review’s sentiment.*")
            
            st.write(f"**Prediction**: {pred_label}")
            st.write(f"**Confidence**: {confidence:.0%}")
        
        st.write("---")
        with st.expander("Limitations:"):
           st.write(
            "*This model was trained on product reviews and may not generalize well to other domains, including slang or abbreviations. "
            "Frequent misclassifications suggest it may not be suitable for your use case.*"
        )

            
            



    # if mode is the preloaded dataset (Amazon test set), display full dash
    if mode == "Preloaded Sample Data":
        df = mu.get_preloaded_data("sample_with_time.csv")
        if df.shape[1] == 2:
            df.columns = ['text', 'date']
        elif 'text' not in df.columns:
            st.error("Uploaded CSV must contain a column named 'text'.")
        else:
            df = df[['text', 'date']]
        display_dash(df)

    # if mode is the uploaded csv, verify input & display full dash
    if mode == "Upload CSV":
        uploaded_file = st.file_uploader("Upload a CSV file", type=['csv'])
        st.write("Please ensure the uploaded CSV meets one of the following:")
        st.write("*a) Contains at least the columns 'text' and 'date' (other columns will be ignored).*")
        st.write("*b) Exactly two columns without headers, assumed to be text and date in order.*")

        if uploaded_file is not None:
            df = pd.read_csv(uploaded_file)
            # if one col name text, else raise error, delete other cols
            if df.shape[1] == 2:
                df.columns = ['text', 'date']
            elif 'text' in df.columns and 'date' in df.columns:
                df = df[['text', 'date']]
            else:
                st.error("Uploaded CSV must contain exactly two columns without headers, or columns named 'text' and 'date'.")
            display_dash(df)




if page == "About":
    # loads markdown about file
    with open("about.md", "r") as file:
        about_md = file.read()
    st.write(about_md)
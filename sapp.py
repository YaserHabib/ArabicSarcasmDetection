import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report
import joblib

# Streamlit webpage layout
st.title("Tweet Analysis and ML Model Training")

# File Upload
uploaded_file = st.file_uploader("Choose a file")
if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)
    st.write(data.head())

    # Feature Extraction
    tfidf_vectorizer = TfidfVectorizer(max_features=1000)
    X = tfidf_vectorizer.fit_transform(data['tweet'])
    y = data['sarcasm']

    # Splitting the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    # Model Training
    if st.button('Train Model'):
        model = MultinomialNB()
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        # Displaying results
        st.write('Model Performance:')
        st.text(classification_report(y_test, y_pred))

        # Saving and downloading the model
        joblib.dump(model, 'trained_model.joblib')
        st.download_button(label='Download Model',
                           data='trained_model.joblib',
                           file_name='trained_model.joblib')

# To run the app, use `streamlit run your_app.py` in your terminal

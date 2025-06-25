import streamlit as st
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

# Page setup
st.set_page_config(page_title="Smart Grocery Advisor", page_icon="🤖", layout="centered")

# Header
st.markdown("<h1 style='text-align: center; color: #2ecc71;'>SMART GROCERY ADVISOR</h1>", unsafe_allow_html=True)
st.image("https://img.icons8.com/emoji/96/robot-emoji.png", width=80)

# Tagline
st.markdown(
    "<p style='text-align: center; font-size: 18px; color: #7f8c8d;'>"
    "Your intelligent assistant for classifying foods and drinks."
    "</p>",
    unsafe_allow_html=True
)

# Load and train model
df = pd.read_csv("food_data.csv")
model = Pipeline([
    ('vectorizer', CountVectorizer()),
    ('classifier', MultinomialNB())
])
model.fit(df['food_name'], df['category'])

# Prediction
def classify_with_confidence(food):
    probs = model.predict_proba([food])[0]
    prediction = model.predict([food])[0]
    confidence = max(probs)
    return prediction, confidence

# Input field
food = st.text_input("🍎 Enter a food name to classify:")

# Button and result
if st.button("CLASSIFY"):
    if food:
        pred, conf = classify_with_confidence(food)
        st.markdown(
            f"<div style='text-align: center; font-size: 20px; color: #27ae60;'>"
            f"✅ <b>{food.capitalize()}</b> is classified as: <b>{pred}</b> ({conf:.2f} confidence)</div>",
            unsafe_allow_html=True
        )
    else:
        st.warning("Please enter a food name.")

# Footer
st.markdown("<hr style='border-color: #ecf0f1;'>", unsafe_allow_html=True)
st.markdown(
    "<p style='text-align: center; font-size: 14px; color: #bdc3c7;'>Created by Nollin Masai Wabuti</p>",
    unsafe_allow_html=True
)

import streamlit as st
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

# Load dataset
df = pd.read_csv("food_data.csv")

# Normalize food names to lowercase for consistent training
df['food_name'] = df['food_name'].str.lower()

# Train the ML pipeline
model = Pipeline([
    ('vectorizer', CountVectorizer()),
    ('classifier', MultinomialNB())
])
model.fit(df['food_name'], df['category'])

# Function to classify new food input
def classify_with_confidence(food):
    food = food.lower()
    probs = model.predict_proba([food])[0]
    prediction = model.predict([food])[0]
    confidence = max(probs)
    return prediction, confidence

# Streamlit App Interface
st.markdown("<h1 style='text-align: center; color: green;'>SMART GROCERY ADVISOR</h1>", unsafe_allow_html=True)
st.image("https://img.icons8.com/emoji/96/robot-emoji.png", width=80)
st.write("Your intelligent assistant for classifying foods and drinks.")

# User input
food = st.text_input("🍎 Enter a food name to classify:")

if st.button("CLASSIFY"):
    if food.strip() == "":
        st.warning("⚠️ Please enter a food name.")
    else:
        prediction, confidence = classify_with_confidence(food)
        st.success(f"✅ **{food.title()}** is classified as: **{prediction}**\n\nConfidence: **{confidence:.2f}**")

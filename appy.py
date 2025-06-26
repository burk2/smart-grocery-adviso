import streamlit as st
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

# Load data
df = pd.read_csv("food_data.csv")

# Lowercase food names for better matching
df['food_name'] = df['food_name'].str.lower()

# Build model
model = Pipeline([
    ('vectorizer', CountVectorizer()),
    ('classifier', MultinomialNB())
])
model.fit(df['food_name'], df['category'])

# Classify with confidence
def classify_with_confidence(food):
    food = food.lower()
    probs = model.predict_proba([food])[0]
    prediction = model.predict([food])[0]
    confidence = max(probs)
    return prediction, confidence

# Streamlit UI
st.markdown("<h1 style='text-align: center; color: green;'>SMART GROCERY ADVISOR</h1>", unsafe_allow_html=True)
st.image("https://img.icons8.com/emoji/96/robot-emoji.png", width=80)

food = st.text_input("🍎 Enter a food name to classify:")

if st.button("CLASSIFY"):
    if food.strip() == "":
        st.warning("⚠️ Please enter a food name.")
    else:
        pred, conf = classify_with_confidence(food)
        if conf < 0.5:
            st.warning(f"🤖 I'm unsure... but it might be: **{pred}** (Confidence: {conf:.2f})")
        else:
            st.success(f"✅ {food.title()} is classified as: **{pred}** ({conf:.2f} confidence)")

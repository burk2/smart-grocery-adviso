import streamlit as st
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

# Load dataset
df = pd.read_csv("food_data.csv")

# Display sample data
st.subheader("📦 Sample of your food data")
st.dataframe(df.head())

# Train model
model = Pipeline([
    ('vectorizer', CountVectorizer()),
    ('classifier', MultinomialNB())
])
model.fit(df['food_name'], df['category'])

# Prediction function
def classify_with_confidence(food):
    probs = model.predict_proba([food])[0]
    prediction = model.predict([food])[0]
    confidence = max(probs)
    return prediction, confidence

# UI
st.markdown("<h1 style='text-align: center; color: green;'>SMART GROCERY ADVISOR</h1>", unsafe_allow_html=True)
st.image("https://img.icons8.com/emoji/96/robot-emoji.png", width=80)

sample_foods = df['food_name'].sample(min(5, len(df))).tolist()
st.caption("💡 Try one of these: " + ", ".join(sample_foods))

food = st.text_input("🍎 Enter a food name to classify:")

if st.button("CLASSIFY"):
    if food:
        pred, conf = classify_with_confidence(food)
        if conf < 0.5:
            st.warning("🤖 Hmm... I'm not sure about this food.")
        else:
            st.success(f"✅ **{food.capitalize()}** is classified as: **{pred}** (confidence: {conf:.2f})")
    else:
        st.warning("⚠️ Please enter a food name.")

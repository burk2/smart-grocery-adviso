import streamlit as st
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

# Load data
df = pd.read_csv("world_food_data_7000.csv")

# Ensure column names are lowercase
df.columns = df.columns.str.lower()

# Lowercase all food names
df['food_name'] = df['food_name'].str.lower()

# Build ML pipeline
model = Pipeline([
    ('vectorizer', CountVectorizer()),
    ('classifier', MultinomialNB())
])

# Train the model
model.fit(df['food_name'], df['category'])

# Classification function
def classify_food(food_input):
    food_input = food_input.lower()
    prediction = model.predict([food_input])[0]
    confidence = max(model.predict_proba([food_input])[0])
    return prediction, confidence

# Streamlit App UI
st.set_page_config(page_title="Smart Grocery Advisor", layout="centered")

st.title("🧠 SMART GROCERY ADVISOR")
st.subheader("Your intelligent assistant for classifying foods and drinks.")
st.markdown("Enter a food or drink name below to get its category classification:")

food_input = st.text_input("🍎 Enter a food name to classify:")

if food_input:
    predicted_category, confidence = classify_food(food_input)
    st.success(f"✅ **{food_input.title()}** is classified as: **{predicted_category}**")
    st.write(f"🔍 **Confidence**: `{confidence:.2f}`")
else:
    st.info("Please enter a food name to see the prediction.")

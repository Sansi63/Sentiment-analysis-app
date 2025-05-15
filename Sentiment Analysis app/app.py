import streamlit as st
import joblib

# Load vectorizer and model
vectorizer = joblib.load('tfidf_vectorizer.pkl')
model = joblib.load('svc_model.pkl')

# Streamlit UI
st.title("🧠 Sentiment Analysis App")
st.markdown("Enter a message below and the model will predict the sentiment.")

# Text input
user_input = st.text_area("✍️ Enter your message here:")

# Predict on button click
if st.button("Predict Sentiment"):
    if user_input.strip() == "":
        st.warning("Please enter some text.")
    else:
        # Vectorize and predict
        text_tfidf = vectorizer.transform([user_input])
        prediction = model.predict(text_tfidf)[0]
        st.success(f"**Predicted Sentiment:** {prediction}")

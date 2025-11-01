import streamlit as st
import pickle
from preprocess_msg import preprocessing_msg as pre_msg

# Load model and vectorizer
tfidf = pickle.load(open('vectorized.pkl', 'rb'))
model = pickle.load(open('model.pkl', 'rb'))

# Page configuration
st.set_page_config(page_title="📧 Email Spam Classifier", page_icon="✉️", layout="centered")

# Custom CSS styling
st.markdown("""
    <style>
    body {
        background-color: #f0f2f6;
    }
    .stTextInput input {
        border: 2px solid #4CAF50;
        border-radius: 10px;
        padding: 10px;
    }
    .stButton>button {
        background-color: #4CAF50;
        color: white;
        border-radius: 10px;
        padding: 8px 16px;
        font-weight: bold;
        transition: 0.3s;
    }
    .stButton>button:hover {
        background-color: #45a049;
        color: white;
        transform: scale(1.05);
    }
    </style>
""", unsafe_allow_html=True)

# Title
st.title("📩 Email & SMS Spam Classifier")
st.write("Type your message below and find out whether it's **Spam or Not Spam** ⚡")

# Input box
msg = st.text_area("Enter your message:", placeholder="Type something like — 'Congratulations! You won a lottery worth $1,000,000!'")

transform_msg=pre_msg.transform_text(msg)
vectorized_input=tfidf.transform([transform_msg])
prediction=model.predict(vectorized_input)
# Predict button
btn=st.button("predict")
if btn and len(msg)>0:
        # Result display
        if prediction == 0:
            st.success("✅ This message is **Not Spam**.")
            st.balloons()
        else:
            st.error("🚨 This message is **Spam!** Be careful.")
            st.snow()

import streamlit as st
import joblib

# Load model & vectorizer
model = joblib.load('spam_classifier_model.pkl')
vectorizer = joblib.load('count_vectorizer.pkl')

# 🎯 App Title
st.markdown("<h1 style='text-align: center; color: orange;'>📩 SMS Spam Classifier 🔍</h1>", unsafe_allow_html=True)
st.markdown("---")

# 💬 User Input
st.markdown("### 📨 Enter your message below:")
user_input = st.text_area("")

# 🔘 Predict Button
if st.button("🚀 Predict"):
    if user_input.strip() == "":
        st.warning("⚠️ Please enter a message!")
    else:
        vect_input = vectorizer.transform([user_input])
        prediction = model.predict(vect_input)[0]

        if prediction == "spam":
            st.error("🚫 This message is classified as: **SPAM**")
        else:
            st.success("✅ This message is classified as: **HAM (Not Spam)**")

st.markdown("---")

# 📁 File Upload
st.markdown("### 📁 Or Upload a File to Classify Multiple Messages")

uploaded_file = st.file_uploader("Upload a .txt or .csv file", type=["txt", "csv"])

if uploaded_file:
    import pandas as pd

    try:
        if uploaded_file.name.endswith(".csv"):
            df = pd.read_csv(uploaded_file)
        else:
            df = pd.read_csv(uploaded_file, header=None, names=["message"])

        st.write("✅ File loaded. Preview:")
        st.dataframe(df.head())

        vects = vectorizer.transform(df['message'])
        preds = model.predict(vects)
        df['prediction'] = preds

        st.success("🎯 Prediction Complete!")
        st.dataframe(df)

        st.download_button(
            label="📥 Download Results as CSV",
            data=df.to_csv(index=False),
            file_name="predictions.csv",
            mime='text/csv'
        )
    except Exception as e:
        st.error("❌ Error reading file. Please upload valid text or CSV.")

st.caption("Made with ❤️ using Python and Streamlit")

import streamlit as st
import re
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from utils import preprocess, create_weighted_pattern

# Configure page
st.set_page_config(page_title="Email Scam Detector", page_icon="📧")

# Load data and create regex pattern (cached for performance)
@st.cache_resource
def load_pattern():
    data = pd.read_csv('email.csv')
    data['Label'] = data['Category'].map({'ham': 0, 'spam': 1})
    data = data.dropna()
    data['Message'] = data['Message'].apply(preprocess)
    
    vectorizer = TfidfVectorizer(
        ngram_range=(1, 3),
        max_features=1000,
        min_df=2,
        max_df=0.95,
        stop_words='english'
    )
    
    return create_weighted_pattern(data['Message'], data['Label'], vectorizer=vectorizer)

regex_pattern = load_pattern()
print(regex_pattern)

# UI Elements
st.title("📧 Email Scam Detector")
st.markdown("""
    This tool analyzes email content using advanced pattern matching to detect potential scams.
    Paste your email text below to check if it's suspicious.
""")

email_input = st.text_area(
    "Paste email content here:",
    height=200,
    placeholder="Enter email text here..."
)

if st.button("Analyze Email"):
    if email_input.strip() == "":
        st.warning("Please enter some email content to analyze")
    else:
        with st.spinner("Analyzing content..."):
            # Preprocess input
            processed_text = preprocess(email_input)
            
            # Check for scam patterns
            is_scam = re.search(regex_pattern, processed_text)
            
            # Display results
            st.subheader("Analysis Results")
            if is_scam:
                st.error("🚨 Warning: This email contains characteristics of a scam!")
                st.markdown("**Potential scam indicators detected:**")
                st.write("The email matches known scam patterns including suspicious keywords, phrases, or patterns.")
            else:
                st.success("✅ This email appears to be safe!")
                st.markdown("**No significant scam indicators detected**")
                
        st.markdown("---")
        st.subheader("How it works")
        st.markdown("""
            - Analyzes text against known scam patterns
            - Checks for suspicious keywords and phrases
            - Uses advanced NLP pattern matching
            - Evaluates message structure and content
        """)
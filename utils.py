import re 
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer


# preprocess the email text by removing special characters, numbers, and converting to lowercase
def preprocess(text):
    text = re.sub(r'\W', ' ', text)
    text = re.sub(r'\d', ' ', text)
    text = text.lower()
    return text

# Create a weighted pattern based on TF-IDF features
def create_weighted_pattern(texts, labels, vectorizer=None):
    # Get TF-IDF features
    X = vectorizer.fit_transform(texts)
    feature_names = vectorizer.get_feature_names_out()
    
    # Calculate spam significance score
    spam_texts = X[labels == 1]
    ham_texts = X[labels == 0]
    
    spam_scores = np.mean(spam_texts.toarray(), axis=0)
    ham_scores = np.mean(ham_texts.toarray(), axis=0)
    
    # Calculate importance ratio
    importance = spam_scores / (ham_scores + 1e-10)
    
    # Select top features
    top_n = 100
    top_indices = (-importance).argsort()[:top_n]
    selected_features = feature_names[top_indices]
    
    # Create pattern with word boundaries
    pattern = r'\b(?:' + '|'.join(
        re.escape(feature) for feature in selected_features
    ) + r')\b'
    
    return pattern

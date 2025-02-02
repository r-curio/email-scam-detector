import streamlit as st
import re
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from utils import preprocess, create_weighted_pattern
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, confusion_matrix
import numpy as np
import csv
import io

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

# Add file upload section
st.markdown("---")
st.subheader("Batch Email Analysis")
st.markdown("""
    Upload a CSV file containing emails to analyze multiple messages at once.
    The CSV file should have:
    - A column named 'email' or 'message' containing the email text
    - (Optional) A column named 'label' or 'category' where:
        - 1 = spam emails
        - 0 = legitimate emails
""")

uploaded_file = st.file_uploader("Choose a CSV file", type='csv')

if uploaded_file is not None:
    try:
        # Read the CSV file
        df = pd.read_csv(uploaded_file)
        
        # Try to find the email column and label column
        email_column = None
        label_column = None
        
        # Check for email column
        possible_email_columns = ['email', 'message', 'text', 'content', 'Email', 'Message']
        possible_label_columns = ['label', 'category', 'spam', 'Label', 'Category', 'Spam']
        
        for col in possible_email_columns:
            if col in df.columns:
                email_column = col
                break
                
        for col in possible_label_columns:
            if col in df.columns:
                label_column = col
                break
        
        if email_column is None:
            st.error("Could not find email column. Please ensure your CSV has a column named 'email' or 'message'.")
        else:
            # Initialize metrics placeholders
            metrics_placeholder = st.empty()
            progress_bar = st.progress(0)
            
            # Initialize counters
            stats = {
                'total': len(df),
                'processed': 0,
                'true_positives': 0,
                'true_negatives': 0,
                'false_positives': 0,
                'false_negatives': 0
            }
            
            # Process all emails
            with st.spinner('Analyzing emails...'):
                results = []
                
                # Create columns for real-time stats
                st.subheader("Real-time Analysis Statistics")
                col1, col2 = st.columns(2)
                col3, col4 = st.columns(2)
                
                # Initialize metric displays
                tp_metric = col1.empty()
                tn_metric = col2.empty()
                fp_metric = col3.empty()
                fn_metric = col4.empty()
                
                for index, row in df.iterrows():
                    email = row[email_column]
                    processed_text = preprocess(str(email))
                    is_scam = bool(re.search(regex_pattern, processed_text))
                    
                    # Update statistics if label column exists
                    if label_column:
                        true_label = 1 if str(row[label_column]).strip() in ['spam', '1', 'true', 'yes'] else 0
                        if is_scam and true_label == 1:
                            stats['true_positives'] += 1
                        elif not is_scam and true_label == 0:
                            stats['true_negatives'] += 1
                        elif is_scam and true_label == 0:
                            stats['false_positives'] += 1
                        else:
                            stats['false_negatives'] += 1
                    
                    results.append({
                        'Email Text': email[:100] + '...' if len(email) > 100 else email,
                        'Prediction': 'SCAM' if is_scam else 'SAFE',
                        'Full Text': email
                    })
                    
                    # Update progress and stats
                    stats['processed'] += 1
                    progress = stats['processed'] / stats['total']
                    progress_bar.progress(progress)
                    
                    # Update real-time metrics with corrected percentage calculations
                    tp_metric.metric("True Positives", 
                                   f"{stats['true_positives']} ({(stats['true_positives']/(stats['true_positives'] + stats['false_negatives'])*100 if stats['true_positives'] + stats['false_negatives'] > 0 else 0):.1f}%)")
                    
                    tn_metric.metric("True Negatives", 
                                   f"{stats['true_negatives']} ({(stats['true_negatives']/(stats['true_negatives'] + stats['false_positives'])*100 if stats['true_negatives'] + stats['false_positives'] > 0 else 0):.1f}%")
                    
                    fp_metric.metric("False Positives", 
                                   f"{stats['false_positives']} ({(stats['false_positives']/(stats['true_negatives'] + stats['false_positives'])*100 if stats['true_negatives'] + stats['false_positives'] > 0 else 0):.1f}%)")
                    
                    fn_metric.metric("False Negatives", 
                                   f"{stats['false_negatives']} ({(stats['false_negatives']/(stats['true_positives'] + stats['false_negatives'])*100 if stats['true_positives'] + stats['false_negatives'] > 0 else 0):.1f}%)")

                # Create results DataFrame
                results_df = pd.DataFrame(results)
                
                # Display results
                st.subheader("Batch Analysis Results")
                
                # Summary statistics
                total = len(results)
                scam_count = sum(1 for r in results if r['Prediction'] == 'SCAM')
                safe_count = total - scam_count
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Total Emails", total)
                with col2:
                    st.metric("Detected Scams", scam_count)
                with col3:
                    st.metric("Safe Emails", safe_count)
                
                # Display detailed results in an expandable section
                with st.expander("View Detailed Results"):
                    # Add a search/filter box
                    search = st.text_input("Filter results (type to search)")
                    
                    # Filter results based on search
                    if search:
                        filtered_df = results_df[
                            results_df['Email Text'].str.contains(search, case=False) |
                            results_df['Prediction'].str.contains(search, case=False)
                        ]
                    else:
                        filtered_df = results_df
                    
                    # Display results with custom formatting
                    for idx, row in filtered_df.iterrows():
                        with st.container():
                            col1, col2 = st.columns([3, 1])
                            with col1:
                                st.text_area(
                                    f"Email {idx + 1}",
                                    row['Full Text'],
                                    height=100,
                                    key=f"email_{idx}"
                                )
                            with col2:
                                if row['Prediction'] == 'SCAM':
                                    st.error("🚨 SCAM")
                                else:
                                    st.success("✅ SAFE")
                            st.markdown("---")
                
                # Add download button for results
                csv = results_df.to_csv(index=False)
                st.download_button(
                    "Download Results CSV",
                    csv,
                    "email_analysis_results.csv",
                    "text/csv",
                    key='download-csv'
                )
                
    except Exception as e:
        st.error(f"Error processing file: {str(e)}")

# Create sidebar for metrics
st.sidebar.title("Model Performance Analysis")

# Move performance metrics to sidebar
st.sidebar.markdown("---")
st.sidebar.subheader("Model Performance Metrics")

@st.cache_data
def calculate_metrics():
    # Load and preprocess the test data
    data = pd.read_csv('email.csv')
    data['Label'] = data['Category'].map({'ham': 0, 'spam': 1})
    data = data.dropna()
    data['Message'] = data['Message'].apply(preprocess)
    
    # Make predictions using regex pattern
    predictions = [1 if re.search(regex_pattern, text) else 0 for text in data['Message']]
    
    # Calculate metrics
    metrics = {
        'Accuracy': accuracy_score(data['Label'], predictions),
        'Precision': precision_score(data['Label'], predictions),
        'Recall': recall_score(data['Label'], predictions),
        'F1 Score': f1_score(data['Label'], predictions)
    }
    return metrics

# Display metrics in sidebar columns
metrics = calculate_metrics()
col1, col2 = st.sidebar.columns(2)
col3, col4 = st.sidebar.columns(2)

with col1:
    st.metric("Accuracy", f"{metrics['Accuracy']:.2%}")
with col2:
    st.metric("Precision", f"{metrics['Precision']:.2%}")
with col3:
    st.metric("Recall", f"{metrics['Recall']:.2%}")
with col4:
    st.metric("F1 Score", f"{metrics['F1 Score']:.2%}")

# Add explanation of metrics in sidebar expander
with st.sidebar.expander("What do these metrics mean?"):
    st.markdown("""
    - **Accuracy**: Percentage of correctly classified emails (both spam and non-spam)
    - **Precision**: When the model predicts spam, how often is it correct?
    - **Recall**: Out of all actual spam emails, how many did we catch?
    - **F1 Score**: Balanced measure between precision and recall
    """)

# Move confusion matrix statistics to sidebar
st.sidebar.markdown("---")
st.sidebar.subheader("Detailed Classification Statistics")

@st.cache_data
def calculate_confusion_stats():
    # Load and preprocess the test data (reusing existing code)
    data = pd.read_csv('email.csv')
    data['Label'] = data['Category'].map({'ham': 0, 'spam': 1})
    data = data.dropna()
    data['Message'] = data['Message'].apply(preprocess)
    
    # Make predictions using regex pattern
    predictions = [1 if re.search(regex_pattern, text) else 0 for text in data['Message']]
    
    # Calculate confusion matrix
    tn, fp, fn, tp = confusion_matrix(data['Label'], predictions).ravel()
    total = len(data['Label'])
    
    return {
        'True Negatives': (tn, tn/total),
        'False Positives': (fp, fp/total),
        'False Negatives': (fn, fn/total),
        'True Positives': (tp, tp/total)
    }

# Display confusion matrix statistics in sidebar grid
conf_stats = calculate_confusion_stats()
col1, col2 = st.sidebar.columns(2)
col3, col4 = st.sidebar.columns(2)

with col1:
    count, ratio = conf_stats['True Negatives']
    st.metric("True Negatives", f"{count} ({ratio:.1%})")
    st.caption("Correctly identified non-spam")
    
with col2:
    count, ratio = conf_stats['False Positives']
    st.metric("False Positives", f"{count} ({ratio:.1%})")
    st.caption("Incorrectly flagged as spam")
    
with col3:
    count, ratio = conf_stats['False Negatives']
    st.metric("False Negatives", f"{count} ({ratio:.1%})")
    st.caption("Missed spam")
    
with col4:
    count, ratio = conf_stats['True Positives']
    st.metric("True Positives", f"{count} ({ratio:.1%})")
    st.caption("Correctly identified spam")

# Add explanation in sidebar expander
with st.sidebar.expander("What do these statistics mean?"):
    st.markdown("""
    - **True Negatives**: Legitimate emails correctly identified as non-spam
    - **False Positives**: Legitimate emails incorrectly flagged as spam
    - **False Negatives**: Spam emails that were missed (classified as legitimate)
    - **True Positives**: Spam emails correctly identified as spam
    
    The percentages show the proportion of each category in the total dataset.
    """)
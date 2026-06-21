import streamlit as st
import joblib
import numpy as np
import pandas as pd

# ── Load all saved models ───────────────────────────────────
kmeans_model = joblib.load('kmeans_model.joblib')
scaler = joblib.load('scaler.joblib')
cluster_labels = joblib.load('cluster_labels.joblib')
part1 = joblib.load('product_similarity_part1.joblib')
part2 = joblib.load('product_similarity_part2.joblib')

# Combine them back together into the full dataframe
product_similarity_df = pd.concat([part1, part2])

# ── App Title ────────────────────────────────────────────────
st.title("🛒 Shopper Spectrum")
st.write("Customer Segmentation and Product Recommendations")

# ── Sidebar Navigation ──────────────────────────────────────
menu = st.sidebar.selectbox("Choose a Module", 
                              ["Product Recommendation", "Customer Segmentation"])

# ── Module 1: Product Recommendation ────────────────────────
if menu == "Product Recommendation":
    st.header("🔍 Product Recommendation")
    
    product_name = st.text_input("Enter Product Name")
    
    if st.button("Get Recommendations"):
        if product_name in product_similarity_df.columns:
            similar_scores = product_similarity_df[product_name].sort_values(ascending=False)
            similar_scores = similar_scores.drop(product_name)
            top_5 = similar_scores.head(5)
            
            st.success("Top 5 Recommended Products:")
            for i, (product, score) in enumerate(top_5.items(), 1):
                st.write(f"{i}. {product}")
        else:
            st.error("Product not found. Please check the spelling.")

# ── Module 2: Customer Segmentation ─────────────────────────
elif menu == "Customer Segmentation":
    st.header("👥 Customer Segmentation")
    
    recency = st.number_input("Recency (in days)", min_value=0, value=30)
    frequency = st.number_input("Frequency (number of purchases)", min_value=0, value=5)
    monetary = st.number_input("Monetary (total spend)", min_value=0.0, value=500.0)
    
    if st.button("Predict Cluster"):
        frequency_log = np.log1p(frequency)
        monetary_log = np.log1p(monetary)
        
        new_customer = [[recency, frequency_log, monetary_log]]
        new_customer_scaled = scaler.transform(new_customer)
        predicted_cluster = kmeans_model.predict(new_customer_scaled)
        predicted_segment = cluster_labels[predicted_cluster[0]]
        
        st.success(f"Predicted Customer Segment: **{predicted_segment}**")

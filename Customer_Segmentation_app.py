import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import streamlit as st

# Title of the app
st.title('Customer Segmentation using KMeans Clustering')

# Load the dataset
@st.cache_data
def load_data():
    df = pd.read_csv('Mall_Customers.csv')  # Make sure to place the CSV file in the same directory
    df.rename({'Annual Income (k$)': 'Income', 'Spending Score (1-100)': 'score'}, axis=1, inplace=True)
    df.drop('CustomerID', axis=1, inplace=True)
    return df

df = load_data()

# Display the dataset
st.subheader('Dataset Preview')
st.write(df.head())

# Show basic statistics
st.subheader('Basic Statistics')
st.write(df.describe())

# Visualizing the distribution of income
st.subheader('Income Distribution')
fig, ax = plt.subplots()
sns.histplot(df.Income, kde=True, ax=ax)
st.pyplot(fig)

# Visualizing the relationship between income and spending score
st.subheader('Income vs Spending Score')
fig, ax = plt.subplots()
sns.scatterplot(x=df['score'], y=df['Income'], ax=ax)
st.pyplot(fig)

# Data Preprocessing (Standardization)
X = df[['Income', 'score']].values
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# KMeans Clustering
st.subheader('KMeans Clustering')
num_clusters = st.slider('Select number of clusters', min_value=2, max_value=10, value=5)

km = KMeans(n_clusters=num_clusters, init='k-means++', random_state=0)
y_pred = km.fit_predict(X_scaled)

# Visualizing Clusters
st.subheader(f'Clusters of Customers (k={num_clusters})')
fig, ax = plt.subplots(figsize=(8, 6))
for i in range(num_clusters):
    ax.scatter(X_scaled[y_pred == i, 0], X_scaled[y_pred == i, 1], s=100, label=f'Cluster {i+1}')

ax.set_title(f'Clusters of Customers (k={num_clusters})')
ax.set_xlabel('Income')
ax.set_ylabel('Spending Score')
ax.legend()
st.pyplot(fig)

# Display the cluster centers
st.subheader('Cluster Centers')
cluster_centers = scaler.inverse_transform(km.cluster_centers_)
st.write(cluster_centers)

# Display customer data with cluster labels
df['Cluster'] = y_pred
st.subheader('Customer Data with Cluster Labels')
st.write(df)

# Optionally, you can also add the option to download the dataset with cluster labels:
@st.cache_data
def create_downloadable_csv(df):
    return df.to_csv(index=False)

csv = create_downloadable_csv(df)
st.download_button(
    label="Download Customer Data with Clusters",
    data=csv,
    file_name='customer_data_with_clusters.csv',
    mime='text/csv'
)

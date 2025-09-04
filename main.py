import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
# --- 1. Synthetic Data Generation ---
np.random.seed(42)  # for reproducibility
num_customers = 500
data = {
    'Recency': np.random.randint(1, 365, num_customers),  # Days since last purchase
    'Frequency': np.random.poisson(lam=5, size=num_customers),  # Number of purchases
    'MonetaryValue': np.random.gamma(shape=2, scale=100, size=num_customers), # Total spending
    'Churn': np.random.binomial(1, 0.2, num_customers) # 0 = Not Churned, 1 = Churned
}
df = pd.DataFrame(data)
# --- 2. Data Cleaning and Preprocessing ---
# No significant cleaning needed for this synthetic data.
# Standardize numerical features for KMeans clustering
scaler = StandardScaler()
df_scaled = scaler.fit_transform(df[['Recency', 'Frequency', 'MonetaryValue']])
# --- 3. Customer Segmentation via KMeans Clustering ---
# Determine optimal number of clusters (e.g., using the Elbow method -  simplified here)
kmeans = KMeans(n_clusters=3, random_state=42) # Choosing 3 clusters as an example.  More robust methods could be used.
df['Cluster'] = kmeans.fit_predict(df_scaled)
# --- 4. Analysis: Churn Rate by Cluster ---
churn_by_cluster = df.groupby('Cluster')['Churn'].mean()
print("Churn Rate by Cluster:")
print(churn_by_cluster)
# --- 5. Visualization ---
plt.figure(figsize=(8, 6))
sns.barplot(x=churn_by_cluster.index, y=churn_by_cluster.values)
plt.title('Churn Rate per Customer Segment')
plt.xlabel('Customer Segment')
plt.ylabel('Churn Rate')
plt.savefig('churn_rate_by_cluster.png')
print("Plot saved to churn_rate_by_cluster.png")
plt.figure(figsize=(8,6))
sns.scatterplot(x='MonetaryValue', y='Frequency', hue='Cluster', data=df, palette='viridis')
plt.title('Customer Segmentation based on Monetary Value and Frequency')
plt.xlabel('Monetary Value')
plt.ylabel('Purchase Frequency')
plt.savefig('customer_segmentation.png')
print("Plot saved to customer_segmentation.png")
# --- 6. Prioritization for Retention Campaigns ---
# Based on churn rate, prioritize clusters with higher churn for targeted retention efforts.
# Further analysis could involve feature importance from a model to understand drivers of churn within each cluster.
#Note: This is a simplified example.  A real-world analysis would involve more sophisticated techniques for optimal cluster selection (e.g., silhouette analysis), handling imbalanced data (if churn is rare), and predictive modeling to understand churn drivers.
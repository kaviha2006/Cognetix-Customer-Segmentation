import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score

# -----------------------------
# Load Dataset
# -----------------------------
df = pd.read_csv("Mall_Customers.csv")

print("\n===== First 5 Rows =====")
print(df.head())

# ---------------------------------------
# Convert Annual Income from $ → ₹
# ---------------------------------------
USD_TO_INR = 83  # approx conversion rate
df['Annual Income (₹)'] = df['Annual Income (k$)'] * 1000 * USD_TO_INR

# ---------------------------------------
# Remove Missing + Duplicate Values
# ---------------------------------------
df = df.dropna()
df = df.drop_duplicates()

# ---------------------------------------
# Remove Outliers using IQR Method
# ---------------------------------------
def remove_outliers(data, column):
    Q1 = data[column].quantile(0.25)
    Q3 = data[column].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5*IQR
    upper = Q3 + 1.5*IQR
    return data[(data[column] >= lower) & (data[column] <= upper)]

df = remove_outliers(df, 'Annual Income (₹)')
df = remove_outliers(df, 'Spending Score (1-100)')

# ---------------------------------------
# Feature Selection
# ---------------------------------------
features = df[['Annual Income (₹)', 'Spending Score (1-100)']]
X = features.values

# Scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ---------------------------------------
# Elbow Method + Silhouette Score
# ---------------------------------------
wcss = []
silhouette_scores = []

K_range = range(2, 11)

print("\n===== Silhouette Scores =====")
for k in K_range:
    kmeans = KMeans(n_clusters=k, random_state=42)
    labels = kmeans.fit_predict(X_scaled)
    wcss.append(kmeans.inertia_)
    sil = silhouette_score(X_scaled, labels)
    silhouette_scores.append(sil)
    print(f"k={k} → Silhouette Score: {sil}")

# Plot WCSS graph
plt.figure()
plt.plot(K_range, wcss, marker='o')
plt.title("Elbow Method")
plt.xlabel("Number of Clusters")
plt.ylabel("WCSS")
plt.show()

# Find best k using best silhouette score
best_k = K_range[np.argmax(silhouette_scores)]
print(f"\nBest number of clusters (based on silhouette score): {best_k}")

# ---------------------------------------
# Final K-Means Model
# ---------------------------------------
model = KMeans(n_clusters=best_k, random_state=42)
df['Cluster'] = model.fit_predict(X_scaled)

print("\n===== Cluster Labels Added =====")
print(df.head())

# ---------------------------------------
# Visualization
# ---------------------------------------
plt.figure(figsize=(8,6))
plt.scatter(df['Annual Income (₹)'], df['Spending Score (1-100)'],
            c=df['Cluster'], s=60)
plt.title("Customer Segmentation Clusters")
plt.xlabel("Annual Income (₹)")
plt.ylabel("Spending Score (1–100)")
plt.show()

# ---------------------------------------
# Cluster Summary
# ---------------------------------------
summary = df.groupby("Cluster")[['Annual Income (₹)', 'Spending Score (1-100)']].mean()
summary['Count'] = df['Cluster'].value_counts()

print("\n===== Cluster Summary (₹) =====")
print(summary)

# ---------------------------------------
# Automatic Cluster Insights (Option 1)
# ---------------------------------------
print("\n===== Cluster Insight Report =====")

cluster_insights = {}

for cid in summary.index:
    avg_income = summary.loc[cid, 'Annual Income (₹)']
    avg_score = summary.loc[cid, 'Spending Score (1-100)']
    
    if avg_income > df['Annual Income (₹)'].mean() and avg_score > 50:
        label = "Premium High-Value Customers"
    elif avg_income < df['Annual Income (₹)'].mean() and avg_score > 50:
        label = "Budget but High Engagement Customers"
    elif avg_income > df['Annual Income (₹)'].mean() and avg_score < 50:
        label = "Rich but Low Engagement Customers"
    elif avg_score < 30:
        label = "Low Value / Low Spending Customers"
    else:
        label = "Average Middle Segment"

    cluster_insights[cid] = label
    print(f"Cluster {cid}: {label}")

# ---------------------------------------
# Prediction Function
# ---------------------------------------
def predict_customer(income_inr, score):
    data = scaler.transform([[income_inr, score]])
    cluster = model.predict(data)[0]
    return cluster

# ---------------------------------------
# Prediction Loop (Option 4)
# ---------------------------------------
while True:
    print("\n===== Predict New Customer Segment =====")
    
    try:
        inc = float(input("Enter Annual Income (₹): "))
        score = float(input("Enter Spending Score (1-100): "))
    except:
        print("Invalid input! Please enter numbers only.")
        continue
    
    cid = predict_customer(inc, score)
    insight = cluster_insights[cid]
    
    print(f"\n➡ Customer belongs to Cluster: {cid}")
    print(f"➡ Segment Meaning: {insight}")
    
    again = input("\nPredict again? (y/n): ").lower()
    if again != 'y':
        print("\nExiting Prediction Menu. Thank you!")
        break

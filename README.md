# Customer Segmentation using K-Means Clustering

## Company
**Cognitix Technology**

## Project Title
Customer Segmentation using K-Means Clustering

## Project Type
Intermediate Level – Machine Learning (Clustering)

---

## 📌 Objective
The goal of this project is to segment customers into meaningful groups based on their purchasing behavior.  
By applying **K-Means clustering**, customers are categorized into groups such as high-value customers, budget buyers, and low-engagement users, helping businesses make data-driven marketing decisions.

---

## 📊 Dataset
- **Source:** Kaggle – Customer Segmentation Tutorial  
- **Features Used:**
  - Annual Income (converted from USD to INR)
  - Spending Score (1–100)

> Note: The dataset is publicly available on Kaggle and is not redistributed in this repository.

---

## ⚙️ Technologies Used
- Python
- Pandas
- NumPy
- Matplotlib
- Scikit-learn

---

## 🔍 Project Workflow
1. Load and explore customer shopping data
2. Clean data (remove missing values and outliers)
3. Convert Annual Income from USD to INR
4. Feature scaling using StandardScaler
5. Determine optimal number of clusters using:
   - Elbow Method
   - Silhouette Score
6. Apply K-Means clustering
7. Visualize customer segments
8. Generate cluster insights
9. Predict segment for new customer inputs

---

## 📈 Graphical Representations
- Elbow Method Graph (WCSS vs K)
- Customer Segmentation Scatter Plot

These visualizations help understand cluster formation and validate the model.

---

## 📋 Sample Output
The program displays:
- Cluster summary table
- Meaningful cluster labels (business insights)
- Interactive prediction for new customers
Example:

<img width="641" height="511" alt="image" src="https://github.com/user-attachments/assets/05f048aa-d6b8-4e8a-a43b-0bdb49f3e235" />
<img width="801" height="632" alt="image" src="https://github.com/user-attachments/assets/78d785d3-9bb0-41a8-a493-d098e90d5161" />
<img width="705" height="556" alt="image" src="https://github.com/user-attachments/assets/ed7316d7-51ef-45da-b07c-5962faf109c8" />
<img width="702" height="596" alt="image" src="https://github.com/user-attachments/assets/4b5b72c5-6fb4-42f0-bc74-2e1cc5646df5" />

## 🚀 How to Run the Project
1. Install dependencies:
pip install pandas numpy matplotlib scikit-learn


2. Run the script:


python customer_segmentation.py


3. Enter new customer details when prompted.

---

## 💡 Key Insights
- High-income customers with high spending score are premium targets
- Some high-income customers show low engagement
- Budget customers with high spending behavior are valuable growth segments

---

## 📌 Conclusion
This project demonstrates how unsupervised learning techniques like **K-Means clustering** can be effectively used for customer behavior analysis.  
It is suitable for real-world business analytics and marketing strategy development.

---

## 👤 Author
**Kaviha R. M**  
Machine Learning & Full Stack Enthusiast  

---

## 🔗 Acknowledgements
- Kaggle for the dataset
- Cognitix Technology for project guidance


Customer Segmentation

### Turning Customer Data into Meaningful Customer Insights

Customer Segmentation is a machine learning project designed to identify customer segments based on demographic characteristics, spending habits, purchasing behavior, and marketing campaign engagement.

Using exploratory data analysis, feature engineering, data preprocessing, and clustering techniques, this project groups customers with similar characteristics to help businesses better understand their audiences and make data-driven marketing decisions.

The project also includes an interactive **Streamlit web application** that allows users to explore customer profiles and receive an estimated segment assignment based on their input.

---

## 🎯 Project Objectives

* Identify meaningful customer segments using unsupervised machine learning.
* Analyze customer demographics, income, spending, and purchase behavior.
* Explore customer engagement with marketing campaigns.
* Compare clustering approaches and evaluate cluster quality.
* Build an interactive web application to demonstrate the segmentation workflow.

## ✨ Key Features

* **Exploratory Data Analysis (EDA):** Explore customer demographics, purchasing patterns, spending behavior, and campaign responses.
* **Feature Engineering:** Create meaningful features such as Age, Total Children, Total Spending, Total Purchases, and Total Campaign Acceptance.
* **Data Preprocessing:** Handle missing values, cap selected outliers, remove irrelevant or constant columns, and standardize numerical features.
* **Clustering Analysis:** Explore K-Means, Hierarchical Clustering, and DBSCAN.
* **Model Evaluation:** Use the Silhouette Score to compare clustering quality.
* **Interactive Web App:** Enter customer attributes and view an estimated customer segment through a Streamlit interface.

## 🧠 Machine Learning Approach

This project explores multiple unsupervised learning techniques:

| Algorithm               | Purpose                                                         |
| ----------------------- | --------------------------------------------------------------- |
| K-Means                 | Establish a baseline for customer segmentation.                 |
| Hierarchical Clustering | Identify groups of customers based on similarity.               |
| DBSCAN                  | Explore density-based clusters and identify potential outliers. |

**Selected model:** Agglomerative Hierarchical Clustering with Ward linkage and two clusters.

The selected configuration achieved a Silhouette Score of approximately **0.275** during model evaluation. This score provides a measure of cluster separation; the results should be interpreted alongside cluster profiles and business usefulness.

Because the selected model does not natively predict clusters for new customers, the Streamlit application uses the nearest cluster representative in standardized feature space to estimate a segment assignment.

## 📊 Features Used for Segmentation

The model uses the following twelve features:

* Income
* Age
* Total Children
* Recency
* Total Spending
* Total Purchases
* Number of Deals Purchases
* Number of Web Purchases
* Number of Catalog Purchases
* Number of Store Purchases
* Number of Web Visits per Month
* Total Campaign Acceptance

These features combine demographic, behavioral, transactional, and marketing-engagement information.

## 🛠️ Technology Stack

* **Language:** Python
* **Data Analysis:** Pandas, NumPy
* **Visualization:** Matplotlib, Seaborn
* **Machine Learning:** Scikit-learn
* **Model Persistence:** Joblib
* **Web Application:** Streamlit
* **Development Environment:** Jupyter Notebook
* **Version Control:** Git and GitHub

## 📁 Project Structure

```text
customer-segmentation/
│
├── app.py
├── requirements.txt
├── README.md
│
├── deployment/
│   └── hierarchical_artifacts.pkl
│
├── notebooks/
│   └── customer_segmentation.ipynb
│
└── data/
    └── marketing_campaign.xlsx
```

*Note: Update this structure to match your actual repository. The dataset and notebook should only appear here if you have included them in the repository.*

## 🚀 Run the Application Locally

### 1. Clone the repository

```bash
git clone https://github.com/YOUR-USERNAME/customer-segmentation.git
cd customer-segmentation
```

Replace `YOUR-USERNAME` with your GitHub username.

### 2. Install dependencies

```bash
python -m pip install -r requirements.txt
```

### 3. Launch Streamlit

```bash
python -m streamlit run app.py
```

The application should open in your browser. If it does not, visit:

```text
http://localhost:8501
```
 📈 Potential Business Applications

Customer segmentation can help businesses:

* Develop more targeted marketing campaigns.
* Understand high-spending and frequent-purchasing customers.
* Explore relationships between customer demographics and purchasing behavior.
* Identify customers who respond to promotional campaigns.
* Design more personalized customer engagement strategies.

These are potential applications of the analysis, not guaranteed outcomes.

## 🔮 Future Improvements

* Develop richer customer personas using cluster-level statistical profiles.
* Compare additional clustering algorithms and feature-selection strategies.
* Add interactive visualizations of cluster characteristics.
* Evaluate segment stability and business interpretability.
* Improve new-customer assignment using a validated predictive approach.
* Incorporate additional customer and campaign data.

## 👩‍💻 About the Project

Customer Segmentation demonstrates an end-to-end data science workflow, from data preparation and exploratory analysis to unsupervised learning and interactive application development.

The goal is to transform raw customer data into understandable segments that can support customer analytics and marketing strategy.

---

⭐ **If you find this project interesting, feel free to star the repository!**

*Built with Python, machine learning, and curiosity about customer behavior.*

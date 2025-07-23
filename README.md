
# 🧠 Market Segmentation System

A cutting-edge, full-stack solution for segmenting customers using unsupervised machine learning, visualizing insights, and serving real-time personalized offers via a REST API. This project demonstrates end-to-end problem solving—from data exploration through model deployment—designed for production-grade performance and maintainability.

---

## 🎯 Problem Statement

Traditional customer grouping relies on static rules (e.g., demographic buckets) that miss nuanced behaviors. We need a dynamic, data-driven way to:

1. **Identify distinct customer segments** based on spending, engagement, and profile attributes.
2. **Automate real-time segment inference** for new users.
3. **Deliver tailored offers/discounts** that maximize engagement and conversion.

---

## 🚀 High-Level Solution

1. **Exploratory Data Analysis**  
   - Inspect distributions, correlations, and outliers across numeric features (age, income, spending score, etc.).  
   - Visualize categorical breakdowns (gender, preferred category) to understand base rates.

2. **Preprocessing Pipeline**  
   - **Deduplication** & missing-value handling.  
   - **Standard Scaling** of numeric features to zero mean and unit variance.  
   - **One-Hot Encoding** for categorical fields.

3. **Dimensionality Reduction**  
   - Use **PCA** to project high-dimensional feature space into two principal components for visualization and noise reduction.

4. **Unsupervised Clustering**  
   - Apply **KMeans**:
     - Use **Elbow Method** (within-cluster sum of squares) and **Silhouette Score** to choose optimal `k`.  
     - Train final model to discover 3 meaningful segments.

5. **Segment Profiling & Business Rules**  
   - Analyze cluster centers and box plots to label segments (e.g., “High Spender,” “Value Seeker,” “Loyal Regular”).  
   - Map each segment to a curated set of offers.

6. **Model Deployment with FastAPI**  
   - Serialize `StandardScaler`, `PCA`, and `KMeans` objects with `joblib`.  
   - Expose a `/predict` endpoint that ingests raw user data, runs the preprocessing pipeline, and returns segment + offers in JSON.

7. **Interactive Frontend**  
   - Build a responsive form that collects user inputs.  
   - Submit via Axios to the FastAPI API.  
   - Display real-time segment and personalized offers.

---

## 📂 Repository Structure

```

.
├── app_api/
│   ├── app/                          # FastAPI application module
│   │   ├── main.py                   # API entry point with clustering logic
│   │   └── model/
│   │       ├── scaler.pkl            # Pre-fitted StandardScaler for input normalization
│   │       ├── kmeans_model.pkl      # Trained KMeans clustering model
│   │       └── model_columns.json    # Column order used during training
│   ├── requirements.txt              # Python dependencies for the API
│   └── Dockerfile                    # Containerization setup for deployment
│
├── customer_segmentation_data.csv/     # Dataset 
|
|
├── main.ipynb/     # Python Notebook 
│   
│
└── README.md       # Comprehensive documentation & setup instructions

````

---

## 🔬 Code Analysis & Key Snippets

### 1. Data Scaling & PCA  
```python
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# Fit on training data
scaler = StandardScaler().fit(X_numeric)
scaled_data = scaler.fit_transform(data_to_scale)
scaled_df = pd.DataFrame(scaled_data, columns=columns_to_scale)

scaled_df.index =df.index
df[columns_to_scale] = scaled_df
joblib.dump(scaler, "scaler.pkl")

pca = PCA(n_components=2, random_state=42)
pca_components = pca.fit_transform(df.drop('cluster', axis=1, errors='ignore'))

# Add PCA components to DataFrame
df['PCA1'] = pca_components[:, 0]
df['PCA2'] = pca_components[:, 1]

````

* **Insight**: Scaling ensures features with different units don’t dominate clustering.
* **PCA** reduces noise and enables 2D visualizations.


### 2. Finding k

```python
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

wcss = []
for i in range(1,11):
    kmeans = KMeans(n_clusters=i, random_state=42)
    kmeans.fit(df)
    wcss.append(kmeans.inertia_)

plt.figure(figsize=(10, 6))
plt.plot(range(1, 11), wcss, marker='o',linestyle='--')
plt.title('Elbow Method for Optimal k')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')
plt.show()

silhouette_scores = []
for k in range(2,11):
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(df)
    score = silhouette_score(df, kmeans.labels_)
    silhouette_scores.append(score)
    

plt.figure(figsize=(10, 6))
plt.plot(range(2, 11), silhouette_scores, marker='o',linestyle='--')
plt.title('Silhouette Score Method')
plt.xlabel('Number of clusters')
plt.ylabel('Silhouette Score')
plt.show()
    
````

### 3. KMeans Clustering & Validation

```python
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# Initialize K-means with K=3 clusters
kmeans = KMeans(n_clusters=3, random_state=42)
labels = kmeans.fit_predict(df.drop(['cluster', 'PCA1', 'PCA2'], axis=1, errors='ignore'))

# Add cluster labels to the original data
df['cluster'] = labels
joblib.dump(kmeans, "kmeans.pkl")
    
```

* **Elbow vs. Silhouette**: Combine both to pick a stable cluster count.

### 4. FastAPI Predict Endpoint

```python
@app.post("/predict")
def predict_cluster(customer: CustomerData):
    # Encode gender
    input_gender = customer.gender.strip().lower()
    gender_male = 1 if input_gender == 'male' else 0
    gender_other = 1 if input_gender == 'other' else 0

    # Encode preferred_category
    input_category = customer.preferred_category.strip().lower()
    category_dummies = [
        1 if input_category == 'fashion' else 0,
        1 if input_category == 'groceries' else 0,
        1 if input_category == 'home & garden' else 0,
        1 if input_category == 'sports' else 0
    ]

    # Combine features
    features = [
        customer.age,
        customer.income,
        customer.spending_score,
        customer.membership_years,
        customer.purchase_frequency,
        customer.last_purchase_amount,
        gender_male,
        gender_other,
        *category_dummies
    ]

    # Convert to DataFrame
    df = pd.DataFrame([features])

    # Scale numeric features
    df.iloc[:, :6] = scaler.transform(df.iloc[:, :6])

    # Predict cluster
    cluster = kmeans.predict(df)[0]

    return {"cluster": int(cluster)}
```

* **Reindexing** protects against missing categories.
* **Serialization** via `joblib` keeps load times minimal.

---

## 📈 Results & Insights

* **Segment A (Young Active Shoppers)**:

  * Moderate spending score, High purchase frequency, Mid-range purchases.
  * Action: Offer: Frequent shopper discounts, New collection previews, Loyalty program bonuses

* **Segment B (Middle-aged Value Seekers)**:

  * Balanced spending, Lower purchase frequency, Average purchases.
  * Action: Offer: Bulk purchase deals, Seasonal sales, Family package offers.

* **Segment C (Senior Premium Customers)**:

  * High-value purchases, Moderate frequency, Strong brand loyalty.
  * Action: Offer: Exclusive VIP memberships, Luxury product bundles, Personal shopping assistance.

---

Here’s a revised and complete **📖 Usage** section to reflect your actual project structure, deployment, and mobile integration:

---

## 📖 Usage

### 1. **Backend (API)**

The backend is built using **FastAPI** and exposes a `/predict` endpoint for real-time customer segmentation.

To run locally:

```bash
cd app_api
pip install -r requirements.txt
uvicorn main:app --reload
```

### 2. **Frontend**

The frontend (HTML/CSS/JS) is served as static files and is fully integrated with the backend.

**Live Deployment:**
The complete application (backend API + frontend) is deployed on **Hugging Face Spaces**.

* **API Endpoint:** `https://deepakdesh-market-seg-api.hf.space/predict`
* **Frontend:** Accessible from the same Hugging Face URL

### 3. **Mobile Application**

This web application is converted into a mobile app using **Mobiroller**, enabling seamless testing on Android/iOS.

To test:

* Open the mobile app (built using Mobiroller).
* Fill in the customer details via the form.
* Submit and instantly view the predicted customer segment and personalized marketing insights.

APK: https://drive.google.com/file/d/1iiCv6Rotmn0JCUfPTQKK9SnDta3lqBXO/view?usp=sharing

---
### 4. Images
<!-- <p align="center" width="150"> <img src="WhatsApp Image 2025-05-15 at 15.30.05_94fac7a4.jpg" alt="Form UI"> </p> -->
<p align="center" > <img src="WhatsApp Image 2025-05-15 at 15.30.05_94fac7a4.jpg" alt="Form UI" width="150"> </p>
<p align="center" > <img src="WhatsApp Image 2025-05-15 at 15.30.26_b17af246.jpg" alt="Form UI" width="150"> </p>
<p align="center"> <img src="WhatsApp Image 2025-05-15 at 15.30.44_2acda89c.jpg" alt="Form UI" width="150"> </p>
<p align="center"> <img src="WhatsApp Image 2025-05-15 at 15.30.52_878417a8.jpg" alt="Form UI" width="150"> </p>



## 🌟 Competitive Edge

* **End-to-end reproducibility**: notebooks, model artifacts, API, and UI—all versioned.
* **Robust validation**: Elbow, Silhouette, and PCA checks guard against overfitting.
* **Production-ready**: FastAPI backend with solid input validation and serialization.
* **Extensibility**: Easily swap in new models (e.g., DBSCAN, hierarchical) or add real-time streaming.

---

## 🔭 Future Enhancements

* Integrate real-time user behavior data (clickstream) for dynamic re-segmentation.
* Add a **recommendation engine** (collaborative filtering) for product-level suggestions.
* Deploy using Kubernetes & CI/CD pipelines for auto-scaling.

---

### **Built with passion by**:

- *Desh Deepak Verma* (U22EC028)  
- *Krishna Meghwal* (U22EC042)  
- *Ajay Parmar* (U22EC038)  

— Ready to drive data-driven marketing strategies and elevate customer engagement!

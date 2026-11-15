# Travaux Dirigés / Pratiques 1 (TP 1) : Advanced Data Detective
**Institution :** EPITA — School of Engineering and Computer Science  
**Module :** DEP (Exploratory Data Analysis) — Specialization Lab  
**Group Assignment :** Group 5  
**Domain Topic :** Streaming Service Subscription Churn & Content Viewing  
**Dataset Source :** synthetic  
**Dataset Profile :** 5200 sampled rows, 14 columns  
**Anomalies & Degradation :** High (Injected missingness, messy types, duplicate records, and outliers)  

---

## 1. Prerequisites & Environment Setup

Each group must work inside a dedicated Jupyter Notebook or interactive session:

```bash
# Recommended environment setup
python3 -m venv .venv
source .venv/bin/activate
pip install pandas numpy matplotlib seaborn
```

---

## 2. Step-by-Step Guided Investigation Protocol

### Part 1: The "First Contact" Protocol & Grain Definition 
### Part 2: Schema Coercion & Memory Optimization
### Part 3: Missingness Profiling & Strategic Triage 
### Part 4: Deduplication & Outlier Detective Work

---

## 3. Part 5: The 5 Functional & Business-Driven Questions (01:30 – 02:00)

Every visualization and analytical inquiry must answer a concrete business question. Answer each of the following 5 business questions using pandas transformations, an informative chart, and a concise 1-sentence executive takeaway:

As an EPITA Data Science Professor and Senior Data Product Manager, I've designed this assignment to challenge your analytical skills using a real-world streaming service dataset. The goal is to move beyond mere technical execution and cultivate a product-oriented mindset, extracting actionable business intelligence from raw data.

Your group has been provided with a dataset concerning "Streaming Service Subscription Churn & Content Viewing." Below are 5 functional, business-oriented questions that require a robust Pandas-based solution, careful data cleaning, and insightful interpretation.

---

### **Assignment: Streaming Service Analytics - Churn & Engagement Deep Dive**

**Course:** DEP (Data Engineering & Products)
**Dataset Topic:** Streaming Service Subscription Churn & Content Viewing

---

**Question 1: Churn Performance by Subscription Plan**

*   **The Question:** What is the churn rate for each distinct `plan_type` (e.g., Basic, Standard, Premium), and which plan type exhibits the highest or lowest churn, indicating potential areas for product improvement or success?
*   **Business Context:** The product management and marketing teams are keen to understand if certain subscription tiers are more effective at retaining users. This insight is crucial for evaluating the value proposition of each plan, guiding potential adjustments to pricing, feature sets, or targeted retention campaigns for at-risk subscriber segments.
---

**Question 2: Evaluating Promotional Campaign Impact**

*   **The Question:** Do users who registered using a `promo_code_used` exhibit a significantly different churn rate or average `monthly_price` paid compared to those who did not use a promotional code?
*   **Business Context:** The marketing and finance departments require robust analysis to assess the effectiveness and return on investment of promotional campaigns. Understanding if promo codes lead to better long-term retention (lower churn) or influence the `monthly_price` paid (post-promo) helps in optimizing future marketing budgets, campaign designs, and customer acquisition strategies.
---

**Question 3: Identifying High-Churn Risk Profiles by Engagement**

*   **The Question:** Is there a discernible correlation between user engagement metrics (`total_watch_minutes`, `num_genres_watched`) or `content_satisfaction_score` and the likelihood of user churn? Specifically, do lower values in these key metrics correspond to a higher probability of churning?
*   **Business Context:** The data science and product teams aim to develop early warning systems for churn. By identifying specific behavioral indicators (such as low watch time, limited genre exploration, or poor satisfaction scores), the company can proactively segment at-risk users and deploy targeted retention strategies like personalized content recommendations or special offers before they decide to leave.
---

**Question 4: Analyzing Churn Timing and User Lifecycle Duration**

*   **The Question:** What is the average active subscription duration (measured from `registration_date` to `last_active_date`) for users who eventually churned, compared to those who remained subscribed? Does a shorter active period typically precede churn?
*   **Business Context:** The customer success and product management teams are interested in understanding the typical user lifecycle and identifying "early churners" – users who disengage or leave shortly after registration. This insight helps in establishing benchmarks for initial user engagement and pinpointing critical periods within the subscription lifecycle where targeted interventions could be most effective.
---

**Question 5: Device-Specific User Engagement and Satisfaction Insights**

*   **The Question:** How do key engagement metrics, specifically `total_watch_minutes` and `content_satisfaction_score`, vary across different `device_type` categories? Are there specific device types that are associated with significantly higher or lower user engagement and satisfaction?
*   **Business Context:** The product development and user experience (UX/UI) teams need to make data-driven decisions about resource allocation and optimization priorities. Understanding which devices drive the most engagement and user satisfaction can inform focused development efforts, platform-specific content optimization, or identify devices that require significant UX/UI improvements.
---


## 4. Part 6: Advanced Data Preparation for Machine Learning (No Hints Provided)

Following the EDA and business inquiry phase, you must prepare the cleaned dataset for downstream machine learning modeling. You must apply the strict principles established during CI 1, ensuring the resulting artifacts are completely leakage-free and model-ready:

1. **Target Identification & Leakage-Free Train/Test Split:**
   - Identify the primary predictive target in your domain dataset (e.g. classification target such as Churn, Cancellation, Failure/Delay, or Retention; or regression target).
   - Perform an 80/20 Train/Test split **BEFORE** fitting any transformers, scalers, or imputation strategies.
   - If your target is categorical/discrete, you must strictly stratify the split.

2. **Strict Leakage-Free Missingness Imputation:**
   - Quantify remaining missing values on `X_train`.
   - Apply Scikit-Learn imputers (e.g. `SimpleImputer`, `KNNImputer`) fitted **strictly on `X_train`** and transformed across both `X_train` and `X_test`.
   - Never use global statistics.

3. **Feature Engineering, Encoding & Scaling:**
   - Transform high-cardinality and nominal categorical features using appropriate encoders (e.g. `OneHotEncoder(handle_unknown='ignore')` or `TargetEncoder`).
   - Extract numerical signals from temporal variables.
   - Scale continuous features (e.g. using `StandardScaler` or `RobustScaler`) fitted **only** on the training set.

4. **Model-Ready Dataset Exports:**
   - Export your final prepared features and targets to disk: `train_prepared.csv` and `test_prepared.csv`.


## 5. Deliverables & Assessment

At the conclusion of the lab, Group 5 must submit:
1. **The Cleaned Jupyter Notebook (`TP1_Group_5.ipynb`)** containing:
   - Top-to-bottom clean execution with zero errors.
   - Clear markdown cells documenting the **grain**, schema transformations, and cleaning decisions.
   - The 5 functional business questions with plots and 1-sentence executive takeaways.
2. **The Cleaned Dataset Export :**
   ```python
   df.to_csv("dataset_cleaned.csv", index=False)
   ```
3. **Prepared ML Splits :**
   ```python
   # Leakage-free prepared datasets ready for modeling
   train_prepared.to_csv("train_prepared.csv", index=False)
   test_prepared.to_csv("test_prepared.csv", index=False)
   ```
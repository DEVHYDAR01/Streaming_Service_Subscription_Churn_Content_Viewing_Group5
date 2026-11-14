# EPITA — DEP : Travaux Dirigés / Pratiques 1 (TP 1)
**Module :** Exploratory Data Analysis  
**Session :** Practical Lab 1 (Multi-Group Investigation)  
**Total Cohorts :** 8 Groups  
**Complexity :** High  

---

## Lab Overview & Instructions
Welcome to TP 1 of Exploratory Data Analysis. Following CI 1, each student group is assigned a unique, complex domain dataset. Your mission is to perform a rigorous end-to-end data audit, clean anomalies, and answer 6 functional business questions.

### Group Dataset Directory & Assignments

| Group | Topic | Source | Records | Assignment Guide |
|---|---|---|---|---|
| Group 1 | **Airline Flight Delays & Passenger Satisfaction** | `synthetic` | 5200 rows | [questions-group.md](group_1/questions-group.md) |
| Group 2 | **Hotel Bookings, Cancellations & Seasonal Pricing** | `synthetic` | 5200 rows | [questions-group.md](group_2/questions-group.md) |
| Group 3 | **Mobile App User Retention & In-App Purchases** | `synthetic` | 5200 rows | [questions-group.md](group_3/questions-group.md) |
| Group 4 | **Automotive Predictive Maintenance & Sensor Data** | `synthetic` | 5200 rows | [questions-group.md](group_4/questions-group.md) |
| Group 5 | **Streaming Service Subscription Churn & Content Viewing** | `synthetic` | 5200 rows | [questions-group.md](group_5/questions-group.md) |
| Group 6 | **Fitness Tracker Activity & Health Metrics** | `synthetic` | 5200 rows | [questions-group.md](group_6/questions-group.md) |
| Group 7 | **Restaurant Reservations, Wait Times & Customer Reviews** | `synthetic` | 5200 rows | [questions-group.md](group_7/questions-group.md) |
| Group 8 | **Supply Chain Logistics, Inventory & Shipping Disruptions** | `synthetic` | 5200 rows | [questions-group.md](group_8/questions-group.md) |

---

---

## Advanced Machine Learning Preparation Requirement
Every group is required to extend their analysis into predictive modeling readiness:
- Implement a strict, leakage-free Train/Test split.
- Handle missing values using Scikit-Learn transformers fitted exclusively on the training set.
- Encode categorical features and scale continuous numeric attributes.
- Export both `train_prepared.csv` and `test_prepared.csv`.
- **Note:** No explicit hints or boilerplate code are provided. Students must architect the cleaning and feature engineering pipelines based on the principles established in CI 1.

## General Deliverables & Submission
Each group must submit by the end of the lab session:
1. **Jupyter Notebook (`TP1_Group_<N>.ipynb`):** Fully reproducible analysis covering Parts 1 through 5.
2. **Cleaned Dataset (`dataset_cleaned.csv`):** Standardized, coerced, deduplicated, and anomaly-triaged dataset.
3. **Prepared Datasets :** `train_prepared.csv` and `test_prepared.csv` (properly encoded, scaled, and leakage-free).

# student-readiness-pilot

# 📚 eCampus Exam Readiness Prediction

## 🔹 Project Overview
This project explores whether we can **predict a student's exam readiness** based on how they interact with the eCampus learning app (watching, listening, reading, practicing, testing, earning points).

- **Goal:** Provide readiness insights for students, parents, teachers, schools, and corporate clients.  
- **Dataset:** User activity logs (60K+ students, 10 years).  
- **Pilot:** 500 students taking in-app tests (2025).  

---

## 🔹 Approach
1. **Exploratory Data Analysis (EDA)**  
   - Analyze user profiles, engagement patterns, quiz/test behavior.  
   - Derive business insights from 60K+ students.  

2. **Feature Engineering**  
   - Build aggregated student dataset (demographics, subscriptions, engagement frequency, quiz/test scores, points).  

3. **Predictive Modeling**  
   - Baseline: Logistic Regression  
   - Intermediate: Random Forest / XGBoost  
   - Advanced: Transformers (transfer learning across cohorts)  

4. **Deployment**  
   - Batch predictions (every 30 days).  
   - Dashboards + API integration for students, parents, and schools.  

---

## 🔹 Repository Structure
- `data/` → raw, interim, processed datasets  
- `notebooks/` → Colab notebooks for EDA, feature engineering, and modelling  
- `scripts/` → reusable Python functions  
- `reports/` → results, figures, summaries  
- `README.md` → project documentation  

---

## 🔹 Tools & Environment
- **Languages:** Python (Colab), SQL (MySQL export)  
- **Libraries:** Pandas, NumPy, Scikit-learn, Seaborn, Matplotlib, Plotly  
- **Version Control:** GitHub  
- **Cloud Hosting (future):** Azure Machine Learning (production)  

---

## 🔹 Next Steps
- [ ] Extract datasets (Core user profile + Exam results).  
- [ ] Run EDA and generate initial business insights.  
- [ ] Define predictive features.  
- [ ] Pilot predictive models with 500 students.  

# ✈️ Unlocking Behavioral Intelligence in Airline Loyalty Programs

> **Consulting & Analytics Club, IIT Guwahati — Summer Projects 2026**

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://airlineloyaltyproject-abswe3zqbfb4qpgnujut6g.streamlit.app/)

---

## 📌 Project Title

**Unlocking Behavioral Intelligence in Airline Loyalty Programs**

A data science and consulting project that builds a churn prediction system, behavioral segmentation framework, and retention playbook for a mid-sized airline's loyalty program.

---

## 🎯 Problem Statement

Airlines use loyalty programs to increase repeat business and improve Customer Lifetime Value (CLV). However, many programs still focus mainly on points and rewards. As a result:

- Many members stay **inactive** without being noticed
- **High-value customers** may stop flying without any early warning
- Marketing teams **react too late** to disengagement signals

This project addresses three core objectives:

1. **Churn Prediction** — Identify members likely to disengage in the coming months before it happens
2. **Customer Segmentation & Value** — Go beyond CLV to understand what makes a customer genuinely valuable going forward, using behavioral and demographic signals
3. **Smart Retention** — Propose specific, actionable interventions for each segment — not vague recommendations, but ones specifying who receives it, when, why, and in what form

> *Dataset: 16,737 Canadian loyalty members | 392,936 flight activity records | 2012–2018*

---

## 📊 Key Results

| Metric | Value |
|--------|-------|
| Total Members Analysed | 16,737 |
| Overall Churn Rate | 14.0% |
| Best Model (Gradient Boosting) AUC | **0.969** |
| CLV Lost to Churn | **$19.2M CAD** |
| At-Risk Valuables Identified | 764 members |
| Passive Loyalists (upgrade opportunity) | 7,456 members |

---

## 🗂️ Customer Segments Identified

| Segment | Members | Avg CLV | Churn Rate | Priority |
|---------|---------|---------|------------|----------|
| Champions | 3,232 | $13,772 | 1.6% | Protect & Elevate |
| Passive Loyalists | 7,456 | $5,669 | 1.4% | Activate & Upgrade |
| At-Risk Valuables | 764 | $7,960 | 10.6% | **Intervene Urgently** |
| New Low-Engagers | 3,268 | $7,443 | 2.7% | Onboard & Engage |
| Churned Members | 2,017 | $8,192 | 100% | Win-back / Analyse |

---

## 🛠️ Technologies Used

| Category | Tools |
|----------|-------|
| **Language** | Python 3.12 |
| **Data Processing** | Pandas, NumPy |
| **Machine Learning** | Scikit-learn (Random Forest, Gradient Boosting, Logistic Regression, K-Means) |
| **Visualisation** | Matplotlib, Seaborn |
| **Dashboard** | Streamlit |
| **Report** | Python-docx (Word document) |
| **Version Control** | Git, GitHub |

### ML Models Used
- **Logistic Regression** — Baseline (AUC: 0.939)
- **Random Forest** — Strong performer (AUC: 0.964)
- **Gradient Boosting** — Best model, selected (AUC: 0.969)
- **K-Means Clustering** — 5-segment behavioral framework

---

## 📁 Project Structure

```
airline_loyalty_project/
│
├── 01_feature_engineering.py     # Data cleaning + feature engineering
├── 02_churn_model.py             # Churn prediction + segmentation
├── dashboard_app.py              # Streamlit dashboard (5 pages)
├── Technical_Report.docx         # 7-section professional report
│
├── final_dataset.csv             # Generated: enriched member dataset
├── segment_summary.csv           # Generated: segment-level aggregations
│
├── requirements.txt              # Python dependencies
└── README.md                     # This file
```

> **Note:** Raw data files (`Customer_Loyalty_History.csv`, `Customer_Flight_Activity.csv`, `Calendar.csv`) are not included in this repo due to size. See dataset link below.

---

## 🚀 How to Run Locally

### Prerequisites
- Python 3.8 or higher
- pip

### Step 1: Clone the repository
```bash
git clone https://github.com/YOUR_USERNAME/airline-loyalty-project.git
cd airline-loyalty-project
```

### Step 2: Install dependencies
```bash
pip install -r requirements.txt
```

### Step 3: Add the raw data files
Download the dataset and place these 3 files in the project folder:
- `Customer_Loyalty_History.csv`
- `Customer_Flight_Activity.csv`
- `Calendar.csv`

### Step 4: Run feature engineering
```bash
python 01_feature_engineering.py
```
*Generates: `cleaned_features.csv`*

### Step 5: Run churn model + segmentation
```bash
python 02_churn_model.py
```
*Generates: `final_dataset.csv`, `segment_summary.csv`, chart PNGs*

### Step 6: Launch the dashboard
```bash
streamlit run dashboard_app.py
```
*Opens at: `http://localhost:8501`*

---

## 🌐 Live Streamlit App

**[👉 Click here to open the live dashboard](https://airlineloyaltyproject-abswe3zqbfb4qpgnujut6g.streamlit.app/)**

The dashboard has 5 pages:

| Page | What it shows |
|------|--------------|
| 📊 Executive Overview | Key metrics, segment distribution, CLV vs churn risk |
| 🎯 Segment Explorer | Drill into any segment — flights, demographics, risk |
| ⚠️ Churn Risk Finder | Filter members by churn probability threshold |
| 🗺️ Provincial Analysis | Churn rates and CLV by Canadian province |
| 📋 Retention Playbook | WHO / WHEN / WHAT / HOW for each segment |

---

## 📋 Deliverables

- [x] **Python scripts** — Feature engineering + churn model + segmentation
- [x] **Technical Report** — 7-section, 8-page Word document with embedded charts
- [x] **Streamlit Dashboard** — 5-page interactive app, publicly deployed
- [x] **Retention Playbook** — Segment-specific interventions with trade-offs

---

## ✉️ Contact

Email: watankumar25@gmail.com

---

*Summer Projects 2026 | Consulting & Analytics Club | IIT Guwahati*

<!-- ===================================================== -->
<!-- Stroke Risk Intelligence — README.md -->
<!-- Premium • Interactive • Recruiter-Ready -->
<!-- Brand-consistent banner • Recruiter-first • Creative visuals -->
<!-- ===================================================== -->

<div align="center">

<img width="100%" alt="Stroke Risk Intelligence premium header"
src="https://capsule-render.vercel.app/api?type=waving&color=0:070A12,45:1E3A8A,75:4F46E5,100:06B6D4&height=230&section=header&text=Stroke%20Risk%20Intelligence&fontSize=60&fontColor=FFFFFF&fontAlignY=40&desc=Healthcare%20Analytics%20•%20Predictive%20Modeling%20•%20Clinical%20Risk%20Signals%20•%20ML%20Evaluation&descAlignY=70&descSize=18&animation=fadeIn" />

<br/>

<img alt="Python" src="https://img.shields.io/badge/Python-ML-3776AB?style=for-the-badge&logo=python&logoColor=white"/>
<img alt="Scikit-Learn" src="https://img.shields.io/badge/Scikit--Learn-Modeling-F7931E?style=for-the-badge&logo=scikitlearn&logoColor=white"/>
<img alt="Pandas" src="https://img.shields.io/badge/Pandas-Data%20Engineering-150458?style=for-the-badge&logo=pandas&logoColor=white"/>
<img alt="Healthcare" src="https://img.shields.io/badge/Domain-Healthcare%20Analytics-16A34A?style=for-the-badge"/>
<img alt="Evaluation" src="https://img.shields.io/badge/Focus-Recall%20%26%20Risk%20Sensitivity-DC2626?style=for-the-badge"/>

<br/><br/>

<a href="#-quickstart"><b>Quickstart</b></a> •
<a href="#-end-to-end-modeling-logic"><b>Modeling Logic</b></a> •
<a href="#-models-benchmarked"><b>Models</b></a> •
<a href="#-analytical-framework"><b>Framework</b></a> •
<a href="#-core-analytical-themes"><b>Themes</b></a> •
<a href="#-recruiter-snapshot"><b>Recruiter Snapshot</b></a> •
<a href="#-contact"><b>Contact</b></a>

</div>

---


<!-- quickstart:start -->
# 🚀 Quickstart

[![tests](https://github.com/Navyagowda2714/Stroke-prediction/actions/workflows/tests.yml/badge.svg)](https://github.com/Navyagowda2714/Stroke-prediction/actions/workflows/tests.yml)

**Requirements:** Python 3.10+ (or Docker). The trained model (`model_pickle.pkl`) and scaler (`scaler.pkl`) are included, so no training is needed to run the app.

```bash
git clone https://github.com/Navyagowda2714/Stroke-prediction.git
cd Stroke-prediction
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# optional: enables the Gemini explanation of each prediction
export GOOGLE_API_KEY=your-key

python app.py            # → http://127.0.0.1:5000
```

**Run with Docker**

```bash
docker build -t stroke-prediction .
docker run -p 5000:5000 -e GOOGLE_API_KEY=your-key stroke-prediction
```

**Call the API directly**

```bash
curl -X POST http://127.0.0.1:5000/predict -H "Content-Type: application/json" \
  -d '{"gender":"Male","age":"67","hypertension":"0","disease":"1","married":"Yes",
       "work":"Private","residence":"Urban","glucose":"228.7","bmi":"36.6","smoking":"formerly smoked"}'
```

**Run the tests**

```bash
pip install -r requirements-dev.txt
pytest -v
```

| Path | What it is |
|---|---|
| `Stroke_Prediction.ipynb` / `stroke_prediction.py` | EDA, preprocessing, model benchmarking and training |
| `app.py` | Flask app: `/` web form, `/predict` JSON API, optional Gemini explanation |
| `model_pickle.pkl`, `scaler.pkl` | Trained classifier and fitted `StandardScaler` (scikit-learn 1.6.1) |
| `templates/` | Front-end HTML/CSS |
| `tests/` | pytest suite for the API and helper logic (runs in CI on every push) |
<!-- quickstart:end -->

---

# 🧬 End-to-End Modeling Logic

<details open>
<summary><b>🧠 Clinical Signal Orchestration (click to collapse)</b></summary>

<br/>

<div align="center">

### ⚙️ The Stroke Risk “Signal Factory”
A view of how raw health data becomes a risk decision.

</div>

```mermaid
flowchart LR
  A["🧾 Patient Signals<br/>Age • BMI • Glucose • HTN • Heart Disease • Smoking"] --> 
  B["🧹 Integrity Gate<br/>Missing values • type fixes • sanity checks"] -->
  C["🧬 Meaning Encoder<br/>Categoricals → one-hot<br/>clinical meaning preserved"] -->
  D["📏 Calibration<br/>scaling • normalization<br/>fair feature influence"] -->
  E["🧪 Evidence Split<br/>train/test boundary"] -->
  F["🏟️ Model Arena<br/>LR • DT • RF • KNN • GB"] -->
  G["⚖️ Risk Court<br/>recall-first selection<br/>error-map review"] -->
  H["🎯 Output<br/>stroke probability"]
```

<div align="center">

#### 🧠 What makes this *not* a generic pipeline
Instead of optimizing for “best accuracy”, this flow is built to reduce **false negatives** — because missing a high-risk patient is the most expensive mistake.

</div>

</details>

---

# ⚔️ Models Benchmarked

<details open>
<summary><b>🏁 “Model League Cards” — not a table (click to collapse)</b></summary>

<br/>

<div align="center">

<table>
<tr>
<td width="20%" valign="top">

### 🩺 Logistic Regression  
**Role:** Baseline clinician  
**Strength:** Probability clarity  
**Wins when:** interpretability + stability matter

</td>
<td width="20%" valign="top">

### 🌳 Decision Tree  
**Role:** Rule explainer  
**Strength:** Human readable paths  
**Wins when:** justification is critical

</td>
<td width="20%" valign="top">

### 🌲 Random Forest  
**Role:** Stability engine  
**Strength:** Robust ensembles  
**Wins when:** noise/variance exists

</td>
<td width="20%" valign="top">

### 🧭 KNN  
**Role:** Similar-patient matcher  
**Strength:** Local pattern discovery  
**Wins when:** clusters are meaningful

</td>
<td width="20%" valign="top">

### ⚡ Gradient Boosting  
**Role:** Precision finisher  
**Strength:** Strong signal learning  
**Wins when:** performance needs refinement

</td>
</tr>
</table>

</div>

<div align="center">
<sub>All models are evaluated under the same boundaries so improvements are real — not accidental.</sub>
</div>

</details>

---

# ⚖️ Analytical Framework

<details open>
<summary><b>🏥 Risk Review Board — evaluation that sounds like healthcare (click to collapse)</b></summary>

<br/>

<div align="center">

### 🧾 “Clinical Evaluation Brief”
A compact decision memo style, not a metric list.

</div>

> **1) Safety First — Recall**  
> If the model misses stroke-positive cases, it fails the healthcare requirement.

> **2) Alarm Discipline — Precision**  
> Too many false alarms creates unnecessary escalation and reduces trust.

> **3) Balanced Judgment — F1**  
> Ensures the model isn’t good at only one thing.

> **4) Error Autopsy — Confusion Matrix**  
> Shows *how* the model fails, not just *how much* it fails.

> **5) Separation Strength — ROC-AUC**  
> Tests whether the model genuinely distinguishes low vs high risk.

<div align="center">
<b>Selection Rule:</b> accuracy never overrides clinical cost.
</div>

</details>

---

# 🧠 Core Analytical Themes

<details open>
<summary><b>🧩 “Theme Board” — styled like a brand panel (click to collapse)</b></summary>

<br/>

<div align="center">

<!-- classy "panels" without mermaid -->
<table>
<tr>
<td width="33%" valign="top">

### 🩺 Risk Sensitivity
- Recall-first mindset  
- False-negative control  
- Minority class awareness  
- Threshold thinking  

</td>
<td width="33%" valign="top">

### 🧱 Engineering Discipline
- Clean preprocessing chain  
- Comparable benchmarks  
- Stable feature handling  
- Reproducible evaluation  

</td>
<td width="33%" valign="top">

### 🧠 Clinical Reasoning
- Metrics mapped to impact  
- Decision framing  
- Error interpretation  
- Responsible ML lens  

</td>
</tr>
</table>

</div>

<div align="center">
<sub>This project demonstrates applied ML behavior — not “train model, print accuracy”.</sub>
</div>

</details>

---

# 🌟 Why This Project Matters

<details open>
<summary><b>✨ Recruiter Story Mode — “Decision System, not a notebook” (click to collapse)</b></summary>

<br/>

<div align="center">

### 🧠 The difference

</div>

```text
Most projects:   "I trained a classifier."
This project:    "I designed a risk-sensitive decision architecture."
```

<div align="center">

### 🎯 What’s special here

</div>

- **Healthcare cost-aware evaluation** (recall prioritized for safety)  
- **Benchmarking discipline** (models compared fairly, same boundaries)  
- **Interpretation mindset** (error behavior analyzed, not hidden)  
- **Recruiter clarity** (communicates impact + engineering maturity)

</details>

---

# 🎯 Recruiter Snapshot

<details open>
<summary><b>📌 What a recruiter should remember in 15 seconds (click to collapse)</b></summary>

<br/>

<div align="center">

<table>
<tr>
<td width="50%" valign="top">

### ✅ What I demonstrated
- End-to-end ML system thinking  
- Class-imbalance awareness  
- Model benchmarking discipline  
- Healthcare-first evaluation logic  
- Clear analytical communication  

</td>
<td width="50%" valign="top">

### 🧠 Why it’s valuable
- Medical ML demands safety metrics  
- False negatives are costlier than false positives  
- Evaluation isn’t reporting — it’s design  
- This reflects applied ML maturity  

</td>
</tr>
</table>

</div>

</details>

---

# 📬 Contact

<div align="center">

<a href="https://www.linkedin.com/in/navyashree-byregowda-472821196/">
<img src="https://img.shields.io/badge/LinkedIn-Connect-1E40AF?style=for-the-badge&logo=linkedin&logoColor=white"/>
</a>

<a href="https://github.com/Navyagowda2714">
<img src="https://img.shields.io/badge/GitHub-Portfolio-111827?style=for-the-badge&logo=github&logoColor=white"/>
</a>

<a href="mailto:navyashreebyregowda@gmail.com">
<img src="https://img.shields.io/badge/Email-Let's%20Talk-DC2626?style=for-the-badge&logo=gmail&logoColor=white"/>
</a>

<br/><br/>
<sub>Stroke Risk Intelligence — responsible healthcare prediction through risk-aware machine learning.</sub>

</div>

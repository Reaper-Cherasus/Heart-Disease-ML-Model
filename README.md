# Heart Disease Prediction Model

A full-stack machine learning pipeline that predicts heart disease using the UCI Cleveland dataset. Includes model training, a REST API, and a web UI.

---

## Tech Stack

| Layer | Technology |
|---|---|
| Language | Python 3.14.7 |
| Environment | uv virtual environment |
| ML | scikit-learn, XGBoost |
| Data | pandas, numpy |
| API | FastAPI |
| Web UI | Streamlit |
| Dependency Management | uv |

---

## Dataset

**UCI Heart Disease Dataset (Cleveland)**
- 303 patient records
- 13 input features (age, sex, chest pain type, blood pressure, cholesterol, etc.)
- Target: binary classification (0 = no disease, 1 = disease) + risk probability score

---

## Project Structure

```
heart-disease-model/
+-- data/
¦   +-- heart.csv               # UCI Cleveland dataset
+-- notebooks/
¦   +-- exploration.ipynb       # EDA and experimentation
+-- src/
¦   +-- data_pipeline.py        # Data loading, cleaning, preprocessing
¦   +-- train.py                # Model training and evaluation
¦   +-- predict.py              # Prediction and probability scoring
¦   +-- model/
¦       +-- best_model.pkl      # Saved trained model
+-- api/
¦   +-- main.py                 # FastAPI REST API
+-- app/
¦   +-- streamlit_app.py        # Streamlit web UI
+-- requirements.txt
+-- pyproject.toml
+-- README.md
```

---

## Build Steps

### Step 1 — Environment Setup

Set up the project using uv and Python 3.14.7.

```powershell
# Install uv (if not already installed)
pip install uv

# Download Python 3.14.7
uv python install 3.14.7

# Create virtual environment
uv venv --python 3.14.7

# Activate (Windows CMD)
.venv\Scripts\activate.bat

# Activate (PowerShell)
Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
.venv\Scripts\activate
```

---

### Step 2 — Install Dependencies

```powershell
uv pip install pandas numpy scikit-learn xgboost fastapi uvicorn streamlit joblib
```

---

### Step 3 — Data Pipeline

Load and preprocess the UCI Cleveland dataset.

- Load `heart.csv`
- Handle missing values
- Encode categorical features (chest pain type, slope, thal, etc.)
- Scale numerical features (age, cholesterol, blood pressure, etc.)
- Split into train/test sets (80/20)

```powershell
python src/data_pipeline.py
```

---

### Step 4 — Model Training & Evaluation

Train and compare multiple models, then select the best performer.

**Models trained:**
- Logistic Regression (interpretable baseline)
- Random Forest (robust ensemble)
- XGBoost (high performance gradient boosting) ? recommended

**Evaluation metrics:**
- Accuracy
- AUC-ROC score
- Confusion matrix
- Feature importance plot

```powershell
python src/train.py
```

The best model is saved to `src/model/best_model.pkl`.

---

### Step 5 — REST API (FastAPI)

A FastAPI endpoint that accepts patient data and returns a prediction and probability score.

**Start the API:**
```powershell
uvicorn api.main:app --reload
```

**Example request:**
```json
POST http://localhost:8000/predict

{
  "age": 55,
  "sex": 1,
  "cp": 2,
  "trestbps": 130,
  "chol": 250,
  "fbs": 0,
  "restecg": 1,
  "thalach": 165,
  "exang": 0,
  "oldpeak": 1.2,
  "slope": 2,
  "ca": 0,
  "thal": 2
}
```

**Example response:**
```json
{
  "prediction": 1,
  "label": "Heart Disease Detected",
  "probability": 0.82
}
```

API docs available at: `http://localhost:8000/docs`

---

### Step 6 — Web UI (Streamlit)

A user-friendly web interface to input patient data and get instant predictions.

**Start the web app:**
```powershell
streamlit run app/streamlit_app.py
```

Opens at: `http://localhost:8501`

**Features:**
- Input form for all 13 patient features
- Real-time prediction (disease / no disease)
- Risk probability score with visual indicator
- Feature descriptions and normal ranges

---

## Features (Model Inputs)

| Feature | Description |
|---|---|
| age | Age in years |
| sex | Sex (1 = male, 0 = female) |
| cp | Chest pain type (0–3) |
| trestbps | Resting blood pressure (mm Hg) |
| chol | Serum cholesterol (mg/dl) |
| fbs | Fasting blood sugar > 120 mg/dl (1 = true) |
| restecg | Resting ECG results (0–2) |
| thalach | Maximum heart rate achieved |
| exang | Exercise induced angina (1 = yes) |
| oldpeak | ST depression induced by exercise |
| slope | Slope of peak exercise ST segment (0–2) |
| ca | Number of major vessels colored by fluoroscopy (0–3) |
| thal | Thalassemia type (0–3) |

---

## Output

| Output | Description |
|---|---|
| prediction | 0 = No Heart Disease, 1 = Heart Disease |
| probability | Risk score between 0.0 and 1.0 |

---

## License

MIT License

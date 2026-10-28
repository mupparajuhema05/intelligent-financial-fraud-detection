import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)

st.set_page_config(page_title="Intelligent Financial Fraud Detection", page_icon="🛡️", layout="wide")

st.title("🛡️ Intelligent Financial Fraud Detection")
st.caption("Python-based web application for transaction data preparation, fraud detection, and model evaluation.")

st.info(
    "Demo project: upload a CSV containing transaction features and a target column that labels "
    "transactions as fraud (1) or legitimate (0). Do not upload real customer financial information."
)

with st.sidebar:
    st.header("Project Modules")
    st.markdown("**1. Data Preprocessing**")
    st.markdown("**2. Fraud Detection**")
    st.markdown("**3. Performance Evaluation**")
    st.divider()
    st.write("**Required target labels:** 0 = Legitimate, 1 = Fraud")

uploaded_file = st.file_uploader("Upload transaction dataset (CSV)", type=["csv"])

def make_demo_data(n=1000, seed=42):
    rng = np.random.default_rng(seed)
    amount = rng.lognormal(mean=3.2, sigma=1.0, size=n).round(2)
    hour = rng.integers(0, 24, size=n)
    distance = rng.exponential(scale=15, size=n).round(2)
    account_age = rng.integers(1, 2500, size=n)
    online = rng.integers(0, 2, size=n)
    # Synthetic labels only for demonstration, not a real fraud model.
    risk = (
        (amount > np.quantile(amount, 0.92)).astype(int)
        + (hour <= 4).astype(int)
        + (distance > 35).astype(int)
        + online
    )
    probability = 1 / (1 + np.exp(-(risk - 2.7) * 1.25))
    fraud = rng.binomial(1, probability)
    return pd.DataFrame({
        "transaction_amount": amount,
        "transaction_hour": hour,
        "distance_from_usual_km": distance,
        "account_age_days": account_age,
        "online_transaction": online,
        "fraud": fraud
    })

if uploaded_file is not None:
    try:
        data = pd.read_csv(uploaded_file)
    except Exception as exc:
        st.error(f"Could not read the CSV file: {exc}")
        st.stop()
else:
    st.warning("No file uploaded. A synthetic sample dataset is shown so you can explore the demo.")
    data = make_demo_data()

st.subheader("Module 1: Data Preprocessing")
left, right = st.columns(2)
with left:
    st.metric("Rows", f"{len(data):,}")
with right:
    st.metric("Columns", f"{len(data.columns):,}")

st.write("Dataset preview")
st.dataframe(data.head(10), use_container_width=True)

missing = data.isna().sum()
prep_summary = pd.DataFrame({
    "Column": data.columns,
    "Data type": [str(dtype) for dtype in data.dtypes],
    "Missing values": [int(missing[c]) for c in data.columns],
    "Unique values": [int(data[c].nunique(dropna=True)) for c in data.columns]
})
st.write("Data quality summary")
st.dataframe(prep_summary, use_container_width=True)

target = st.selectbox(
    "Select the target/label column",
    options=list(data.columns),
    index=(list(data.columns).index("fraud") if "fraud" in data.columns else len(data.columns)-1)
)

if st.button("Prepare dataset"):
    st.session_state["prepared"] = True
    st.success("Dataset inspection completed. Missing values will be handled by the training pipeline.")

st.subheader("Module 2: Fraud Detection")
st.write("Choose the target column above. For this demo, target labels must be coded as 0 (legitimate) and 1 (fraud).")

if target not in data.columns:
    st.error("Please select a valid target column.")
    st.stop()

model_data = data.dropna(subset=[target]).copy()
try:
    y_raw = model_data[target]
    unique_labels = set(pd.Series(y_raw).dropna().unique().tolist())
    if unique_labels.issubset({0, 1, False, True}) and len(unique_labels) == 2:
        y = y_raw.astype(int)
    else:
        st.warning("The selected target is not encoded as 0 and 1. Map legitimate transactions to 0 and fraud transactions to 1 before training.")
        st.stop()
except Exception:
    st.error("Could not interpret the target labels. Use 0 and 1 labels.")
    st.stop()

if y.nunique() < 2:
    st.error("The target column must contain both classes: 0 and 1.")
    st.stop()

X = model_data.drop(columns=[target])
if X.shape[1] == 0:
    st.error("The dataset needs at least one feature column besides the target.")
    st.stop()

# Drop columns that are entirely empty and prevent unsupported target leakage.
X = X.dropna(axis=1, how="all")
if X.shape[1] == 0:
    st.error("No usable feature columns were found.")
    st.stop()

test_size = st.slider("Test data percentage", min_value=10, max_value=40, value=20, step=5)
n_estimators = st.slider("Number of Random Forest trees", min_value=50, max_value=300, value=100, step=50)

if st.button("Train fraud detection model", type="primary"):
    try:
        if y.value_counts().min() < 2:
            st.error("Each class needs at least two rows for a stratified train/test split.")
            st.stop()

        numeric_cols = X.select_dtypes(include=["number", "bool"]).columns.tolist()
        categorical_cols = [c for c in X.columns if c not in numeric_cols]

        numeric_pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler())
        ])
        categorical_pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore"))
        ])

        transformers = []
        if numeric_cols:
            transformers.append(("numeric", numeric_pipeline, numeric_cols))
        if categorical_cols:
            transformers.append(("categorical", categorical_pipeline, categorical_cols))

        preprocess = ColumnTransformer(transformers=transformers)
        model = Pipeline([
            ("preprocessing", preprocess),
            ("classifier", RandomForestClassifier(
                n_estimators=n_estimators, random_state=42, class_weight="balanced"
            ))
        ])

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size / 100, random_state=42, stratify=y
        )
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)

        st.session_state["model"] = model
        st.session_state["feature_columns"] = list(X.columns)
        st.session_state["target_column"] = target
        st.session_state["metrics"] = {
            "accuracy": accuracy_score(y_test, predictions),
            "precision": precision_score(y_test, predictions, zero_division=0),
            "recall": recall_score(y_test, predictions, zero_division=0),
            "f1": f1_score(y_test, predictions, zero_division=0),
            "confusion": confusion_matrix(y_test, predictions, labels=[0, 1]),
            "report": classification_report(y_test, predictions, labels=[0, 1],
                                            target_names=["Legitimate (0)", "Fraud (1)"],
                                            zero_division=0)
        }
        st.success("Model training completed.")
    except Exception as exc:
        st.error(f"Model training failed: {exc}")
        st.caption("Check that the dataset has enough rows, both target classes, and valid feature columns.")

if "model" in st.session_state:
    st.subheader("Check a single transaction")
    st.write("Enter values for the feature columns to estimate whether this transaction resembles the fraud class in the training data.")
    feature_cols = st.session_state["feature_columns"]
    current_data = X[feature_cols]
    inputs = {}
    cols = st.columns(2)
    for i, col_name in enumerate(feature_cols):
        series = current_data[col_name]
        with cols[i % 2]:
            if pd.api.types.is_numeric_dtype(series) or pd.api.types.is_bool_dtype(series):
                default = float(series.median()) if pd.api.types.is_numeric_dtype(series) else 0
                inputs[col_name] = st.number_input(col_name, value=default)
            else:
                choices = series.dropna().astype(str).unique().tolist()
                choices = choices[:100] if choices else ["Unknown"]
                inputs[col_name] = st.selectbox(col_name, options=choices)

    if st.button("Check transaction"):
        sample = pd.DataFrame([inputs], columns=feature_cols)
        try:
            pred = int(st.session_state["model"].predict(sample)[0])
            if pred == 1:
                st.error("Model result: Potentially fraudulent transaction")
            else:
                st.success("Model result: Likely legitimate transaction")
            if hasattr(st.session_state["model"], "predict_proba"):
                probability = st.session_state["model"].predict_proba(sample)[0]
                classes = list(st.session_state["model"].named_steps["classifier"].classes_)
                if 1 in classes:
                    fraud_prob = float(probability[classes.index(1)])
                    st.metric("Estimated fraud probability", f"{fraud_prob:.1%}")
        except Exception as exc:
            st.error(f"Could not score this transaction: {exc}")

    st.subheader("Module 3: Performance Evaluation")
    metrics = st.session_state["metrics"]
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Accuracy", f"{metrics['accuracy']:.3f}")
    m2.metric("Precision", f"{metrics['precision']:.3f}")
    m3.metric("Recall", f"{metrics['recall']:.3f}")
    m4.metric("F1-score", f"{metrics['f1']:.3f}")

    st.write("Confusion matrix (actual rows × predicted columns)")
    cm = pd.DataFrame(
        metrics["confusion"],
        index=["Actual legitimate (0)", "Actual fraud (1)"],
        columns=["Predicted legitimate (0)", "Predicted fraud (1)"]
    )
    st.dataframe(cm, use_container_width=True)
    st.write("Classification report")
    st.code(metrics["report"])

    st.download_button(
        "Download evaluation metrics (CSV)",
        data=pd.DataFrame([{
            "accuracy": metrics["accuracy"],
            "precision": metrics["precision"],
            "recall": metrics["recall"],
            "f1_score": metrics["f1"]
        }]).to_csv(index=False),
        file_name="fraud_detection_metrics.csv",
        mime="text/csv"
    )
else:
    st.info("Train the model to enable single-transaction prediction and performance evaluation.")

st.divider()
st.caption("Educational prototype only. Model predictions are not proof of fraud and should not be used as the sole basis for financial decisions.")

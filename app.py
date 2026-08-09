import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, matthews_corrcoef, roc_auc_score,
    ConfusionMatrixDisplay
)

st.title("Wine Quality Classification App 🍷")

# Sidebar images
st.sidebar.image("images/redwine_glass.jpeg", use_container_width=True)
st.sidebar.image("images/wine_glass.webp", use_container_width=True)

# Load models
models = {
    "Logistic Regression": joblib.load("model/logistic_regression.pkl"),
    "Decision Tree": joblib.load("model/decision_tree.pkl"),
    "KNN": joblib.load("model/knn.pkl"),
    "Naive Bayes": joblib.load("model/naive_bayes.pkl"),
    "Random Forest": joblib.load("model/random_forest.pkl")
}

# --- Dataset selection ---
st.subheader("Dataset Selection")

dataset_choice = st.selectbox("Choose a built-in dataset", ["Red Wine", "White Wine"])
if dataset_choice == "Red Wine":
    test_data = pd.read_csv("winequality-red.csv", sep=";")
else:
    test_data = pd.read_csv("winequality-white.csv", sep=";")

uploaded_file = st.file_uploader("Or upload your own CSV file", type=["csv"])
if uploaded_file is not None:
    test_data = pd.read_csv(uploaded_file)

st.write("Test Data Preview:", test_data.head())

# Separate features and labels
if "quality" in test_data.columns:
    test_data['label'] = (test_data['quality'] >= 6).astype(int)
    X_test = test_data.drop(['quality','label'], axis=1)
    y_test = test_data['label']
else:
    st.error("Dataset must contain a 'quality' column.")
    st.stop()

# --- Sidebar summary ---
st.sidebar.markdown("## 📊 **Dataset & Model Info**")
st.sidebar.markdown(f"**Dataset:** :red[{dataset_choice if uploaded_file is None else 'Uploaded File'}]")
st.sidebar.markdown(f"**Rows:** :red[{test_data.shape[0]}]")
st.sidebar.markdown(f"**Features:** :red[{X_test.shape[1]}]")

class_counts = y_test.value_counts().to_dict()
st.sidebar.markdown("**Classes:**")
fig_pie, ax_pie = plt.subplots()
ax_pie.pie(class_counts.values(), labels=[f"Class {k}" for k in class_counts.keys()],
           autopct='%1.1f%%', colors=["#ff9999","#66b3ff"])
ax_pie.set_title("Class Distribution")
st.sidebar.pyplot(fig_pie)
plt.close(fig_pie)

st.sidebar.markdown("**Models Available:**")
for m in models.keys():
    st.sidebar.markdown(f"- :red[{m}]")

# --- Model selection ---
model_choice = st.selectbox("Choose a model", list(models.keys()))
model = models[model_choice]

y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:,1] if hasattr(model, "predict_proba") else y_pred

# --- Single model metrics ---
st.subheader(f"Evaluation Metrics for {model_choice}")
st.write(f"Accuracy: {accuracy_score(y_test, y_pred):.3f}")
st.write(f"AUC: {roc_auc_score(y_test, y_prob):.3f}")
st.write(f"Precision: {precision_score(y_test, y_pred):.3f}")
st.write(f"Recall: {recall_score(y_test, y_pred):.3f}")
st.write(f"F1 Score: {f1_score(y_test, y_pred):.3f}")
st.write(f"MCC: {matthews_corrcoef(y_test, y_pred):.3f}")

metrics = {
    "Accuracy": accuracy_score(y_test, y_pred),
    "Precision": precision_score(y_test, y_pred),
    "Recall": recall_score(y_test, y_pred),
    "F1": f1_score(y_test, y_pred),
    "MCC": matthews_corrcoef(y_test, y_pred),
    "AUC": roc_auc_score(y_test, y_prob)
}

# --- Download single model metrics ---
df_single = pd.DataFrame(metrics, index=[model_choice])
csv_single = df_single.to_csv().encode('utf-8')
st.download_button(
    label=f"📥 Download {model_choice} Metrics as CSV",
    data=csv_single,
    file_name=f"{model_choice.replace(' ','_').lower()}_metrics.csv",
    mime="text/csv"
)

# --- Create columns depending on model ---
if model_choice == "Random Forest":
    col1, col2, col3 = st.columns(3)
else:
    col1, col2 = st.columns(2)

with col1:
    fig, ax = plt.subplots()
    ax.bar(metrics.keys(), metrics.values(), color="maroon")
    ax.set_ylabel("Score")
    ax.set_title(f"{model_choice} Metrics")
    st.pyplot(fig)
    plt.close(fig)

with col2:
    fig_cm, ax_cm = plt.subplots()
    ConfusionMatrixDisplay.from_predictions(y_test, y_pred, ax=ax_cm, cmap="Reds")
    ax_cm.set_title(f"{model_choice} Confusion Matrix")
    st.pyplot(fig_cm)
    plt.close(fig_cm)

if model_choice == "Random Forest":
    with col3:
        st.subheader("Feature Importance")
        importances = model.feature_importances_
        indices = np.argsort(importances)[::-1]
        fig, ax = plt.subplots()
        ax.bar(range(len(importances)), importances[indices], color="maroon")
        ax.set_xticks(range(len(importances)))
        ax.set_xticklabels(X_test.columns[indices], rotation=45, ha="right")
        ax.set_ylabel("Importance")
        ax.set_title("Random Forest Feature Importance")
        st.pyplot(fig)
        plt.close(fig)

# --- Compare all models button ---
if st.button("Compare All Models"):
    results = {}
    for name, mdl in models.items():
        y_pred_all = mdl.predict(X_test)
        y_prob_all = mdl.predict_proba(X_test)[:,1] if hasattr(mdl, "predict_proba") else y_pred_all
        results[name] = {
            "Accuracy": accuracy_score(y_test, y_pred_all),
            "Precision": precision_score(y_test, y_pred_all),
            "Recall": recall_score(y_test, y_pred_all),
            "F1": f1_score(y_test, y_pred_all),
            "MCC": matthews_corrcoef(y_test, y_pred_all),
            "AUC": roc_auc_score(y_test, y_prob_all)
        }

    # Convert results dict to DataFrame
    df_results = pd.DataFrame(results).T

    st.subheader("📊 Model Comparison")
    st.dataframe(df_results)

    # Optional: allow download
    csv_results = df_results.to_csv().encode('utf-8')
    st.download_button(
        label="📥 Download Comparison Metrics as CSV",
        data=csv_results,
        file_name="model_comparison.csv",
        mime="text/csv"
    )

    # --- Grouped bar chart for all metrics ---
    st.subheader("📊 Metrics Comparison (Grouped Bar Chart)")
    fig_all, ax_all = plt.subplots(figsize=(10,6))
    df_results.plot(kind="bar", ax=ax_all)
    ax_all.set_ylabel("Score")
    ax_all.set_title("Comparison of All Metrics Across Models")
    ax_all.legend(loc="upper right")
    st.pyplot(fig_all)
    plt.close(fig_all)

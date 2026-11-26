import streamlit as st
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    confusion_matrix,
    ConfusionMatrixDisplay,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)

# --------------------------------------------------
# Page setup
# --------------------------------------------------

st.set_page_config(page_title="Penguins Decision Tree Explorer")

st.title("Penguins Decision Tree Explorer")

st.write(
    """
    Explore how feature selection and hyperparameters
    affect a Decision Tree classifier trained on the
    Penguins dataset.
    """
)

# --------------------------------------------------
# Load data
# --------------------------------------------------

df = sns.load_dataset("penguins")

df.dropna(inplace=True,ignore_index=True)

feature_names = [
    "bill_length_mm",
    "bill_depth_mm",
    "flipper_length_mm",
    "body_mass_g",
]

target_name = "species"

# --------------------------------------------------
# Feature selection
# --------------------------------------------------

st.header("1. Select Features")

selected_features = st.multiselect(
    "Choose features to use:",
    feature_names,
    default=feature_names,
)

if len(selected_features) == 0:
    st.warning("Please select at least one feature.")
    st.stop()

# --------------------------------------------------
# Hyperparameters
# --------------------------------------------------

st.header("2. Decision Tree") 

st.subheader("Hyperparameters")

max_depth = st.slider(
    "Maximum Depth",
    min_value=1,
    max_value=15,
    value=3,
)

min_samples_split = st.slider(
    "Minimum Samples Split",
    min_value=2,
    max_value=20,
    value=2,
)

criterion = st.selectbox(
    "Splitting Criterion",
    ["gini", "entropy"],
)

# --------------------------------------------------
# Train model
# --------------------------------------------------

st.subheader("Decision Tree")

X = df[selected_features]
y = df[target_name]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.3,
    random_state=42,
    stratify=y,
)

model = DecisionTreeClassifier(
    max_depth=max_depth,
    min_samples_split=min_samples_split,
    criterion=criterion,
    random_state=42,
)

model.fit(X_train, y_train)

# --------------------------------------------------
# Plot tree
# --------------------------------------------------

fig_tree, ax_tree = plt.subplots(
    figsize=(14, 8)
)

plot_tree(
    model,
    feature_names=selected_features,
    class_names=model.classes_,
    filled=True,
    rounded=True,
    ax=ax_tree,
)

st.pyplot(fig_tree)

# --------------------------------------------------
# Feature importance
# --------------------------------------------------

st.subheader("Feature Importance")

importance_df = pd.DataFrame(
    {
        "Feature": selected_features,
        "Importance": model.feature_importances_,
    }
).sort_values(
    by="Importance",
    ascending=False,
)

fig_imp, ax_imp = plt.subplots(figsize=(6, 4))

sns.barplot(
    data=importance_df,
    x="Importance",
    y="Feature",
    ax=ax_imp,
)

ax_imp.set_title("Decision Tree Feature Importance")

st.pyplot(fig_imp)

st.dataframe(
    importance_df.style.format(
        {"Importance": "{:.3f}"}
    )
)

# --------------------------------------------------
# Model performance
# --------------------------------------------------

st.header("3. Model Performance")

y_pred = model.predict(X_test)

# --------------------------------------------------
# Confusion matrix
# --------------------------------------------------

st.subheader("Confusion Matrix")

fig_cm, ax_cm = plt.subplots(figsize=(5, 4))

cm = confusion_matrix(y_test, y_pred)

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=model.classes_,
)

disp.plot(ax=ax_cm)

st.pyplot(fig_cm)

# --------------------------------------------------
# Metrics
# --------------------------------------------------

accuracy = accuracy_score(y_test, y_pred)

precision = precision_score(
    y_test,
    y_pred,
    average="weighted",
)

recall = recall_score(
    y_test,
    y_pred,
    average="weighted",
)

f1 = f1_score(
    y_test,
    y_pred,
    average="weighted",
)

col1, col2 = st.columns(2)

with col1:
    st.metric("Accuracy", f"{accuracy:.3f}")
    st.metric("Precision", f"{precision:.3f}")

with col2:
    st.metric("Recall", f"{recall:.3f}")
    st.metric("F1 Score", f"{f1:.3f}")

# --------------------------------------------------
# Prediction section
# --------------------------------------------------

st.header("4. Make a Prediction")

st.write(
    "Enter measurements for a new penguin."
)

user_values = {}

for feature in selected_features:

    min_val = float(df[feature].min())
    max_val = float(df[feature].max())

    user_values[feature] = st.number_input(
        feature.replace("_", " ").title(),
        min_value=min_val,
        max_value=max_val,
    )

if st.button("Predict Species"):

    new_sample = pd.DataFrame(
        [user_values]
    )

    prediction = model.predict(new_sample)[0]

    probabilities = model.predict_proba(new_sample)[0]

    st.success(
        f"Predicted species: {prediction}"
    )

    prob_df = pd.DataFrame(
        {
            "Species": model.classes_,
            "Probability": probabilities,
        }
    )

    st.subheader("Prediction Probabilities")

    st.dataframe(
        prob_df.style.format(
            {"Probability": "{:.3f}"}
        )
    )

    st.bar_chart(
        prob_df.set_index("Species")
    )

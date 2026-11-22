
# ============================================================
# BRITISH AIRWAYS 
# ============================================================



import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os

from sklearn.model_selection import (
    train_test_split,
    StratifiedKFold,
    cross_validate
)

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
    classification_report,
    ConfusionMatrixDisplay
)

from sklearn.inspection import permutation_importance



df = pd.read_csv("customer_booking.csv", encoding="latin1")

print("DATASET LOADED SUCCESSFULLY")
print("=" * 50)



print("\nFIRST FIVE ROWS")
print(df.head())

print("\nLAST FIVE ROWS")
print(df.tail())

print("\nDATASET SHAPE")
print(df.shape)

print("\nCOLUMN NAMES")
print(df.columns.tolist())

print("\nDATA TYPES")
print(df.dtypes)

print("\nDATASET INFORMATION")
df.info()

print("\nSTATISTICAL SUMMARY")
print(df.describe(include="all").T)

print("\nMISSING VALUES")
print(df.isnull().sum())

print("\nDUPLICATE ROWS")
print(df.duplicated().sum())

print("\nTARGET DISTRIBUTION")
print(df["booking_complete"].value_counts())

print("\nTARGET PERCENTAGE")
print(df["booking_complete"].value_counts(normalize=True) * 100)




df = df.drop_duplicates().copy()



df = df.dropna(subset=["booking_complete"])



df["booking_complete"] = df["booking_complete"].astype(int)

print("\nCLEANED DATASET SHAPE")
print(df.shape)




df["total_addons"] = (
    df["wants_extra_baggage"].fillna(0)
    + df["wants_preferred_seat"].fillna(0)
    + df["wants_in_flight_meals"].fillna(0)
)

df["has_any_addon"] = (
    df["total_addons"] > 0
).astype(int)

df["purchase_lead_category"] = pd.cut(
    df["purchase_lead"],
    bins=[-1, 7, 30, 90, np.inf],
    labels=["Last Minute", "Short Term",
            "Medium Term", "Early Booking"]
)


df["stay_category"] = pd.cut(
    df["length_of_stay"],
    bins=[-1, 3, 7, 14, np.inf],
    labels=["Short Stay", "One Week",
            "Two Weeks", "Long Stay"]
)



df["flight_period"] = pd.cut(
    df["flight_hour"],
    bins=[-1, 5, 11, 17, 23],
    labels=["Night", "Morning", "Afternoon", "Evening"]
)

print("\nFEATURE ENGINEERING COMPLETED")
print(df.head())


os.makedirs("output", exist_ok=True)



plt.figure(figsize=(7, 5))

sns.countplot(
    data=df,
    x="booking_complete"
)

plt.title("Customer Booking Distribution")
plt.xlabel("Booking Complete (0 = No, 1 = Yes)")
plt.ylabel("Number of Customers")

plt.tight_layout()
plt.savefig("output/booking_distribution.png", dpi=200)
plt.show()

print("\nBOOKING RATE:")
print(df["booking_complete"].mean() * 100, "%")

plt.figure(figsize=(8, 5))

sns.barplot(
    data=df,
    x="sales_channel",
    y="booking_complete",
    errorbar=None
)

plt.title("Booking Rate by Sales Channel")
plt.ylabel("Booking Rate")

plt.tight_layout()
plt.savefig("output/booking_by_channel.png", dpi=200)
plt.show()


plt.figure(figsize=(8, 5))

sns.barplot(
    data=df,
    x="trip_type",
    y="booking_complete",
    errorbar=None
)

plt.title("Booking Rate by Trip Type")
plt.ylabel("Booking Rate")

plt.tight_layout()
plt.savefig("output/booking_by_trip_type.png", dpi=200)
plt.show()


X = df.drop(columns=["booking_complete"])

y = df["booking_complete"]

print("\nFEATURE SHAPE:", X.shape)
print("TARGET SHAPE:", y.shape)



categorical_features = X.select_dtypes(
    include=["object", "category"]
).columns.tolist()

numerical_features = X.select_dtypes(
    include=["number"]
).columns.tolist()

print("\nCATEGORICAL FEATURES")
print(categorical_features)

print("\nNUMERICAL FEATURES")
print(numerical_features)



numeric_transformer = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="median"))
    ]
)

categorical_transformer = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(
            handle_unknown="ignore",
            min_frequency=50
        ))
    ]
)

preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, numerical_features),
        ("cat", categorical_transformer, categorical_features)
    ]
)



X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nTRAINING DATA:", X_train.shape)
print("TESTING DATA:", X_test.shape)



rf_model = RandomForestClassifier(
    n_estimators=200,
    max_depth=18,
    min_samples_split=10,
    min_samples_leaf=5,
    max_features="sqrt",
    class_weight="balanced_subsample",
    random_state=42,
    n_jobs=-1
)

# Complete machine learning pipeline

model = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("classifier", rf_model)
    ]
)


cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

scoring = {
    "accuracy": "accuracy",
    "precision": "precision",
    "recall": "recall",
    "f1": "f1",
    "roc_auc": "roc_auc",
    "pr_auc": "average_precision"
}

print("\nRUNNING 5-FOLD CROSS VALIDATION...")

cv_results = cross_validate(
    model,
    X_train,
    y_train,
    cv=cv,
    scoring=scoring,
    n_jobs=-1
)

print("\nCROSS VALIDATION RESULTS")
print("=" * 50)

cv_summary = {}

for metric in scoring:

    scores = cv_results["test_" + metric]

    cv_summary[metric] = scores.mean()

    print(
        f"{metric.upper()}: "
        f"{scores.mean():.4f} "
        f"+/- {scores.std():.4f}"
    )

print("\nTRAINING RANDOM FOREST MODEL...")

model.fit(X_train, y_train)

print("MODEL TRAINED SUCCESSFULLY")


y_pred = model.predict(X_test)

y_probability = model.predict_proba(X_test)[:, 1]

accuracy = accuracy_score(y_test, y_pred)

precision = precision_score(
    y_test, y_pred, zero_division=0
)

recall = recall_score(
    y_test, y_pred, zero_division=0
)

f1 = f1_score(
    y_test, y_pred, zero_division=0
)

roc_auc = roc_auc_score(
    y_test, y_probability
)

pr_auc = average_precision_score(
    y_test, y_probability
)

print("\nFINAL MODEL PERFORMANCE")
print("=" * 50)

print("Accuracy:", round(accuracy, 4))
print("Precision:", round(precision, 4))
print("Recall:", round(recall, 4))
print("F1 Score:", round(f1, 4))
print("ROC-AUC:", round(roc_auc, 4))
print("PR-AUC:", round(pr_auc, 4))

print("\nCLASSIFICATION REPORT")

print(
    classification_report(
        y_test,
        y_pred,
        zero_division=0
    )
)

cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(7, 5))

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=["Not Booked", "Booked"],
    yticklabels=["Not Booked", "Booked"]
)

plt.title("Random Forest Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")

plt.tight_layout()
plt.savefig("output/confusion_matrix.png", dpi=200)
plt.show()



print("\nCALCULATING FEATURE IMPORTANCE...")


perm = permutation_importance(
    model,
    X_test,
    y_test,
    scoring="average_precision",
    n_repeats=5,
    random_state=42,
    n_jobs=-1
)

importance = pd.Series(
    perm.importances_mean,
    index=X_test.columns
).sort_values(ascending=False)

print("\nFEATURE IMPORTANCE RANKING")
print(importance)

# Save feature importance

importance.to_csv(
    "output/feature_importance.csv",
    header=["importance"]
)

# Plot top 10 features

plt.figure(figsize=(10, 7))

importance.head(10).sort_values().plot(
    kind="barh"
)

plt.title("Top 10 Features Influencing Customer Booking")
plt.xlabel("Mean Decrease in Average Precision")

plt.tight_layout()

plt.savefig(
    "output/feature_importance.png",
    dpi=300
)

plt.show()


predictions = pd.DataFrame({
    "Actual_Booking": y_test.values,
    "Predicted_Booking": y_pred,
    "Booking_Probability": y_probability
})

predictions.to_csv(
    "output/booking_predictions.csv",
    index=False
)



cv_table = pd.DataFrame({
    "Metric": list(scoring.keys()),
    "Mean_Score": [
        cv_results["test_" + m].mean()
        for m in scoring
    ],
    "Standard_Deviation": [
        cv_results["test_" + m].std()
        for m in scoring
    ]
})

cv_table.to_csv(
    "output/cross_validation_results.csv",
    index=False
)

joblib.dump(
    model,
    "output/british_airways_booking_model.pkl"
)

print("\nMODEL SAVED SUCCESSFULLY")


from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.dml.color import RGBColor

prs = Presentation()



slide = prs.slides.add_slide(
    prs.slide_layouts[0]
)

slide.shapes.title.text = "CUSTOMER BOOKING PREDICTION"

slide.placeholders[1].text = (
    "British Airways | Data Science Project\n"
    "Forage Task 2 | November 2026"
)


slide = prs.slides.add_slide(
    prs.slide_layouts[6]
)

title_box = slide.shapes.add_textbox(
    Inches(0.5), Inches(0.2),
    Inches(12), Inches(0.7)
)

title_box.text_frame.text = "MODEL RESULTS & EVALUATION"

title_box.text_frame.paragraphs[0].font.size = Pt(25)
title_box.text_frame.paragraphs[0].font.bold = True


summary = slide.shapes.add_textbox(
    Inches(0.5), Inches(1.2),
    Inches(5.5), Inches(5.5)
)

tf = summary.text_frame
tf.word_wrap = True

summary_points = [
    f"Model: Random Forest Classifier",
    f"5-Fold CV ROC-AUC: {cv_summary['roc_auc']:.3f}",
    f"5-Fold CV PR-AUC: {cv_summary['pr_auc']:.3f}",
    f"Test Accuracy: {accuracy:.3f}",
    f"Test Precision: {precision:.3f}",
    f"Test Recall: {recall:.3f}",
    f"Test F1 Score: {f1:.3f}",
    f"Booking rate: {y.mean():.1%}",
    f"Top predictor: {importance.index[0]}",
    "Recommendation: Test threshold tuning and validate on future data."
]

for i, point in enumerate(summary_points):

    if i == 0:
        p = tf.paragraphs[0]
    else:
        p = tf.add_paragraph()

    p.text = "• " + point
    p.font.size = Pt(16)
    p.space_after = Pt(10)


slide.shapes.add_picture(
    "output/feature_importance.png",
    Inches(6.3),
    Inches(1.2),
    width=Inches(6.4)
)


prs.save(
    "output/British_Airways_Booking_Results.pptx"
)



print("\n" + "=" * 55)
print("BRITISH AIRWAYS PROJECT COMPLETED SUCCESSFULLY")
print("=" * 55)

print("\nGENERATED FILES:")

for file in os.listdir("output"):
    print(file)

print("\nTHANK YOU!")


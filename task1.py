# ==========================================================
# Task 1: Data Cleaning and Preprocessing
# AI & ML Internship - Elevate Labs
# ==========================================================

# Import Required Libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler

print("=" * 60)
print("TASK 1 : DATA CLEANING & PREPROCESSING")
print("=" * 60)

# ==========================================================
# Step 1: Load Titanic Dataset
# ==========================================================

df = pd.read_csv("dataset/Titanic-Dataset.csv")

print("\nFirst 5 Rows:")
print(df.head())

print("\nDataset Shape:")
print(df.shape)

print("\nDataset Information:")
print(df.info())

# ==========================================================
# Step 2: Check Missing Values
# ==========================================================

print("\nMissing Values Before Cleaning:")
print(df.isnull().sum())

# Visualize Missing Values
missing_values = df.isnull().sum()

plt.figure(figsize=(8, 5))
sns.barplot(
    x=missing_values.index,
    y=missing_values.values
)

plt.xticks(rotation=90)
plt.title("Missing Values in Dataset")
plt.tight_layout()

plt.savefig("Screenshots/missing_values.png")
plt.close()

# ==========================================================
# Step 3: Handle Missing Values
# ==========================================================

# Fill missing Age values using Median
df["Age"] = df["Age"].fillna(
    df["Age"].median()
)

# Fill missing Embarked values using Mode
df["Embarked"] = df["Embarked"].fillna(
    df["Embarked"].mode()[0]
)

# Drop Cabin column because it contains
# too many missing values
if "Cabin" in df.columns:
    df.drop("Cabin", axis=1, inplace=True)

print("\nMissing Values After Cleaning:")
print(df.isnull().sum())

# ==========================================================
# Step 4: Encode Categorical Features
# ==========================================================

encoder = LabelEncoder()

df["Sex"] = encoder.fit_transform(
    df["Sex"]
)

df["Embarked"] = encoder.fit_transform(
    df["Embarked"]
)

# ==========================================================
# Step 5: Standardize Numerical Features
# ==========================================================

scaler = StandardScaler()

df[["Age", "Fare"]] = scaler.fit_transform(
    df[["Age", "Fare"]]
)

# ==========================================================
# Step 6: Detect Outliers Using Boxplot
# ==========================================================

plt.figure(figsize=(8, 5))

sns.boxplot(x=df["Fare"])

plt.title("Boxplot Before Outlier Removal")

plt.savefig(
    "Screenshots/boxplot_before.png"
)

plt.close()

# ==========================================================
# Step 7: Remove Outliers Using IQR Method
# ==========================================================

Q1 = df["Fare"].quantile(0.25)
Q3 = df["Fare"].quantile(0.75)

IQR = Q3 - Q1

lower_bound = Q1 - (1.5 * IQR)
upper_bound = Q3 + (1.5 * IQR)

df_cleaned = df[
    (df["Fare"] >= lower_bound)
    &
    (df["Fare"] <= upper_bound)
]

# ==========================================================
# Step 8: Visualize Data After Outlier Removal
# ==========================================================
 
plt.figure(figsize=(8, 5))

sns.boxplot(x=df_cleaned["Fare"])

plt.title("Boxplot After Outlier Removal")

plt.savefig(
    "Screenshots/boxplot_after.png"
)

plt.close()

# ==========================================================
# Step 9: Save Cleaned Dataset
# ==========================================================

df_cleaned.to_csv(
    "dataset/Cleaned_titanic_Dataset.csv",
    index=False
)

# ==========================================================
# Final Results
# ==========================================================

print("\nOriginal Dataset Shape :", df.shape)
print("Cleaned Dataset Shape :", df_cleaned.shape)

print("\nCleaned Dataset Saved Successfully")
print("Location : dataset/Cleaned_titanic_Dataset.csv")

print("\n✅Task 1 Completed Successfully!")
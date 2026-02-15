import pandas as pd

# Load dataset
df = pd.read_csv("breast-cancer-wisconsin.data")

print("\n===== BASIC INFO =====")
print("Shape:", df.shape)
print("\nColumns:")
print(df.columns.tolist())

print("\n===== FIRST 5 ROWS =====")
print(df.head())

print("\n===== DATA TYPES =====")
print(df.dtypes)

# Check for ID-like column
print("\n===== POSSIBLE ID COLUMN =====")
for col in df.columns:
    if df[col].is_unique and df[col].dtype != object:
        print(f"Column '{col}' might be an ID column")

# Check class label distribution
# Usually the last column for this dataset
class_col = df.columns[-1]

print("\n===== CLASS LABEL CHECK =====")
print(f"Class column: {class_col}")
print("Unique class labels:", df[class_col].unique())
print("\nClass counts:")
print(df[class_col].value_counts())

# Check for missing values
print("\n===== MISSING VALUE CHECK =====")
print(df.isnull().sum())

# Check for '?' values (common in Bare Nuclei)
print("\n===== '?' STRING CHECK =====")
for col in df.columns:
    if df[col].dtype == object:
        count_q = (df[col] == '?').sum()
        if count_q > 0:
            print(f"Column '{col}' has {count_q} '?' values")

# Basic statistics (numeric only)
print("\n===== BASIC STATISTICS =====")
print(df.describe())

# Verify class labels explicitly
print("\n===== CLASS LABEL INTERPRETATION =====")
print("Expected:")
print("2 = Benign")
print("4 = Malignant")

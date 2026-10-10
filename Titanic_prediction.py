
import os
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

# Load dataset
train = pd.read_csv("train.csv")

print("Dataset loaded successfully!")
print(train.head())
print("\nDataset shape:", train.shape)

# Select features and target
features = ["Pclass", "Sex", "Age", "SibSp",
            "Parch", "Fare", "Embarked"]

X = train[features]
y = train["Survived"]

numeric_features = ["Pclass", "Age", "SibSp", "Parch", "Fare"]
categorical_features = ["Sex", "Embarked"]

# Handle missing values and encode text
numeric_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median"))
])

categorical_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer([
    ("num", numeric_pipeline, numeric_features),
    ("cat", categorical_pipeline, categorical_features)
])

# Create model
model = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", RandomForestClassifier(
        n_estimators=100, random_state=42
    ))
])

# Split data
X_train, X_valid, y_train, y_valid = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Train and evaluate
model.fit(X_train, y_train)
predictions = model.predict(X_valid)

print("\nModel Accuracy:",
      round(accuracy_score(y_valid, predictions) * 100, 2), "%")

print("\nClassification Report:")
print(classification_report(y_valid, predictions))

# Train final model on all training data
model.fit(X, y)

# Predict test dataset
if os.path.exists("test.csv"):
    test = pd.read_csv("test.csv")
    test_predictions = model.predict(test[features])

    result = pd.DataFrame({
        "PassengerId": test["PassengerId"],
        "Survived": test_predictions
    })

    result.to_csv("submission.csv", index=False)
    print("\nPredictions saved to submission.csv")
    print(result.head())

print("\nTask 1 completed successfully!")

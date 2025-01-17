import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt

# Load the dataset
data_file_path = "Datathon_data-2025-Raw.xlsx"
data = pd.ExcelFile(data_file_path)
data_main = data.parse("Data")

# Replace missing values represented as ".." with NaN
data_main_cleaned = data_main.replace("..", pd.NA)

# Define target variable
target = "Logistics performance index: Overall (1=low to 5=high) [LP.LPI.OVRL.XQ]"

# Drop rows where target is NaN
data_main_cleaned = data_main_cleaned.dropna(subset=[target])

# Identify numerical and categorical columns
numerical_cols = data_main_cleaned.select_dtypes(include=["float64", "int64"]).columns
categorical_cols = data_main_cleaned.select_dtypes(include=["object"]).columns

# Exclude metadata columns
categorical_cols = [col for col in categorical_cols if col not in ["Country Name", "Country Code"]]
numerical_cols = [col for col in numerical_cols if col != target]

# Preprocess categorical columns: Fill missing values and convert to string
data_main_cleaned[categorical_cols] = data_main_cleaned[categorical_cols].fillna("missing").astype(str)

# Define preprocessing for numerical and categorical columns
numerical_transformer = SimpleImputer(strategy="mean")
categorical_transformer = OneHotEncoder(handle_unknown="ignore", sparse_output=False)

# Combine preprocessing in a ColumnTransformer
preprocessor = ColumnTransformer(
    transformers=[
        ("num", numerical_transformer, numerical_cols),
        ("cat", categorical_transformer, categorical_cols)
    ]
)

# Define Random Forest Regressor pipeline
rf_pipeline = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("regressor", RandomForestRegressor(random_state=42))
])

# Define hyperparameter grid for Random Forest
param_grid_rf = {
    "regressor__n_estimators": [100, 200, 300],
    "regressor__max_depth": [10, 20, 30],
    "regressor__min_samples_split": [2, 5, 10]
}

# Convert target variable to numeric
y_cleaned = pd.to_numeric(data_main_cleaned[target], errors="coerce")

# Drop rows with NaN in target
valid_indices = ~y_cleaned.isna()
X_cleaned = data_main_cleaned.loc[valid_indices]
y_cleaned = y_cleaned.loc[valid_indices]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_cleaned, y_cleaned, test_size=0.2, random_state=42)

# Use GridSearchCV to find the best hyperparameters
grid_search_rf = GridSearchCV(estimator=rf_pipeline, param_grid=param_grid_rf, cv=3, scoring="r2", verbose=1, n_jobs=-1)
grid_search_rf.fit(X_train, y_train)

# Get the best model
best_rf_model = grid_search_rf.best_estimator_

# Make predictions
y_pred_rf = best_rf_model.predict(X_test)

# Evaluate the model
mse_rf = mean_squared_error(y_test, y_pred_rf)
mae_rf = mean_absolute_error(y_test, y_pred_rf)
r2_rf = r2_score(y_test, y_pred_rf)

# Print performance metrics
print("Random Forest Model Performance:")
print(f"Mean Squared Error (MSE): {mse_rf}")
print(f"Mean Absolute Error (MAE): {mae_rf}")
print(f"R² Score: {r2_rf}")

# Extract feature importances
preprocessor = best_rf_model.named_steps["preprocessor"]
regressor = best_rf_model.named_steps["regressor"]

# Get feature names
encoded_cat_features = preprocessor.named_transformers_["cat"].get_feature_names_out(categorical_cols)
all_feature_names = list(numerical_cols) + list(encoded_cat_features)

# Map feature importances
feature_importances = regressor.feature_importances_
feature_importance_df = pd.DataFrame({
    "Feature": all_feature_names,
    "Importance": feature_importances
}).sort_values(by="Importance", ascending=False)

# Display top 10 features
top_features = feature_importance_df.head(10)
print("\nTop 10 Most Important Features:")
print(top_features)

# Plot feature importance
plt.figure(figsize=(10, 6))
plt.barh(top_features["Feature"], top_features["Importance"], color="skyblue", edgecolor="black")
plt.xlabel("Feature Importance")
plt.ylabel("Features")
plt.title("Top 10 Most Important Features")
plt.gca().invert_yaxis()
plt.tight_layout()
plt.show()

import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
import joblib  # for saving model

# ----------------------
# Load data
# ----------------------
df = pd.read_json("./iot-dataset/iot-train-set.json", lines=True)
unused_features = ['timestamp', 'zone_id', "day_of_week", 'temperature_celsius', 'special_event_flag']
categorical_cols = ["time_of_day", "weather_condition"]
df.drop(unused_features, axis=1, inplace=True)

X = df.drop("lighting_action_class", axis=1)
y = df["lighting_action_class"]

# Identify numeric columns
numeric_cols = [col for col in X.columns if col not in categorical_cols]

# ----------------------
# Preprocessing: one-hot encode categoricals
# ----------------------
preprocessor = ColumnTransformer(
    transformers=[
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols),
        ("num", "passthrough", numeric_cols)
    ]
)

# ----------------------
# Gradient Boosting model
# ----------------------
gb_model = GradientBoostingClassifier(random_state=42)

# Create pipeline
pipeline = Pipeline(steps=[("preprocessor", preprocessor),
                           ("classifier", gb_model)])

# ----------------------
# Train model
# ----------------------
pipeline.fit(X, y)

# ----------------------
# Save trained model
# ----------------------
joblib.dump(pipeline, "./gradient_boosting_iot_model.pkl")
print("Model saved to gradient_boosting_iot_model.pkl")

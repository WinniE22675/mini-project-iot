from river import preprocessing
import pandas as pd
from river import compose

# ----------------------
# Load data
# ----------------------
df = pd.read_csv("./iot-dataset/iot-train-set.csv", sep=";")

# Drop unused features
unused_features = ['timestamp', 'zone_id', "day_of_week", 'temperature_celsius', 'special_event_flag']
df.drop(unused_features, axis=1, inplace=True)

# Define features and target
X = df.drop("lighting_action_class", axis=1)
y = df["lighting_action_class"]

categorical_cols = ["time_of_day", "weather_condition"]
numeric_cols = [col for col in X.columns if col not in categorical_cols]
# ----------------------
# Preprocessing branches
# ----------------------
categorical_branch = compose.Select(*categorical_cols) | preprocessing.OneHotEncoder()
numeric_branch = compose.Select(*numeric_cols)  # passthrough numeric features

# Merge branches and send to ARFClassifier
preprocessor = compose.TransformerUnion(categorical_branch, numeric_branch)

model = compose.Pipeline(
    preprocessor,
)

for x in X[:2].to_dict(orient="records"):
    model.learn_one(x)
    print(model.transform_one(x))
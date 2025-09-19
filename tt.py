import pandas as pd

df = pd.read_csv("./iot-dataset/iot-train-set.csv", sep=";")
unused_features = ['timestamp', 'zone_id', "day_of_week", 'temperature_celsius', 'special_event_flag']
df.drop(unused_features, axis=1, inplace=True)

X = df.drop("lighting_action_class", axis=1)
y = df["lighting_action_class"]

categorical_cols = ["time_of_day", "weather_condition"]
numeric_cols = [col for col in X.columns if col not in categorical_cols]

print(categorical_cols)
print(numeric_cols)
import joblib
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix, ConfusionMatrixDisplay


# ----------------------
# Load data
# ----------------------
df = pd.read_json("./iot-dataset/iot-train-set.json", lines=True)
unused_features = ['timestamp', 'zone_id', "day_of_week", 'temperature_celsius', 'special_event_flag']
df.drop(unused_features, axis=1, inplace=True)

X_test = df.drop("lighting_action_class", axis=1)
y_test = df["lighting_action_class"]

loaded_model = joblib.load("./gradient_boosting_iot_model.pkl")
y_pred = loaded_model.predict(X_test)  # use on new data


# ----------------------
# Evaluate
# ----------------------
print("Accuracy:", accuracy_score(y_test, y_pred))
print("F1 Macro:", f1_score(y_test, y_pred, average="macro"))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# Confusion matrix
cm = confusion_matrix(y_test, y_pred, labels=y_test.unique())
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=y_test.unique())
disp.plot(cmap=plt.cm.Blues, values_format='d')
plt.title("Gradient Boosting - Confusion Matrix")
plt.show()
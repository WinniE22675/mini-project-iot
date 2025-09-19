import pandas as pd

df = pd.read_csv("./iot-dataset/iot-test-set.csv", sep=";")

df.to_json("./iot-dataset/iot-test-set.json", orient='records', lines=True)

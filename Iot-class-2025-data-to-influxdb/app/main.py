from quixstreams import Application
from influxdb_client import InfluxDBClient, Point, WriteOptions
from datetime import datetime, timezone
import os
import json
from datetime import datetime
import logging

# Load environment variables (useful when working locally)
from dotenv import load_dotenv
# load_dotenv(os.path.dirname(os.path.abspath(__file__))+"/.env")
load_dotenv(".env")

# Logggin env
log_level_str = os.getenv("LOG_LEVEL", "INFO").upper()
log_level = getattr(logging, log_level_str, logging.INFO)

logging.basicConfig(
    level=log_level,
    format='[%(asctime)s] [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

# Config 
KAFKA_BROKER = os.getenv("KAFKA_BROKER", "localhost:9092")
KAFKA_INPUT_TOPIC = os.getenv("KAFKA_INPUT_TOPIC", "iot-frames-model")

# --- InfluxDB Setup ---
INFLUX_URL = os.getenv("INFLUX_URL", "http://localhost:8086")
INFLUX_TOKEN = os.getenv("INFLUX_TOKEN", "your_token")
INFLUX_ORG = os.getenv("INFLUX_ORG", "your_org")
INFLUX_BUCKET = os.getenv("INFLUX_BUCKET", "iot_data")

influx_client = InfluxDBClient(url=INFLUX_URL, token=INFLUX_TOKEN, org=INFLUX_ORG)
influx_write = influx_client.write_api(write_options=WriteOptions(batch_size=1, flush_interval=1))

# --- Quix Setup ---
# Config
KAFKA_BROKER = os.getenv("KAFKA_BROKER", "localhost:9092")
KAFKA_INPUT_TOPIC = os.getenv("KAFKA_INPUT_TOPIC", "event-frames-model")

app = Application(broker_address=KAFKA_BROKER,
                loglevel="INFO",
                auto_offset_reset="earliest",
                state_dir=os.path.dirname(os.path.abspath(__file__))+"/state/",
                consumer_group="model-influxdb"
      )
input_topic = app.topic(KAFKA_INPUT_TOPIC, value_deserializer="json")


def process_event(data):
    try:
        payload = data.get("payload", {})

        # --- Handle timestamp ---
        ts_raw = payload.get("timestamp", None)
        if ts_raw is not None:
            try:
                # ถ้าเป็น epoch ms
                timestamp = datetime.fromtimestamp(float(ts_raw) / 1000.0, tz=timezone.utc)
            except:
                # ถ้าเป็น string เช่น "2024-09-12 08:00:00"
                timestamp = datetime.strptime(ts_raw, "%Y-%m-%d %H:%M:%S").replace(tzinfo=timezone.utc)
        else:
            timestamp = datetime.utcnow()

        # --- InfluxDB Point ---
        point = (
            Point(KAFKA_INPUT_TOPIC)
            .tag("sensor_id", data.get("id", "unknown"))
            .tag("place_id", data.get("place_id", "unknown"))
            .tag("name", data.get("name", "unknown"))
            # --- Fields จาก dataset ---
            .field("zone_id", str(payload.get("zone_id", "unknown")))
            .field("ambient_light_lux", float(payload.get("ambient_light_lux", 0)))
            .field("motion_detected", int(payload.get("motion_detected", 0)))
            .field("temperature_celsius", float(payload.get("temperature_celsius", 0)))
            .field("occupancy_count", int(payload.get("occupancy_count", 0)))
            .field("day_of_week", str(payload.get("day_of_week", "unknown")))
            .field("time_of_day", str(payload.get("time_of_day", "unknown")))
            .field("weather_condition", str(payload.get("weather_condition", "unknown")))
            .field("special_event_flag", str(payload.get("special_event_flag", "0")))
            .field("energy_price_per_kwh", float(payload.get("energy_price_per_kwh", 0)))
            .field("prev_hour_energy_usage_kwh", float(payload.get("prev_hour_energy_usage_kwh", 0)))
            .field("traffic_density", float(payload.get("traffic_density", 0)))
            .field("avg_pedestrian_speed", float(payload.get("avg_pedestrian_speed", 0)))
            .field("adjusted_light_intensity", float(payload.get("adjusted_light_intensity", 0)))
            .field("energy_consumption_kwh", float(payload.get("energy_consumption_kwh", 0)))
            .field("lighting_action_class", int(payload.get("lighting_action_class", "unknown")))
            .time(timestamp)
        )

        influx_write.write(bucket=INFLUX_BUCKET, org=INFLUX_ORG, record=point)
        logging.info(f"[✓] Wrote to InfluxDB: {point.to_line_protocol()}")

    except Exception as e:
        logging.error(f"❌ Error processing message: {e}")




# Stream
sdf = app.dataframe(input_topic)
sdf = sdf.apply(process_event)

logging.info(f"Connecting to ...{KAFKA_BROKER}")
logging.info(f"🚀 Listening to Kafka topic: {KAFKA_INPUT_TOPIC}")
app.run()

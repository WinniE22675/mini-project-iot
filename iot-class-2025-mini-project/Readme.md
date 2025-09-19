# IoT Data Platform Stack

## 📌 Overview
This project provides a lightweight IoT data platform using Docker Compose.  
It is designed to collect, process, and visualize data from IoT devices in real time.

### Components
- **VerneMQ** → MQTT Broker for IoT device connectivity  
- **Kafka (KRaft mode)** → Event streaming platform for reliable, scalable data pipelines  
- **Kafka-UI** → Web interface for managing Kafka topics, messages, and consumers  
- **InfluxDB 2.7** → Time-series database for storing sensor data  
- **Prometheus** → Metrics collector for monitoring the stack  
- **Grafana** → Dashboard visualization and alerting  

---

## ⚡ Services
| Service     | Host Port | Description |
|-------------|-----------|-------------|
| VerneMQ     | `1883`    | MQTT Broker |
| VerneMQ API | `8888`    | HTTP API |
| Kafka       | `9092`    | Kafka external listener |
| Kafka       | `29092`   | Kafka internal listener (Docker network) |
| Kafka-UI    | `8080`    | Kafka management UI |
| InfluxDB    | `8086`    | Time-series DB |
| Prometheus  | `9090`    | Metrics server |
| Grafana     | `3000`    | Visualization dashboards |

---

## 🚀 Quick Start

### 1. Clone the repository
```bash
git clone https://github.com/hanattaw/iot-class-2025-mini-project
cd iot-class-2025-mini-project

# Smart Electricity Grid Anomaly Detection System

> **Real-time anomaly detection for residential smart meter telemetry using LSTM autoencoders, edge inference, and MQTT streaming.**

---

## Architecture

```
┌─────────────────┐     MQTT (QoS 1)     ┌──────────────────┐     InfluxDB Write
│                 │ ──────────────────►  │                  │ ──────────────────►  InfluxDB
│   Simulator     │   grid/meter/#       │  Edge Processor  │
│   (50 meters)   │                      │  (TFLite LSTM)   │     Webhook
│                 │                      │                  │ ──────────────────►  Alerts
└─────────────────┘                      └──────────────────┘
        │                                         │
        ▼                                         ▼
   Mosquitto Broker                          Grafana Dashboards
   (port 1883)                               (port 3000)
```

**Data flow:** Simulated meters → MQTT → Edge anomaly detection → InfluxDB → Grafana

---

## Tech Stack

| Component        | Technology                     | Purpose                              |
|------------------|--------------------------------|--------------------------------------|
| Simulation       | Python 3.11, NumPy, Pandas     | Residential meter fleet simulation   |
| Transport        | Eclipse Mosquitto 2.0 (MQTT)   | Telemetry message broker             |
| Edge Inference   | TensorFlow Lite                | LSTM autoencoder anomaly detection   |
| Time-Series DB   | InfluxDB 2.7                   | Meter readings + anomaly storage     |
| Visualization    | Grafana 10.4                   | Real-time dashboards and alerting    |
| Orchestration    | Docker Compose 3.9             | Full-stack container management      |

---

## Quick Start

### Option 1: Docker (Recommended)

```bash
# 1. Clone and configure
cp .env.example .env     # Edit .env with your secrets

# 2. Launch the full stack
docker-compose up -d

# 3. Verify services
docker-compose ps

# 4. Monitor logs
docker-compose logs -f simulator
```

**Access points:**
- Grafana Dashboard: [http://localhost:3000](http://localhost:3000) (admin/admin)
- InfluxDB UI: [http://localhost:8086](http://localhost:8086)
- MQTT Broker: `localhost:1883`

### Option 2: Local Python Development

```bash
# 1. Create virtual environment
python -m venv .venv

# Windows
.venv\Scripts\activate

# Linux/macOS
source .venv/bin/activate

# 2. Install dependencies
pip install -r simulator/requirements.txt

# 3. Start Mosquitto locally (separate terminal)
mosquitto -c mosquitto/mosquitto.conf

# 4. Run the simulator
python -m simulator.cli_entrypoint --mode live

# 5. Subscribe to readings (another terminal)
mosquitto_sub -t "grid/meter/#"
```

---

## CLI Commands

```bash
# Live MQTT streaming (50 meters, 5s interval)
python -m simulator.cli_entrypoint --mode live

# Custom fleet size and interval
python -m simulator.cli_entrypoint --mode live --meters 10 --interval 2

# Fast-forward dataset generation (7 days → CSV)
python -m simulator.cli_entrypoint --mode fastforward --days 7

# Inject anomaly on a specific meter
python -m simulator.cli_entrypoint --mode live --inject-anomaly M_005:extreme_spike:15

# Fast-forward with random anomaly injection
python -m simulator.cli_entrypoint --mode fastforward --days 30 --random-anomalies
```

---

## Project Structure

```
smart-grid-anomaly/
│
├── config/                         # Centralized configuration
│   ├── simulation_config.py        #   Fleet sizing, timing, load profile ranges
│   ├── mqtt_settings.py            #   Broker connection, topics, QoS
│   └── inference_config.py         #   Model inference thresholds
│
├── simulator/                      # Meter simulation + MQTT publishing
│   ├── residential_meter_model.py  #   Single-meter diurnal load model
│   ├── anomaly_injection_engine.py #   6 anomaly types with lifecycle mgmt
│   ├── mqtt_stream_publisher.py    #   Paho MQTT client with reconnection
│   ├── meter_fleet_orchestrator.py #   Fleet coordination (live + fast-forward)
│   ├── cli_entrypoint.py           #   Argparse CLI interface
│   ├── requirements.txt            #   Module-specific dependencies
│   └── Dockerfile                  #   Production container image
│
├── edge/                           # Real-time stream processing
│   ├── edge_stream_processor.py    #   MQTT subscriber + processing loop
│   ├── anomaly_rule_classifier.py  #   Rule-based anomaly classification
│   ├── influxdb_write_client.py    #   InfluxDB v2 write operations
│   ├── anomaly_alert_dispatcher.py #   Webhook-based alert dispatch
│   ├── requirements.txt            #   Module-specific dependencies
│   └── Dockerfile                  #   Production container image
│
├── model/                          # ML training pipeline
│   ├── training/
│   │   ├── lstm_autoencoder.py     #   LSTM autoencoder architecture
│   │   └── training_pipeline.py    #   End-to-end training workflow
│   ├── inference/
│   │   └── tflite_inference_wrapper.py  # TFLite inference API
│   ├── artifacts/                  #   Trained models (.h5, .tflite)
│   └── requirements.txt            #   Module-specific dependencies
│
├── data/                           # Datasets (gitignored)
│   ├── raw/                        #   Simulated readings CSV
│   ├── processed/                  #   Feature-engineered training data
│   └── evaluation/                 #   Evaluation sets and anomaly logs
│
├── utils/                          # Shared utilities
│   ├── logging_setup.py            #   Structured logging configuration
│   ├── time_utils.py               #   UTC timestamp utilities
│   └── data_transformations.py     #   Data preprocessing helpers
│
├── mosquitto/                      # MQTT broker configuration
│   └── mosquitto.conf              #   Listener, auth, persistence settings
│
├── grafana/                        # Dashboard provisioning
│   ├── provisioning/               #   Datasources + dashboard configs
│   └── dashboards/                 #   JSON dashboard definitions
│
├── docker-compose.yml              # Full-stack orchestration
├── Makefile                        # Development convenience targets
├── .env                            # Local secrets (gitignored)
├── .env.example                    # Safe env template (committed)
├── .gitignore                      # VCS exclusion rules
└── README.md                       # This file
```

---

## MQTT Payload Schema

Every meter reading published to `grid/meter/{meter_id}/reading`:

```json
{
  "meter_id": "M_023",
  "timestamp": "2026-03-28T14:32:10Z",
  "kwh": 3.47,
  "voltage": 229.8,
  "current": 15.12,
  "seq_no": 98432
}
```

---

## Supported Anomaly Types

| Type                  | Effect             | Real-World Scenario                |
|-----------------------|--------------------|------------------------------------|
| `energy_theft_drop`   | kWh × 0.10         | Tampered meter / bypass wiring     |
| `extreme_spike`       | kWh × 4.5          | Faulty CT clamp / firmware bug     |
| `flatline_failure`    | Constant values     | Frozen hardware / stuck register   |
| `sustained_overload`  | kWh × 2.5          | Illegal load / unauthorized EV     |
| `nocturnal_spike`     | kWh × 3.5 (1–5 AM) | Crypto mining / grow operation     |
| `voltage_instability` | Voltage σ = 15V     | Transformer fault / loose neutral  |

---

## License

Internal project — all rights reserved.

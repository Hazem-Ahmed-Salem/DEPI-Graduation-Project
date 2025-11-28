# monitoring_module.py
import logging
import time
import csv
import os

# ---------------- Monitoring Setup ----------------
logging.basicConfig(
    filename="Monitoring.log",
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# CSV file setup
CSV_FILE = "Monitoring Data.csv"

# Create CSV file with header if it doesn't exist
if not os.path.exists(CSV_FILE):
    with open(CSV_FILE, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "timestamp", "model", "side", "gesture",
            "confidence_percent", "frame_time_ms", "inference_ms"
        ])

MIN_CONFIDENCE = 0.70
MAX_LOW_CONFIDENCE_COUNT = 20
low_conf_count = 0
last_timestamp = time.time()

def send_alert(message):
    print("ALERT:", message)
    logging.warning(message)

def check_model_performance(conf_ratio, gesture, side, model_name, inference_time_ms):
    global low_conf_count, last_timestamp

    current_time = time.time()
    frame_interval = current_time - last_timestamp
    last_timestamp = current_time

    confidence_percent = conf_ratio * 100.0

    # ---------- Write to LOG file ----------
    log_msg = (
        f"MODEL={model_name} | SIDE={side} | GESTURE={gesture} | "
        f"CONFIDENCE={confidence_percent:.1f}% | TIME={frame_interval*1000:.1f}ms | "
        f"INFERENCE={inference_time_ms:.1f}ms"
    )
    logging.info(log_msg)

    # ---------- Write to CSV file ----------
    with open(CSV_FILE, mode="a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            time.strftime("%Y-%m-%d %H:%M:%S"),
            model_name,
            side,
            gesture,
            round(confidence_percent, 2),
            round(frame_interval * 1000, 2),
            round(inference_time_ms, 2)
        ])

    # ---------- Trigger alert logic ----------
    if conf_ratio < MIN_CONFIDENCE:
        low_conf_count += 1
        if low_conf_count >= MAX_LOW_CONFIDENCE_COUNT:
            send_alert(
                f"Drift detected: Low confidence reached {MAX_LOW_CONFIDENCE_COUNT}. "
                f"Last gesture={gesture}, conf={confidence_percent:.1f}%"
            )
            low_conf_count = 0
    else:
        low_conf_count = 0

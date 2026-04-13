import requests
import json
import time

BASE_URL = "http://localhost:8000/api/v1"

def test_ingest():
    # Example payload from ESP32
    payload = {
        "device_id": "COW-101",
        "schema_version": "1.0",
        "imu": {
            "accel": [0.05, 0.1, 9.8],
            "gyro": [0.0, 0.0, 0.0]
        },
        "temperature": 40.2, # Fever logic should trigger
        "rumination_score": 40
    }

    try:
        response = requests.post(f"{BASE_URL}/ingest", json=payload)
        print("Status Code:", response.status_code)
        print("Response:", json.dumps(response.json(), indent=2))
    except Exception as e:
        print("Error connecting to server. Is it running?")
        print(e)

if __name__ == "__main__":
    # In a real scenario, you'd run this multiple times to build a window
    print("Testing Ingestion Hook...")
    test_ingest()

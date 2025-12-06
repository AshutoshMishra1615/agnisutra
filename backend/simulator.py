import time
import requests
import random

# Configuration
API_URL = "http://127.0.0.1:8000/iot/update"
FARM_ID = 1

def simulate_sensor():
    print(f"Starting sensor simulation...")
    print(f"Sending data to {API_URL}")
    
    while True:
        # Simulate sensor data
        moisture = round(random.uniform(20.0, 60.0), 2)
        nitrogen = round(random.uniform(10.0, 100.0), 2)
        phosphorus = round(random.uniform(10.0, 100.0), 2)
        potassium = round(random.uniform(10.0, 100.0), 2)
        
        payload = {
            "moisture": moisture,
            "nitrogen": nitrogen,
            "phosphorus": phosphorus,
            "potassium": potassium
        }
        
        try:
            response = requests.post(API_URL, json=payload)
            if response.status_code == 200:
                print(f"Sent: {payload} | Response: {response.json()}")
            else:
                print(f"Failed: {response.status_code} | {response.text}")
        except Exception as e:
            print(f"Error: {e}")
            
        # Wait for 2 seconds
        time.sleep(2)

if __name__ == "__main__":
    simulate_sensor()

import requests
import time
from datetime import datetime, timedelta
import json

AGRO_API_KEY = "5e2ed96e32afbcac715fccb11814026b"

def fetch_ndvi_debug(lat, lon):
    # 1. Create Polygon (Mocking the ID we found earlier for Bhubaneswar to save time/calls)
    # Found existing polygon ID: 6936aba53502732b3242f5c7
    poly_id = "6936aba53502732b3242f5c7" 
    
    print(f"Using Polygon ID: {poly_id}")
    
    if poly_id:
        end_date = int(time.time())
        start_date = int((datetime.now() - timedelta(days=60)).timestamp())
        
        # --- Fetch Stats (Value) ---
        history_url = f"http://api.agromonitoring.com/agro/1.0/ndvi/history?start={start_date}&end={end_date}&polyid={poly_id}&appid={AGRO_API_KEY}"
        
        # --- Fetch Image (Visual) ---
        image_url_api = f"http://api.agromonitoring.com/image/1.0/search?start={start_date}&end={end_date}&polyid={poly_id}&appid={AGRO_API_KEY}"
        
        print(f"DEBUG: NDVI History URL: {history_url}")
        
        try:
            # A. Fetch History
            response = requests.get(history_url)
            print(f"DEBUG: NDVI History Response Status: {response.status_code}")
            
            if response.status_code == 400 and "end can not be after now" in response.text:
                print("DEBUG: System clock appears ahead. Retrying with -1 year shift.")
                end_date -= 31536000
                start_date -= 31536000
                history_url = f"http://api.agromonitoring.com/agro/1.0/ndvi/history?start={start_date}&end={end_date}&polyid={poly_id}&appid={AGRO_API_KEY}"
                image_url_api = f"http://api.agromonitoring.com/image/1.0/search?start={start_date}&end={end_date}&polyid={poly_id}&appid={AGRO_API_KEY}"
                
                print(f"DEBUG: Retry NDVI History URL: {history_url}")
                response = requests.get(history_url)
                print(f"DEBUG: Retry Response Status: {response.status_code}")

            ndvi_val = None
            if response.status_code == 200:
                data = response.json()
                if isinstance(data, list) and len(data) > 0:
                    data.sort(key=lambda x: x.get('dt', 0), reverse=True)
                    for item in data:
                        if 'data' in item and 'mean' in item['data']:
                            ndvi_val = float(item['data']['mean'])
                            print(f"DEBUG: Found NDVI value: {ndvi_val}")
                            break
            
            # B. Fetch Image URL
            ndvi_image_url = None
            try:
                print(f"DEBUG: Fetching Image URL: {image_url_api}")
                img_response = requests.get(image_url_api)
                print(f"DEBUG: Image Response Status: {img_response.status_code}")
                if img_response.status_code != 200:
                    print(f"DEBUG: Image Response Text: {img_response.text}")
                
                if img_response.status_code == 200:
                    img_data = img_response.json()
                    print(f"DEBUG: Image Data Length: {len(img_data)}")
                    if isinstance(img_data, list) and len(img_data) > 0:
                        img_data.sort(key=lambda x: x.get('dt', 0), reverse=True)
                        for img in img_data:
                            # print(f"DEBUG: Checking image: {img.get('image', {}).keys()}")
                            if 'image' in img and 'ndvi' in img['image']:
                                ndvi_image_url = img['image']['ndvi']
                                print(f"DEBUG: Found NDVI Image URL: {ndvi_image_url}")
                                break
            except Exception as e:
                print(f"DEBUG: Error fetching image: {e}")

            return {
                "ndvi": ndvi_val, 
                "ndvi_image": ndvi_image_url,
                "source": "satellite_realtime"
            }

        except Exception as e:
            print(f"Error fetching NDVI: {e}")
            pass
            
    return {"ndvi": 0.72, "ndvi_image": None, "source": "mock_fallback"}

if __name__ == "__main__":
    # Test with known good parameters from previous logs
    # Polygon: 6936a7cefd6b85402bb1c6c5
    # Time: Jan 2024 (1704067200 to 1706572800)
    print("Testing with known good parameters (Jan 2024)")
    
    poly_id = "6936a7cefd6b85402bb1c6c5"
    start_date = 1704067200
    end_date = 1706572800
    
    image_url_api = f"http://api.agromonitoring.com/image/1.0/search?start={start_date}&end={end_date}&polyid={poly_id}&appid={AGRO_API_KEY}"
    history_url = f"http://api.agromonitoring.com/agro/1.0/ndvi/history?start={start_date}&end={end_date}&polyid={poly_id}&appid={AGRO_API_KEY}"
    
    print(f"DEBUG: Fetching History URL: {history_url}")
    try:
        hist_response = requests.get(history_url)
        print(f"DEBUG: History Response Status: {hist_response.status_code}")
    except Exception as e:
        print(f"DEBUG: Error: {e}")

    print(f"DEBUG: Fetching Image URL: {image_url_api}")
    try:
        img_response = requests.get(image_url_api)
        print(f"DEBUG: Image Response Status: {img_response.status_code}")
        if img_response.status_code == 200:
            print(f"DEBUG: Image Data Length: {len(img_response.json())}")
        else:
            print(f"DEBUG: Image Response Text: {img_response.text}")
    except Exception as e:
        print(f"DEBUG: Error: {e}")

    # print("Testing with lat=20.296, lon=85.824 (Bhubaneswar)")
    # result = fetch_ndvi_debug(20.296, 85.824)
    # print(f"Result: {result}")

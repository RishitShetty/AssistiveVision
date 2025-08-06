# location_service.py
import requests
import json

class LocationService:
    @staticmethod
    def get_current_location():
        """Get current location based on IP"""
        try:
            response = requests.get("https://ipinfo.io/json")
            loc_data = response.json()
            location_str = f"{loc_data.get('city', 'Unknown')}, {loc_data.get('region', 'Unknown')}, {loc_data.get('country', 'Unknown')}"
            return {"status": "location_sent", "location": location_str}
        except Exception as e:
            return {"status": "error", "message": "Error fetching location."}

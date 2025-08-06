# services/face_rec.py
import requests
from functools import lru_cache
import logging
from config import Config

logger = logging.getLogger(__name__)


class LocationService:
    def __init__(self):
        self.api_key = Config.IPINFO_API_KEY

    @lru_cache(maxsize=1)  # Cache location for 5 minutes
    def get_location(self, retries=3):
        """Get current location with retry mechanism"""
        for attempt in range(retries):
            try:
                response = requests.get(
                    f"https://ipinfo.io/json?token={self.api_key}",
                    timeout=5
                )
                response.raise_for_status()
                data = response.json()

                return {
                    'city': data.get('city', 'Unknown'),
                    'region': data.get('region', 'Unknown'),
                    'country': data.get('country', 'Unknown'),
                    'loc': data.get('loc', '')
                }

            except requests.exceptions.RequestException as e:
                logger.error(f"Location service error (attempt {attempt + 1}/{retries}): {str(e)}")
                if attempt == retries - 1:
                    return None

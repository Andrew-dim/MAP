import requests
from requests.auth import HTTPBasicAuth
import os
from tqdm import tqdm
import time

class CopernicusDataSpace:
    """Handler for Copernicus Data Space Ecosystem API"""

    def __init__(self, username, password):
        self.username = username
        self.password = password
        self.base_url = "https://catalogue.dataspace.copernicus.eu/odata/v1"
        self.download_url = "https://zipper.dataspace.copernicus.eu/odata/v1"
        self.session = requests.Session()
        self.session.auth = HTTPBasicAuth(username, password)
        self.access_token = None

    def get_access_token(self):
        """Get access token for Copernicus Data Space"""
        token_url = "https://identity.dataspace.copernicus.eu/auth/realms/CDSE/protocol/openid-connect/token"

        data = {
            "grant_type": "password",
            "username": self.username,
            "password": self.password,
            "client_id": "cdse-public"
        }

        try:
            response = requests.post(token_url, data=data, timeout=30)
            response.raise_for_status()
            self.access_token = response.json()["access_token"]
            return True
        except Exception as e:
            print(f"Error getting access token: {e}")
            return False

    def search_products(self, polygon, date_from, date_to, max_cloud_cover=100, satellite='SENTINEL-2'):
        """Search for Sentinel products (S1 SAR or S2 optical)

        Args:
            polygon: Shapely polygon for AOI
            date_from: Start datetime
            date_to: End datetime
            max_cloud_cover: Maximum cloud cover % (only for Sentinel-2)
            satellite: 'SENTINEL-1' or 'SENTINEL-2' (default)

        Note: Sentinel-2 supports cloud cover filtering, Sentinel-1 does not (radar penetrates clouds)
        """

        if not self.access_token:
            if not self.get_access_token():
                raise Exception("Failed to get access token")

        wkt = polygon.wkt
        bounds = polygon.bounds
        print(f"  Polygon bounds: Lon({bounds[0]:.4f} to {bounds[2]:.4f}), "
              f"Lat({bounds[1]:.4f} to {bounds[3]:.4f})")

        date_from_str = date_from.strftime("%Y-%m-%dT%H:%M:%S.000Z")
        date_to_str = date_to.strftime("%Y-%m-%dT%H:%M:%S.999Z")

        if satellite == 'SENTINEL-1':
            # Sentinel-1 query - SIMPLIFIED (no product type filter for testing)
            # This will return ALL S1 products (GRD, SLC, RAW, etc.)
            query = (
                f"Collection/Name eq 'SENTINEL-1' and "
                f"OData.CSC.Intersects(area=geography'SRID=4326;{wkt}') and "
                f"ContentDate/Start gt {date_from_str} and "
                f"ContentDate/Start lt {date_to_str}"
            )
            print(f"  DEBUG: Simplified S1 query (all product types)")
        else:
            # Sentinel-2 query (with cloud cover)
            query = (
                f"Collection/Name eq 'SENTINEL-2' and "
                f"OData.CSC.Intersects(area=geography'SRID=4326;{wkt}') and "
                f"ContentDate/Start gt {date_from_str} and "
                f"ContentDate/Start lt {date_to_str} and "
                f"Attributes/OData.CSC.DoubleAttribute/any(att:att/Name eq 'cloudCover' and "
                f"att/OData.CSC.DoubleAttribute/Value le {max_cloud_cover})"
            )

        url = f"{self.base_url}/Products?$filter={query}&$top=100"
        print(f"  Query URL (truncated): {url[:150]}...")

        headers = {"Authorization": f"Bearer {self.access_token}"}

        try:
            response = requests.get(url, headers=headers, timeout=60)
            response.raise_for_status()
            data = response.json()
            return data.get('value', [])
        except requests.exceptions.Timeout:
            print("  ✗ Connection timeout. Please check your internet connection.")
            return []
        except requests.exceptions.HTTPError as e:
            print(f"  ✗ HTTP Error: {e}")
            print(f"     This might indicate invalid coordinates or query format")
            return []
        except Exception as e:
            print(f"  ✗ Search error: {e}")
            return []

    def download_product(self, product_id, output_path, max_retries=3):
        """Download a product with retry logic and token refresh"""

        for attempt in range(max_retries):
            try:
                if not self.get_access_token():
                    raise Exception("Failed to refresh access token")

                download_url = f"{self.download_url}/Products({product_id})/$value"
                headers = {"Authorization": f"Bearer {self.access_token}"}

                response = requests.get(download_url, headers=headers, stream=True, timeout=300)
                response.raise_for_status()

                total_size = int(response.headers.get('content-length', 0))
                temp_path = str(output_path) + ".tmp"

                with open(temp_path, 'wb') as f, tqdm(
                        total=total_size,
                        unit='B',
                        unit_scale=True,
                        unit_divisor=1024,
                        desc=f"      Downloading (attempt {attempt + 1}/{max_retries})"
                ) as pbar:
                    for chunk in response.iter_content(chunk_size=8192):
                        if chunk:
                            f.write(chunk)
                            pbar.update(len(chunk))

                os.replace(temp_path, output_path)
                return True

            except (requests.exceptions.ConnectionError,
                    requests.exceptions.ChunkedEncodingError,
                    ConnectionResetError) as e:
                print(f"      ⚠ Connection error on attempt {attempt + 1}/{max_retries}: {type(e).__name__}")

                if os.path.exists(temp_path):
                    os.remove(temp_path)

                if attempt < max_retries - 1:
                    wait_time = 2 ** attempt
                    print(f"      ⏳ Retrying in {wait_time} seconds...")
                    time.sleep(wait_time)
                else:
                    print(f"      ✗ Download failed after {max_retries} attempts")
                    return False

            except Exception as e:
                print(f"      ✗ Download error: {e}")
                if os.path.exists(temp_path):
                    os.remove(temp_path)
                return False

        return False
#!/usr/bin/env python3
"""
Sentinel-1 & Sentinel-2 Data Downloader and Processor

Sentinel-2 (Optical):
- Max cloud cover filtering (API-compliant)
- True color RGB composites
- 13 bands statistics
- TCIs saved as PNG

Sentinel-1 (SAR):
- GRD and SLC product support
- VV, VH, HH, HV polarizations
- Radiometric calibration
- Speckle filtering (Lee filter)
- Grayscale and false-color RGB outputs
- Backscatter statistics

Common Features:
- Automatic ZIP deletion after extraction
- Automatic .SAFE folder cleanup
- Statistics and histograms
- Clipped outputs to user AOI
"""

from pymongo import MongoClient
from bson import ObjectId
import os, smtplib
from email.message import EmailMessage
import secrets
from datetime import timedelta
from flask import jsonify
import json

from datetime import datetime
from pathlib import Path
from SentinelDownloader import SentinelDownloader

try:

    from PIL import Image
    import matplotlib

    matplotlib.use('Agg')  # Non-interactive backend

except ImportError as e:
    print(f"Missing required library: {e}")
    print("\nPlease install required packages:")
    print("pip install requests shapely geopandas rasterio pillow tqdm matplotlib scipy")
    exit(1)

# Fix PIL decompression bomb warning
Image.MAX_IMAGE_PIXELS = None


def load_config(config_path="config.json"):
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            return json.load(f)
    return None


def save_config(username, password, config_path="config.json"):
    config = {
        "username": username,
        "password": password,
        "note": "Keep this file secure and don't commit to version control!"
    }
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    print(f"\n✓ Credentials saved to {config_path}")
    print("  (Add this file to .gitignore!)")


def send_email(to_email: str, subject: str, body: str):
    msg = EmailMessage()
    msg["From"] = os.getenv("SMTP_FROM")
    msg["To"] = to_email
    msg["Subject"] = subject
    msg.set_content(body)

    with smtplib.SMTP(os.getenv("SMTP_HOST"), int(os.getenv("SMTP_PORT"))) as server:
        server.starttls()
        server.login(
            os.getenv("SMTP_USER"),
            os.getenv("SMTP_PASS")
        )
        server.send_message(msg)


def main(user_id):
    print("\n" + "=" * 70)
    print(" " * 20 + "Sentinel-2 Data Downloader")
    print(" " * 15 + "(Copernicus Data Space Ecosystem)")
    print("=" * 70)

    config = load_config()

    if config and 'username' in config and 'password' in config:
        print("\n✓ Using credentials from config.json (batch mode)")
        username = config['username']
        password = config['password']
    else:
        print("\nEnter your Copernicus Data Space credentials:")
        print("(Register at: https://dataspace.copernicus.eu/)")
        username = input("Username: ").strip()
        password = input("Password: ").strip()

        if not username or not password:
            print("\n✗ Error: Username and password required")
            return

        save = input("\nSave credentials for batch mode? (y/n): ").strip().lower()
        if save == 'y':
            save_config(username, password)

    script_dir = Path(__file__).parent
    selections_file = script_dir / f"selections_{user_id}.txt"

    if not selections_file.exists():
        print(f"\n✗ Error: {selections_file} not found!")
        print(f"   Looking in: {script_dir.absolute()}")
        print("Please run the GUI application first.")
        return

    try:
        downloader = SentinelDownloader(username, password, output_dir=f'sentinel_data_{user_id}')
        downloader.process_all_selections(str(selections_file))
    except Exception as e:
        print(f"\n✗ Fatal error: {e}")
        import traceback
        traceback.print_exc()

    client = MongoClient('mongodb://localhost:27017/')
    db = client["Credentials"]
    downloads = db["download_tokens"]
    users = db["users"]

    # Create one-time download token valid for 24 hours
    token = secrets.token_urlsafe(32)
    expires_at = datetime.utcnow() + timedelta(hours=24)

    downloads.insert_one({
        "user_id": user_id,
        "token": token,
        "expires_at": expires_at,
        "used": False,
        "created_at": datetime.utcnow()
    })

    user = users.find_one({"_id": ObjectId(user_id)})
    body = f"http://127.0.0.1:5000/api/copernicus/download?token={token}"
    to_email = str(user.get("email")) if user else None
    subject = 'Sentinel Data Ready for Download'
    send_email(to_email, subject, body)


import sys

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: Copernicus2.py <user_id>")
        sys.exit(1)

    user_id = sys.argv[1]
    main(user_id)
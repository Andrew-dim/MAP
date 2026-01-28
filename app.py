"""
Multi-Sensor SAR Analysis Platform - Backend
"""

from flask import (
    Flask, render_template, request, jsonify, redirect,
    url_for, session, send_file
)
from flask_cors import CORS
from flask_socketio import SocketIO
from pymongo import MongoClient
from datetime import datetime, timedelta
from werkzeug.security import generate_password_hash, check_password_hash
from functools import wraps
import random
import string
import json
import os
import sys
import uuid
import threading
import subprocess
import zipfile
import shutil
from pathlib import Path
from bson import ObjectId

app = Flask(__name__)
app.secret_key = os.environ.get('FLASK_SECRET_KEY', 'your-secret-key-change-in-production')
CORS(app, supports_credentials=True)
socketio = SocketIO(app, cors_allowed_origins="*", async_mode='threading')

MONGO_URI = os.environ.get('MONGO_URI', 'mongodb://localhost:27017/')
client = MongoClient(MONGO_URI)
db = client["Credentials"]
users_collection = db["Logins"]
jobs_collection = db["jobs"]
otps_collection = db["otps"]

UPLOAD_FOLDER = Path('uploads')
OUTPUT_FOLDER = Path('sentinel_data')
NISAR_OUTPUT_FOLDER = Path('nisar_outputs')
SR_OUTPUT_FOLDER = Path('sr_outputs')

UPLOAD_FOLDER.mkdir(exist_ok=True)
OUTPUT_FOLDER.mkdir(exist_ok=True)
NISAR_OUTPUT_FOLDER.mkdir(exist_ok=True)
SR_OUTPUT_FOLDER.mkdir(exist_ok=True)

app.config['UPLOAD_FOLDER'] = str(UPLOAD_FOLDER)
app.config['MAX_CONTENT_LENGTH'] = 500 * 1024 * 1024

active_jobs = {}


def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'user_id' not in session:
            return jsonify({'error': 'Authentication required'}), 401
        return f(*args, **kwargs)
    return decorated_function


def get_current_user():
    if 'user_id' in session:
        try:
            return users_collection.find_one({'_id': ObjectId(session['user_id'])})
        except:
            return users_collection.find_one({'_id': session['user_id']})
    return None


@app.route('/')
def index():
    if 'user_id' in session:
        return redirect(url_for('map_page'))
    return redirect(url_for('login_page'))


@app.route('/login')
def login_page():
    if 'user_id' in session:
        return redirect(url_for('map_page'))
    return render_template('login.html')


@app.route('/register')
def register_page():
    return render_template('register.html')


@app.route('/map')
@login_required
def map_page():
    return render_template('map.html')


@app.route('/profile')
@login_required
def profile_page():
    user = get_current_user()
    if not user:
        return redirect(url_for('login_page'))
    user_data = {
        'username': user.get('username', ''),
        'email': user.get('email', ''),
        'name': user.get('name', ''),
        'lastname': user.get('lastname', ''),
        'created_at': user.get('created_at', ''),
        'verified': user.get('verified', False)
    }
    return render_template('profile.html', user=user_data)


@app.route('/verify')
def verify_page():
    return render_template('verify.html')


@app.route('/api/login', methods=['POST'])
def api_login():
    data = request.get_json()
    if not data:
        return jsonify({'ok': False, 'error': 'No data provided'}), 400
    username = data.get('username')
    password = data.get('password')
    if not username or not password:
        return jsonify({'ok': False, 'error': 'Username and password required'}), 400
    user = users_collection.find_one({'$or': [{'username': username}, {'email': username}]})
    if not user:
        return jsonify({'ok': False, 'error': 'User not found'}), 404
    if not check_password_hash(user['password'], password):
        return jsonify({'ok': False, 'error': 'Invalid password'}), 401
    if not user.get('verified', False):
        session['pending_user_id'] = str(user['_id'])
        return jsonify({'ok': False, 'error': 'Please verify your email first', 'redirect': '/verify'}), 403
    session['user_id'] = str(user['_id'])
    session['username'] = user['username']
    return jsonify({'ok': True, 'message': 'Login successful'})


@app.route('/api/register', methods=['POST'])
def api_register():
    data = request.get_json()
    if not data:
        return jsonify({'ok': False, 'error': 'No data provided'}), 400
    username = data.get('username')
    email = data.get('email')
    password = data.get('password')
    password2 = data.get('password2')
    name = data.get('name', '')
    lastname = data.get('lastname', '')
    if not all([username, email, password]):
        return jsonify({'ok': False, 'error': 'All fields required'}), 400
    if password != password2:
        return jsonify({'ok': False, 'error': 'Passwords do not match'}), 400
    if users_collection.find_one({'username': username}):
        return jsonify({'ok': False, 'error': 'Username already exists'}), 409
    if users_collection.find_one({'email': email}):
        return jsonify({'ok': False, 'error': 'Email already registered'}), 409
    user_doc = {
        'username': username, 'email': email,
        'password': generate_password_hash(password),
        'name': name, 'lastname': lastname,
        'created_at': datetime.utcnow(), 'verified': False
    }
    result = users_collection.insert_one(user_doc)
    otp = ''.join(random.choices(string.digits, k=6))
    otps_collection.insert_one({
        'user_id': str(result.inserted_id), 'otp': otp,
        'created_at': datetime.utcnow(),
        'expires_at': datetime.utcnow() + timedelta(minutes=10)
    })
    session['pending_user_id'] = str(result.inserted_id)
    print(f"\n{'='*50}\n[OTP CODE] For {email}: {otp}\n{'='*50}\n")
    return jsonify({'ok': True, 'message': 'Registration successful. Check terminal for OTP.'})


@app.route('/api/verify-otp', methods=['POST'])
def verify_otp():
    data = request.get_json()
    code = data.get('code', data.get('otp', ''))
    user_id = session.get('pending_user_id')
    if not user_id:
        return jsonify({'ok': False, 'error': 'Session expired'}), 400
    if not code:
        return jsonify({'ok': False, 'error': 'OTP required'}), 400
    otp_doc = otps_collection.find_one({
        'user_id': user_id, 'otp': code,
        'expires_at': {'$gt': datetime.utcnow()}
    })
    if not otp_doc:
        return jsonify({'ok': False, 'error': 'Invalid or expired OTP'}), 400
    try:
        users_collection.update_one({'_id': ObjectId(user_id)}, {'$set': {'verified': True}})
    except:
        users_collection.update_one({'_id': user_id}, {'$set': {'verified': True}})
    otps_collection.delete_one({'_id': otp_doc['_id']})
    session.pop('pending_user_id', None)
    return jsonify({'ok': True, 'message': 'Email verified'})


@app.route('/api/resend-otp', methods=['POST'])
def resend_otp():
    user_id = session.get('pending_user_id')
    if not user_id:
        return jsonify({'ok': False, 'error': 'No pending verification'}), 400
    otps_collection.delete_many({'user_id': user_id})
    otp = ''.join(random.choices(string.digits, k=6))
    otps_collection.insert_one({
        'user_id': user_id, 'otp': otp,
        'created_at': datetime.utcnow(),
        'expires_at': datetime.utcnow() + timedelta(minutes=10)
    })
    print(f"\n{'='*50}\n[NEW OTP]: {otp}\n{'='*50}\n")
    return jsonify({'ok': True, 'message': 'New OTP sent'})


@app.route('/logout', methods=['POST'])
def logout():
    session.clear()
    return jsonify({'ok': True})


def update_job_status(job_id, status, progress, message, products=None):
    update = {'status': status, 'progress': progress, 'message': message, 'updated_at': datetime.utcnow()}
    if products:
        update['products'] = products
    jobs_collection.update_one({'job_id': job_id}, {'$set': update})


def collect_products(output_dir):
    products = []
    output_path = Path(output_dir)
    if not output_path.exists():
        return products
    for ext in ['*.tif', '*.tiff', '*.png', '*.jpg', '*.jpeg', '*.geojson', '*.json', '*.pdf', '*.zip', '*.html', '*.txt']:
        for f in output_path.rglob(ext):
            if 'config' not in f.name.lower() and 'selection' not in f.name.lower():
                try:
                    products.append({
                        'name': f.name,
                        'path': str(f.relative_to(output_path)),
                        'size': f.stat().st_size,
                        'type': f.suffix[1:].upper()
                    })
                except:
                    pass
    return products


def format_selections_file(selections, satellite, user_id, output_path, resolution_mode="original", date_from="", date_to="", cloud_max=30):
    with open(output_path, "w", encoding="utf-8") as f:
        for i, sel in enumerate(selections, 1):
            coords = sel.get("coords", [])
            f.write(f"Selection {i} - Satellite\n")
            f.write(f"Constellation: {satellite}\n")
            if date_from and date_to:
                time_from = date_from.replace("T", " ")[:16]
                time_to = date_to.replace("T", " ")[:16]
                f.write(f"Time Range: {time_from} to {time_to}\n")
            f.write(f"Max Cloud Cover: {cloud_max}%\n")
            for coord in coords:
                if isinstance(coord, dict):
                    lon = coord.get("lon", coord.get("lng", 0))
                    lat = coord.get("lat", 0)
                    f.write(f"{lon}, {lat}\n")
            f.write("\n")
    return output_path


def run_processor(job_id, user_id, satellite, output_dir, selections_file, resolution_mode='original'):
    try:
        update_job_status(job_id, 'processing', 5, 'Starting processor...')
        project_root = Path(__file__).parent.resolve()

        if satellite == 'NISAR':
            config_file = Path(output_dir) / f"nisar_config_{user_id}.json"
            cmd = ['python', 'NISARProcessor.py', user_id, str(config_file.resolve())]
            process = subprocess.Popen(
                cmd, cwd=str(project_root), stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT, text=True, env=os.environ.copy()
            )
            for line in process.stdout:
                if line.strip():
                    print(f"[NISAR] {line.strip()}")
            process.wait()
            if process.returncode == 0:
                products = collect_products(output_dir)
                update_job_status(job_id, 'completed', 100, 'Complete', products=products)
            else:
                update_job_status(job_id, 'error', 0, f'Exit code {process.returncode}')

        elif satellite in ['Sentinel-1', 'Sentinel-2']:
            update_job_status(job_id, 'processing', 10, f'Starting {satellite}...')
            copernicus_script = project_root / 'COPERNICUS' / 'Copernicus.py'
            
            if copernicus_script.exists():
                # Copy selections file to COPERNICUS folder
                shutil.copy(selections_file, project_root / "COPERNICUS" / Path(selections_file).name)
                cmd = ['python', 'COPERNICUS/Copernicus.py', user_id]
                process = subprocess.Popen(
                    cmd, cwd=str(project_root), stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT, text=True, env=os.environ.copy()
                )
                for line in process.stdout:
                    if line.strip():
                        print(f"[Sentinel] {line.strip()}")
                        if 'Searching' in line:
                            update_job_status(job_id, 'processing', 15, 'Searching...')
                        elif 'Downloading' in line:
                            update_job_status(job_id, 'processing', 30, 'Downloading...')
                        elif 'Processing' in line:
                            update_job_status(job_id, 'processing', 60, 'Processing...')
                process.wait()

                if process.returncode == 0:
                    # Check if enhanced mode - run SR processing
                    if resolution_mode == 'enhanced':
                        update_job_status(job_id, 'processing', 70, 'Applying Super-Resolution...')
                        sr_output = SR_OUTPUT_FOLDER / f"sr_{satellite.lower()}_{user_id}_{job_id[:8]}"
                        sr_output.mkdir(parents=True, exist_ok=True)

                        # Find downloaded images in previews folder
                        previews_dir = Path(output_dir) / 'previews'
                        if not previews_dir.exists():
                            previews_dir = Path(output_dir)

                        sr_cmd = [
                            sys.executable, 'SuperResolution.py',
                            '--user-id', user_id,
                            '--satellite', satellite,
                            '--output', str(sr_output),
                            '--input', str(previews_dir)
                        ]
                        print(f"[SR] Running: {' '.join(sr_cmd)}")
                        sr_process = subprocess.Popen(
                            sr_cmd, cwd=str(project_root), stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT, text=True, env=os.environ.copy()
                        )
                        for line in sr_process.stdout:
                            if line.strip():
                                print(f"[SR] {line.strip()}")
                        sr_process.wait()

                        if sr_process.returncode == 0:
                            products = collect_products(str(sr_output))
                            update_job_status(job_id, 'completed', 100, 'SR Enhancement complete', products=products)
                        else:
                            # SR failed, but download succeeded
                            products = collect_products(output_dir)
                            update_job_status(job_id, 'completed', 100, 'Download complete (SR failed)', products=products)
                    else:
                        # Original mode - just download
                        products = collect_products(output_dir)
                        update_job_status(job_id, 'completed', 100, 'Complete', products=products)
                else:
                    update_job_status(job_id, 'error', 0, f'Exit code {process.returncode}')
            else:
                update_job_status(job_id, 'error', 0, 'Copernicus.py not found')
        else:
            update_job_status(job_id, 'error', 0, f'Unknown satellite: {satellite}')
    except Exception as e:
        import traceback
        traceback.print_exc()
        update_job_status(job_id, 'error', 0, str(e))
    finally:
        if job_id in active_jobs:
            del active_jobs[job_id]


@app.route('/api/submit', methods=['POST'])
@login_required
def submit_analysis():
    data = request.get_json()
    if not data:
        return jsonify({'error': 'No data provided'}), 400
    user_id = session['user_id']
    satellite = data.get('satellite', 'Sentinel-2')
    selections = data.get('selections', [])
    resolution_mode = data.get('resolution_mode', 'original')
    date_from = data.get('date_from', '')
    date_to = data.get('date_to', '')
    cloud_max = data.get('cloud_max', 30)
    if not selections:
        return jsonify({'error': 'No selections provided'}), 400

    job_id = str(uuid.uuid4())
    if satellite == 'NISAR':
        output_dir = NISAR_OUTPUT_FOLDER / f"nisar_{user_id}_{job_id[:8]}"
    else:
        output_dir = OUTPUT_FOLDER / f"sentinel_data_{user_id}"
    output_dir.mkdir(parents=True, exist_ok=True)

    selections_file = output_dir / f"selections_{user_id}.txt"
    format_selections_file(selections, satellite, user_id, selections_file, resolution_mode, date_from, date_to, cloud_max)

    job_doc = {
        'job_id': job_id, 'user_id': user_id, 'satellite': satellite,
        'resolution_mode': resolution_mode, 'status': 'queued', 'progress': 0,
        'message': 'Queued', 'selections_count': len(selections),
        'created_at': datetime.utcnow(), 'updated_at': datetime.utcnow(),
        'output_dir': str(output_dir), 'selections_file': str(selections_file),
        'products': [], 'errors': []
    }

    if satellite == 'NISAR':
        nisar_config = {
            'frequency': data.get('frequency', 'L-band'),
            'level': data.get('level', 'L2-GCOV'),
            'analysis': data.get('analysis', 'basic'),
            'military_features': data.get('military_features', {}),
            'selections': selections,
            'date_from': date_from,
            'date_to': date_to
        }
        config_file = output_dir / f"nisar_config_{user_id}.json"
        with open(config_file, 'w') as f:
            json.dump(nisar_config, f, indent=2, default=str)
        job_doc['nisar_config'] = nisar_config
        job_doc['config_file'] = str(config_file)

    jobs_collection.insert_one(job_doc)

    thread = threading.Thread(
        target=run_processor,
        args=(job_id, user_id, satellite, str(output_dir), str(selections_file), resolution_mode)
    )
    thread.daemon = True
    thread.start()
    active_jobs[job_id] = thread

    return jsonify({
        'success': True, 'ok': True, 'job_id': job_id,
        'resolution_mode': resolution_mode,
        'message': f'{satellite} job submitted'
    })


@app.route('/submit-polygons', methods=['POST'])
@login_required
def submit_polygons_legacy():
    return submit_analysis()


@app.route('/api/jobs', methods=['GET'])
@login_required
def list_jobs():
    user_id = session['user_id']
    jobs = list(jobs_collection.find({'user_id': user_id}, {'_id': 0}).sort('created_at', -1).limit(50))
    for job in jobs:
        if 'created_at' in job:
            job['created_at'] = job['created_at'].isoformat()
        if 'updated_at' in job:
            job['updated_at'] = job['updated_at'].isoformat()
    return jsonify({'jobs': jobs})


@app.route('/api/jobs/<job_id>', methods=['GET'])
@login_required
def get_job_status(job_id):
    user_id = session['user_id']
    job = jobs_collection.find_one({'job_id': job_id, 'user_id': user_id}, {'_id': 0})
    if not job:
        return jsonify({'error': 'Job not found'}), 404
    if 'created_at' in job:
        job['created_at'] = job['created_at'].isoformat()
    if 'updated_at' in job:
        job['updated_at'] = job['updated_at'].isoformat()
    return jsonify(job)


@app.route('/api/jobs/<job_id>/cancel', methods=['POST'])
@login_required
def cancel_job(job_id):
    user_id = session['user_id']
    job = jobs_collection.find_one({'job_id': job_id, 'user_id': user_id})
    if not job:
        return jsonify({'error': 'Job not found'}), 404
    update_job_status(job_id, 'cancelled', 0, 'Cancelled by user')
    return jsonify({'success': True})


@app.route('/api/jobs/<job_id>/download/<path:filename>', methods=['GET'])
@login_required
def download_product(job_id, filename):
    user_id = session['user_id']
    job = jobs_collection.find_one({'job_id': job_id, 'user_id': user_id})
    if not job:
        return jsonify({'error': 'Job not found'}), 404
    file_path = Path(job['output_dir']) / filename
    if not file_path.exists():
        return jsonify({'error': 'File not found'}), 404
    return send_file(file_path, as_attachment=True)


@app.route('/api/jobs/<job_id>/download-all', methods=['GET'])
@login_required
def download_all_products(job_id):
    user_id = session['user_id']
    job = jobs_collection.find_one({'job_id': job_id, 'user_id': user_id})
    if not job:
        return jsonify({'error': 'Job not found'}), 404
    output_dir = Path(job['output_dir'])
    zip_filename = f"{job['satellite']}_{job_id[:8]}_products.zip"
    zip_path = OUTPUT_FOLDER / zip_filename
    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for f in output_dir.rglob('*'):
            if f.is_file():
                zipf.write(f, f.relative_to(output_dir))
    return send_file(zip_path, as_attachment=True)


@app.route('/api/satellites', methods=['GET'])
def list_satellites():
    return jsonify({
        'Sentinel-1': {'name': 'Sentinel-1', 'type': 'C-band SAR', 'resolution': '5-40m'},
        'Sentinel-2': {'name': 'Sentinel-2', 'type': 'Optical', 'resolution': '10-60m'},
        'NISAR': {'name': 'NISAR', 'type': 'L+S band SAR', 'resolution': '3-10m'}
    })


@app.route('/api/health', methods=['GET'])
def health_check():
    return jsonify({'status': 'healthy', 'timestamp': datetime.utcnow().isoformat()})


@socketio.on('connect', namespace='/jobs')
def handle_connect():
    print("Client connected")


@socketio.on('disconnect', namespace='/jobs')
def handle_disconnect():
    print("Client disconnected")


@app.errorhandler(404)
def not_found(e):
    if request.path.startswith('/api/'):
        return jsonify({'error': 'Not found'}), 404
    return '<h1>404</h1>', 404


@app.errorhandler(500)
def internal_error(e):
    return jsonify({'error': 'Server error'}), 500


if __name__ == '__main__':
    print("=" * 60)
    print("Hellas - Multi-Sensor SAR Platform")
    print("=" * 60)
    socketio.run(app, host='0.0.0.0', port=5000, debug=True, allow_unsafe_werkzeug=True)

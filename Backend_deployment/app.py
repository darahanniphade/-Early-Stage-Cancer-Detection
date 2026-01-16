from flask import Flask, render_template, request, jsonify, send_from_directory
from werkzeug.utils import secure_filename
import os
from pathlib import Path
from datetime import datetime
import json

# Import prediction functions
from prediction.predict_brain import predict_image as predict_brain_image
from prediction.predict_skin import predict_image as predict_skin_image

app = Flask(__name__)

# ========== CONFIGURATION ==========
BASE_DIR = Path(__file__).resolve().parent
UPLOAD_FOLDER = BASE_DIR / 'static' / 'uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'bmp'}

app.config['UPLOAD_FOLDER'] = str(UPLOAD_FOLDER)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Create upload directory if it doesn't exist
UPLOAD_FOLDER.mkdir(parents=True, exist_ok=True)

# ========== HELPER FUNCTIONS ==========
def allowed_file(filename):
    """Check if file extension is allowed"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def save_upload_history(prediction_type, filename, result):
    """Save prediction history to JSON file"""
    history_file = BASE_DIR / 'static' / 'history.json'
    
    try:
        if history_file.exists():
            with open(history_file, 'r') as f:
                history = json.load(f)
        else:
            history = []
        
        entry = {
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'type': prediction_type,
            'filename': filename,
            'prediction': result.get('predicted_class', 'Unknown'),
            'confidence': result.get('confidence', 0)
        }
        
        history.append(entry)
        
        # Keep only last 100 entries
        history = history[-100:]
        
        with open(history_file, 'w') as f:
            json.dump(history, f, indent=2)
    except Exception as e:
        print(f"Error saving history: {e}")

# ========== ROUTES ==========

@app.route('/')
def index():
    """Home page"""
    return render_template('index.html')

@app.route('/brain')
def brain():
    """Brain tumor detection page"""
    return render_template('brain.html')

@app.route('/skin')
def skin():
    """Skin cancer detection page"""
    return render_template('skin.html')

@app.route('/predict/brain', methods=['POST'])
def predict_brain():
    """Handle brain tumor prediction"""
    if 'file' not in request.files:
        return jsonify({'success': False, 'error': 'No file uploaded'})
    
    file = request.files['file']
    
    if file.filename == '':
        return jsonify({'success': False, 'error': 'No file selected'})
    
    if not allowed_file(file.filename):
        return jsonify({'success': False, 'error': 'Invalid file type. Please upload an image.'})
    
    try:
        # Save file with timestamp to avoid conflicts
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = secure_filename(file.filename)
        name, ext = os.path.splitext(filename)
        unique_filename = f"brain_{timestamp}_{name}{ext}"
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)
        
        file.save(filepath)
        
        # Make prediction
        result = predict_brain_image(filepath)
        
        if result['success']:
            # Add image URL to result
            result['image_url'] = f'/static/uploads/{unique_filename}'
            result['filename'] = unique_filename
            
            # Save to history
            save_upload_history('brain', unique_filename, result)
        
        return jsonify(result)
        
    except Exception as e:
        return jsonify({'success': False, 'error': f'Server error: {str(e)}'})

@app.route('/predict/skin', methods=['POST'])
def predict_skin():
    """Handle skin cancer prediction"""
    if 'file' not in request.files:
        return jsonify({'success': False, 'error': 'No file uploaded'})
    
    file = request.files['file']
    
    if file.filename == '':
        return jsonify({'success': False, 'error': 'No file selected'})
    
    if not allowed_file(file.filename):
        return jsonify({'success': False, 'error': 'Invalid file type. Please upload an image.'})
    
    try:
        # Save file with timestamp to avoid conflicts
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = secure_filename(file.filename)
        name, ext = os.path.splitext(filename)
        unique_filename = f"skin_{timestamp}_{name}{ext}"
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)
        
        file.save(filepath)
        
        # Make prediction
        result = predict_skin_image(filepath)
        
        if result['success']:
            # Add image URL to result
            result['image_url'] = f'/static/uploads/{unique_filename}'
            result['filename'] = unique_filename
            
            # Save to history
            save_upload_history('skin', unique_filename, result)
        
        return jsonify(result)
        
    except Exception as e:
        return jsonify({'success': False, 'error': f'Server error: {str(e)}'})

@app.route('/history')
def get_history():
    """Get prediction history"""
    history_file = BASE_DIR / 'static' / 'history.json'
    
    try:
        if history_file.exists():
            with open(history_file, 'r') as f:
                history = json.load(f)
            return jsonify({'success': True, 'history': history})
        else:
            return jsonify({'success': True, 'history': []})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/static/uploads/<filename>')
def uploaded_file(filename):
    """Serve uploaded files"""
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

@app.errorhandler(413)
def request_entity_too_large(error):
    """Handle file too large error"""
    return jsonify({'success': False, 'error': 'File too large. Maximum size is 16MB.'}), 413

@app.errorhandler(404)
def not_found(error):
    """Handle 404 errors"""
    return render_template('index.html'), 404

@app.errorhandler(500)
def internal_error(error):
    """Handle 500 errors"""
    return jsonify({'success': False, 'error': 'Internal server error'}), 500

# ========== MAIN ==========
if __name__ == '__main__':
    print(f"""
    {'='*60}
    🏥 Medical AI Application Starting
    {'='*60}
    
    📁 Upload folder: {UPLOAD_FOLDER}
    🌐 Access the application at: http://localhost:5000
    
    Available endpoints:
    • Home:        http://localhost:5000/
    • Brain:       http://localhost:5000/brain
    • Skin:        http://localhost:5000/skin
    
    {'='*60}
    """)
    
    app.run(debug=True, host='0.0.0.0', port=5000)
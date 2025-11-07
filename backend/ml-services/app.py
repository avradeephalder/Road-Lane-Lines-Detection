from flask import Flask, request, jsonify, send_file
from werkzeug.utils import secure_filename
import os
from inference import load_model, process_image, process_video

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 500 * 1024 * 1024  # 500MB max
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['OUTPUT_FOLDER'] = 'outputs'

os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['OUTPUT_FOLDER'], exist_ok=True)

# Load model at startup
MODEL_PATH = "models/unet_tusimple_amp.pth"
load_model(MODEL_PATH)

ALLOWED_EXTENSIONS = {'mp4', 'avi', 'mov', 'jpg', 'jpeg', 'png'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/health', methods=['GET'])
def health():
    return jsonify({"status": "ok", "message": "ML service running"}), 200

@app.route('/predict', methods=['POST'])
def predict():
    """Main prediction endpoint"""
    if 'file' not in request.files:
        return jsonify({"error": "No file provided"}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "Empty filename"}), 400
    
    if not allowed_file(file.filename):
        return jsonify({"error": "Invalid file type"}), 400
    
    try:
        # Save uploaded file
        filename = secure_filename(file.filename)
        input_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(input_path)
        
        # Determine output path
        name, ext = os.path.splitext(filename)
        output_filename = f"{name}_processed{ext}"
        output_path = os.path.join(app.config['OUTPUT_FOLDER'], output_filename)
        
        # Process based on file type
        file_ext = ext.lower()
        if file_ext in ['.jpg', '.jpeg', '.png']:
            result_path = process_image(input_path, output_path)
        elif file_ext in ['.mp4', '.avi', '.mov']:
            result_path = process_video(input_path, output_path)
        else:
            return jsonify({"error": "Unsupported file type"}), 400
        
        # Clean up input
        os.remove(input_path)
        
        # Return processed file
        return send_file(result_path, mimetype='application/octet-stream', as_attachment=True, download_name=output_filename)
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    finally:
        # Cleanup output after sending (optional - can keep for caching)
        if os.path.exists(output_path):
            try:
                os.remove(output_path)
            except:
                pass

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)

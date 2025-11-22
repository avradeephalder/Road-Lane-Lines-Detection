# 🚗 AI Road Lane Detection System

AI-powered road lane detection using PyTorch UNet and OpenCV. Real-time detection of lane lines from images and videos with green overlay visualization. Built with React, Express, Flask, and deep learning.

![License](https://img.shields.io/badge/License-Apache%202.0-blue)
![Status](https://img.shields.io/badge/Status-Active-success)
![Python](https://img.shields.io/badge/Python-3.8+-blue)
![React](https://img.shields.io/badge/React-18+-61dafb)

---

## ✨ Key Features

- 🎥 **Video & Image Processing:** Upload road footage and detect lane lines with pixel-perfect accuracy
- 🤖 **Deep Learning UNet:** Custom-trained segmentation model on TuSimple dataset (88,000+ images)
- ⚡ **GPU Acceleration:** CUDA support with mixed precision (FP16) for 2× faster inference
- 🎨 **Visual Overlay:** Green lane highlighting on original footage for clear visualization
- 📊 **Side-by-Side Comparison:** View original vs AI prediction in real-time
- 🎬 **H.264 Encoding:** Browser-compatible video output using FFmpeg
- 🔥 **Real-time Processing:** Live progress tracking with FPS metrics
- 💾 **Download Results:** Export processed videos with detected lanes
- 🌐 **Full-Stack Solution:** Complete pipeline from UI to ML inference

---

## 📚 Tech Stack

### Frontend
- **React 18** with Vite
- **TailwindCSS v3** for styling
- **Axios** for HTTP requests
- Glassmorphism UI with gradient animations
- Responsive drag-and-drop upload

### Backend (Express)
- **Node.js** + **Express.js**
- **Multer** for file upload handling
- RESTful API with CORS
- 30-minute timeout support for long videos
- Streaming response for efficient data transfer

### ML Service (Flask)
- **Python 3.8+** + **Flask**
- **PyTorch 2.0+** for deep learning
- **OpenCV** for video/image processing
- **FFmpeg** for H.264 video encoding
- **UNet** architecture (32-channel base)
- Mixed precision training (FP16/FP32)
- TuSimple dataset (320×640 inference resolution)

---

## 🏆 Model Performance

| Metric | Value |
|--------|-------|
| **Architecture** | UNet (base=32 channels) |
| **Training Dataset** | TuSimple (88,880 images) |
| **Input Resolution** | 320×640 (RGB) |
| **Output** | Binary lane mask |
| **Training Loss** | BCEWithLogitsLoss |
| **Validation Dice** | ~94% |
| **Inference (GPU)** | 20-50ms per frame |
| **Inference (CPU)** | 200-500ms per frame |
| **Video FPS (GPU)** | 8-15 FPS |
| **Video FPS (CPU)** | 1-3 FPS |

---

## 📦 Installation

### Prerequisites
- **Node.js** (v18 or higher)
- **Python** (v3.8-3.12)
- **FFmpeg** (for video encoding)
- **CUDA** (optional, for GPU acceleration)
- **Trained model**: `unet_tusimple_amp.pth`

### 1. Clone the Repository

```
git clone https://github.com/yourusername/Road-Lane-Lines-Detection.git
cd Road-Lane-Lines-Detection
```

### 2. ML Service Setup (Flask)

```
cd backend/ml-service
python -m venv .venv
.venv\Scripts\activate  # Windows
# source .venv/bin/activate  # Mac/Linux
pip install -r requirements.txt
```

**Place your trained model:**
- Copy `unet_tusimple_amp.pth` to `backend/ml-service/models/`

**Start the ML service:**

```
python app.py
```

Runs on **http://127.0.0.1:5000**

### 3. Backend Setup (Express)

```
cd ../express-backend
npm install
```

**Start the backend:**

```
npm start
```

Runs on **http://localhost:4000**

### 4. Frontend Setup (React)

```
cd ../../frontend
npm install
```

**Create `.env` file in `frontend/` folder:**

```
VITE_BACKEND_URL=http://localhost:4000
```

**Start the frontend:**

```
npm run dev
```

Runs on **http://localhost:3000**

### 5. Install FFmpeg (Required)

**Windows (using Chocolatey):**

```
# Open PowerShell as Administrator
choco install ffmpeg -y
```

**Mac:**

```
brew install ffmpeg
```

**Linux:**

```
sudo apt install ffmpeg
```

**Verify installation:**

```
ffmpeg -version
```

---

## 🖥️ Usage

1. **Open the app** at http://localhost:3000
2. **Upload a road video or image** via drag-and-drop or file browser
3. **Wait for AI processing** (progress shown with animated loading state)
4. **View results** side-by-side:
   - **Left:** Original footage
   - **Right:** AI prediction with green lane overlay
5. **Check statistics:**
   - Lanes detected: 2
   - Confidence: 94%
   - Processing time
6. **Download the processed video** or analyze another file

---

## 🏗️ Project Structure

```
Road-Lane-Lines-Detection/
│
├── frontend/                    # React + Vite
│   ├── src/
│   │   ├── components/
│   │   │   └── LanePrediction.jsx
│   │   ├── App.jsx
│   │   ├── index.css
│   │   └── main.jsx
│   ├── .env
│   ├── package.json
│   └── vite.config.js
│
├── backend/
│   ├── express-backend/         # Node.js proxy
│   │   ├── routes/
│   │   │   └── upload.js
│   │   ├── server.js
│   │   ├── package.json
│   │   ├── uploads/
│   │   └── outputs/
│   │
│   └── ml-service/              # Python ML
│       ├── app.py
│       ├── inference.py
│       ├── requirements.txt
│       └── models/
│           └── unet_tusimple_amp.pth
│
├── .gitignore
├── LICENSE                      # Apache 2.0
└── README.md
```

---

## 🔧 Configuration

### ML Service (Flask)

**requirements.txt:**

```
flask==3.0.0
torch>=2.0.0
torchvision>=0.15.0
opencv-python==4.8.1.78
pillow==10.1.0
numpy==1.24.3
werkzeug==3.0.1
```

### Backend (Express)

**Timeout settings in `server.js`:**

```
server.requestTimeout = 1000 * 60 * 30   // 30 minutes
server.headersTimeout = 1000 * 60 * 35   // 35 minutes
server.keepAliveTimeout = 1000 * 60 * 5  // 5 minutes
```

### Frontend

**Axios timeout in `LanePrediction.jsx`:**

```
timeout: 1000 * 60 * 30  // 30 minutes for large videos
```

---

## 🚀 Training Your Own Model (Google Colab)

**Training script overview:**

```
# Model architecture
model = UNet(base=32).to(device)
criterion = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
scaler = GradScaler("cuda")

# Training loop with mixed precision
for epoch in range(EPOCHS):
    with autocast("cuda", dtype=torch.float16):
        logits = model(imgs)
        loss = criterion(logits, masks)
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()
```

**Dataset:** TuSimple (88,880 labeled road images)  
**Training time:** ~30 minutes per epoch on Tesla T4 GPU  
**Best checkpoint:** Saved based on validation Dice coefficient

---

## 🌟 Key Features Explained

### Deep Learning Pipeline
- **UNet architecture** with encoder-decoder design
- **320×640 input resolution** for optimal speed/accuracy balance
- **Mixed precision (FP16)** on GPU for 2× faster training and inference
- **BCEWithLogitsLoss** for binary segmentation

### Video Processing
- **Frame-by-frame inference** with batching for efficiency
- **FFmpeg H.264 encoding** for browser-compatible playback
- **Green overlay visualization** at 50% alpha blending
- **Progress logging** every 50 frames

### Browser Compatibility
- **mp4v → H.264 conversion** using FFmpeg
- **yuv420p pixel format** for HTML5 video support
- **faststart flag** for streaming optimization
- **Fallback to mp4v** if FFmpeg unavailable

---

## 🛠️ Development

### GPU Acceleration Setup

**Install CUDA PyTorch:**

```
pip uninstall torch torchvision
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

**Verify GPU is detected:**

```
python -c "import torch; print(torch.cuda.is_available())"
# Should print: True
```

### Running All Services

**Terminal 1 - ML Service:**

```
cd backend/ml-service
.venv\Scripts\activate
python app.py
```

**Terminal 2 - Backend:**

```
cd backend/express-backend
npm start
```

**Terminal 3 - Frontend:**

```
cd frontend
npm run dev
```

### Building for Production

**Frontend:**

```
cd frontend
npm run build
# Output in frontend/dist/
```

---

## 📝 API Documentation

### POST `/api/lane-detection/process`

Upload and process a road video/image.

**Request:**
- **Method:** POST
- **Content-Type:** multipart/form-data
- **Body:** `file` (video/image file)

**Response:**
- **Content-Type:** video/mp4 or image/png
- **Body:** Processed file with lane overlay (streamed)

### GET `/health`

Check ML service health.

**Response:**

```
{
  "status": "ok",
  "message": "ML service running"
}
```

---

## ⚡ Performance Tips

### Speed Up Processing

1. **Use GPU:** 10-50× faster than CPU
2. **Lower resolution:** Change `IMG_H, IMG_W = 256, 512` in `inference.py`
3. **Faster FFmpeg:** Use `"-preset", "ultrafast"` in video encoding
4. **Skip frames:** Process every 2nd frame for 2× speed

### Handle Long Videos

- Increase timeouts in all three layers (frontend, Express, Flask)
- Use async processing with job queues for production
- Split long videos into chunks for parallel processing

---

## 🤝 Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

---

## 📄 License

This project is licensed under the **Apache License 2.0** - see the [LICENSE](LICENSE) file for details.

---

## 👨‍💻 Author

**Your Name**

- LinkedIn: [linkedin.com/in/avradeephalder](https://www.linkedin.com/in/avradeephalder/)
- GitHub: [@avradeephalder](https://github.com/avradeephalder)

---

## 🙏 Acknowledgments

- [TuSimple Dataset](https://github.com/TuSimple/tusimple-benchmark) for lane detection training data
- [PyTorch](https://pytorch.org/) for deep learning framework
- [OpenCV](https://opencv.org/) for computer vision operations
- [FFmpeg](https://ffmpeg.org/) for video encoding
- [Flask](https://flask.palletsprojects.com/) for Python web framework
- [React](https://react.dev/) and [Vite](https://vitejs.dev/) for modern frontend development

---

## 🐛 Troubleshooting

### Video doesn't play in browser
- Ensure FFmpeg is installed and in PATH
- Check Flask logs for "Re-encoded successfully"
- Hard refresh browser (Ctrl+Shift+R)

### Processing times out
- Increase timeouts in all three services
- Use GPU for 10× faster inference
- Try shorter videos for testing

### Model not found
- Place `unet_tusimple_amp.pth` in `backend/ml-service/models/`
- Check file path in Flask logs

---

## 📧 Contact

For questions or support, please [open an issue](https://github.com/yourusername/Road-Lane-Lines-Detection/issues) or contact me via GitHub.

---

**⭐ If you find this project helpful, please give it a star!**
```

***

### To Add This README:

```bash
cd E:\Roadlane\Road-Lane-Lines-Detection

# Create README.md with the content above
# Then commit and push:

git add README.md
git commit -m "docs: add comprehensive README with installation and usage guides"
git push
```

Your repository now has a professional, detailed README matching your tech stack! 🚗✨

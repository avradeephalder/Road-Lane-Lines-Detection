import os
import time
import subprocess
import torch
from torch import nn
import torchvision.transforms as T
from PIL import Image
import cv2
import numpy as np
from torch.amp import autocast

# -----------------------------
# Model definition (same as training)
# -----------------------------
class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )
    def forward(self, x):
        return self.net(x)

class UNet(nn.Module):
    def __init__(self, base=32):
        super().__init__()
        self.d1 = DoubleConv(3, base)
        self.p1 = nn.MaxPool2d(2)
        self.d2 = DoubleConv(base, base * 2)
        self.p2 = nn.MaxPool2d(2)
        self.d3 = DoubleConv(base * 2, base * 4)
        self.p3 = nn.MaxPool2d(2)
        self.d4 = DoubleConv(base * 4, base * 8)
        self.p4 = nn.MaxPool2d(2)
        self.b = DoubleConv(base * 8, base * 16)

        self.u4 = nn.ConvTranspose2d(base * 16, base * 8, 2, 2)
        self.c4 = DoubleConv(base * 16, base * 8)
        self.u3 = nn.ConvTranspose2d(base * 8, base * 4, 2, 2)
        self.c3 = DoubleConv(base * 8, base * 4)
        self.u2 = nn.ConvTranspose2d(base * 4, base * 2, 2, 2)
        self.c2 = DoubleConv(base * 4, base * 2)
        self.u1 = nn.ConvTranspose2d(base * 2, base, 2, 2)
        self.c1 = DoubleConv(base * 2, base)
        self.out = nn.Conv2d(base, 1, 1)

    def forward(self, x):
        d1 = self.d1(x); p1 = self.p1(d1)
        d2 = self.d2(p1); p2 = self.p2(d2)
        d3 = self.d3(p2); p3 = self.p3(d3)
        d4 = self.d4(p3); p4 = self.p4(d4)
        b = self.b(p4)
        u4 = self.u4(b);  c4 = self.c4(torch.cat([u4, d4], dim=1))
        u3 = self.u3(c4); c3 = self.c3(torch.cat([u3, d3], dim=1))
        u2 = self.u2(c3); c2 = self.c2(torch.cat([u2, d2], dim=1))
        u1 = self.u1(c2); c1 = self.c1(torch.cat([u1, d1], dim=1))
        return self.out(c1)

# -----------------------------
# Globals and diagnostics
# -----------------------------
device = "cuda" if torch.cuda.is_available() else "cpu"

def _device_banner():
    ver = getattr(torch, "__version__", "unknown")
    if torch.cuda.is_available():
        try:
            name = torch.cuda.get_device_name(0)
            cap = torch.cuda.get_device_capability(0)
            print(f"[ML] CUDA enabled: {name} (CC {cap[0]}.{cap[1]})")
        except Exception as e:
            print(f"[ML] CUDA enabled but failed to read device info: {e}")
        print(f"[ML] torch version: {ver}")
        print(f"[ML] autocast available: {hasattr(torch.cuda, 'amp')}")
    else:
        print("[ML] CUDA not available -> using CPU")
        print(f"[ML] torch version: {ver}")

_device_banner()

model = None
IMG_H, IMG_W = 320, 640

# -----------------------------
# Utilities
# -----------------------------
def load_model(model_path: str):
    """Load trained model once at startup"""
    global model
    print(f"[ML] Loading model: {model_path}")
    model = UNet(base=32).to(device)

    state = torch.load(model_path, map_location=device)
    # Support checkpoints that wrap state in "state_dict"
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]

    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing:
        print(f"[ML][warn] Missing keys: {missing}")
    if unexpected:
        print(f"[ML][warn] Unexpected keys: {unexpected}")

    model.eval()
    amp_note = "with AMP fp16" if device == "cuda" else "no AMP (CPU)"
    print(f"[ML] Model ready on {device} ({amp_note})")

def overlay(img_np: np.ndarray, mask_np: np.ndarray, alpha: float = 0.5):
    """Overlay green mask on image"""
    mask_col = np.zeros_like(img_np)
    mask_col[..., 1] = (mask_np * 255).astype(np.uint8)
    return cv2.addWeighted(img_np, 1.0, mask_col, alpha, 0)

def _forward(tens: torch.Tensor, context_hint: str = "") -> torch.Tensor:
    """One forward pass with proper autocast depending on device"""
    if device == "cuda":
        if context_hint:
            print(f"[ML] Using CUDA autocast(fp16) for {context_hint}")
        with autocast("cuda", dtype=torch.float16):
            logits = model(tens)
    else:
        if context_hint:
            print(f"[ML] Using CPU (no autocast) for {context_hint}")
        logits = model(tens)
    return logits

# -----------------------------
# Image path
# -----------------------------
def process_image(input_path: str, output_path: str):
    """Process single image and save visualization"""
    if model is None:
        raise RuntimeError("Model not loaded")

    t0 = time.time()
    print(f"[ML] process_image start | device={device} | input={os.path.basename(input_path)}")

    img = Image.open(input_path).convert("RGB")
    orig_w, orig_h = img.size
    img_resized = T.Resize((IMG_H, IMG_W), interpolation=T.InterpolationMode.BILINEAR)(img)
    tens = T.ToTensor()(img_resized).unsqueeze(0).to(device, non_blocking=True)

    with torch.no_grad():
        logits = _forward(tens, "image")
        prob = torch.sigmoid(logits)[0, 0].detach().float().cpu().numpy()
        pred_small = (prob > 0.5).astype(np.uint8) * 255

    pred_full = cv2.resize(pred_small, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)
    img_np = np.array(img)
    result = overlay(img_np, (pred_full > 127).astype(np.uint8))

    result_bgr = cv2.cvtColor(result, cv2.COLOR_RGB2BGR)
    cv2.imwrite(output_path, result_bgr)

    dt = time.time() - t0
    print(f"[ML] process_image done in {dt:.3f}s -> {os.path.basename(output_path)}")
    return output_path

# -----------------------------
# Video path (with FFmpeg re-encode to H.264)
# -----------------------------
def process_video(input_path: str, output_path: str):
    """Process video frame by frame, re-encode with FFmpeg to H.264 for browser playback"""
    if model is None:
        raise RuntimeError("Model not loaded")

    print(f"[ML] process_video start | device={device} | input={os.path.basename(input_path)}")
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {input_path}")

    # Write intermediate file with mp4v to keep OpenCV path fast and portable
    base, ext = os.path.splitext(output_path)
    temp_output = base + "_temp.mp4"
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    writer = cv2.VideoWriter(temp_output, fourcc, fps, (W, H))

    frame_idx = 0
    t0 = time.time()
    print("[ML] Processing frames...")
    with torch.no_grad():
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(rgb)
            img_r = T.Resize((IMG_H, IMG_W), interpolation=T.InterpolationMode.BILINEAR)(img)
            tens = T.ToTensor()(img_r).unsqueeze(0).to(device, non_blocking=True)

            logits = _forward(tens, "video" if frame_idx == 0 else "")
            prob = torch.sigmoid(logits)[0, 0].detach().float().cpu().numpy()
            pred_small = (prob > 0.5).astype(np.uint8) * 255
            pred_full = cv2.resize(pred_small, (W, H), interpolation=cv2.INTER_NEAREST)
            vis = overlay(rgb, (pred_full > 127).astype(np.uint8))
            out_bgr = cv2.cvtColor(vis, cv2.COLOR_RGB2BGR)
            writer.write(out_bgr)

            frame_idx += 1
            if frame_idx % 50 == 0:
                elapsed = time.time() - t0
                avg_fps = frame_idx / max(1e-6, elapsed)
                print(f"[ML] frames={frame_idx} | elapsed={elapsed:.2f}s | avg_fps={avg_fps:.2f}")

    cap.release()
    writer.release()

    # Re-encode with FFmpeg to H.264 for browser compatibility
    print("[ML] Re-encoding with FFmpeg for browser compatibility...")
    try:
        # Ensure .mp4 extension on final output
        final_output = base + ".mp4"

        # Run FFmpeg with web-friendly settings
        completed = subprocess.run(
            [
                "ffmpeg", "-y",
                "-i", temp_output,
                "-c:v", "libx264",
                "-preset", "fast",
                "-crf", "23",
                "-pix_fmt", "yuv420p",
                "-movflags", "+faststart",
                final_output,
            ],
            check=True,
            capture_output=True,
            text=True,
        )

        # Clean up temp file
        if os.path.exists(temp_output):
            os.remove(temp_output)

        print(f"[ML] Re-encoded successfully: {os.path.basename(final_output)}")

        total = time.time() - t0
        avg_fps = frame_idx / max(1e-6, total)
        print(f"[ML] process_video done | frames={frame_idx} | time={total:.2f}s | avg_fps={avg_fps:.2f}")
        return final_output

    except subprocess.CalledProcessError as e:
        # FFmpeg printed an error; log it and fallback to temp file
        print(f"[ML][ERROR] FFmpeg failed:\n{e.stderr}")
        if os.path.exists(temp_output):
            # Use the mp4v file (may not play in some browsers)
            os.replace(temp_output, output_path)
            print("[ML] Falling back to mp4v output (some browsers may not play it).")
            return output_path
        else:
            raise

    except FileNotFoundError:
        # FFmpeg not installed or not in PATH
        print("[ML][ERROR] FFmpeg not found in PATH. Install FFmpeg and ensure it's available in the system PATH.")
        if os.path.exists(temp_output):
            os.replace(temp_output, output_path)
            print("[ML] Falling back to mp4v output (some browsers may not play it).")
            return output_path
        else:
            raise

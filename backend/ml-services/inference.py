import os
import torch
from torch import nn
import torchvision.transforms as T
from PIL import Image
import cv2
import numpy as np
from torch.amp import autocast

# Model definition (same as training)
class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1), nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1), nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
        )
    def forward(self, x): return self.net(x)

class UNet(nn.Module):
    def __init__(self, base=32):
        super().__init__()
        self.d1 = DoubleConv(3, base)
        self.p1 = nn.MaxPool2d(2)
        self.d2 = DoubleConv(base, base*2)
        self.p2 = nn.MaxPool2d(2)
        self.d3 = DoubleConv(base*2, base*4)
        self.p3 = nn.MaxPool2d(2)
        self.d4 = DoubleConv(base*4, base*8)
        self.p4 = nn.MaxPool2d(2)
        self.b  = DoubleConv(base*8, base*16)
        
        self.u4 = nn.ConvTranspose2d(base*16, base*8, 2, 2)
        self.c4 = DoubleConv(base*16, base*8)
        self.u3 = nn.ConvTranspose2d(base*8, base*4, 2, 2)
        self.c3 = DoubleConv(base*8, base*4)
        self.u2 = nn.ConvTranspose2d(base*4, base*2, 2, 2)
        self.c2 = DoubleConv(base*4, base*2)
        self.u1 = nn.ConvTranspose2d(base*2, base, 2, 2)
        self.c1 = DoubleConv(base*2, base)
        self.out = nn.Conv2d(base, 1, 1)
    
    def forward(self, x):
        d1 = self.d1(x); p1 = self.p1(d1)
        d2 = self.d2(p1); p2 = self.p2(d2)
        d3 = self.d3(p2); p3 = self.p3(d3)
        d4 = self.d4(p3); p4 = self.p4(d4)
        b  = self.b(p4)
        u4 = self.u4(b);  c4 = self.c4(torch.cat([u4, d4], dim=1))
        u3 = self.u3(c4); c3 = self.c3(torch.cat([u3, d3], dim=1))
        u2 = self.u2(c3); c2 = self.c2(torch.cat([u2, d2], dim=1))
        u1 = self.u1(c2); c1 = self.c1(torch.cat([u1, d1], dim=1))
        return self.out(c1)

# Global model variable
device = "cuda" if torch.cuda.is_available() else "cpu"
model = None
IMG_H, IMG_W = 320, 640

def load_model(model_path):
    """Load trained model once at startup"""
    global model
    model = UNet(base=32).to(device)
    state = torch.load(model_path, map_location=device)
    model.load_state_dict(state)
    model.eval()
    print(f"Model loaded on {device}")

def overlay(img_np, mask_np, alpha=0.5):
    """Overlay green mask on image"""
    mask_col = np.zeros_like(img_np)
    mask_col[..., 1] = (mask_np * 255).astype(np.uint8)
    return cv2.addWeighted(img_np, 1.0, mask_col, alpha, 0)

def process_image(input_path, output_path):
    """Process single image"""
    if model is None:
        raise RuntimeError("Model not loaded")
    
    # Read and preprocess
    img = Image.open(input_path).convert("RGB")
    orig_w, orig_h = img.size
    img_resized = T.Resize((IMG_H, IMG_W), interpolation=T.InterpolationMode.BILINEAR)(img)
    tens = T.ToTensor()(img_resized).unsqueeze(0).to(device)
    
    # Inference
    with torch.no_grad():
        if device == "cuda":
            with autocast("cuda", dtype=torch.float16):
                logits = model(tens)
        else:
            logits = model(tens)
        prob = torch.sigmoid(logits)[0, 0].cpu().numpy()
        pred_small = (prob > 0.5).astype(np.uint8) * 255
    
    # Resize mask back to original and overlay
    pred_full = cv2.resize(pred_small, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)
    img_np = np.array(img)
    result = overlay(img_np, (pred_full > 127).astype(np.uint8))
    
    # Save
    result_bgr = cv2.cvtColor(result, cv2.COLOR_RGB2BGR)
    cv2.imwrite(output_path, result_bgr)
    return output_path

def process_video(input_path, output_path):
    """Process video frame by frame"""
    if model is None:
        raise RuntimeError("Model not loaded")
    
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {input_path}")
    
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    writer = cv2.VideoWriter(output_path, fourcc, fps, (W, H))
    
    with torch.no_grad():
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(rgb)
            img_r = T.Resize((IMG_H, IMG_W), interpolation=T.InterpolationMode.BILINEAR)(img)
            tens = T.ToTensor()(img_r).unsqueeze(0).to(device)
            
            if device == "cuda":
                with autocast("cuda", dtype=torch.float16):
                    logits = model(tens)
            else:
                logits = model(tens)
            
            prob = torch.sigmoid(logits)[0, 0].cpu().numpy()
            pred_small = (prob > 0.5).astype(np.uint8) * 255
            pred_full = cv2.resize(pred_small, (W, H), interpolation=cv2.INTER_NEAREST)
            vis = overlay(rgb, (pred_full > 127).astype(np.uint8))
            out_bgr = cv2.cvtColor(vis, cv2.COLOR_RGB2BGR)
            writer.write(out_bgr)
    
    cap.release()
    writer.release()
    return output_path

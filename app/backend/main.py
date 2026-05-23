import sys
from pathlib import Path
import os
import torch
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import shutil
import tempfile
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional

# Add root to sys path
_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from models.meta_learner import PhishFusion
from utils.url_processor import URLProcessor

app = FastAPI(title="PhishFusion API")

# Configure CORS for Vite frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins, adjust for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = None
processor = None
stat_scaler = None

@app.on_event("startup")
def load_model():
    global model, processor, stat_scaler
    try:
        model = PhishFusion().to(DEVICE)
        weights_path = os.path.join(_ROOT, "weights", "phishfusion_best.pth")
        
        # Check if weights exist before loading
        if os.path.exists(weights_path):
            model.load_state_dict(torch.load(weights_path, map_location=DEVICE), strict=False)
            print("Model weights loaded successfully.")
        else:
            print(f"WARNING: No trained weights found at {weights_path}. Using uninitialized model.")
            
        model.eval()
        
        processor = URLProcessor(max_len=128)
        print("URL processor initialized.")

        # Initialize and fit MinMaxScaler from train data
        train_csv_path = os.path.join(_ROOT, "data", "processed", "train2_features.csv")
        if os.path.exists(train_csv_path):
            print(f"Fitting MinMaxScaler from {train_csv_path}...")
            train_df = pd.read_csv(train_csv_path)
            stat_scaler = MinMaxScaler()
            stat_scaler.fit(train_df.iloc[:, 129:151].values.astype(np.float64, copy=True))
            print("MinMaxScaler fitted successfully.")
        else:
            print(f"WARNING: Train dataset not found at {train_csv_path}. Statistical features will NOT be scaled!")
            
    except Exception as e:
        print(f"Error loading model: {e}")

class PredictResponse(BaseModel):
    risk_score: float
    is_phishing: bool
    decoded_url: Optional[str] = None

@app.post("/predict", response_model=PredictResponse)
async def predict(url: Optional[str] = Form(None), file: Optional[UploadFile] = File(None)):
    global model, processor, stat_scaler
    if not url and not file:
        raise HTTPException(status_code=400, detail="Must provide either URL or QR image.")

    try:
        qr_image_path = None
        decoded_url = url
        
        # If file is provided, save it temporarily and process
        if file:
            suffix = Path(file.filename).suffix
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp_img:
                shutil.copyfileobj(file.file, temp_img)
                qr_image_path = temp_img.name
            
            # If no URL provided but file is, decode URL from QR
            if not decoded_url:
                try:
                    decoded_url = processor.extract_url_from_qr(qr_image_path)
                except ValueError as ve:
                    # Cleanup
                    os.remove(qr_image_path)
                    raise HTTPException(status_code=400, detail=str(ve))

        if not decoded_url:
            if qr_image_path and os.path.exists(qr_image_path):
                os.remove(qr_image_path)
            raise HTTPException(status_code=400, detail="Could not determine URL from input.")

        stat_features, lexical_tokens, qr_features = processor.process(decoded_url, qr_image_path=qr_image_path)
        
        # Cleanup temp file
        if qr_image_path and os.path.exists(qr_image_path):
            os.remove(qr_image_path)

        # Apply MinMaxScaler to statistical features if scaler is available
        if stat_scaler is not None:
            stat_features = stat_scaler.transform([stat_features])[0]

        # Prepare tensors
        stat_tensor = torch.FloatTensor(stat_features).unsqueeze(0).to(DEVICE)
        lexical_tensor = torch.LongTensor(lexical_tokens).unsqueeze(0).to(DEVICE)
        qr_tensor = torch.FloatTensor(qr_features).unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            output = model(qr_tensor, lexical_tensor, stat_tensor)
            risk_score = torch.sigmoid(output).item() if output.max() > 1 else output.item()
            is_phishing = risk_score > 0.5
            
        return PredictResponse(
            risk_score=risk_score,
            is_phishing=is_phishing,
            decoded_url=decoded_url if file else None
        )

    except HTTPException:
        raise
    except Exception as e:
        if 'qr_image_path' in locals() and qr_image_path and os.path.exists(qr_image_path):
            os.remove(qr_image_path)
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

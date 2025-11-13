# server.py
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from phishing_model import predict_email  # uses your saved pipeline

app = FastAPI()

# Allow Chrome extension to call this API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # you can tighten this later
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class EmailRequest(BaseModel):
    text: str

@app.post("/predict-phishing-pipeline")
def predict_phishing(req: EmailRequest):
    """
    Request body: {"text": "email content"}
    """
    return predict_email(req.text)

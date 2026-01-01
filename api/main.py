from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from api.inference import predict

app = FastAPI()

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# API Routes First
@app.post("/predict")
def predict_api(payload: dict):
    # EMPTY INPUT VALIDATION
    if not payload or not any(str(v).strip() for v in payload.values()):
        return {
            "error": "Please enter at least one field to analyze difficulty."
        }

    return predict(payload)


# SERVE FRONTEND AT /ui
app.mount("/ui", StaticFiles(directory="frontend", html=True), name="frontend")

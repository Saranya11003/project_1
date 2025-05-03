from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import base64, io, time
from PIL import Image
import os
from models import load_model


app = FastAPI(title="posecloud")

class PoseRequest(BaseModel):
    id : str
    image : str

class PoseResponseJson(BaseModel):
    id : str
    count : str
    boxes : list
    keypoints : list
    speed_preprocess: float
    speed_inference: float

class PoseResponseImage(BaseModel):
    id: str
    image: str


def get_model_choice():
    return os.getenv("MODEL_CHOICE", "model1")


@app.on_event("startup")
async def startup_event():
    global model
    model = load_model(get_model_choice())


@app.post("/api/pose/json", response_model=PoseResponseJson)
async def pose_json(request: PoseRequest):
    start = time.time()
    try:
        data = base64.b64decode(request.image)
        img = Image.open(io.BytesIO(data)).convert("RGB")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid image data: {e}")
    t1 = time.time()
    boxes, keypoints = model.predict(img)
    t2 = time.time()
    print(request.id)
    print(boxes)
    print(keypoints)
    try :
        response = PoseResponseJson(
        id = request.id,
        count=str(len(boxes)),
        boxes=boxes,
        keypoints=keypoints,
        speed_preprocess=t1 - start,
        speed_inference=t2 - t1
        )
        return response
    except Exception as e:
        print(e)
        return None
        



@app.post("/api/pose/annotated", response_model=PoseResponseImage)
async def pose_annotated(request: PoseRequest):
    try:
        data = base64.b64decode(request.image)
        img = Image.open(io.BytesIO(data)).convert("RGB")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid image data: {e}")
    
    boxes, keypoints = model.predict(img)
    annotated = model.draw_annotations(img, boxes, keypoints)
    buf = io.BytesIO()
    annotated.save(buf, format="JPEG")
    encoded_img = base64.b64encode(buf.getvalue()).decode()
    return PoseResponseImage(
        id = request.id,
        image = encoded_img
    )
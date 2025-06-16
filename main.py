from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse, StreamingResponse
from PIL import Image, UnidentifiedImageError
import io
import torch
from ultralytics import YOLO
import cv2
import numpy as np
import os

app = FastAPI()

# Проверка наличия модели
MODEL_PATH = "best.pt"
if not os.path.isfile(MODEL_PATH):
    raise FileNotFoundError(f"Модель не найдена по пути: {MODEL_PATH}")

# Загрузка модели
try:
    model = YOLO(MODEL_PATH)
except Exception as e:
    raise RuntimeError(f"Ошибка загрузки модели: {e}")

@app.post("/get_detected_json")
async def get_detected_json(file: UploadFile = File(...)):
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Файл должен быть изображением.")

    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")
    except UnidentifiedImageError:
        raise HTTPException(status_code=400, detail="Не удалось распознать изображение.")

    results = model.predict(image)[0]
    boxes = results.boxes.xyxy.cpu().numpy()
    confs = results.boxes.conf.cpu().numpy()

    detections = [
        {"x1": float(x1), "y1": float(y1), "x2": float(x2), "y2": float(y2), "confidence": float(conf)}
        for (x1, y1, x2, y2), conf in zip(boxes, confs)
    ]
    return JSONResponse(content={"detections": detections})

@app.post("/get_detected_image")
async def get_detected_image(file: UploadFile = File(...)):
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Файл должен быть изображением.")

    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")
    except UnidentifiedImageError:
        raise HTTPException(status_code=400, detail="Не удалось распознать изображение.")

    results = model.predict(image)[0]
    boxes = results.boxes.xyxy.cpu().numpy()
    confs = results.boxes.conf.cpu().numpy()

    # Отрисовка
    image_np = np.array(image)
    for (x1, y1, x2, y2), conf in zip(boxes, confs):
        label = f"{conf:.2f}"
        cv2.rectangle(image_np, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
        cv2.putText(image_np, label, (int(x1), int(y1)-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)

    _, encoded_img = cv2.imencode(".jpg", cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR))
    image_bytes = encoded_img.tobytes()

    detections = [
        {"x1": float(x1), "y1": float(y1), "x2": float(x2), "y2": float(y2), "confidence": float(conf)}
        for (x1, y1, x2, y2), conf in zip(boxes, confs)
    ]

    return StreamingResponse(
        content=io.BytesIO(image_bytes),
        media_type="image/jpeg",
        headers={"X-Detections": str(detections)}  # можно доставать отдельно в клиенте
    )
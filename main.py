from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.openapi.models import Example
from fastapi.responses import Response
from PIL import Image, UnidentifiedImageError
import io
import torch
from ultralytics import YOLO
import cv2
import numpy as np
import os
import base64
import asyncio
from config import MODEL_PATH

app = FastAPI()

# Проверка наличия модели
if not os.path.isfile(MODEL_PATH):
    raise FileNotFoundError(f"Модель не найдена по пути: {MODEL_PATH}")

# Загрузка модели
try:
    model = YOLO(MODEL_PATH)
except Exception as e:
    raise RuntimeError(f"Ошибка загрузки модели: {e}")

@app.post("/get_detected_json",
          summary="Получить координаты объектов",
          response_description="JSON с координатами",
          responses={
              200: {
                  "content": {
                      "application/json": {
                          "example": {
                              "detections": [
                                  {"x1": 123.0, "y1": 456.0, "x2": 789.0, "y2": 999.0, "confidence": 0.87}
                              ]
                          }
                      }
                  }
              }
          })
async def get_detected_json(file: UploadFile = File(..., description="jpg изображение, на котором нужно найти объекты")):
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Файл должен быть изображением.")

    try:
        contents = await file.read()
        image = await asyncio.to_thread(lambda: Image.open(io.BytesIO(contents)).convert("RGB"))
    except UnidentifiedImageError:
        raise HTTPException(status_code=400, detail="Не удалось распознать изображение.")

    results = model.predict(image)[0]
    boxes = results.boxes.xyxy.cpu().numpy()
    confs = results.boxes.conf.cpu().numpy()

    detections = [
        {"x1": float(x1), "y1": float(y1), "x2": float(x2), "y2": float(y2), "confidence": float(conf)}
        for (x1, y1, x2, y2), conf in zip(boxes, confs)
    ]

    if not detections:
        return JSONResponse(content={"message": "Объекты не найдены", "detections": []})

    return JSONResponse(content={"detections": detections})

@app.post("/get_detected_image",
          summary="Получить изображение с отмеченными объектами",
          response_description="JPEG с разметкой и координатами в заголовке",
          responses={
              200: {
                  "content": {
                      "image/jpeg": {
                          "example": "<binary JPEG изображение>"
                      }
                  },
                  "headers": {
                      "detections": {
                          "description": "Список координат в виде JSON",
                          "schema": {
                              "type": "string"
                          }
                      }
                  }
              }
          })
async def get_detected_image(file: UploadFile = File(..., description="jpg изображение, на котором нужно найти объекты")):
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Файл должен быть изображением.")

    try:
        contents = await file.read()
        image = await asyncio.to_thread(lambda: Image.open(io.BytesIO(contents)).convert("RGB"))
    except UnidentifiedImageError:
        raise HTTPException(status_code=400, detail="Не удалось распознать изображение.")

    results = model.predict(image)[0]
    boxes = results.boxes.xyxy.cpu().numpy()
    confs = results.boxes.conf.cpu().numpy()

    image_np = np.array(image)
    detections = []
    for (x1, y1, x2, y2), conf in zip(boxes, confs):
        label = f"{conf:.2f}"
        cv2.rectangle(image_np, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
        cv2.putText(image_np, label, (int(x1), int(y1)-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)
        detections.append({"x1": float(x1), "y1": float(y1), "x2": float(x2), "y2": float(y2), "confidence": float(conf)})

    if not detections:
        return JSONResponse(content={"message": "Объекты не найдены", "detections": []})

    _, encoded_img = cv2.imencode(".jpg", cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR))
    image_bytes = encoded_img.tobytes()

    return StreamingResponse(
        content=io.BytesIO(image_bytes),
        media_type="image/jpeg",
        headers={"detections": str(detections)}
    )

@app.post("/get_detected_base64",
          summary="Получить JSON и изображение в base64",
          response_description="JSON с координатами и base64 изображением",
          responses={
              200: {
                  "content": {
                      "application/json": {
                          "example": {
                              "detections": [
                                  {"x1": 100.0, "y1": 200.0, "x2": 300.0, "y2": 400.0, "confidence": 0.91}
                              ],
                              "image_base64": "<base64 encoded JPEG>"
                          }
                      }
                  }
              }
          })
async def get_detected_full(file: UploadFile = File(..., description="jpg изображение, на котором нужно найти объекты")):
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Файл должен быть изображением.")

    try:
        contents = await file.read()
        image = await asyncio.to_thread(lambda: Image.open(io.BytesIO(contents)).convert("RGB"))
    except UnidentifiedImageError:
        raise HTTPException(status_code=400, detail="Не удалось распознать изображение.")

    results = model.predict(image)[0]
    boxes = results.boxes.xyxy.cpu().numpy()
    confs = results.boxes.conf.cpu().numpy()

    image_np = np.array(image)
    detections = []
    for (x1, y1, x2, y2), conf in zip(boxes, confs):
        label = f"{conf:.2f}"
        cv2.rectangle(image_np, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
        cv2.putText(image_np, label, (int(x1), int(y1)-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)
        detections.append({"x1": float(x1), "y1": float(y1), "x2": float(x2), "y2": float(y2), "confidence": float(conf)})

    _, encoded_img = cv2.imencode(".jpg", cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR))
    image_bytes = encoded_img.tobytes()
    img_base64 = base64.b64encode(image_bytes).decode("utf-8")

    return {"detections": detections, "image_base64": img_base64}


@app.get("/health", summary="Проверка состояния сервера", response_description="Статус сервера")
async def health_check():
    return {"status": "ok", "message": "Server is up"}


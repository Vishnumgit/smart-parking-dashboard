# /// script
# requires-python = "==3.11.*"
# dependencies = [
#   "codewords-client==0.4.6",
#   "fastapi==0.116.1",
#   "opencv-python-headless==4.10.0.84",
#   "numpy==1.26.4",
#   "httpx==0.28.1",
# ]
# [tool.env-checker]
# env_vars = [
#   "PORT=8000",
#   "LOGLEVEL=INFO",
#   "CODEWORDS_API_KEY",
#   "CODEWORDS_RUNTIME_URI"
# ]
# ///

import asyncio
import time
from datetime import datetime, timezone
from typing import Optional

import cv2
import httpx
import numpy as np
from codewords_client import AsyncCodewordsClient, logger, run_service
from fastapi import FastAPI
from pydantic import BaseModel, Field


# ---------- FastAPI Application ----------
app = FastAPI(
    title="Smart Parking Analyzer",
    description="Analyzes parking lot camera feeds to detect occupied/free spaces with green/red overlays, motion detection, and availability prediction.",
    version="1.0.0",
)


# ---------- Models ----------
class ParkingSpace(BaseModel):
    id: str = Field(..., description="Unique space identifier like A1, B2")
    x: int = Field(..., description="Top-left X coordinate")
    y: int = Field(..., description="Top-left Y coordinate")
    w: int = Field(..., description="Width of parking space")
    h: int = Field(..., description="Height of parking space")


class AnalysisRequest(BaseModel):
    """Request model for parking analysis."""
    camera_image: str = Field(
        ...,
        description="Parking lot camera image to analyze",
        json_schema_extra={"contentMediaType": "image/*"},
    )
    parking_spaces: Optional[list[ParkingSpace]] = Field(
        default=None,
        description="Manually defined parking space zones. If empty, auto-detection is used."
    )
    edge_threshold: float = Field(
        default=0.06,
        description="Edge density threshold for occupancy detection (higher = stricter)",
        ge=0.01, le=0.5
    )
    std_threshold: float = Field(
        default=30.0,
        description="Standard deviation threshold for occupancy detection",
        ge=5.0, le=100.0
    )
    previous_frame: Optional[str] = Field(
        default=None,
        description="Previous camera frame for motion detection comparison",
        json_schema_extra={"contentMediaType": "image/*"},
    )


class SpaceStatus(BaseModel):
    id: str
    occupied: bool
    confidence: float
    edge_density: float
    std_dev: float


class MotionZone(BaseModel):
    x: int
    y: int
    w: int
    h: int
    intensity: float


class PredictionData(BaseModel):
    current_occupancy_pct: float
    trend: str
    predicted_available_in_5min: int
    recommendation: str


class AnalysisResponse(BaseModel):
    """Response model with full parking analysis."""
    annotated_image_url: str = Field(..., description="URL to annotated parking image with green/red boxes")
    total_spaces: int
    occupied_spaces: int
    available_spaces: int
    occupancy_percentage: float
    spaces: list[SpaceStatus]
    motion_detected: bool
    motion_zones: list[MotionZone]
    prediction: PredictionData
    analysis_timestamp: str
    processing_time_ms: int


# ---------- Helper Functions ----------

async def download_image(url: str) -> np.ndarray:
    """Download image from URL and return as OpenCV array."""
    async with httpx.AsyncClient(timeout=30.0, follow_redirects=True) as client:
        resp = await client.get(url)
        resp.raise_for_status()
        img_array = np.frombuffer(resp.content, np.uint8)
        img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError("Could not decode image from URL")
        return img


def auto_detect_spaces(img: np.ndarray) -> list[dict]:
    """Auto-detect parking space grid from image dimensions."""
    h, w = img.shape[:2]
    space_w = max(60, w // 14)
    space_h = max(120, h // 5)
    gap = max(5, space_w // 10)
    margin_x = max(20, w // 20)
    margin_y = max(80, h // 8)
    cols = min(12, (w - 2 * margin_x) // (space_w + gap))
    rows = min(3, (h - 2 * margin_y) // (space_h + gap + 20))
    row_labels = "ABCDEFGHIJ"
    spaces = []
    for r in range(rows):
        for c in range(cols):
            x = margin_x + c * (space_w + gap)
            y = margin_y + r * (space_h + gap + 20)
            if x + space_w <= w and y + space_h <= h:
                spaces.append({"id": f"{row_labels[r]}{c+1}", "x": x, "y": y, "w": space_w, "h": space_h})
    return spaces


def analyze_space(gray: np.ndarray, space: dict, edge_thresh: float, std_thresh: float) -> dict:
    """Analyze a single parking space for occupancy using edge detection + pixel variance."""
    x, y, w, h = space["x"], space["y"], space["w"], space["h"]
    img_h, img_w = gray.shape[:2]
    x = max(0, min(x, img_w - 1))
    y = max(0, min(y, img_h - 1))
    w = min(w, img_w - x)
    h = min(h, img_h - y)
    roi = gray[y:y+h, x:x+w]
    if roi.size == 0:
        return {"id": space["id"], "occupied": False, "confidence": 0.0, "edge_density": 0.0, "std_dev": 0.0}
    edges = cv2.Canny(roi, 50, 150)
    edge_density = float(np.sum(edges > 0) / max(1, w * h))
    std_dev = float(np.std(roi))
    is_occupied = edge_density > edge_thresh or std_dev > std_thresh
    confidence = min(1.0, max(
        (edge_density / edge_thresh) if edge_density > edge_thresh else 0,
        (std_dev / std_thresh) if std_dev > std_thresh else 0
    ) * 0.7)
    if is_occupied:
        confidence = max(0.5, confidence)
    return {
        "id": space["id"], "occupied": is_occupied,
        "confidence": round(confidence, 2), "edge_density": round(edge_density, 4), "std_dev": round(std_dev, 2)
    }


def draw_overlays(img: np.ndarray, spaces: list[dict], results: list[dict]) -> np.ndarray:
    """Draw green (free) / red (occupied) bounding boxes on the image."""
    annotated = img.copy()
    result_map = {r["id"]: r for r in results}
    for space in spaces:
        r = result_map.get(space["id"], {})
        is_occ = r.get("occupied", False)
        color = (0, 0, 255) if is_occ else (0, 255, 0)
        x, y, w, h = space["x"], space["y"], space["w"], space["h"]
        cv2.rectangle(annotated, (x, y), (x + w, y + h), color, 3)
        overlay = annotated.copy()
        fill_color = (0, 0, 180) if is_occ else (0, 180, 0)
        cv2.rectangle(overlay, (x, y), (x + w, y + h), fill_color, -1)
        cv2.addWeighted(overlay, 0.15, annotated, 0.85, 0, annotated)
        label = f"{space['id']}"
        status = "OCC" if is_occ else "FREE"
        cv2.putText(annotated, label, (x + 4, y + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        cv2.putText(annotated, status, (x + 4, y + 36), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
    return annotated


def draw_header(img: np.ndarray, total: int, occupied: int, available: int) -> np.ndarray:
    """Draw stats header bar on the image."""
    h, w = img.shape[:2]
    cv2.rectangle(img, (0, 0), (w, 55), (20, 20, 20), -1)
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    text = f"SMART PARKING | Total: {total} | Free: {available} | Occupied: {occupied} | {ts}"
    cv2.putText(img, text, (15, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    pct = (occupied / total * 100) if total > 0 else 0
    bar_w = int((w - 30) * pct / 100)
    cv2.rectangle(img, (15, 48), (w - 15, 53), (60, 60, 60), -1)
    bar_color = (0, 255, 0) if pct < 60 else (0, 200, 255) if pct < 85 else (0, 0, 255)
    cv2.rectangle(img, (15, 48), (15 + bar_w, 53), bar_color, -1)
    return img


def detect_motion(current: np.ndarray, previous: np.ndarray) -> tuple[bool, list[dict]]:
    """Detect motion between two frames using frame differencing."""
    gray1 = cv2.cvtColor(previous, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(current, cv2.COLOR_BGR2GRAY)
    gray1 = cv2.GaussianBlur(gray1, (21, 21), 0)
    gray2 = cv2.GaussianBlur(gray2, (21, 21), 0)
    diff = cv2.absdiff(gray1, gray2)
    thresh = cv2.threshold(diff, 30, 255, cv2.THRESH_BINARY)[1]
    thresh = cv2.dilate(thresh, None, iterations=3)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    motion_zones = []
    for c in contours:
        area = cv2.contourArea(c)
        if area > 500:
            bx, by, bw, bh = cv2.boundingRect(c)
            intensity = min(1.0, area / 5000)
            motion_zones.append({"x": int(bx), "y": int(by), "w": int(bw), "h": int(bh), "intensity": round(intensity, 2)})
    return len(motion_zones) > 0, motion_zones


def generate_prediction(occupied: int, total: int) -> dict:
    """Generate simple parking availability prediction."""
    pct = (occupied / total * 100) if total > 0 else 0
    if pct < 50:
        return {"current_occupancy_pct": round(pct, 1), "trend": "low_demand",
                "predicted_available_in_5min": max(0, total - occupied - 1),
                "recommendation": "Plenty of parking available. No rush needed."}
    elif pct < 75:
        return {"current_occupancy_pct": round(pct, 1), "trend": "moderate_demand",
                "predicted_available_in_5min": max(0, total - occupied - 2),
                "recommendation": "Moderate occupancy. Spaces filling up gradually."}
    elif pct < 90:
        return {"current_occupancy_pct": round(pct, 1), "trend": "high_demand",
                "predicted_available_in_5min": max(0, total - occupied - 3),
                "recommendation": "High demand! Consider arriving soon for best selection."}
    else:
        return {"current_occupancy_pct": round(pct, 1), "trend": "near_full",
                "predicted_available_in_5min": max(0, (total - occupied) - 1),
                "recommendation": "Nearly full! Very few spaces remain. Check alternative lots."}


async def upload_image(img: np.ndarray, filename: str) -> str:
    """Encode image and upload to CodeWords storage."""
    _, buffer = cv2.imencode(".png", img)
    async with AsyncCodewordsClient() as client:
        url = await client.upload_file_content(filename=filename, file_content=buffer.tobytes())
    return url


# ---------- Main Endpoint ----------
@app.post("/", response_model=AnalysisResponse)
async def analyze_parking(request: AnalysisRequest):
    """Analyze a parking lot camera feed for occupancy, motion, and predictions."""
    start = time.time()
    logger.info("STEPLOG START capture_frame")
    logger.info("Starting parking analysis", camera_image=request.camera_image)

    # Step 1: Download camera frame
    current_frame = await download_image(request.camera_image)
    logger.info("Frame downloaded", shape=str(current_frame.shape))

    # Step 2: Determine parking spaces
    logger.info("STEPLOG START detect_spaces")
    if request.parking_spaces and len(request.parking_spaces) > 0:
        spaces = [s.model_dump() for s in request.parking_spaces]
        logger.info("Using manually defined spaces", count=len(spaces))
    else:
        spaces = await asyncio.to_thread(auto_detect_spaces, current_frame)
        logger.info("Auto-detected spaces", count=len(spaces))

    # Step 3: Analyze each space
    logger.info("STEPLOG START analyze_spaces")
    def _analyze_all_spaces():
        gray = cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY)
        return [analyze_space(gray, space, request.edge_threshold, request.std_threshold) for space in spaces]

    results = await asyncio.to_thread(_analyze_all_spaces)
    total = len(results)
    occupied = sum(1 for r in results if r["occupied"])
    available = total - occupied
    logger.info("Space analysis complete", total=total, occupied=occupied, available=available)

    # Step 4: Motion detection
    logger.info("STEPLOG START detect_motion")
    motion_detected = False
    motion_zones: list[dict] = []
    if request.previous_frame:
        try:
            prev_frame = await download_image(request.previous_frame)
            if prev_frame.shape == current_frame.shape:
                motion_detected, motion_zones = await asyncio.to_thread(detect_motion, current_frame, prev_frame)
                logger.info("Motion detection complete", detected=motion_detected, zones=len(motion_zones))
        except (httpx.HTTPStatusError, httpx.RequestError, ValueError) as e:
            logger.warning("Motion detection skipped due to frame retrieval issue", error=str(e))

    # Step 5: Draw overlays
    logger.info("STEPLOG START draw_overlays")
    def _draw_all_overlays():
        annotated = draw_overlays(current_frame, spaces, results)
        if motion_detected:
            for mz in motion_zones:
                cv2.rectangle(annotated, (mz["x"], mz["y"]), (mz["x"] + mz["w"], mz["y"] + mz["h"]), (255, 165, 0), 2)
                cv2.putText(annotated, "MOTION", (mz["x"], mz["y"] - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 165, 0), 1)
        return draw_header(annotated, total, occupied, available)

    annotated = await asyncio.to_thread(_draw_all_overlays)

    # Step 6: Upload annotated image
    logger.info("STEPLOG START upload_result")
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    image_url = await upload_image(annotated, f"parking_analysis_{ts}.png")
    logger.info("Annotated image uploaded", url=image_url)

    # Step 7: Generate predictions
    logger.info("STEPLOG START predict")
    prediction = generate_prediction(occupied, total)
    elapsed_ms = int((time.time() - start) * 1000)

    return AnalysisResponse(
        annotated_image_url=image_url, total_spaces=total,
        occupied_spaces=occupied, available_spaces=available,
        occupancy_percentage=round(occupied / total * 100, 1) if total > 0 else 0.0,
        spaces=[SpaceStatus(**r) for r in results],
        motion_detected=motion_detected, motion_zones=[MotionZone(**mz) for mz in motion_zones],
        prediction=PredictionData(**prediction),
        analysis_timestamp=datetime.now(timezone.utc).isoformat(),
        processing_time_ms=elapsed_ms
    )


if __name__ == "__main__":
    run_service(app)

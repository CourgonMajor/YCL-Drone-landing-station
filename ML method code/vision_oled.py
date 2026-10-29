"""Top-down drone-to-pad guidance. Displays operator guidance only; never controls a drone."""
import argparse
import json
import math
import platform
import time
from collections import deque

import cv2
import numpy as np
import serial
from ultralytics import YOLO


# The supplied printable target is 300 mm across. Change only if you print a
# different target size. Camera must be mounted straight down and remain fixed.
TARGET_DIAMETER_MM = 300.0
DRONE_CLASS_ID = 0  # Set to the class ID used for your trained drone model.
ARRIVAL_TOLERANCE_MM = 20.0
MAX_GUIDANCE_MM = 999


def ring_level(gray, x, y, radius, fraction):
    vals = []
    for angle in range(0, 360, 10):
        a = math.radians(angle)
        px = round(x + radius * fraction * math.cos(a))
        py = round(y + radius * fraction * math.sin(a))
        if 0 <= px < gray.shape[1] and 0 <= py < gray.shape[0]:
            vals.append(gray[py, px])
    return float(np.median(vals)) if len(vals) >= 24 else None


def find_bullseye(frame):
    """Return (center, outer_radius, score) for the strongest concentric target."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.medianBlur(gray, 5)
    h, w = gray.shape
    circles = cv2.HoughCircles(
        gray, cv2.HOUGH_GRADIENT, dp=1.2, minDist=max(30, min(w, h) // 5),
        param1=100, param2=30, minRadius=max(20, min(w, h) // 12),
        maxRadius=max(21, min(w, h) // 2),
    )
    if circles is None:
        return None
    best = None
    best_score = -1.0
    for cx_f, cy_f, radius_f in circles[0]:
        cx, cy, radius = int(cx_f), int(cy_f), int(radius_f)
        # This target has alternating white/black rings, center outward.
        levels = [ring_level(gray, cx, cy, radius, f) for f in (0.25, 0.48, 0.70, 0.91)]
        if any(v is None for v in levels):
            continue
        white_a, black_a, white_b, black_b = levels
        contrast = min(white_a - black_a, white_b - black_b)
        if contrast < 28 or white_a < black_a + 20 or white_b < black_b + 20:
            continue
        score = contrast + abs((white_a + white_b) / 2 - (black_a + black_b) / 2) * 0.2
        if score > best_score:
            best_score = score
            best = ((cx_f, cy_f), float(radius_f), score)
    return best


def open_camera(index):
    # CAP_ANY selects the platform's normal backend; this avoids forcing DSHOW
    # on Windows and works with OpenCV's available macOS backends.
    cap = cv2.VideoCapture(index, cv2.CAP_ANY)
    if not cap.isOpened():
        cap.release()
        raise RuntimeError(
            f"Could not open camera {index}. Close other camera apps, allow camera access, "
            "and try --camera 1 (or another index)."
        )
    return cap


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--weights", required=True, help="Path to your trained YOLO model file, e.g. best.pt")
    p.add_argument("--port", required=True, help="Arduino Nano USB serial port, e.g. COM5 or /dev/cu.wchusbserialXXXX")
    p.add_argument("--camera", type=int, default=0)
    p.add_argument("--confidence", type=float, default=0.65)
    p.add_argument("--deadband-mm", type=float, default=ARRIVAL_TOLERANCE_MM)
    args = p.parse_args()

    model = YOLO(args.weights)
    cap = open_camera(args.camera)
    link = None
    history = deque(maxlen=5)
    last_send = 0.0
    try:
        link = serial.Serial(args.port, 115200, timeout=0.02, write_timeout=0.2)
        time.sleep(2)  # Nano resets when its USB serial port opens.
        while True:
            ok, frame = cap.read()
            if not ok:
                raise RuntimeError("Camera stopped returning images")
            target = find_bullseye(frame)
            result = model.predict(frame, verbose=False, conf=args.confidence)[0]
            drone = None
            if result.boxes is not None and len(result.boxes):
                candidates = []
                for box in result.boxes:
                    cls = int(box.cls.item())
                    conf = float(box.conf.item())
                    if cls == DRONE_CLASS_ID and conf >= args.confidence:
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                        candidates.append((conf, ((x1 + x2) / 2, (y1 + y2) / 2), (x1, y1, x2, y2)))
                if candidates:
                    _, drone, bounds = max(candidates, key=lambda item: item[0])
                    x1, y1, x2, y2 = map(int, bounds)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 220, 0), 2)

            guidance = "HOLD: FIND TARGET"
            payload = "HOLD"
            if target:
                (tx, ty), radius, score = target
                cv2.circle(frame, (round(tx), round(ty)), round(radius), (0, 255, 255), 2)
                cv2.circle(frame, (round(tx), round(ty)), 5, (0, 255, 255), -1)
                if drone is not None:
                    dx_mm = (drone[0] - tx) * TARGET_DIAMETER_MM / (2 * radius)
                    dy_mm = (drone[1] - ty) * TARGET_DIAMETER_MM / (2 * radius)
                    history.append((dx_mm, dy_mm))
                    sx, sy = np.median(np.array(history), axis=0)
                    # Directions tell the pilot which way to move the drone.
                    move_x = -sx
                    move_y = -sy
                    if abs(sx) <= args.deadband_mm and abs(sy) <= args.deadband_mm:
                        guidance, payload = "CENTERED - HOLD", "CENTERED"
                    else:
                        parts = []
                        if abs(sx) > args.deadband_mm:
                            parts.append(f"{'LEFT' if move_x < 0 else 'RIGHT'} {min(MAX_GUIDANCE_MM, round(abs(sx)))} mm")
                        if abs(sy) > args.deadband_mm:
                            parts.append(f"{'UP' if move_y < 0 else 'DOWN'} {min(MAX_GUIDANCE_MM, round(abs(sy)))} mm")
                        guidance = "MOVE " + " / ".join(parts)
                        # Compact line protocol understood by the Arduino sketch.
                        payload = f"GUIDE|{parts[0].replace(' ', ':')}|{parts[1].replace(' ', ':') if len(parts) > 1 else '-'}"
                    cv2.circle(frame, (round(drone[0]), round(drone[1])), 5, (0, 220, 0), -1)
                    cv2.arrowedLine(frame, (round(tx), round(ty)), (round(drone[0]), round(drone[1])), (255, 120, 0), 2)
                    cv2.putText(frame, f"Offset from pad: {sx:.0f}, {sy:.0f} mm", (15, 62), cv2.FONT_HERSHEY_SIMPLEX, .6, (255, 255, 255), 2)
                else:
                    history.clear()
                    guidance, payload = "HOLD: DRONE NOT FOUND", "HOLD"
            else:
                history.clear()

            cv2.putText(frame, guidance, (15, 30), cv2.FONT_HERSHEY_SIMPLEX, .65, (0, 255, 0), 2)
            cv2.imshow("SkyDock guidance only - press Q to quit", frame)
            now = time.monotonic()
            if now - last_send >= 0.15:
                link.write((payload + "\n").encode("ascii"))
                last_send = now
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        cap.release()
        if link:
            link.write(b"HOLD\n")
            link.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

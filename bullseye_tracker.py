import cv2
import numpy as np
import math

camera = cv2.VideoCapture(0)

if not camera.isOpened():
    print("Could not open the camera.")
    raise SystemExit


def ring_brightness(gray, cx, cy, radius, fraction):
    """Sample image brightness around one ring inside the bullseye."""
    values = []

    for angle in range(0, 360, 15):
        radians = math.radians(angle)
        x = int(cx + radius * fraction * math.cos(radians))
        y = int(cy + radius * fraction * math.sin(radians))

        if 0 <= x < gray.shape[1] and 0 <= y < gray.shape[0]:
            values.append(gray[y, x])

    return float(np.median(values)) if values else 0


while True:
    success, frame = camera.read()

    if not success:
        print("Could not read from the camera.")
        break

    height, width = frame.shape[:2]
    target_x = width // 2
    target_y = height // 2

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.medianBlur(gray, 5)

    circles = cv2.HoughCircles(
        gray,
        cv2.HOUGH_GRADIENT,
        dp=1.2,
        minDist=35,
        param1=100,
        param2=25,
        minRadius=12,
        maxRadius=max(13, min(width, height) // 3),
    )

    best_marker = None
    best_contrast = 0

    if circles is not None:
        for circle in circles[0]:
            cx, cy, radius = [int(value) for value in circle]

            # The printed pattern from center outward is:
            # white, black, white, black.
            samples = [
                ring_brightness(gray, cx, cy, radius, fraction)
                for fraction in (0.15, 0.42, 0.67, 0.90)
            ]

            white_level = (samples[0] + samples[2]) / 2
            black_level = (samples[1] + samples[3]) / 2
            contrast = white_level - black_level

            # Keep candidates with the expected alternating rings.
            if (
                contrast > best_contrast
                and samples[0] > samples[1] + 15
                and samples[2] > samples[1] + 15
                and samples[2] > samples[3] + 15
                and contrast > 35
            ):
                best_marker = (cx, cy, radius)
                best_contrast = contrast

    cv2.drawMarker(
        frame,
        (target_x, target_y),
        (255, 255, 0),
        cv2.MARKER_CROSS,
        30,
        2,
    )

    if best_marker:
        dot_x, dot_y, radius = best_marker
        offset_x = dot_x - target_x
        offset_y = dot_y - target_y

        cv2.circle(frame, (dot_x, dot_y), radius, (0, 255, 0), 2)
        cv2.circle(frame, (dot_x, dot_y), 4, (0, 255, 0), -1)

        if abs(offset_x) < 20 and abs(offset_y) < 20:
            message = "PATTERN CENTERED"
        else:
            horizontal = "RIGHT" if offset_x > 0 else "LEFT"
            vertical = "DOWN IN IMAGE" if offset_y > 0 else "UP IN IMAGE"

            if abs(offset_x) >= 20 and abs(offset_y) >= 20:
                message = f"{vertical}, {horizontal}"
            elif abs(offset_x) >= 20:
                message = horizontal
            else:
                message = vertical

        cv2.putText(
            frame, message, (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2
        )
        cv2.putText(
            frame, f"Offset: x={offset_x}px, y={offset_y}px",
            (20, 75), cv2.FONT_HERSHEY_SIMPLEX,
            0.6, (255, 255, 255), 2
        )
    else:
        cv2.putText(
            frame, "Bullseye not found", (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2
        )

    cv2.imshow("SkyDock bullseye test — press Q to close", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

camera.release()
cv2.destroyAllWindows()

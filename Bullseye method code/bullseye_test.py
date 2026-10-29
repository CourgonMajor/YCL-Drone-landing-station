"""Test the pad bullseye, small drone bullseye, HC-SR04, and Nano switches."""
import argparse
import json
import math
import time

import cv2
import numpy as np
import serial


PAD_DIAMETER_MM = 300.0
CENTER_TOLERANCE_MM = 20.0
SWITCH_PINS = (2, 3, 4, 5)


def sample_ring(gray, cx, cy, radius, fraction):
    values = []
    for angle in range(0, 360, 10):
        a = math.radians(angle)
        x = round(cx + radius * fraction * math.cos(a))
        y = round(cy + radius * fraction * math.sin(a))
        if 0 <= x < gray.shape[1] and 0 <= y < gray.shape[0]:
            values.append(gray[y, x])
    return float(np.median(values)) if len(values) >= 24 else None


def ring_contrast(gray, cx, cy, radius):
    levels = [sample_ring(gray, cx, cy, radius, f) for f in (0.25, 0.48, 0.70, 0.91)]
    if any(value is None for value in levels):
        return None
    a, b, c, d = levels
    return min(a - b, c - b, c - d)


def refine_marker_center(gray, cx, cy, radius):
    """Shift a Hough center slightly to maximize the bullseye ring contrast."""
    best_x, best_y = float(cx), float(cy)
    best_score = ring_contrast(gray, best_x, best_y, radius)
    if best_score is None:
        return best_x, best_y

    extent = max(2, round(radius * 0.12))
    step = max(1, round(radius * 0.03))
    for offset_y in range(-extent, extent + 1, step):
        for offset_x in range(-extent, extent + 1, step):
            test_x = cx + offset_x
            test_y = cy + offset_y
            score = ring_contrast(gray, test_x, test_y, radius)
            if score is not None and score > best_score:
                best_x, best_y, best_score = float(test_x), float(test_y), score

    return best_x, best_y


def detect_markers(frame, pad_roi=None, previous_pad=None, previous_drone=None):
    """Find the largest valid pad bullseye and a smaller valid drone bullseye."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.medianBlur(gray, 5)
    h, w = gray.shape
    if previous_pad is not None:
        # Pad is stationary: keep its original center and avoid a costly full
        # frame Hough scan on every camera frame.
        pad = previous_pad
    else:
        minimum = max(5, min(h, w) // 160)
        circles = cv2.HoughCircles(
            gray, cv2.HOUGH_GRADIENT, dp=1.2,
            minDist=max(8, min(h, w) // 70), param1=100, param2=24,
            minRadius=minimum, maxRadius=max(minimum + 1, min(h, w) // 2),
        )
        if circles is None:
            return None, None
        strict, loose_pad = [], []
        for cx, cy, radius in circles[0]:
            levels = [sample_ring(gray, cx, cy, radius, f) for f in (0.25, 0.48, 0.70, 0.91)]
            if any(value is None for value in levels):
                continue
            a, b, c, d = levels
            if min(a - b, c - b, c - d) >= 24:
                strict.append((float(cx), float(cy), float(radius), min(a - b, c - b, c - d)))
            if b < c - 24 and d < c - 24 and c > b + 24:
                loose_pad.append((float(cx), float(cy), float(radius), min(c - b, c - d)))
        if pad_roi is not None:
            roi_x, roi_y, roi_w, roi_h = pad_roi
            strict = [item for item in strict if roi_x <= item[0] <= roi_x + roi_w and roi_y <= item[1] <= roi_y + roi_h]
            loose_pad = [item for item in loose_pad if roi_x <= item[0] <= roi_x + roi_w and roi_y <= item[1] <= roi_y + roi_h]
        pad_candidates = strict + loose_pad
        if not pad_candidates:
            return None, None
        pad = max(pad_candidates, key=lambda item: item[2])
        # Hough's circle center can be a few pixels off. Refine it against
        # the pad's concentric rings before locking the yellow reference.
        pad_x, pad_y = refine_marker_center(gray, pad[0], pad[1], pad[2])
        pad = (pad_x, pad_y, pad[2], pad[3])

    # Search only inside the pad neighborhood for the small 50 mm target.
    # This cuts Hough work substantially and excludes unrelated objects.
    pad_x, pad_y, pad_radius, _ = pad
    margin = int(pad_radius * 1.02)
    left = max(0, int(pad_x) - margin)
    top = max(0, int(pad_y) - margin)
    right = min(w, int(pad_x) + margin + 1)
    bottom = min(h, int(pad_y) + margin + 1)
    search_gray = gray[top:bottom, left:right]
    drone_min = max(4, int(pad_radius * 0.08))
    drone_max = max(drone_min + 1, int(pad_radius * 0.32))
    small_circles = cv2.HoughCircles(
        search_gray,
        cv2.HOUGH_GRADIENT,
        dp=1.2,
        minDist=max(6, min(search_gray.shape) // 100),
        param1=80,
        param2=17,
        minRadius=drone_min,
        maxRadius=drone_max,
    )
    drone_candidates = []
    if small_circles is not None:
        for local_x, local_y, radius in small_circles[0]:
            cx, cy = local_x + left, local_y + top
            ratio = radius / pad[2]
            if not 0.08 <= ratio <= 0.30:
                continue
            # The small marker should be over the pad, not another object
            # elsewhere in the camera view.
            if math.hypot(cx - pad[0], cy - pad[1]) > pad[2] * 0.95:
                continue
            levels = [sample_ring(gray, cx, cy, radius, f) for f in (0.25, 0.48, 0.70, 0.91)]
            if any(value is None for value in levels):
                continue
            a, b, c, d = levels
            contrast = min(a - b, c - b, c - d)
            if contrast >= 18:
                drone_candidates.append((float(cx), float(cy), float(radius), contrast))

    if previous_drone is not None:
        old_x, old_y, old_radius, _ = previous_drone
        nearby = [
            item for item in drone_candidates
            if math.hypot(item[0] - old_x, item[1] - old_y) <= max(32.0, old_radius * 4.0)
        ]
    else:
        nearby = []

    candidates_to_rank = nearby if nearby else drone_candidates
    drone = None
    if candidates_to_rank:
        selected = min(
            candidates_to_rank,
            key=lambda item: (
                math.hypot(item[0] - previous_drone[0], item[1] - previous_drone[1]) / max(1.0, previous_drone[2])
                if nearby and previous_drone is not None
                else 0.0,
                abs(item[2] / pad[2] - (50.0 / PAD_DIAMETER_MM)),
                -item[3],
            ),
        )
        refined_x, refined_y = refine_marker_center(
            gray, selected[0], selected[1], selected[2]
        )
        drone = (refined_x, refined_y, selected[2], selected[3])

    return pad, drone


def open_camera(index):
    cap = cv2.VideoCapture(index, cv2.CAP_ANY)
    if not cap.isOpened():
        cap.release()
        raise RuntimeError(
            f"Could not open camera {index}. Close other camera apps, allow camera access, "
            "and try --camera 1 or --camera 2."
        )
    # Smaller frames make Hough circle detection much faster on high-resolution
    # Mac webcams while retaining enough pixels for a 50 mm printed marker.
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 960)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 540)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    return cap


def draw_sensor_panel(frame, reading, fresh):
    """Draw a high-contrast dashboard for distance and the four switches."""
    height, width = frame.shape[:2]
    top = max(0, height - 150)
    overlay = frame.copy()
    cv2.rectangle(overlay, (10, top), (width - 10, height - 10), (18, 24, 30), -1)
    cv2.addWeighted(overlay, 0.86, frame, 0.14, 0, frame)

    white = (245, 245, 245)
    muted = (175, 185, 190)
    cv2.putText(frame, "ULTRASONIC", (24, top + 28), cv2.FONT_HERSHEY_SIMPLEX, 0.52, muted, 1, cv2.LINE_AA)
    if not fresh:
        distance_text, distance_color = "NO DATA", (60, 80, 255)
    elif reading.get("distance_cm") is None:
        distance_text, distance_color = "NO ECHO", (0, 190, 255)
    else:
        distance_text = f"{float(reading['distance_cm']):.1f} cm"
        distance_color = (100, 235, 120)
    cv2.putText(frame, distance_text, (24, top + 78), cv2.FONT_HERSHEY_SIMPLEX, 0.9, distance_color, 2, cv2.LINE_AA)
    grid_x = max(190, int(width * 0.31))
    cv2.putText(frame, "SWITCHES  (green = pressed)", (grid_x, top + 28), cv2.FONT_HERSHEY_SIMPLEX, 0.48, muted, 1, cv2.LINE_AA)

    grid_y = top + 38
    gap_x = 8
    cell_w = max(48, (width - grid_x - 24 - 3 * gap_x) // 4)
    cell_h = 54
    for index, pin in enumerate(SWITCH_PINS):
        x = grid_x + index * (cell_w + gap_x)
        y = grid_y + 8
        pressed = bool(fresh and reading.get(f"sw{pin}", 0))
        fill = (38, 105, 58) if pressed else (55, 63, 69)
        cv2.rectangle(frame, (x, y), (x + cell_w, y + cell_h), fill, -1)
        cv2.putText(frame, f"D{pin}", (x + 8, y + 17), cv2.FONT_HERSHEY_SIMPLEX, 0.42, white, 1, cv2.LINE_AA)
        state = "PRESSED" if pressed else ("open" if fresh else "--")
        state_color = (150, 255, 165) if pressed else white
        cv2.putText(frame, state, (x + 8, y + 35), cv2.FONT_HERSHEY_SIMPLEX, 0.40, state_color, 1, cv2.LINE_AA)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", required=True, help="Nano port: COM5 or /dev/cu.wchusbserial...")
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--deadband-mm", type=float, default=CENTER_TOLERANCE_MM)
    args = parser.parse_args()

    cap = open_camera(args.camera)
    nano = None
    last_reading = None
    last_reading_time = 0.0
    last_send = 0.0
    last_command = ""
    serial_buffer = bytearray()
    started_at = time.monotonic()
    no_data_reported = False
    last_pad = None
    last_drone = None
    last_drone_time = 0.0
    try:
        nano = serial.Serial(args.port, 115200, timeout=0.01, write_timeout=0.2)
        time.sleep(2.0)  # The Nano resets when its USB serial port opens.
        nano.reset_input_buffer()
        ok, first_frame = cap.read()
        if not ok:
            raise RuntimeError("The camera stopped returning images")
        if first_frame.shape[1] > 960:
            scale = 960.0 / first_frame.shape[1]
            first_frame = cv2.resize(
                first_frame,
                (960, round(first_frame.shape[0] * scale)),
                interpolation=cv2.INTER_AREA,
            )
        selection_window = "Select pad 1: drag a box around the large bullseye, then press ENTER"
        pad_roi = cv2.selectROI(
            selection_window,
            first_frame,
            showCrosshair=True,
            fromCenter=False,
        )
        cv2.destroyWindow(selection_window)
        if pad_roi[2] <= 0 or pad_roi[3] <= 0:
            raise RuntimeError("Pad selection was cancelled. Restart and select pad 1.")
        first_frame_pending = True

        while True:
            if first_frame_pending:
                frame = first_frame
                first_frame_pending = False
            else:
                ok, frame = cap.read()
                if not ok:
                    raise RuntimeError("The camera stopped returning images")

            if frame.shape[1] > 960:
                scale = 960.0 / frame.shape[1]
                frame = cv2.resize(frame, (960, round(frame.shape[0] * scale)), interpolation=cv2.INTER_AREA)

            # Keep up with Nano telemetry while the camera preview is running.
            # Drain all complete telemetry lines so slow image processing
            # cannot leave the reader steadily behind the Nano.
            waiting = nano.in_waiting
            if waiting:
                serial_buffer.extend(nano.read(waiting))
            while b"\n" in serial_buffer:
                raw_line, _, remainder = serial_buffer.partition(b"\n")
                serial_buffer = bytearray(remainder)
                try:
                    parsed = json.loads(raw_line.decode("utf-8", errors="ignore").strip())
                    if isinstance(parsed, dict) and "distance_cm" in parsed:
                        last_reading = parsed
                        last_reading_time = time.monotonic()
                except (ValueError, UnicodeDecodeError):
                    continue
            if len(serial_buffer) > 512:
                serial_buffer.clear()
            if last_reading is None and not no_data_reported and time.monotonic() - started_at > 3.0:
                print(
                    f"No Nano JSON telemetry received on {args.port}. Check that the matching "
                    "bullseye firmware is uploaded, the selected port is the Nano, and Arduino "
                    "Serial Monitor/Plotter is closed."
                )
                no_data_reported = True

            frame_time = time.monotonic()
            tracking_drone = (
                last_drone
                if last_drone is not None and frame_time - last_drone_time <= 0.60
                else None
            )
            pad, detected_drone = detect_markers(
                frame,
                pad_roi,
                last_pad,
                tracking_drone,
            )
            if pad is not None:
                last_pad = pad
            if detected_drone is not None:
                if tracking_drone is not None:
                    # Favor the current frame so quick movement does not
                    # leave the displayed marker trailing behind.
                    smoothing = 0.92
                    detected_drone = (
                        tracking_drone[0] * (1.0 - smoothing) + detected_drone[0] * smoothing,
                        tracking_drone[1] * (1.0 - smoothing) + detected_drone[1] * smoothing,
                        tracking_drone[2] * (1.0 - smoothing) + detected_drone[2] * smoothing,
                        detected_drone[3],
                    )
                last_drone = detected_drone
                last_drone_time = frame_time
                drone = detected_drone
            elif last_drone is not None and frame_time - last_drone_time <= 0.50:
                # Bridge one or two missed Hough detections while the marker
                # remains visible; discard it quickly if it actually leaves.
                drone = last_drone
            else:
                last_drone = None
                drone = None
            command = "HOLD"
            roi_x, roi_y, roi_w, roi_h = pad_roi
            cv2.rectangle(
                frame,
                (roi_x, roi_y),
                (roi_x + roi_w, roi_y + roi_h),
                (0, 165, 255),
                1,
            )
            if pad:
                px, py, pr, _ = pad
                cv2.circle(frame, (round(px), round(py)), round(pr), (0, 255, 255), 2)
                cv2.circle(frame, (round(px), round(py)), 5, (0, 255, 255), -1)
                if drone:
                    dx, dy, dr, _ = drone
                    cv2.circle(frame, (round(dx), round(dy)), round(dr), (0, 220, 0), 2)
                    cv2.circle(frame, (round(dx), round(dy)), 4, (0, 220, 0), -1)
                    off_x = (dx - px) * PAD_DIAMETER_MM / (2.0 * pr)
                    off_y = (dy - py) * PAD_DIAMETER_MM / (2.0 * pr)
                    cv2.arrowedLine(frame, (round(px), round(py)), (round(dx), round(dy)), (255, 120, 0), 2)
                    if abs(off_x) <= args.deadband_mm and abs(off_y) <= args.deadband_mm:
                        status = "CENTERED - HOLD POSITION"
                        command = "CENTERED"
                    else:
                        moves = []
                        if abs(off_x) > args.deadband_mm:
                            moves.append(("LEFT" if off_x > 0 else "RIGHT", round(abs(off_x))))
                        if abs(off_y) > args.deadband_mm:
                            moves.append(("UP" if off_y > 0 else "DOWN", round(abs(off_y))))
                        status = "MOVE DRONE: " + " / ".join(f"{direction} {distance} mm" for direction, distance in moves)
                        x_message = f"{moves[0][0]}:{moves[0][1]}:mm" if moves else "-"
                        y_message = f"{moves[1][0]}:{moves[1][1]}:mm" if len(moves) > 1 else "-"
                        command = f"GUIDE|{x_message}|{y_message}"
                    cv2.putText(frame, status, (12, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (0, 255, 0), 2)
                    cv2.putText(frame, f"Offset from pad: X {off_x:+.0f} mm  Y {off_y:+.0f} mm", (12, 58), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                else:
                    status = "PAD FOUND - WAITING FOR DRONE MARKER"
                    cv2.putText(frame, status, (12, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 200, 255), 2)
            else:
                cv2.putText(frame, "HOLD - PAD BULLSEYE NOT FOUND", (12, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

            sensor_fresh = last_reading is not None and time.monotonic() - last_reading_time < 1.0
            draw_sensor_panel(frame, last_reading or {}, sensor_fresh)

            now = time.monotonic()
            if command != last_command or now - last_send >= 0.2:
                nano.write((command + "\n").encode("ascii"))
                last_command, last_send = command, now

            cv2.imshow("SkyDock bullseye bench test - press Q to quit", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        cap.release()
        if nano:
            try:
                nano.write(b"HOLD\n")
                nano.close()
            except serial.SerialException:
                pass
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
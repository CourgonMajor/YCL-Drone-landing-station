import cv2
import numpy as np

camera = cv2.VideoCapture(0)

if not camera.isOpened():
    print("Could not open the camera.")
    raise SystemExit

while True:
    success, frame = camera.read()

    if not success:
        print("Could not read from the camera.")
        break

    height, width = frame.shape[:2]

    # For this first test, use the middle of the image as the target.
    target_x = width // 2
    target_y = height // 2

    # Red appears at both ends of OpenCV's hue scale, so use two ranges.
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    lower_red_1 = np.array([0, 100, 80])
    upper_red_1 = np.array([10, 255, 255])
    lower_red_2 = np.array([170, 100, 80])
    upper_red_2 = np.array([179, 255, 255])

    mask_1 = cv2.inRange(hsv, lower_red_1, upper_red_1)
    mask_2 = cv2.inRange(hsv, lower_red_2, upper_red_2)
    red_mask = cv2.bitwise_or(mask_1, mask_2)

    # Ignore tiny red specks.
    red_mask = cv2.erode(red_mask, None, iterations=1)
    red_mask = cv2.dilate(red_mask, None, iterations=2)

    contours, _ = cv2.findContours(
        red_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    # Draw the target cross in the middle of the camera image.
    cv2.drawMarker(
        frame, (target_x, target_y), (255, 255, 0),
        cv2.MARKER_CROSS, 30, 2
    )

    if contours:
        largest = max(contours, key=cv2.contourArea)

        if cv2.contourArea(largest) > 80:
            moments = cv2.moments(largest)

            if moments["m00"] > 0:
                dot_x = int(moments["m10"] / moments["m00"])
                dot_y = int(moments["m01"] / moments["m00"])

                offset_x = dot_x - target_x
                offset_y = dot_y - target_y

                cv2.circle(frame, (dot_x, dot_y), 10, (0, 255, 0), 2)

                if abs(offset_x) < 20 and abs(offset_y) < 20:
                    message = "DOT CENTERED"
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
            frame, "No red dot found", (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2
        )

    cv2.imshow("SkyDock red-dot test — press Q to close", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

camera.release()
cv2.destroyAllWindows()

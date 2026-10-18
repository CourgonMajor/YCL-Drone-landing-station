import cv2
import numpy as np

WIDTH, HEIGHT = 800, 650
PAD_SIZE = 460
TARGET_X, TARGET_Y = WIDTH // 2, HEIGHT // 2 + 20
CENTER_TOLERANCE = 20
MOVE_STEP = 10

# This is simulated drone position, not a real detection.
drone_x, drone_y = TARGET_X, TARGET_Y


def draw_scene():
    image = np.full((HEIGHT, WIDTH, 3), (245, 248, 250), dtype=np.uint8)

    # Draw the landing pad.
    half = PAD_SIZE // 2
    left, top = TARGET_X - half, TARGET_Y - half
    right, bottom = TARGET_X + half, TARGET_Y + half
    cv2.rectangle(image, (left, top), (right, bottom), (220, 230, 238), -1)
    cv2.rectangle(image, (left, top), (right, bottom), (90, 110, 130), 3)

    # Pad center and acceptable centered zone.
    cv2.circle(image, (TARGET_X, TARGET_Y), CENTER_TOLERANCE, (0, 180, 0), 2)
    cv2.drawMarker(
        image, (TARGET_X, TARGET_Y), (255, 120, 0),
        cv2.MARKER_CROSS, 32, 2
    )

    # Draw a simple top-down drone shape.
    arm = 65
    propeller_radius = 22
    propeller_points = [
        (drone_x - arm, drone_y - arm),
        (drone_x + arm, drone_y - arm),
        (drone_x - arm, drone_y + arm),
        (drone_x + arm, drone_y + arm),
    ]

    for px, py in propeller_points:
        cv2.line(image, (drone_x, drone_y), (px, py), (70, 75, 85), 8)
        cv2.circle(image, (px, py), propeller_radius, (100, 110, 125), 3)

    cv2.rectangle(
        image, (drone_x - 24, drone_y - 18),
        (drone_x + 24, drone_y + 18), (70, 75, 85), -1
    )

    # Simulated model prediction of the chosen alignment point.
    cv2.circle(image, (drone_x, drone_y), 7, (0, 200, 0), -1)
    cv2.circle(image, (drone_x, drone_y), 12, (0, 120, 0), 2)

    # Offset from pad center. Positive y is lower in the image.
    offset_x = drone_x - TARGET_X
    offset_y = drone_y - TARGET_Y

    if abs(offset_x) <= CENTER_TOLERANCE and abs(offset_y) <= CENTER_TOLERANCE:
        guidance = "PILOT GUIDANCE: CENTERED"
        guidance_color = (0, 130, 0)
    else:
        corrections = []

        if offset_x > CENTER_TOLERANCE:
            corrections.append("LEFT")
        elif offset_x < -CENTER_TOLERANCE:
            corrections.append("RIGHT")

        if offset_y > CENTER_TOLERANCE:
            corrections.append("UP")
        elif offset_y < -CENTER_TOLERANCE:
            corrections.append("DOWN")

        guidance = "PILOT GUIDANCE: MOVE " + " + ".join(corrections)
        guidance_color = (0, 80, 200)

    cv2.putText(
        image, "SIMULATION - NO CAMERA OR DRONE CONTROL",
        (25, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (40, 50, 65), 2
    )
    cv2.putText(
        image, guidance, (25, 75),
        cv2.FONT_HERSHEY_SIMPLEX, 0.8, guidance_color, 2
    )
    cv2.putText(
        image, f"Offset: x={offset_x}px, y={offset_y}px",
        (25, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (40, 50, 65), 2
    )
    cv2.putText(
        image, "W/A/S/D move the simulated drone | R reset | Q quit",
        (25, HEIGHT - 20), cv2.FONT_HERSHEY_SIMPLEX,
        0.55, (40, 50, 65), 1
    )

    return image


while True:
    cv2.imshow("SkyDock alignment simulator", draw_scene())
    key = cv2.waitKey(20) & 0xFF

    if key == ord("q"):
        break
    elif key == ord("w"):
        drone_y -= MOVE_STEP
    elif key == ord("s"):
        drone_y += MOVE_STEP
    elif key == ord("a"):
        drone_x -= MOVE_STEP
    elif key == ord("d"):
        drone_x += MOVE_STEP
    elif key == ord("r"):
        drone_x, drone_y = TARGET_X, TARGET_Y

cv2.destroyAllWindows()

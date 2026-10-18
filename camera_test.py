import cv2

camera = cv2.VideoCapture(0)

if not camera.isOpened():
    print("Camera not found. Check camera permissions or try camera number 1.")
    raise SystemExit

while True:
    success, image = camera.read()

    if not success:
        print("Could not read from camera.")
        break

    cv2.imshow("Camera test — press Q to close", image)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

camera.release()
cv2.destroyAllWindows()

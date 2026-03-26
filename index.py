import cv2
import numpy as np

# Open video
# -------------------------------
video = cv2.VideoCapture("2024_1011_113958_187.MP4")

if not video.isOpened():
    print("Error: Could not open video file.")
    exit()

ret, prev_frame = video.read()
if not ret:
    print("Error: Could not read first frame.")
    exit()


# Preprocessing setup
# -------------------------------
clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))

prev_hsv = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2HSV)
prev_gray = prev_hsv[:, :, 2]
prev_gray = clahe.apply(prev_gray)

frame_index = 0

# Main loop
# -------------------------------
while True:
    ret, frame = video.read()
    if not ret:
        break

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    gray = hsv[:, :, 2]
    gray = clahe.apply(gray)

    # Farneback Flow
    # -------------------------------
    flow = cv2.calcOpticalFlowFarneback(
        prev_gray,
        gray,
        None,
        0.5,   # pyramid scale
        3,     # pyramid levels
        21,    # window size
        3,     # iterations
        5,     # poly_n
        1.2,   # poly_sigma
        0
    )

    dx = flow[..., 0]
    dy = flow[..., 1]

    mag, ang = cv2.cartToPolar(dx, dy)

    # Filter Weak Motion && Calculate Results
    # -------------------------------
    threshold = 0.2
    mask = mag > threshold

    if np.any(mask):
        avg_dx = np.mean(dx[mask])
        avg_dy = np.mean(dy[mask])
        avg_speed = np.mean(mag[mask])
    else:
        avg_dx, avg_dy, avg_speed = 0, 0, 0

    print(f"Frame {frame_index}: "
          f"magnitude=({avg_dx:.4f}, {avg_dy:.4f}) "
          f"speed={avg_speed:.4f}")


    # Flow arrows
    # -------------------------------
    vis = frame.copy()
    step = 16

    for y in range(0, flow.shape[0], step):
        for x in range(0, flow.shape[1], step):
            fx, fy = flow[y, x]

            if mag[y, x] > threshold:
                cv2.arrowedLine(
                    vis,
                    (x, y),
                    (int(x + fx * 3), int(y + fy * 3)),
                    (0, 255, 0),
                    1,
                    tipLength=0.3
                )


    # Show Video && Update Video
    # -------------------------------
    cv2.imshow("Original", frame)
    cv2.imshow("Flow Field", vis)

    if cv2.waitKey(25) & 0xFF == ord('q'):
        break

    prev_gray = gray.copy()
    frame_index += 1


video.release()
cv2.destroyAllWindows()
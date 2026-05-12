import cv2
import csv
import numpy as np
from scipy.spatial.distance import cdist

name = "laminar.MP4"
video = cv2.VideoCapture(f"videos/{name}")

if not video.isOpened():
    print("Error: Could not open video file.")
    exit()

ret, first_frame = video.read()
if not ret:
    print("Error: Could not read first frame.")
    exit()

SCREEN_X, SCREEN_Y  = 1000, 800
ADAPT_BLOCK = 41
ADAPT_C = -4
CLAHE_CLIP = 2.0

MIN_BLOB_AREA = 250
MAX_BLOB_AREA = 8000
CIRCULARITY_MIN = 0.4

MAX_FRAME = 155

MAX_MATCH_DIST = 80
REAPPEAR_DIST  = 120
MAX_VELOCITY   = 25
GHOST_FRAMES   = 20

VELOCITY_SMOOTH = 0.8
CONFIRM_FRAMES = 3

particles  = {}
all_particle_histories = {}
next_id    = 0
frame_index = 0

COLORS = [
    (0,255,255),(255,150,0),(0,255,150),(200,0,255),
    (255,255,0),(100,180,255),(255,0,180),(255,255,255),(0,255,80),
]

clahe = cv2.createCLAHE(clipLimit=CLAHE_CLIP, tileGridSize=(16, 16))
kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))

prev_gray = None
static_accumulator = None
alpha_static = 0.02

WRITE_TRACKED_VIDEO = True
output_writer = None

csv_file_per_particle = open(f"{name}_per-particle_summary.csv", "w", newline="")
csv_writer_per_particle = csv.writer(csv_file_per_particle)
csv_writer_per_particle.writerow(["particle_id", "frames_tracked", "avg_speed", "std_speed", "avg_angle", "std_angle", "avg_x_velocity", "avg_y_velocity"])

csv_file_summary = open(f"{name}_summary_particle_data.csv", "w", newline="")
csv_writer_summary = csv.writer(csv_file_summary)
csv_writer_summary.writerow(["particles_tracked", "avg_vx", "avg_vy", "avg_speed", "angle"])

if WRITE_TRACKED_VIDEO:
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    #fourcc = cv2.VideoWriter_fourcc(*'XVID')
    fps = video.get(cv2.CAP_PROP_FPS)
    output_writer = cv2.VideoWriter(
        "tracked_output.mp4", fourcc, fps, (SCREEN_X, SCREEN_Y)
    )

def detect_particles(frame, motion_mask, static_mask):
    frame = cv2.resize(frame, (SCREEN_X, SCREEN_Y))

    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    l = clahe.apply(lab[:, :, 0])

    binary = cv2.adaptiveThreshold(
        l, 255,
        cv2.ADAPTIVE_THRESH_MEAN_C,
        cv2.THRESH_BINARY,
        ADAPT_BLOCK, ADAPT_C
    )

    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel_open)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel_close)

    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    blobs = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if not (MIN_BLOB_AREA <= area <= MAX_BLOB_AREA):
            continue

        M = cv2.moments(cnt)
        if M["m00"] == 0:
            continue

        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])

        perimeter = cv2.arcLength(cnt, True)
        if perimeter == 0:
            continue

        circularity = 4 * np.pi * area / (perimeter ** 2)
        if circularity < CIRCULARITY_MIN:
            continue

        mask = np.zeros_like(motion_mask)
        cv2.drawContours(mask, [cnt], -1, 255, -1)

        motion_score = cv2.mean(motion_mask, mask=mask)[0]
        static_score = cv2.mean(static_mask, mask=mask)[0]

        if motion_score < 2:
            continue

        if static_score < 5:
            continue

        blobs.append((cx, cy, area, cnt))

    return blobs, binary

def velocity_to_angle(vx, vy):
    angle = np.degrees(np.arctan2(-vy, vx)) % 360
    return angle

def avg_velocities_from_hist(v):
    sum_x = 0
    sum_y = 0
    size = 0
    for vx, vy in v:
        sum_x = sum_y + vx
        sum_y = sum_y + vy
        size += 1

    sum_x = sum_x / size
    sum_y = sum_y / size

    return (sum_x, sum_y)

def predict_position(p):
    cx, cy = p["center"]
    vx, vy = p["velocity"]
    ghost = p.get("ghost", 0)
    
    scale = 1.0 if ghost == 0 else max(0.5, 1.0 - ghost * 0.08)
    return cx + vx * scale, cy + vy * scale

def retire_particle(pid, p):
    """Write per-particle summary row and store history when a particle dies."""
    hist = p["history"]
    if hist:
        all_particle_histories[pid] = hist
        speeds = [np.hypot(vx, vy) for vx, vy in hist]
        angles = [velocity_to_angle(vx, vy) for vx, vy in hist]
        avg_velocity_x, avg_velocity_y = avg_velocities_from_hist(hist)
        csv_writer_per_particle.writerow([
            pid,
            len(hist),
            f"{np.mean(speeds):.4f}",
            f"{np.std(speeds):.4f}",
            f"{np.mean(angles):.4f}",
            f"{np.std(angles):.4f}",
            f"{avg_velocity_x:.4f}",
            f"{avg_velocity_y:.4f}",
        ])

def match_and_update(particles, detections, next_id):
    global frame_index

    if not detections:
        for pid, p in particles.items():
            p["ghost"] += 1

        new_particles = {}
        for pid, p in particles.items():
            if p["ghost"] < GHOST_FRAMES:
                new_particles[pid] = p
            else:
                retire_particle(pid, p)

        return new_particles, next_id

    det_centers = np.array([(d[0], d[1]) for d in detections], dtype=float)

    if not particles:
        for cx, cy, area, cnt in detections:
            particles[next_id] = {
                "center": (cx, cy),
                "velocity": (0.0, 0.0),
                "ghost": 0,
                "moving_frames": 0,
                "confirmed": False,
                "area": area,
                "history": [],
            }
            next_id += 1
        return particles, next_id

    par_ids = list(particles.keys())
    predicted = np.array([predict_position(particles[pid]) for pid in par_ids], dtype=float)
    dists = cdist(predicted, det_centers)

    matched_pars, matched_dets = set(), set()

    for flat_idx in np.argsort(dists, axis=None):
        pi, di = divmod(flat_idx, len(detections))

        if pi in matched_pars or di in matched_dets:
            continue

        pid = par_ids[pi]
        ghost = particles[pid]["ghost"]

        if dists[pi, di] > (MAX_MATCH_DIST if ghost == 0 else REAPPEAR_DIST):
            continue

        cx, cy, area, cnt = detections[di]
        old_cx, old_cy = particles[pid]["center"]

        raw_vx = cx - old_cx
        raw_vy = cy - old_cy

        speed = np.hypot(raw_vx, raw_vy)
        if speed > MAX_VELOCITY:
            scale = MAX_VELOCITY / speed
            raw_vx *= scale
            raw_vy *= scale

        prev_vx, prev_vy = particles[pid]["velocity"]

        svx = VELOCITY_SMOOTH * raw_vx + (1 - VELOCITY_SMOOTH) * prev_vx
        svy = VELOCITY_SMOOTH * raw_vy + (1 - VELOCITY_SMOOTH) * prev_vy

        speed = np.hypot(svx, svy)

        if speed > 0.5:
            particles[pid]["moving_frames"] += 1
        else:
            particles[pid]["moving_frames"] = max(0, particles[pid]["moving_frames"] - 1)

        if particles[pid]["moving_frames"] >= CONFIRM_FRAMES:
            particles[pid]["confirmed"] = True

        particles[pid]["history"].append((svx, svy))
        particles[pid].update({
            "center": (cx, cy),
            "velocity": (svx, svy),
            "ghost": 0,
            "area": area,
        })

        matched_pars.add(pi)
        matched_dets.add(di)

    new_particles = {}
    for pi, pid in enumerate(par_ids):
        p = particles[pid]

        if pi not in matched_pars:
            p["ghost"] += 1

        if p["ghost"] < GHOST_FRAMES:
            new_particles[pid] = p
        else:
            retire_particle(pid, p)

    for di, (cx, cy, area, cnt) in enumerate(detections):
        if di not in matched_dets:
            new_particles[next_id] = {
                "center": (cx, cy),
                "velocity": (0.0, 0.0),
                "ghost": 0,
                "moving_frames": 0,
                "confirmed": False,
                "area": area,
                "history": [],
            }
            next_id += 1

    return new_particles, next_id

while True and frame_index <= MAX_FRAME:
    ret, frame = video.read()
    if not ret:
        break

    frame = cv2.resize(frame, (SCREEN_X, SCREEN_Y))
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    if prev_gray is None:
        prev_gray = gray.copy()

    motion_mask = cv2.absdiff(gray, prev_gray)
    _, motion_mask = cv2.threshold(motion_mask, 15, 255, cv2.THRESH_BINARY)
    motion_mask = cv2.morphologyEx(motion_mask, cv2.MORPH_OPEN, kernel_open)

    if static_accumulator is None:
        static_accumulator = gray.astype(np.float32)

    static_accumulator = cv2.addWeighted(
        static_accumulator, 1 - alpha_static,
        gray.astype(np.float32), alpha_static,
        0
    )

    static_mask = cv2.absdiff(gray, cv2.convertScaleAbs(static_accumulator))
    _, static_mask = cv2.threshold(static_mask, 10, 255, cv2.THRESH_BINARY)

    detections, binary = detect_particles(frame, motion_mask, static_mask)
    particles, next_id = match_and_update(particles, detections, next_id)

    vis = frame.copy()
    for pid, p in particles.items():
        # if not p["confirmed"]:      Confirmed particle flag is not working atm
        #     continue

        cx, cy = p["center"]
        vx, vy = p["velocity"]
        speed = np.hypot(vx, vy)

        color = COLORS[pid % len(COLORS)]
        radius = max(12, int((p["area"] / np.pi) ** 0.5) + 4)

        cv2.circle(vis, (cx, cy), radius, color, 2)

        cv2.putText(
            vis,
            f"ID {pid}",
            (cx + 10, cy - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            1,
            cv2.LINE_AA
        )

    cv2.imshow("Tracking", vis)
    # cv2.imshow("Mask", binary)
    # cv2.imshow("Motion", motion_mask)
    # cv2.imshow("Static Filter", static_mask)

    if WRITE_TRACKED_VIDEO:
        output_writer.write(vis)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    prev_gray = gray.copy()
    frame_index += 1


video.release()
if output_writer:
    output_writer.release()
cv2.destroyAllWindows()

for pid, p in particles.items():
    retire_particle(pid, p)

all_velocities = [v for hist in all_particle_histories.values() for v in hist]

if all_velocities:
    avg_vx = np.mean([v[0] for v in all_velocities])
    avg_vy = np.mean([v[1] for v in all_velocities])
    avg_speed = np.mean([np.hypot(v[0], v[1]) for v in all_velocities])

    print("\n===== TRACKING RESULTS =====")
    print(f"Particles tracked: {len(all_particle_histories)}")
    print(f"Average velocity: {avg_vx:.2f}, {avg_vy:.2f}")
    print(f"Average speed: {avg_speed:.2f}")
    print(f"Angle: {velocity_to_angle(avg_vx, avg_vy)}")

    csv_writer_summary.writerow([
        len(all_particle_histories),
        f"{avg_vx:.2f}",
        f"{avg_vy:.2f}",
        f"{avg_speed:.2f}",
        f"{velocity_to_angle(avg_vx, avg_vy):.2f}",
    ])
else:
    print("No motion data recorded.")

csv_file_per_particle.close()
csv_file_summary.close()

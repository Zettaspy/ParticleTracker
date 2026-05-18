import cv2
import csv
import numpy as np
from math import sqrt
import matplotlib.pyplot as mpylt
from scipy.spatial.distance import cdist
from scipy.io import loadmat
import imutils

# name = "Calibration_0010.tif"
# path = "tests/20240814_CameraA_frames_extracted20260407"

name = "turbulent.MP4"
path = "videos/"

video = cv2.VideoCapture(f"{path}/{name}")

calibration_data_name = "opencv_params.mat"

calibration_data = loadmat(f'./cameraCalibration/{calibration_data_name}', simplify_cells=True)
calibration_output = calibration_data["convertedCP"]

K = calibration_output["CameraMatrix"]
radial = calibration_output["RadialDistortion"]
tangential = calibration_output["TangentialDistortion"]

dist = np.array([
    radial[0],
    radial[1],
    tangential[0],
    tangential[1],
    radial[2] if len(radial) > 2 else 0
], dtype=np.float32)

if not video.isOpened():
    print("Error: Could not open video file.")
    exit()

SCREEN_X, SCREEN_Y  = 1000, 800
CAMERA_ANGLE_ROTATION_FROM_CALIBRATION = 90  #Currently only works with 90 and -90 and 0
ADAPT_BLOCK = 41
ADAPT_C = -4
CLAHE_CLIP = 2.0

DECTECT_CIRCLES = True
SAME_COLOR = False

MIN_BLOB_AREA = 250
MAX_BLOB_AREA = 8000
CIRCULARITY_MIN = 0.4
MOTION_MIN = 2
DRAWN_MOTION_MIN = 0.05
STATIC_MIN = 5

MAX_FRAME = 155
FRAME_RATE = video.get(cv2.CAP_PROP_FPS)

MAX_MATCH_DIST = 80
REAPPEAR_DIST  = 120
MAX_VELOCITY   = 25
MIN_VELOCITY   = 0.5
GHOST_FRAMES   = 20

VELOCITY_SMOOTH = 0.8
CONFIRM_FRAMES = 3

PATH_SEGMENTS = 2

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

csv_file_per_particle = open(f"output/csv/{name}_per-particle_summary.csv", "w", newline="")
csv_writer_per_particle = csv.writer(csv_file_per_particle)
csv_writer_per_particle.writerow(["particle_id", "frames_tracked", "avg_speed", "std_speed", "avg_angle", "std_angle", "avg_x_velocity", "avg_y_velocity"])

csv_file_summary = open(f"output/csv/{name}_summary_particle_data.csv", "w", newline="")
csv_writer_summary = csv.writer(csv_file_summary)
csv_writer_summary.writerow(["particles_tracked", "avg_vx", "avg_vy", "avg_speed", "angle", "video_frame_rate"])

if WRITE_TRACKED_VIDEO:
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    #fourcc = cv2.VideoWriter_fourcc(*'XVID')
    fps = video.get(cv2.CAP_PROP_FPS)
    output_writer = cv2.VideoWriter(
        "output/videos/tracked_output.mp4", fourcc, fps, (SCREEN_X, SCREEN_Y)
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

        if motion_score < MOTION_MIN:
            continue

        if static_score < STATIC_MIN:
            continue

        blobs.append((cx, cy, area, cnt))

    return blobs, binary
#


def velocity_to_angle(vx, vy):
    angle = np.degrees(np.arctan2(-vy, vx)) % 360
    return angle
#


def predict_position(p):
    cx, cy = p["center"]
    vx, vy = p["velocity"]
    ghost = p.get("ghost", 0)
    
    scale = 1.0 if ghost == 0 else max(0.5, 1.0 - ghost * 0.08)
    return cx + vx * scale, cy + vy * scale

#

def retire_particle(pid, p):
    """Write per-particle summary row and store history when a particle dies."""
    hist = p["history"]
    if hist:
        all_particle_histories[pid] = {
            "history": hist,
            "start_center": p.get("start_center", p["center"])
        }

        speeds = [np.hypot(vx, vy) for vx, vy in hist]
        angles = [velocity_to_angle(vx, vy) for vx, vy in hist]
        avg_velocity_x = [vx for vx, _ in hist]
        avg_velocity_y = [vy for _, vy in hist]

        csv_writer_per_particle.writerow([
            pid,
            len(hist),
            f"{np.average(speeds):.4f}",
            f"{np.std(speeds):.4f}",
            f"{np.average(angles):.4f}",
            f"{np.std(angles):.4f}",
            f"{np.average(avg_velocity_x):.4f}",
            f"{np.average(avg_velocity_y):.4f}",
        ])
#

def match_and_update(particles, detections, next_id):
    global frame_index

    if not detections:
        for pid, p in particles.items():
            p["ghost"] += 1
            p["confirmed"] = False

        new_particles = {}
        for pid, p in particles.items():
            if p["ghost"] < GHOST_FRAMES:
                #p["confirmed"] = True                     # Reinitalize if you want to maintain the last position of ghost particle
                new_particles[pid] = p
            else:
                retire_particle(pid, p)

        return new_particles, next_id

    det_centers = np.array([(d[0], d[1]) for d in detections], dtype=float)

    if not particles:
        for cx, cy, area, cnt in detections:
            particles[next_id] = {
                "center": (cx, cy),
                "start_center": (cx, cy),
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

        if speed > MIN_VELOCITY:
            particles[pid]["moving_frames"] += 1
        else:
            particles[pid]["moving_frames"] = max(0, particles[pid]["moving_frames"] - 1)

        """ To stop drawing if the value is a ghosted value (determines via the history of the particle)"""
        try:
            p_hist = particles[pid]["history"][len(particles[pid]["history"]) - 1]
            p_hist_result = sqrt(((speed - p_hist[len(p_hist) - 1]) ** 2))
        except:
            p_hist_result = 0

        if particles[pid]["moving_frames"] >= CONFIRM_FRAMES or p_hist_result >= DRAWN_MOTION_MIN:
            particles[pid]["confirmed"] = True
        else:
            particles[pid]["confirmed"] = False

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
                "start_center": (cx, cy),
                "velocity": (0.0, 0.0),
                "ghost": 0,
                "moving_frames": 0,
                "confirmed": False,
                "area": area,
                "history": [],
            }
            next_id += 1

    return new_particles, next_id
#

def plot_paths():
    if not all_particle_histories:
        print("No particle history to plot.")
        return

    fig1, ax1 = mpylt.subplots(figsize=(10, 8))
    cmap = mpylt.get_cmap('gist_rainbow')
    colors = [cmap(i) for i in np.linspace(0, 1, len(all_particle_histories.items()))]

    all_path_displacement = []
    i = 0
    for pid, data in all_particle_histories.items():
        hist = data["history"]
        start_cx, start_cy = data["start_center"]

        if not hist:
            continue

        xs = [start_cx]
        ys = [start_cy]
        for vx, vy in hist:
            xs.append(xs[-1] + vx)
            ys.append(ys[-1] + vy)

        dx = xs[-1] - xs[0]
        dy = ys[-1] - ys[0]
        displacement = np.hypot(dx, dy)
        all_path_displacement.append(displacement)
        
        color = colors[i]
        ax1.plot(xs, ys, color=color, linewidth=1.2, alpha=0.7)
        ax1.plot(xs[0], ys[0], 'o', color=color, markersize=5) 
        i += 1

    ax1.set_title(f"{name} Particle Paths")
    ax1.set_xlabel("X Position (px)")
    ax1.set_ylabel("Y Position (px)")
    ax1.invert_yaxis()
    ax1.set_xlim(0, SCREEN_X)
    ax1.set_ylim(SCREEN_Y, 0)
    ax1.grid(True)
    mpylt.savefig(f"output/graphs/{name}_particle_paths.png", dpi=300, bbox_inches="tight")

    all_speeds = []
    all_path_lengths = []
    for pid, data in all_particle_histories.items():
        hist = data["history"]
        if not hist:
            continue

        speeds = [np.hypot(vx, vy) for vx, vy in hist]
        all_speeds.extend(speeds)
        all_path_lengths.append(sum(speeds))

    fig2, (ax_speed, ax_length, ax_disp) = mpylt.subplots(1, 3, figsize=(14, 6))

    plot_bell(ax_speed,  all_speeds, "Speed Distribution", "Speed (px/frame)", "#00cfff")
    plot_bell(ax_length, all_path_lengths, "Path Length Distribution", "Path Length (px)",  "#ffaa00")
    plot_bell(ax_disp, all_path_displacement, "Path Displacement Distribution", "Path Displacement (px)",  "#33dd1b")
    
    fig1.suptitle(f"{name} Speed & Path Length Distributions", fontsize=13)
    mpylt.tight_layout()
    mpylt.savefig(f"output/graphs/{name}_distributions.png", dpi=300, bbox_inches="tight")
    #mpylt.show()
#


def plot_bell(ax, data, title, xlabel, color):
    if len(data) < 2:
        return
    data = np.array(data)
    mu, sigma = np.mean(data), np.std(data)

    ax.hist(data, bins=30, density=True, alpha=0.45, color=color, edgecolor="white")

    x_range = np.linspace(mu - 4 * sigma, mu + 4 * sigma, 300)
    gaussian = (1 / (sigma * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x_range - mu) / sigma) ** 2)
    ax.plot(x_range, gaussian, color=color, linewidth=2.5)

    ax.axvline(mu, color="white", linestyle="--", linewidth=1.5, label=f"μ = {mu:.2f}")
    ax.axvline(mu - sigma, color="gray", linestyle=":", linewidth=1.0, label=f"σ = {sigma:.2f}")
    ax.axvline(mu + sigma, color="gray", linestyle=":", linewidth=1.0)

    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Probability Density")
    ax.legend(fontsize=9)
    ax.grid(True)
#

def rectify_image(rotated_image, original_width, original_height, angle):
    center = (original_width / 2, original_height / 2)
    
    rotation_mat = cv2.getRotationMatrix2D(center, -angle, 1.0)
    
    abs_cos = abs(rotation_mat[0, 0])
    abs_sin = abs(rotation_mat[0, 1])
    bound_w = int(original_height * abs_sin + original_width * abs_cos)
    bound_h = int(original_height * abs_cos + original_width * abs_sin)
    
    rotation_mat[0, 2] += bound_w / 2 - center[0]
    rotation_mat[1, 2] += bound_h / 2 - center[1]
    
    rectified_image = cv2.warpAffine(
        rotated_image, 
        rotation_mat, 
        (original_width, original_height), 
        flags=cv2.INTER_LINEAR
    )
    
    return rectified_image

# -- MAIN --
first_pass = True
while True and frame_index <= MAX_FRAME:
    ret, frame = video.read()
    if not ret:
        break

    if first_pass:
        h, w = frame.shape[:2]

        if CAMERA_ANGLE_ROTATION_FROM_CALIBRATION:
            newcameramtx_normal, roi = cv2.getOptimalNewCameraMatrix(K, dist, (h,w), 0, (h,w))
            map1, map2 = cv2.initUndistortRectifyMap(K, dist, None, newcameramtx_normal, (h,w), cv2.CV_32FC1)
        else:
            newcameramtx_normal, roi = cv2.getOptimalNewCameraMatrix(K, dist, (w,h), 0, (w,h))
            map1, map2 = cv2.initUndistortRectifyMap(K, dist, None, newcameramtx_normal, (w,h), cv2.CV_32FC1)


    if CAMERA_ANGLE_ROTATION_FROM_CALIBRATION:
        frame = imutils.rotate_bound(frame, angle=CAMERA_ANGLE_ROTATION_FROM_CALIBRATION)

    rectified_frame = cv2.remap(frame, map1, map2, cv2.INTER_NEAREST)

    if CAMERA_ANGLE_ROTATION_FROM_CALIBRATION:
        rectified_frame = imutils.rotate_bound(rectified_frame, angle=-CAMERA_ANGLE_ROTATION_FROM_CALIBRATION)

    if first_pass:
        if CAMERA_ANGLE_ROTATION_FROM_CALIBRATION:
            frame = imutils.rotate_bound(frame, angle=-CAMERA_ANGLE_ROTATION_FROM_CALIBRATION)
        cv2.imwrite("./output/images/normal_frame.png", frame)
        cv2.imwrite("./output/images/rectified_frame.png", rectified_frame)
        first_pass = False

    
    rectified_frame = cv2.resize(rectified_frame, (SCREEN_X, SCREEN_Y))

    gray = cv2.cvtColor(rectified_frame, cv2.COLOR_BGR2GRAY)

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

    detections, binary = detect_particles(rectified_frame, motion_mask, static_mask)
    particles, next_id = match_and_update(particles, detections, next_id)

    vis = rectified_frame.copy()
    for pid, p in particles.items():

        cx, cy = p["center"]
        vx, vy = p["velocity"]
        speed = np.hypot(vx, vy)
        radius = max(12, int((p["area"] / np.pi) ** 0.5) + 4)

        if not p["confirmed"]:
            if DECTECT_CIRCLES:
                color = 200
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
            continue

        if SAME_COLOR:
            color = (50, 255, 50)
        else:
            color = COLORS[pid % len(COLORS)]

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
#


video.release()
if output_writer:
    output_writer.release()
cv2.destroyAllWindows()

for pid, p in particles.items():
    retire_particle(pid, p)

all_velocities = [v for hist in all_particle_histories.values() for v in hist["history"]]

if all_velocities:
    avg_vx = np.average([v[0] for v in all_velocities])
    avg_vy = np.average([v[1] for v in all_velocities])
    avg_speed = np.average([np.hypot(v[0], v[1]) for v in all_velocities])
    reynolds_number = 0  # add later to determine if laminar or turbulent

    print("\n===== TRACKING RESULTS =====")
    print(f"Particles tracked: {len(all_particle_histories)}")
    print(f"Average velocity: {avg_vx:.2f}, {avg_vy:.2f}")
    print(f"Average speed: {avg_speed:.2f}")
    print(f"Angle: {velocity_to_angle(avg_vx, avg_vy)}")
    print(f"Reynalds Number: {reynolds_number:.2f}")
    if reynolds_number <= 2300:
        print(f"Flow Type: Laminar")
    elif reynolds_number > 4000:
        print(f"Flow Type: Turbulent")
    else:
        print(f"Flow Type: Mixed")

    csv_writer_summary.writerow([
        len(all_particle_histories),
        f"{avg_vx:.2f}",
        f"{avg_vy:.2f}",
        f"{avg_speed:.2f}",
        f"{velocity_to_angle(avg_vx, avg_vy):.2f}",
        f"{FRAME_RATE}",
    ])

    # Add Plots
    plot_paths()
    
else:
    print("No motion data recorded.")

csv_file_per_particle.close()
csv_file_summary.close()

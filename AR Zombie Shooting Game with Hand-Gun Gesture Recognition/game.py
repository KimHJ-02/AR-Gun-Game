#! python3.12

import random
import time
from collections import deque
from pathlib import Path

import cv2 as cv
import numpy as np
import torch

from model import get_model
from utils import (
    draw_crosshair,
    draw_status,
    overlay_image_alpha,
    preprocess_roi_for_model,
)


# MVP에서 사용할 고정 설정값이다.
ROI_SIZE = 320
GAME_TIME_SECONDS = 60
INFERENCE_INTERVAL = 3
HIT_COOLDOWN_SECONDS = 0.6
FINGERTIP_TARGET_RADIUS = 28
WINDOW_NAME = "AR Zombie Shooting Game"
ZOMBIE_SIZE = (120, 120)  # cv.resize는 (width, height) 순서를 사용한다.
MIN_HAND_CONTOUR_AREA = 900
PREDICTION_HISTORY_SIZE = 5
MIN_GUN_VOTES = 4
GUN_CONFIDENCE_THRESHOLD = 0.90
FORCE_NON_GUN_WHEN_NO_HAND = True

# 이 파일이 있는 프로젝트 폴더를 기준으로 모델과 에셋 경로를 만든다.
PROJECT_DIR = Path(__file__).resolve().parent
MODELS_DIR = PROJECT_DIR / "models"
ASSETS_DIR = PROJECT_DIR / "assets"
DATASET_DIR = PROJECT_DIR / "dataset"
GUN_DIR = DATASET_DIR / "gun"
NON_GUN_DIR = DATASET_DIR / "non_gun"
MODEL_PATH = MODELS_DIR / "hand_gun_model.pth"
ZOMBIE_IMAGE_PATH = ASSETS_DIR / "zombie.png"
DATA_WINDOW_NAME = "Collect Reliable Hand Gesture Data"
DATA_COLLECTION_SECONDS = 60
GUN_COLLECTION_SECONDS = 30
AUTO_CAPTURE_INTERVAL = 0.12


def create_project_folders():
    """게임 실행에 필요한 폴더가 없으면 자동 생성한다."""
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    ASSETS_DIR.mkdir(parents=True, exist_ok=True)


def create_dataset_folders():
    """Create dataset folders automatically if they do not exist."""
    GUN_DIR.mkdir(parents=True, exist_ok=True)
    NON_GUN_DIR.mkdir(parents=True, exist_ok=True)


def load_image_unicode(path):
    """Windows 한글 경로에서도 이미지를 읽기 위한 함수.

    cv.imread는 일부 Windows 환경에서 한글 경로를 제대로 읽지 못할 수 있다.
    np.fromfile과 cv.imdecode를 사용하면 이런 문제를 줄일 수 있다.
    """
    if not path.exists():
        return None

    image_bytes = np.fromfile(str(path), dtype=np.uint8)
    image = cv.imdecode(image_bytes, cv.IMREAD_UNCHANGED)
    return image


def load_zombie_image():
    """assets/zombie.png가 있으면 읽고, 없으면 None을 반환한다."""
    zombie_image = load_image_unicode(ZOMBIE_IMAGE_PATH)

    if zombie_image is None:
        print("Info: assets/zombie.png not found. Drawing zombie as a rectangle.")
        return None

    # 좀비 이미지를 MVP에서 쓰기 좋은 고정 크기로 맞춘다.
    zombie_image = cv.resize(zombie_image, ZOMBIE_SIZE)
    return zombie_image


def load_hand_gun_model(device):
    """학습된 모델을 불러온다.

    모델 파일이 없으면 게임은 종료하지 않고 debug mode로 계속 실행한다.
    """
    if not MODEL_PATH.exists():
        print("먼저 py train.py를 실행하세요")
        print("Model file not found. You can still test the game with debug mode by pressing t.")
        return None, {"gun": 0, "non_gun": 1}

    model = get_model(num_classes=2).to(device)

    try:
        checkpoint = torch.load(MODEL_PATH, map_location=device)

        # train.py는 model_state_dict와 class_to_idx를 함께 저장한다.
        # 혹시 state_dict만 저장된 파일도 읽을 수 있도록 fallback을 둔다.
        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
            model.load_state_dict(checkpoint["model_state_dict"])
            class_to_idx = checkpoint.get("class_to_idx", {"gun": 0, "non_gun": 1})
        else:
            model.load_state_dict(checkpoint)
            class_to_idx = {"gun": 0, "non_gun": 1}

        model.eval()
        print(f"Model loaded: {MODEL_PATH}")
        print(f"class_to_idx: {class_to_idx}")
        return model, class_to_idx

    except Exception as error:
        print("Error: failed to load model.")
        print(f"Reason: {error}")
        print("You can still test the game with debug mode by pressing t.")
        return None, {"gun": 0, "non_gun": 1}


def get_center_roi(frame):
    """프레임 중앙에서 224x224 ROI를 잘라낸다."""
    frame_h, frame_w = frame.shape[:2]  # frame.shape는 (height, width, channel) 순서이다.

    x1 = max((frame_w - ROI_SIZE) // 2, 0)
    y1 = max((frame_h - ROI_SIZE) // 2, 0)
    x2 = min(x1 + ROI_SIZE, frame_w)
    y2 = min(y1 + ROI_SIZE, frame_h)

    roi = frame[y1:y2, x1:x2].copy()
    return roi, (x1, y1, x2, y2)


def count_jpg_files(folder):
    """Return how many jpg images are currently saved in a folder."""
    return len(list(folder.glob("*.jpg")))


def get_next_file_number(folder, prefix):
    """Find the next file number, such as gun_0505.jpg -> 506."""
    max_number = 0

    for file_path in folder.glob(f"{prefix}_*.jpg"):
        number_text = file_path.stem.replace(f"{prefix}_", "", 1)

        if number_text.isdigit():
            max_number = max(max_number, int(number_text))

    return max_number + 1


def save_roi_image(roi, folder, prefix, file_number):
    """Save the current ROI as a jpg image using the same ROI size as the game."""
    resized_roi = cv.resize(roi, (ROI_SIZE, ROI_SIZE))
    save_path = folder / f"{prefix}_{file_number:04d}.jpg"

    # cv.imencode + tofile is safer than cv.imwrite on Korean Windows paths.
    success, encoded_image = cv.imencode(".jpg", resized_roi)

    if not success:
        print(f"Failed to save: {save_path}")
        return False

    with save_path.open("wb") as file:
        encoded_image.tofile(file)

    print(f"Saved: {save_path}")
    return True


def draw_collection_text(frame, text, position, color=(255, 255, 255), scale=0.7):
    """Draw readable text with a black shadow on the webcam frame."""
    font = cv.FONT_HERSHEY_SIMPLEX
    x, y = position

    cv.putText(frame, text, (x + 2, y + 2), font, scale, (0, 0, 0), 3)
    cv.putText(frame, text, (x, y), font, scale, color, 2)


def run_timed_reliable_data_collection():
    """Collect reliable data for 1 minute: 30 seconds gun, then 30 seconds non_gun."""
    create_dataset_folders()

    gun_count = count_jpg_files(GUN_DIR)
    non_gun_count = count_jpg_files(NON_GUN_DIR)
    next_gun_number = get_next_file_number(GUN_DIR, "gun")
    next_non_gun_number = get_next_file_number(NON_GUN_DIR, "non_gun")
    session_gun_count = 0
    session_non_gun_count = 0

    cap = cv.VideoCapture(0)
    cap.set(cv.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv.CAP_PROP_FRAME_HEIGHT, 480)

    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    print("Timed data collection started.")
    print("0-30 sec: keep HAND-GUN pose inside the green ROI.")
    print("30-60 sec: remove hand / show non-gun background inside the green ROI.")
    print("Important: keep the same background for both 30-second phases.")
    print("Press q or ESC to cancel early.")

    start_time = time.time()
    last_save_time = 0.0

    while True:
        ret, frame = cap.read()

        if not ret:
            print("Error: Could not read frame from webcam.")
            break

        current_time = time.time()
        elapsed_time = current_time - start_time

        if elapsed_time >= DATA_COLLECTION_SECONDS:
            break

        roi, (x1, y1, x2, y2) = get_center_roi(frame)
        display_frame = frame.copy()

        is_gun_phase = elapsed_time < GUN_COLLECTION_SECONDS
        total_time_left = max(0.0, DATA_COLLECTION_SECONDS - elapsed_time)

        if is_gun_phase:
            phase_name = "GUN pose"
            phase_instruction = "Make a hand-gun pose inside ROI"
            phase_color = (0, 255, 255)
            target_folder = GUN_DIR
            target_prefix = "gun"
        else:
            phase_name = "NON_GUN"
            phase_instruction = "Remove hand / keep same background"
            phase_color = (0, 180, 255)
            target_folder = NON_GUN_DIR
            target_prefix = "non_gun"

        if current_time - last_save_time >= AUTO_CAPTURE_INTERVAL:
            if is_gun_phase:
                if save_roi_image(roi, target_folder, target_prefix, next_gun_number):
                    next_gun_number += 1
                    gun_count += 1
                    session_gun_count += 1
            else:
                if save_roi_image(roi, target_folder, target_prefix, next_non_gun_number):
                    next_non_gun_number += 1
                    non_gun_count += 1
                    session_non_gun_count += 1

            last_save_time = current_time

        cv.rectangle(display_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        draw_collection_text(display_frame, "1-minute reliable data collection", (20, 35))
        draw_collection_text(display_frame, f"Phase: {phase_name}", (20, 70), phase_color, 0.8)
        draw_collection_text(display_frame, phase_instruction, (20, 105), phase_color)
        draw_collection_text(display_frame, "Keep the same background in both phases", (20, 140), (255, 255, 0), 0.65)
        draw_collection_text(display_frame, f"Time left: {total_time_left:04.1f}s", (20, 175))
        draw_collection_text(display_frame, f"Saved this session - gun: {session_gun_count}", (20, 210))
        draw_collection_text(display_frame, f"Saved this session - non_gun: {session_non_gun_count}", (20, 245))
        draw_collection_text(display_frame, f"Total dataset - gun: {gun_count}, non_gun: {non_gun_count}", (20, 280))
        draw_collection_text(display_frame, "q or ESC: cancel", (20, 315), (200, 200, 200), 0.6)

        cv.imshow(DATA_WINDOW_NAME, display_frame)

        key = cv.waitKey(1) & 0xFF

        if key == ord("q") or key == 27:
            print("Data collection cancelled by user.")
            break

    cap.release()
    cv.destroyAllWindows()

    print("Timed data collection finished.")
    print(f"New gun images: {session_gun_count}")
    print(f"New non_gun images: {session_non_gun_count}")
    print(f"Total gun images: {gun_count}")
    print(f"Total non_gun images: {non_gun_count}")
    print("Next step: run py train.py to retrain the model with the new data.")


def estimate_index_fingertip_from_roi(roi, roi_box):
    """Estimate the index fingertip point from the hand contour inside the ROI.

    This is an OpenCV-only MVP approach. It does not use MediaPipe or a hand
    landmark model. The idea is:
    1. Find likely skin pixels.
    2. Find the largest hand-like contour.
    3. Prefer convex hull points that look like the index fingertip.

    When both the thumb and index finger are extended, the thumb can also be a
    strong contour point. To reduce thumb selection, this function prefers
    points that are far from the hand center, not near the wrist/bottom edge,
    and extended sideways or upward like an index finger.

    The result is returned in full-frame coordinates, not ROI coordinates.
    """
    roi_x1, roi_y1, _, _ = roi_box
    roi_h, roi_w = roi.shape[:2]

    # YCrCb skin threshold works better than raw BGR for simple skin detection.
    ycrcb = cv.cvtColor(roi, cv.COLOR_BGR2YCrCb)
    lower_skin = np.array([0, 133, 77], dtype=np.uint8)
    upper_skin = np.array([255, 173, 127], dtype=np.uint8)
    ycrcb_mask = cv.inRange(ycrcb, lower_skin, upper_skin)

    # Add an HSV mask to make the result less sensitive to lighting changes.
    hsv = cv.cvtColor(roi, cv.COLOR_BGR2HSV)
    hsv_mask = cv.inRange(
        hsv,
        np.array([0, 20, 40], dtype=np.uint8),
        np.array([25, 180, 255], dtype=np.uint8),
    )
    # AND is stricter than OR. This prevents wood, desk, and bright background
    # regions from being treated as hand too easily.
    mask = cv.bitwise_and(ycrcb_mask, hsv_mask)

    # Remove small noise and fill small holes in the hand mask.
    kernel = np.ones((5, 5), np.uint8)
    mask = cv.morphologyEx(mask, cv.MORPH_OPEN, kernel)
    mask = cv.morphologyEx(mask, cv.MORPH_CLOSE, kernel)
    mask = cv.GaussianBlur(mask, (5, 5), 0)

    contours, _ = cv.findContours(mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

    if not contours:
        return None

    hand_contour = max(contours, key=cv.contourArea)
    contour_area = cv.contourArea(hand_contour)

    if contour_area < MIN_HAND_CONTOUR_AREA:
        return None

    skin_ratio = cv.countNonZero(mask) / float(roi_w * roi_h)

    if skin_ratio < 0.02 or skin_ratio > 0.55:
        return None

    x, y, w, h = cv.boundingRect(hand_contour)

    # Large background areas often touch most of the ROI. A real hand can be
    # large, but it usually does not fill almost the whole 224x224 ROI.
    if contour_area > roi_w * roi_h * 0.55:
        return None

    if w > roi_w * 0.95 and h > roi_h * 0.70:
        return None

    if h > roi_h * 0.95 and w > roi_w * 0.70:
        return None

    moments = cv.moments(hand_contour)

    if moments["m00"] == 0:
        return None

    center_x = int(moments["m10"] / moments["m00"])
    center_y = int(moments["m01"] / moments["m00"])

    # Convex hull gives outer boundary candidates. Fingertips usually appear on
    # the convex hull, while inner contour noise is ignored.
    hull_points = cv.convexHull(hand_contour, returnPoints=True).reshape(-1, 2)

    best_point = None
    best_score = float("-inf")
    center = np.array([center_x, center_y])
    min_distance = max(28.0, np.sqrt(contour_area) * 0.35)

    for point in hull_points:
        px, py = int(point[0]), int(point[1])

        # Ignore bottom-edge points because they are often wrist/arm, not a fingertip.
        if py > roi_h - 12:
            continue

        dx = px - center_x
        dy = py - center_y
        distance = float(np.linalg.norm(point - center))

        if distance < min_distance:
            continue

        horizontal_extension = abs(dx)
        upward_extension = max(0, center_y - py)
        downward_penalty = max(0, py - center_y)

        # Index finger in a hand-gun gesture is usually the longest outer point.
        # The score favors long, sideways/upward points and penalizes wrist-like
        # points below the hand center.
        score = (
            distance
            + 0.75 * horizontal_extension
            + 0.35 * upward_extension
            - 0.80 * downward_penalty
        )

        if score > best_score:
            best_score = score
            best_point = (px, py)

    if best_point is None:
        # Fallback: use the farthest non-bottom contour point.
        contour_points = hand_contour.reshape(-1, 2)
        contour_points = contour_points[contour_points[:, 1] <= roi_h - 12]

        if len(contour_points) == 0:
            return None

        distances = np.linalg.norm(contour_points - center, axis=1)
        best_point = tuple(contour_points[int(np.argmax(distances))])

    fingertip_x, fingertip_y = best_point
    return (int(roi_x1 + fingertip_x), int(roi_y1 + fingertip_y))


def create_random_zombie_box(frame_w, frame_h, spawn_area=None):
    """화면 안쪽의 랜덤 위치에 좀비 bounding box를 만든다.

    spawn_area가 있으면 좀비는 그 사각형 영역 안에서만 나온다.
    여기서는 초록색 ROI 안에 좀비를 생성해서 손끝 조준으로 맞추기 쉽게 한다.
    """
    zombie_w, zombie_h = ZOMBIE_SIZE

    # 화면이 아주 작아도 좌표 계산이 깨지지 않도록 최소값을 보정한다.
    max_x = max(frame_w - zombie_w, 0)
    max_y = max(frame_h - zombie_h, 0)

    if spawn_area is None:
        x1 = random.randint(0, max_x)
        y1 = random.randint(0, max_y)
    else:
        area_x1, area_y1, area_x2, area_y2 = spawn_area

        min_x = max(area_x1, 0)
        max_area_x = min(area_x2 - zombie_w, max_x)
        min_y = max(area_y1, 0)
        max_area_y = min(area_y2 - zombie_h, max_y)

        if min_x <= max_area_x:
            x1 = random.randint(min_x, max_area_x)
        else:
            x1 = random.randint(0, max_x)

        if min_y <= max_area_y:
            y1 = random.randint(min_y, max_area_y)
        else:
            y1 = random.randint(0, max_y)

    x2 = x1 + zombie_w
    y2 = y1 + zombie_h

    return (x1, y1, x2, y2)


def is_fingertip_target_hit(box, center, radius):
    """손끝 주변의 작은 원형 타겟이 좀비 박스와 겹치는지 확인한다."""
    center_x, center_y = center
    x1, y1, x2, y2 = box

    # 원 중심에서 좀비 박스에 가장 가까운 점을 찾는다.
    nearest_x = min(max(center_x, x1), x2)
    nearest_y = min(max(center_y, y1), y2)

    dx = center_x - nearest_x
    dy = center_y - nearest_y

    return (dx * dx) + (dy * dy) <= radius * radius


def draw_zombie(frame, zombie_box, zombie_image):
    """좀비 이미지를 화면에 표시한다.

    zombie.png가 있으면 PNG alpha를 고려해서 합성하고,
    이미지가 없으면 사각형으로 표시한다.
    """
    x1, y1, x2, y2 = zombie_box

    if zombie_image is not None:
        overlay_image_alpha(frame, zombie_image, x1, y1)
    else:
        # 이미지가 없을 때도 게임 로직을 테스트할 수 있도록 사각형으로 대체한다.
        cv.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), thickness=-1)
        cv.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), thickness=3)
        cv.putText(
            frame,
            "ZOMBIE",
            (x1 + 15, y1 + 65),
            cv.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2,
        )

    return frame


def draw_fingertip_target(frame, center, radius):
    """실제 hit 판정에 쓰이는 손끝 주변 타겟 범위를 표시한다."""
    cv.circle(frame, center, radius, (255, 255, 0), 2)
    return frame


def draw_debug_info(
    frame,
    model_label,
    model_confidence,
    debug_gun,
    effective_gun,
    aim_from_fingertip,
    gun_votes,
    history_size,
):
    """모델 예측과 debug gun 상태를 구분해서 화면에 표시한다."""
    font = cv.FONT_HERSHEY_SIMPLEX
    font_scale = 0.6
    thickness = 2
    x = 20
    y = 195

    lines = [
        f"Model prediction: {model_label} ({model_confidence:.2f})",
        f"Debug gun toggle: {'ON' if debug_gun else 'OFF'}",
        f"Used as gun: {'YES' if effective_gun else 'NO'}",
        f"Gun votes: {gun_votes}/{history_size}",
        f"Gun threshold: {GUN_CONFIDENCE_THRESHOLD:.2f}",
        f"Aim point: {'index fingertip' if aim_from_fingertip else 'center fallback'}",
        "t: toggle debug gun | r: respawn zombie",
    ]

    for i, text in enumerate(lines):
        text_y = y + (i * 28)
        cv.putText(frame, text, (x + 2, text_y + 2), font, font_scale, (0, 0, 0), thickness + 1)
        cv.putText(frame, text, (x, text_y), font, font_scale, (255, 255, 255), thickness)


def predict_gesture(model, roi, device, idx_to_class):
    """ROI 이미지를 모델에 넣어 gun / non_gun을 예측한다."""
    if model is None:
        return "model unavailable", 0.0, False

    # 모델 추론에서는 gradient가 필요 없으므로 torch.no_grad()를 사용한다.
    with torch.no_grad():
        input_tensor = preprocess_roi_for_model(roi, device)
        outputs = model(input_tensor)
        probabilities = torch.softmax(outputs, dim=1)

        confidence_tensor, predicted_tensor = torch.max(probabilities, dim=1)
        confidence = confidence_tensor.item()
        predicted_index = predicted_tensor.item()

    pred_label = idx_to_class.get(predicted_index, f"class_{predicted_index}")
    model_gun = pred_label == "gun"

    return pred_label, confidence, model_gun


def run_hand_gun_game():
    create_project_folders()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model, class_to_idx = load_hand_gun_model(device)
    idx_to_class = {class_index: class_name for class_name, class_index in class_to_idx.items()}
    zombie_image = load_zombie_image()

    # 0번 카메라는 보통 기본 웹캠이다.
    cap = cv.VideoCapture(0)
    cap.set(cv.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv.CAP_PROP_FRAME_HEIGHT, 480)

    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    ret, first_frame = cap.read()

    if not ret:
        print("Error: Could not read frame from webcam.")
        cap.release()
        return

    frame_h, frame_w = first_frame.shape[:2]
    crosshair_center = (frame_w // 2, frame_h // 2)
    _, first_roi_box = get_center_roi(first_frame)
    zombie_box = create_random_zombie_box(frame_w, frame_h, first_roi_box)

    score = 0
    frame_count = 0
    start_time = time.time()
    last_hit_time = 0.0

    # 모델 예측 상태와 debug 상태는 따로 관리한다.
    model_label = "waiting"
    model_confidence = 0.0
    model_gun = False
    prediction_history = deque(maxlen=PREDICTION_HISTORY_SIZE)
    debug_gun = False

    print("Game started.")
    print("Controls: t = toggle debug gun, r = respawn zombie, q or ESC = quit")

    while True:
        ret, frame = cap.read()

        if not ret:
            print("Error: Could not read frame from webcam.")
            break

        frame_count += 1
        frame_h, frame_w = frame.shape[:2]

        elapsed_time = time.time() - start_time
        time_left = GAME_TIME_SECONDS - elapsed_time

        if time_left <= 0:
            break

        roi, (roi_x1, roi_y1, roi_x2, roi_y2) = get_center_roi(frame)
        fingertip_point = estimate_index_fingertip_from_roi(roi, (roi_x1, roi_y1, roi_x2, roi_y2))

        # If fingertip detection fails, keep the old center crosshair as a safe fallback.
        if fingertip_point is not None:
            crosshair_center = fingertip_point
            aim_from_fingertip = True
        else:
            crosshair_center = (frame_w // 2, frame_h // 2)
            aim_from_fingertip = False

        # 속도가 느려지는 것을 줄이기 위해 3프레임마다 한 번만 모델 추론을 한다.
        if frame_count % INFERENCE_INTERVAL == 0:
            raw_model_label, raw_model_confidence, raw_model_gun = predict_gesture(
                model=model,
                roi=roi,
                device=device,
                idx_to_class=idx_to_class,
            )

            if FORCE_NON_GUN_WHEN_NO_HAND and not aim_from_fingertip:
                model_label = "non_gun(no hand)"
                model_confidence = raw_model_confidence
                model_gun = False
            else:
                model_label = raw_model_label
                model_confidence = raw_model_confidence
                model_gun = raw_model_gun

            confident_gun = model_gun and model_confidence >= GUN_CONFIDENCE_THRESHOLD
            prediction_history.append(confident_gun)

        gun_votes = sum(prediction_history)
        smoothed_model_gun = gun_votes >= MIN_GUN_VOTES

        # debug_gun이 True이면 모델 예측과 관계없이 gun 상태로 처리한다.
        effective_gun = debug_gun or smoothed_model_gun

        # 모델이 없을 때는 debug_gun만으로 게임 로직을 테스트한다.
        if model is None:
            effective_gun = debug_gun
            gun_votes = 0

        # 실제 모델 사격은 손끝이 잡힌 경우에만 허용한다.
        # 이렇게 하면 손이 없는데 모델이 gun으로 튀는 경우 점수가 오르는 일을 줄일 수 있다.
        current_frame_gun = model_gun and model_confidence >= GUN_CONFIDENCE_THRESHOLD
        can_shoot = debug_gun or (
            effective_gun
            and current_frame_gun
            and aim_from_fingertip
        )

        # gun 상태이고 중앙 crosshair가 좀비 박스 안에 있으면 hit 처리한다.
        current_time = time.time()

        if (
            can_shoot
            and is_fingertip_target_hit(zombie_box, crosshair_center, FINGERTIP_TARGET_RADIUS)
            and current_time - last_hit_time >= HIT_COOLDOWN_SECONDS
        ):
            score += 1
            last_hit_time = current_time
            print(
                f"Hit! Score: {score} "
                f"label={model_label} conf={model_confidence:.2f} "
                f"votes={gun_votes}/{PREDICTION_HISTORY_SIZE} "
                f"aim={'fingertip' if aim_from_fingertip else 'fallback'}"
            )
            zombie_box = create_random_zombie_box(frame_w, frame_h, (roi_x1, roi_y1, roi_x2, roi_y2))

        # 화면 표시용 요소를 그린다.
        draw_zombie(frame, zombie_box, zombie_image)
        cv.rectangle(frame, (roi_x1, roi_y1), (roi_x2, roi_y2), (0, 255, 0), 2)
        draw_fingertip_target(frame, crosshair_center, FINGERTIP_TARGET_RADIUS)
        draw_crosshair(frame, crosshair_center)

        # draw_status의 Prediction은 모델 자체의 예측을 보여준다.
        # debug gun 상태는 아래 draw_debug_info에서 따로 표시한다.
        debug_mode_visible = model is None or debug_gun
        draw_status(
            frame=frame,
            score=score,
            time_left=time_left,
            pred_label=model_label,
            confidence=model_confidence,
            debug_mode=debug_mode_visible,
        )

        # 모델 예측과 debug 상태를 따로 보여준다.
        draw_debug_info(
            frame=frame,
            model_label=model_label,
            model_confidence=model_confidence,
            debug_gun=debug_gun,
            effective_gun=effective_gun,
            aim_from_fingertip=aim_from_fingertip,
            gun_votes=gun_votes,
            history_size=PREDICTION_HISTORY_SIZE,
        )

        cv.imshow(WINDOW_NAME, frame)

        key = cv.waitKey(1) & 0xFF

        if key == ord("t"):
            debug_gun = not debug_gun
            print(f"Debug gun: {'ON' if debug_gun else 'OFF'}")

        elif key == ord("r"):
            zombie_box = create_random_zombie_box(frame_w, frame_h, (roi_x1, roi_y1, roi_x2, roi_y2))
            print("Zombie respawned.")

        elif key == ord("q") or key == 27:
            break

    cap.release()
    cv.destroyAllWindows()

    print("Game over.")
    print(f"Final score: {score}")


def choose_game_mode():
    """Ask the player whether to play or collect more data."""
    print("=" * 60)
    print("AR Zombie Shooting Game")
    print("=" * 60)
    print("1. Play hand-gun gesture zombie shooting")
    print("d. Collect 1-minute reliable data")
    print("q. Quit")
    print()

    while True:
        choice = input("Select option (1/d/q, Enter = 1): ").strip().lower()

        if choice == "":
            return "1"

        if choice in {"1", "d", "data", "q", "quit"}:
            return choice

        print("Please enter 1, d, or q.")


def main():
    """Game launcher.

    The hand-gun gesture game is the main mode.
    The data option collects 30 seconds of gun and 30 seconds of non_gun images.
    """
    choice = choose_game_mode()

    if choice == "1":
        run_hand_gun_game()
    elif choice in {"d", "data"}:
        run_timed_reliable_data_collection()
    else:
        print("Game launcher closed.")


if __name__ == "__main__":
    main()

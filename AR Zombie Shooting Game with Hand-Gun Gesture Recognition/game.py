#! python3.12

import random
import time
from pathlib import Path

import cv2 as cv
import numpy as np
import torch

from model import get_model
from utils import (
    draw_crosshair,
    draw_status,
    is_point_inside_box,
    overlay_image_alpha,
    preprocess_roi_for_model,
)


# MVP에서 사용할 고정 설정값이다.
ROI_SIZE = 224
GAME_TIME_SECONDS = 60
INFERENCE_INTERVAL = 3
WINDOW_NAME = "AR Zombie Shooting Game"
ZOMBIE_SIZE = (120, 120)  # cv.resize는 (width, height) 순서를 사용한다.

# 이 파일이 있는 프로젝트 폴더를 기준으로 모델과 에셋 경로를 만든다.
PROJECT_DIR = Path(__file__).resolve().parent
MODELS_DIR = PROJECT_DIR / "models"
ASSETS_DIR = PROJECT_DIR / "assets"
MODEL_PATH = MODELS_DIR / "hand_gun_model.pth"
ZOMBIE_IMAGE_PATH = ASSETS_DIR / "zombie.png"


def create_project_folders():
    """게임 실행에 필요한 폴더가 없으면 자동 생성한다."""
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    ASSETS_DIR.mkdir(parents=True, exist_ok=True)


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


def create_random_zombie_box(frame_w, frame_h):
    """화면 안쪽의 랜덤 위치에 좀비 bounding box를 만든다."""
    zombie_w, zombie_h = ZOMBIE_SIZE

    # 화면이 아주 작아도 좌표 계산이 깨지지 않도록 최소값을 보정한다.
    max_x = max(frame_w - zombie_w, 0)
    max_y = max(frame_h - zombie_h, 0)

    x1 = random.randint(0, max_x)
    y1 = random.randint(0, max_y)
    x2 = x1 + zombie_w
    y2 = y1 + zombie_h

    return (x1, y1, x2, y2)


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


def draw_debug_info(frame, model_label, model_confidence, debug_gun, effective_gun):
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


def main():
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
    zombie_box = create_random_zombie_box(frame_w, frame_h)

    score = 0
    frame_count = 0
    start_time = time.time()

    # 모델 예측 상태와 debug 상태는 따로 관리한다.
    model_label = "waiting"
    model_confidence = 0.0
    model_gun = False
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
        crosshair_center = (frame_w // 2, frame_h // 2)

        elapsed_time = time.time() - start_time
        time_left = GAME_TIME_SECONDS - elapsed_time

        if time_left <= 0:
            break

        roi, (roi_x1, roi_y1, roi_x2, roi_y2) = get_center_roi(frame)

        # 속도가 느려지는 것을 줄이기 위해 3프레임마다 한 번만 모델 추론을 한다.
        if frame_count % INFERENCE_INTERVAL == 0:
            model_label, model_confidence, model_gun = predict_gesture(
                model=model,
                roi=roi,
                device=device,
                idx_to_class=idx_to_class,
            )

        # debug_gun이 True이면 모델 예측과 관계없이 gun 상태로 처리한다.
        effective_gun = debug_gun or model_gun

        # 모델이 없을 때는 debug_gun만으로 게임 로직을 테스트한다.
        if model is None:
            effective_gun = debug_gun

        # gun 상태이고 중앙 crosshair가 좀비 박스 안에 있으면 hit 처리한다.
        if effective_gun and is_point_inside_box(crosshair_center, zombie_box):
            score += 1
            print(f"Hit! Score: {score}")
            zombie_box = create_random_zombie_box(frame_w, frame_h)

        # 화면 표시용 요소를 그린다.
        draw_zombie(frame, zombie_box, zombie_image)
        cv.rectangle(frame, (roi_x1, roi_y1), (roi_x2, roi_y2), (0, 255, 0), 2)
        draw_crosshair(frame, crosshair_center)

        # draw_status는 기본 게임 정보를 표시한다.
        effective_label = "gun" if effective_gun else "non_gun"
        debug_mode_visible = model is None or debug_gun
        draw_status(
            frame=frame,
            score=score,
            time_left=time_left,
            pred_label=effective_label,
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
        )

        cv.imshow(WINDOW_NAME, frame)

        key = cv.waitKey(1) & 0xFF

        if key == ord("t"):
            debug_gun = not debug_gun
            print(f"Debug gun: {'ON' if debug_gun else 'OFF'}")

        elif key == ord("r"):
            zombie_box = create_random_zombie_box(frame_w, frame_h)
            print("Zombie respawned.")

        elif key == ord("q") or key == 27:
            break

    cap.release()
    cv.destroyAllWindows()

    print("Game over.")
    print(f"Final score: {score}")


if __name__ == "__main__":
    main()

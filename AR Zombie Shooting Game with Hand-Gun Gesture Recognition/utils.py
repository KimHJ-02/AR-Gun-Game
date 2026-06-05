import cv2 as cv
import numpy as np
import torch


def overlay_image_alpha(background, overlay, x, y):
    """background 위에 overlay 이미지를 올린다.

    OpenCV 이미지는 보통 BGR 순서이고, PNG 이미지는 투명도(alpha)를 포함한
    BGRA 형태일 수 있다. 이 함수는 overlay가 화면 밖으로 살짝 나가도
    잘리는 영역만 계산해서 안전하게 합성한다.
    """
    bg_h, bg_w = background.shape[:2]
    overlay_h, overlay_w = overlay.shape[:2]

    # overlay 이미지가 실제 화면에서 차지할 영역을 계산한다.
    x1 = max(x, 0)
    y1 = max(y, 0)
    x2 = min(x + overlay_w, bg_w)
    y2 = min(y + overlay_h, bg_h)

    # 화면과 겹치는 부분이 없으면 아무것도 하지 않고 원본을 반환한다.
    if x1 >= x2 or y1 >= y2:
        return background

    # overlay 이미지에서 잘라서 사용할 영역을 계산한다.
    overlay_x1 = x1 - x
    overlay_y1 = y1 - y
    overlay_x2 = overlay_x1 + (x2 - x1)
    overlay_y2 = overlay_y1 + (y2 - y1)

    overlay_crop = overlay[overlay_y1:overlay_y2, overlay_x1:overlay_x2]
    background_crop = background[y1:y2, x1:x2]

    if overlay_crop.shape[2] == 4:
        # BGRA 이미지의 4번째 채널은 투명도(alpha)이다.
        overlay_bgr = overlay_crop[:, :, :3]
        alpha = overlay_crop[:, :, 3] / 255.0

        # 계산하기 쉽게 alpha를 (height, width, 1) 형태로 바꾼다.
        alpha = alpha[:, :, np.newaxis]

        blended = (overlay_bgr * alpha) + (background_crop * (1.0 - alpha))
        background[y1:y2, x1:x2] = blended.astype(np.uint8)
    else:
        # BGR 3채널 이미지는 투명도가 없으므로 그대로 덮어쓴다.
        background[y1:y2, x1:x2] = overlay_crop

    return background


def is_point_inside_box(point, box):
    """point가 box 안에 있는지 확인한다."""
    x, y = point
    x1, y1, x2, y2 = box

    return x1 <= x <= x2 and y1 <= y <= y2


def draw_crosshair(frame, center):
    """화면에 조준점을 그린다."""
    x, y = center
    color = (0, 255, 255)  # OpenCV는 BGR 순서이므로 노란색은 (0, 255, 255)이다.
    thickness = 2
    line_length = 20

    # 가로선과 세로선을 그려 십자선을 만든다.
    cv.line(frame, (x - line_length, y), (x + line_length, y), color, thickness)
    cv.line(frame, (x, y - line_length), (x, y + line_length), color, thickness)

    # 중앙에 작은 원을 그려 조준점을 더 잘 보이게 한다.
    cv.circle(frame, (x, y), 6, color, thickness)

    return frame


def draw_status(frame, score, time_left, pred_label, confidence, debug_mode=False):
    """게임 상태 정보를 화면 왼쪽 위에 표시한다."""
    font = cv.FONT_HERSHEY_SIMPLEX
    font_scale = 0.7
    thickness = 2
    color = (255, 255, 255)
    line_gap = 30
    x = 20
    y = 35

    # 잘 보이도록 검은색 그림자 글씨를 먼저 그리고, 흰색 글씨를 위에 그린다.
    status_lines = [
        f"Score: {score}",
        f"Time: {max(0, int(time_left))}s",
        f"Prediction: {pred_label}",
        f"Confidence: {confidence:.2f}",
    ]

    if debug_mode:
        status_lines.append("DEBUG MODE")

    for i, text in enumerate(status_lines):
        text_y = y + (i * line_gap)
        cv.putText(frame, text, (x + 2, text_y + 2), font, font_scale, (0, 0, 0), thickness + 1)
        cv.putText(frame, text, (x, text_y), font, font_scale, color, thickness)

    return frame


def preprocess_roi_for_model(roi, device):
    """OpenCV ROI 이미지를 PyTorch 모델 입력 형태로 변환한다.

    최종 결과는 1 x 3 x 224 x 224 형태이다.
    """
    # OpenCV는 BGR을 사용하지만, 일반적인 PyTorch 이미지 모델은 RGB를 사용한다.
    rgb = cv.cvtColor(roi, cv.COLOR_BGR2RGB)

    # cv.resize의 size 인자는 (width, height) 순서이다.
    resized = cv.resize(rgb, (224, 224))

    # ToTensor와 같은 방식: H x W x C, 0~255 이미지를 C x H x W, 0~1 tensor로 바꾼다.
    tensor = torch.from_numpy(resized).float() / 255.0
    tensor = tensor.permute(2, 0, 1)

    # ImageNet에서 자주 쓰는 평균과 표준편차로 normalize한다.
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    tensor = (tensor - mean) / std

    # batch dimension을 추가해서 1 x 3 x 224 x 224로 만든 뒤 device로 이동한다.
    tensor = tensor.unsqueeze(0).to(device)

    return tensor

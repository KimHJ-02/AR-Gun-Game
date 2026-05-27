#! python3.12

from pathlib import Path

import cv2 as cv


# 1차 MVP에서는 모델 입력 크기를 224x224로 고정한다.
ROI_SIZE = 224
WINDOW_NAME = "Collect Hand Gesture Data"

# 이 파일이 있는 프로젝트 폴더를 기준으로 dataset 경로를 만든다.
PROJECT_DIR = Path(__file__).resolve().parent
DATASET_DIR = PROJECT_DIR / "dataset"
GUN_DIR = DATASET_DIR / "gun"
NON_GUN_DIR = DATASET_DIR / "non_gun"


def create_dataset_folders():
    """데이터 저장 폴더가 없으면 자동으로 만든다."""
    GUN_DIR.mkdir(parents=True, exist_ok=True)
    NON_GUN_DIR.mkdir(parents=True, exist_ok=True)


def count_jpg_files(folder):
    """현재 폴더 안에 저장된 jpg 이미지 개수를 센다."""
    return len(list(folder.glob("*.jpg")))


def get_next_file_number(folder, prefix):
    """기존 파일명을 확인해서 다음 저장 번호를 구한다.

    예를 들어 gun_0001.jpg, gun_0002.jpg가 있으면 다음 번호는 3이다.
    """
    max_number = 0

    for file_path in folder.glob(f"{prefix}_*.jpg"):
        number_text = file_path.stem.replace(f"{prefix}_", "", 1)

        if number_text.isdigit():
            max_number = max(max_number, int(number_text))

    return max_number + 1


def get_center_roi(frame):
    """프레임 중앙에서 ROI 영역을 잘라낸다.

    화면에 그리는 사각형 좌표와 저장에 사용하는 좌표를 같게 만들어야
    사용자가 보는 영역과 실제 저장 이미지가 일관된다.
    """
    frame_h, frame_w = frame.shape[:2]

    x1 = max((frame_w - ROI_SIZE) // 2, 0)
    y1 = max((frame_h - ROI_SIZE) // 2, 0)
    x2 = min(x1 + ROI_SIZE, frame_w)
    y2 = min(y1 + ROI_SIZE, frame_h)

    roi = frame[y1:y2, x1:x2].copy()
    return roi, (x1, y1, x2, y2)


def save_roi_image(roi, folder, prefix, file_number):
    """ROI 이미지를 224x224 jpg 파일로 저장한다."""
    resized_roi = cv.resize(roi, (ROI_SIZE, ROI_SIZE))
    save_path = folder / f"{prefix}_{file_number:04d}.jpg"

    # OpenCV의 JPEG 인코더를 사용하고, Python 파일 객체로 저장해서
    # Windows 한글 경로에서도 저장 문제가 덜 생기게 한다.
    success, encoded_image = cv.imencode(".jpg", resized_roi)

    if success:
        with save_path.open("wb") as file:
            encoded_image.tofile(file)
        print(f"Saved: {save_path}")
        return True

    print(f"Failed to save: {save_path}")
    return False


def draw_text(frame, text, position, color=(255, 255, 255)):
    """화면에 잘 보이도록 그림자와 함께 글자를 표시한다."""
    font = cv.FONT_HERSHEY_SIMPLEX
    font_scale = 0.65
    thickness = 2
    x, y = position

    cv.putText(frame, text, (x + 2, y + 2), font, font_scale, (0, 0, 0), thickness + 1)
    cv.putText(frame, text, (x, y), font, font_scale, color, thickness)


def main():
    create_dataset_folders()

    gun_count = count_jpg_files(GUN_DIR)
    non_gun_count = count_jpg_files(NON_GUN_DIR)
    next_gun_number = get_next_file_number(GUN_DIR, "gun")
    next_non_gun_number = get_next_file_number(NON_GUN_DIR, "non_gun")

    # 0번 카메라는 보통 노트북 기본 웹캠 또는 첫 번째 연결 카메라이다.
    cap = cv.VideoCapture(0)
    cap.set(cv.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv.CAP_PROP_FRAME_HEIGHT, 480)

    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    print("Data collection started.")
    print("Press g to save gun, n to save non_gun, q or ESC to quit.")

    while True:
        ret, frame = cap.read()

        if not ret:
            print("Error: Could not read frame from webcam.")
            break

        # 원본 frame에서 ROI를 먼저 잘라둔다. 화면 표시용 그림은 복사본에만 그린다.
        roi, (x1, y1, x2, y2) = get_center_roi(frame)
        display_frame = frame.copy()

        # 중앙 ROI 사각형을 표시한다.
        cv.rectangle(display_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        draw_text(display_frame, "Put your hand inside the ROI", (20, 35))
        draw_text(display_frame, "g: save gun | n: save non_gun | q/ESC: quit", (20, 65))
        draw_text(display_frame, f"Gun images: {gun_count}", (20, 100), (0, 255, 255))
        draw_text(display_frame, f"Non-gun images: {non_gun_count}", (20, 130), (0, 255, 255))
        draw_text(display_frame, f"ROI save size: {ROI_SIZE} x {ROI_SIZE}", (20, 160))

        cv.imshow(WINDOW_NAME, display_frame)

        key = cv.waitKey(1) & 0xFF

        if key == ord("g"):
            if save_roi_image(roi, GUN_DIR, "gun", next_gun_number):
                next_gun_number += 1
                gun_count += 1

        elif key == ord("n"):
            if save_roi_image(roi, NON_GUN_DIR, "non_gun", next_non_gun_number):
                next_non_gun_number += 1
                non_gun_count += 1

        elif key == ord("q") or key == 27:
            break

    cap.release()
    cv.destroyAllWindows()
    print("Data collection finished.")


if __name__ == "__main__":
    main()

# AR Zombie Shooting Game with Hand-Gun Gesture Recognition

Python, OpenCV, PyTorch를 사용하는 컴퓨터 비전 텀프로젝트입니다.

이번 단계에서는 1차 MVP 구현을 위한 프로젝트 폴더 구조와 의존성 파일만 준비합니다.

## Folder Structure

```text
project/
├── README.md
├── requirements.txt
├── collect_data.py
├── model.py
├── train.py
├── utils.py
├── game.py
├── dataset/
│   ├── gun/
│   └── non_gun/
├── models/
├── assets/
│   └── zombie.png
└── screenshots/
```

## Files

- `collect_data.py`: 웹캠으로 gun / non_gun 학습 이미지를 수집할 파일입니다.
- `model.py`: PyTorch CNN 모델 구조를 정의할 파일입니다.
- `train.py`: 수집한 데이터로 모델을 학습하고 저장할 파일입니다.
- `utils.py`: ROI 처리, 폴더 생성, 이미지 전처리 등 공통 함수를 둘 파일입니다.
- `game.py`: 웹캠 기반 AR 좀비 슈팅 게임을 실행할 파일입니다.
- `dataset/gun/`: 손 총 모양 이미지가 저장될 폴더입니다.
- `dataset/non_gun/`: 손 총이 아닌 이미지가 저장될 폴더입니다.
- `models/`: 학습된 모델 파일이 저장될 폴더입니다.
- `assets/`: 좀비 이미지 등 게임 리소스를 둘 폴더입니다.
- `screenshots/`: 실행 화면 캡처를 저장할 폴더입니다.

## Windows Run Commands

```powershell
py collect_data.py
py train.py
py game.py
```

## Requirements

필요한 라이브러리는 `requirements.txt`에 정리되어 있습니다.

```powershell
py -m pip install -r requirements.txt
```

## Note

`dataset/gun`, `dataset/non_gun`, `models`, `assets`, `screenshots` 폴더는 코드 실행 시에도 자동 생성되도록 구현할 예정입니다.

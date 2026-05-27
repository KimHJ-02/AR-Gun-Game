#! python3.12

from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

from model import get_model


# 프로젝트에서 공통으로 사용할 기본 설정값이다.
IMAGE_SIZE = 224
BATCH_SIZE = 16
EPOCHS = 10
LEARNING_RATE = 0.001
RANDOM_SEED = 42

# 이 파일이 있는 폴더를 기준으로 경로를 만든다.
PROJECT_DIR = Path(__file__).resolve().parent
DATASET_DIR = PROJECT_DIR / "dataset"
GUN_DIR = DATASET_DIR / "gun"
NON_GUN_DIR = DATASET_DIR / "non_gun"
MODELS_DIR = PROJECT_DIR / "models"
MODEL_SAVE_PATH = MODELS_DIR / "hand_gun_model.pth"

# ImageFolder가 읽을 수 있는 대표적인 이미지 확장자이다.
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}


def count_images(folder):
    """폴더 안에 있는 이미지 파일 개수를 센다."""
    return sum(
        1
        for file_path in folder.rglob("*")
        if file_path.is_file() and file_path.suffix.lower() in IMAGE_EXTENSIONS
    )


def check_dataset():
    """학습 전에 dataset 폴더가 올바른지 확인한다.

    초보자가 가장 자주 만나는 문제는 폴더 이름이 다르거나 이미지가 없는 경우이다.
    그래서 학습을 시작하기 전에 친절한 메시지를 출력하고 멈춘다.
    """
    if not DATASET_DIR.exists():
        print(f"Error: dataset folder not found: {DATASET_DIR}")
        print("Please run py collect_data.py first, or create dataset/gun and dataset/non_gun folders.")
        return False

    missing_folders = []

    if not GUN_DIR.exists():
        missing_folders.append(str(GUN_DIR))

    if not NON_GUN_DIR.exists():
        missing_folders.append(str(NON_GUN_DIR))

    if missing_folders:
        print("Error: required class folder is missing.")
        for folder in missing_folders:
            print(f"- {folder}")
        print("Expected folder structure: dataset/gun and dataset/non_gun")
        return False

    gun_count = count_images(GUN_DIR)
    non_gun_count = count_images(NON_GUN_DIR)

    if gun_count == 0 or non_gun_count == 0:
        print("Error: both classes need at least one image.")
        print(f"gun images: {gun_count}")
        print(f"non_gun images: {non_gun_count}")
        print("Collect data with py collect_data.py before training.")
        return False

    if gun_count + non_gun_count < 2:
        print("Error: at least 2 images are needed for train/validation split.")
        return False

    return True


def make_transforms():
    """학습용 transform과 검증용 transform을 만든다."""
    # 학습 transform은 데이터 다양성을 늘리기 위해 augmentation을 포함한다.
    train_transform = transforms.Compose(
        [
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ]
    )

    # 검증 transform은 모델 성능을 일정하게 평가하기 위해 랜덤 변형을 넣지 않는다.
    val_transform = transforms.Compose(
        [
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ]
    )

    return train_transform, val_transform


def make_train_val_indices(dataset_size):
    """전체 데이터를 8:2 비율로 train/validation으로 나눈다."""
    generator = torch.Generator().manual_seed(RANDOM_SEED)
    indices = torch.randperm(dataset_size, generator=generator).tolist()

    train_size = int(dataset_size * 0.8)
    val_size = dataset_size - train_size

    # 데이터가 아주 적어도 validation set이 1장 이상 생기도록 보정한다.
    if val_size == 0:
        val_size = 1
        train_size = dataset_size - val_size

    train_indices = indices[:train_size]
    val_indices = indices[train_size:]

    return train_indices, val_indices


def calculate_accuracy(outputs, labels):
    """모델 출력과 정답 label을 비교해서 accuracy를 계산한다."""
    _, predicted = torch.max(outputs, dim=1)
    correct = (predicted == labels).sum().item()
    total = labels.size(0)
    return correct, total


def train_one_epoch(model, data_loader, criterion, optimizer, device):
    """1 epoch 동안 모델을 학습한다."""
    model.train()

    running_loss = 0.0
    correct_count = 0
    total_count = 0

    for images, labels in data_loader:
        images = images.to(device)
        labels = labels.to(device)

        # 이전 batch에서 계산된 gradient를 지운다.
        optimizer.zero_grad()

        # forward: 이미지를 모델에 넣어 예측값을 얻는다.
        outputs = model(images)
        loss = criterion(outputs, labels)

        # backward: loss를 기준으로 gradient를 계산한다.
        loss.backward()

        # optimizer가 gradient를 사용해서 모델 weight를 업데이트한다.
        optimizer.step()

        batch_size = images.size(0)
        running_loss += loss.item() * batch_size

        correct, total = calculate_accuracy(outputs, labels)
        correct_count += correct
        total_count += total

    avg_loss = running_loss / total_count
    accuracy = correct_count / total_count

    return avg_loss, accuracy


def evaluate(model, data_loader, device):
    """validation 데이터로 모델 정확도를 평가한다."""
    model.eval()

    correct_count = 0
    total_count = 0

    # 평가할 때는 gradient가 필요 없으므로 no_grad를 사용해서 메모리를 아낀다.
    with torch.no_grad():
        for images, labels in data_loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)

            correct, total = calculate_accuracy(outputs, labels)
            correct_count += correct
            total_count += total

    accuracy = correct_count / total_count
    return accuracy


def main():
    # models 폴더가 없으면 자동 생성한다.
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    if not check_dataset():
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    train_transform, val_transform = make_transforms()

    try:
        # class_to_idx 확인을 위해 transform 없는 dataset을 먼저 만든다.
        base_dataset = datasets.ImageFolder(root=str(DATASET_DIR))
    except Exception as error:
        print("Error: failed to load dataset with ImageFolder.")
        print(f"Reason: {error}")
        return

    if len(base_dataset.classes) != 2:
        print("Error: this project expects exactly 2 classes: gun and non_gun.")
        print(f"Found classes: {base_dataset.classes}")
        return

    print(f"Found classes: {base_dataset.classes}")
    print(f"class_to_idx: {base_dataset.class_to_idx}")
    print(f"Total images: {len(base_dataset)}")

    train_indices, val_indices = make_train_val_indices(len(base_dataset))

    if len(train_indices) == 0 or len(val_indices) == 0:
        print("Error: train/validation split failed because there are too few images.")
        print("Please collect more images for both classes.")
        return

    print(f"Train images: {len(train_indices)}")
    print(f"Validation images: {len(val_indices)}")

    # 같은 폴더를 읽되, train과 validation에 서로 다른 transform을 적용한다.
    train_full_dataset = datasets.ImageFolder(root=str(DATASET_DIR), transform=train_transform)
    val_full_dataset = datasets.ImageFolder(root=str(DATASET_DIR), transform=val_transform)

    train_dataset = Subset(train_full_dataset, train_indices)
    val_dataset = Subset(val_full_dataset, val_indices)

    # Windows에서는 num_workers=0이 가장 문제 없이 동작한다.
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
    )

    model = get_model(num_classes=2).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    best_val_accuracy = -1.0

    print("Training started.")

    for epoch in range(1, EPOCHS + 1):
        train_loss, train_accuracy = train_one_epoch(
            model=model,
            data_loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
        )
        val_accuracy = evaluate(model, val_loader, device)

        print(
            f"Epoch [{epoch}/{EPOCHS}] "
            f"Train Loss: {train_loss:.4f} "
            f"Train Acc: {train_accuracy:.4f} "
            f"Val Acc: {val_accuracy:.4f}"
        )

        # validation accuracy가 가장 높을 때의 모델만 저장한다.
        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "class_to_idx": base_dataset.class_to_idx,
                    "best_val_accuracy": best_val_accuracy,
                    "image_size": IMAGE_SIZE,
                },
                MODEL_SAVE_PATH,
            )
            print(f"Best model saved: {MODEL_SAVE_PATH}")

    print("Training finished.")
    print(f"Best validation accuracy: {best_val_accuracy:.4f}")
    print(f"Model path: {MODEL_SAVE_PATH}")

    # ImageFolder는 폴더명 알파벳 순서로 class index를 만들기 때문에 꼭 확인해야 한다.
    print(f"Final class_to_idx: {base_dataset.class_to_idx}")


if __name__ == "__main__":
    main()

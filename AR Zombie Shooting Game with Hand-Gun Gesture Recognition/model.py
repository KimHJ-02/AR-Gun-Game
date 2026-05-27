import torch.nn as nn
from torchvision import models


class SimpleHandGunCNN(nn.Module):
    """gun / non_gun을 분류하는 간단한 CNN 모델.

    입력 shape:
        batch x 3 x 224 x 224

    출력 shape:
        batch x 2

    출력 2개는 각각 non_gun, gun 클래스 점수(logit)로 사용할 예정이다.
    """

    def __init__(self, num_classes=2):
        super().__init__()

        # feature_extractor는 이미지에서 선, 모서리, 손 모양 같은 특징을 뽑는 부분이다.
        self.feature_extractor = nn.Sequential(
            # 첫 번째 Conv layer: RGB 3채널 이미지를 16개의 특징 맵으로 바꾼다.
            nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, padding=1),
            nn.ReLU(),
            # MaxPool은 이미지 크기를 절반으로 줄여 계산량을 줄이고 중요한 특징을 남긴다.
            nn.MaxPool2d(kernel_size=2, stride=2),  # 224 x 224 -> 112 x 112

            # 두 번째 Conv layer: 더 복잡한 손 모양 특징을 학습한다.
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 112 x 112 -> 56 x 56

            # 세 번째 Conv layer: 더 높은 수준의 특징을 학습한다.
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 56 x 56 -> 28 x 28

            # 네 번째 Conv layer: 최종 분류에 사용할 특징을 더 풍부하게 만든다.
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1),
            nn.ReLU(),

            # AdaptiveAvgPool2d는 입력 이미지 크기에 크게 의존하지 않도록
            # 각 채널을 1 x 1 크기의 평균 특징으로 압축한다.
            nn.AdaptiveAvgPool2d((1, 1)),
        )

        # classifier는 뽑힌 특징을 이용해서 non_gun / gun 중 하나로 분류하는 부분이다.
        self.classifier = nn.Sequential(
            # 128 x 1 x 1 텐서를 128 길이의 벡터로 펼친 뒤 Linear layer에 넣는다.
            nn.Flatten(),
            nn.Linear(128, 64),
            nn.ReLU(),
            # 마지막 출력은 클래스 개수와 같아야 한다. MVP에서는 2개 클래스이다.
            nn.Linear(64, num_classes),
        )

    def forward(self, x):
        """입력 이미지를 받아 클래스 점수를 출력한다."""
        features = self.feature_extractor(x)
        output = self.classifier(features)
        return output


def get_model(num_classes=2):
    """기본 학습과 게임 추론에서 사용할 간단한 CNN 모델을 반환한다."""
    return SimpleHandGunCNN(num_classes=num_classes)


def get_mobilenet_model(num_classes=2):
    """선택 사항: MobileNetV2 transfer learning 모델을 반환한다.

    SimpleHandGunCNN보다 성능이 좋을 수 있지만, 처음 MVP 학습 코드는
    이해하기 쉬운 get_model()의 SimpleHandGunCNN을 기본으로 사용한다.
    """
    # weights=None으로 두면 인터넷 다운로드 없이 모델 구조만 만든다.
    model = models.mobilenet_v2(weights=None)

    # MobileNetV2의 마지막 classifier를 gun / non_gun 2개 출력으로 교체한다.
    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, num_classes)

    return model

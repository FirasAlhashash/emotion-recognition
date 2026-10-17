"""Wertet das trainierte ResNet34 auf dem FER-2013-Testset aus.

Aufruf:
    python evaluate_resnet.py
    python evaluate_resnet.py --data data/test --weights models/resnet34_fer2013.pth

Schreibt den Klassifikationsbericht nach results/ und speichert die Konfusionsmatrix als Bild.
"""
import argparse
import os

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from sklearn.metrics import ConfusionMatrixDisplay, accuracy_score, classification_report, confusion_matrix
from torchvision import datasets, transforms
from torchvision.models import resnet34


class EmotionResNet(nn.Module):
    # ResNet34 mit 1 Eingangskanal (Graustufen) und 7 Ausgängen
    def __init__(self, num_classes=7):
        super().__init__()
        self.model = resnet34(weights=None)
        self.model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.model.fc = nn.Linear(512, num_classes)

    def forward(self, x):
        return self.model(x)


# Gleiche Vorverarbeitung wie beim Training
test_transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((96, 96)),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5]),
])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/test")
    parser.add_argument("--weights", default="models/resnet34_fer2013.pth")
    parser.add_argument("--out", default="results")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = EmotionResNet()
    model.load_state_dict(torch.load(args.weights, map_location="cpu"))
    model.to(device).eval()

    test_set = datasets.ImageFolder(args.data, transform=test_transform)
    loader = torch.utils.data.DataLoader(test_set, batch_size=256)

    y_true, y_pred = [], []
    with torch.no_grad():
        for images, labels in loader:
            outputs = model(images.to(device))
            y_pred += outputs.argmax(dim=1).cpu().tolist()
            y_true += labels.tolist()

    acc = accuracy_score(y_true, y_pred)
    report = classification_report(y_true, y_pred, target_names=test_set.classes, digits=3)
    print(f"Testbilder: {len(test_set)}")
    print(f"Accuracy: {acc:.4f}\n")
    print(report)

    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, "resnet34_report.txt"), "w", encoding="utf-8") as f:
        f.write(f"Testbilder: {len(test_set)}\nAccuracy: {acc:.4f}\n\n{report}")

    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(8, 7))
    ConfusionMatrixDisplay(cm, display_labels=test_set.classes).plot(ax=ax, cmap="Blues", colorbar=False)
    ax.set_xlabel("Vorhergesagt")
    ax.set_ylabel("Wahr")
    plt.xticks(rotation=45)
    plt.title(f"ResNet34 – Testset (Accuracy {acc:.1%})")
    plt.tight_layout()
    plt.savefig(os.path.join(args.out, "resnet34_confusion_matrix.png"), dpi=120)


if __name__ == "__main__":
    main()

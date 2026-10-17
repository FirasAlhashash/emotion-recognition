"""Trainiert ResNet34 auf FER-2013 mit eigenem Validierungs-Split.

Ablauf:
  1. data/train wird stratifiziert in Training (90 %) und Validierung (10 %) geteilt
  2. Training mit Augmentation, Klassengewichten, Label Smoothing und Cosine-LR
  3. Das beste Modell wird nur anhand der Validierungs-Accuracy ausgewählt
  4. data/test wird hier nie angefasst, das macht danach evaluate_resnet.py genau einmal

Nach jeder Epoche wird ein Checkpoint gespeichert. Mit --resume soll es nach einer
Unterbrechung (z. B. Colab-Timeout) an derselben Stelle weitergehen.
Achtung: Das Fortsetzen ist noch nicht praktisch getestet, der Checkpoint wurde bisher
nur geschrieben, aber nie wieder geladen.

Aufruf:
    python train_resnet.py
    python train_resnet.py --checkpoint /content/drive/MyDrive/emotion/checkpoint.pth --resume
"""
import argparse
import json
import os
import random
import time

import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from torchvision.models import ResNet34_Weights, resnet34

from evaluate_resnet import EmotionResNet, evaluate, test_transform

train_transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((96, 96)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5]),
])


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_model(pretrained):
    model = EmotionResNet()
    if pretrained:
        # ImageNet-Gewichte übernehmen, conv1 von RGB auf 1 Kanal umrechnen (Summe über die Farbkanäle)
        imagenet = resnet34(weights=ResNet34_Weights.IMAGENET1K_V1).state_dict()
        imagenet["conv1.weight"] = imagenet["conv1.weight"].sum(dim=1, keepdim=True)
        del imagenet["fc.weight"], imagenet["fc.bias"]
        model.model.load_state_dict(imagenet, strict=False)
    return model


def atomic_save(obj, path):
    # Erst in eine temporäre Datei schreiben, damit bei einem Abbruch kein halber Checkpoint übrig bleibt
    tmp = path + ".tmp"
    torch.save(obj, tmp)
    os.replace(tmp, path)


def write_history(path, args, state, finished):
    with open(path, "w", encoding="utf-8") as f:
        json.dump({
            "args": vars(args),
            "finished": finished,
            "best_epoch": state["best_epoch"],
            "best_val_acc": state["best_val_acc"],
            "history": state["history"],
        }, f, indent=2)


def run_epoch(model, loader, criterion, optimizer, scaler, device, max_batches=None):
    model.train()
    total_loss, correct, seen = 0.0, 0, 0
    use_amp = scaler is not None
    for i, (images, labels) in enumerate(loader):
        if max_batches and i >= max_batches:
            break
        images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)
        optimizer.zero_grad()
        with torch.autocast(device_type=device, enabled=use_amp):
            outputs = model(images)
            loss = criterion(outputs, labels)
        if use_amp:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()
        total_loss += loss.item() * len(labels)
        correct += (outputs.argmax(dim=1) == labels).sum().item()
        seen += len(labels)
    return total_loss / seen, correct / seen


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--val-size", type=float, default=0.1)
    parser.add_argument("--patience", type=int, default=7, help="Early Stopping auf Val-Accuracy")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--no-pretrained", action="store_true")
    parser.add_argument("--no-amp", action="store_true", help="Mixed Precision auf der GPU abschalten")
    parser.add_argument("--max-batches", type=int, default=None, help="nur zum Testen: Batches pro Epoche begrenzen")
    parser.add_argument("--checkpoint", default="checkpoints/checkpoint.pth", help="wird nach jeder Epoche überschrieben")
    parser.add_argument("--resume", action="store_true", help="vom Checkpoint aus weitermachen, falls vorhanden")
    parser.add_argument("--weights-out", default="models/resnet34_fer2013.pth", help="Gewichte des besten Modells")
    parser.add_argument("--out", default="results")
    args = parser.parse_args()

    set_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Gerät: {device}" + (f" ({torch.cuda.get_device_name(0)})" if device == "cuda" else ""))

    # Zwei Sichten auf denselben Ordner: mit Augmentation fürs Training, ohne für die Validierung
    train_full = datasets.ImageFolder(os.path.join(args.data, "train"), transform=train_transform)
    val_full = datasets.ImageFolder(os.path.join(args.data, "train"), transform=test_transform)
    targets = np.array(train_full.targets)

    # Der Split hängt nur vom Seed ab und ist deshalb auch nach --resume derselbe
    train_idx, val_idx = train_test_split(
        np.arange(len(targets)), test_size=args.val_size, stratify=targets, random_state=args.seed
    )
    train_set = Subset(train_full, train_idx)
    val_set = Subset(val_full, val_idx)
    print(f"Training: {len(train_set)}  Validierung: {len(val_set)}  (Testset wird hier nicht benutzt)")

    g = torch.Generator().manual_seed(args.seed)
    pin = device == "cuda"
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.workers, generator=g, pin_memory=pin)
    val_loader = DataLoader(val_set, batch_size=256, num_workers=args.workers, pin_memory=pin)

    # Klassengewichte gegen die Unausgeglichenheit (disgusted hat nur ~400 Bilder)
    counts = np.bincount(targets[train_idx], minlength=len(train_full.classes))
    class_weights = torch.tensor(counts.sum() / (len(counts) * counts), dtype=torch.float32).to(device)

    model = build_model(pretrained=not args.no_pretrained).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.1)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    scaler = None
    if device == "cuda" and not args.no_amp:
        # torch.amp.GradScaler gibt es erst ab PyTorch 2.3, davor torch.cuda.amp
        scaler = torch.amp.GradScaler("cuda") if hasattr(torch.amp, "GradScaler") else torch.cuda.amp.GradScaler()

    for path in (args.checkpoint, args.weights_out):
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    os.makedirs(args.out, exist_ok=True)
    history_path = os.path.join(args.out, "resnet34_training_history.json")

    state = {"epoch": 0, "best_val_acc": 0.0, "best_epoch": 0, "epochs_without_improvement": 0,
             "history": [], "finished": False}

    if args.resume and os.path.exists(args.checkpoint):
        # RNG-Zustände müssen auf der CPU bleiben, Modell und Optimizer verschieben sich beim Laden selbst
        ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        scheduler.load_state_dict(ckpt["scheduler"])
        if scaler is not None and ckpt.get("scaler") is not None:
            scaler.load_state_dict(ckpt["scaler"])
        g.set_state(ckpt["loader_generator"])
        torch.set_rng_state(ckpt["torch_rng"])
        if device == "cuda" and ckpt.get("cuda_rng") is not None:
            torch.cuda.set_rng_state_all(ckpt["cuda_rng"])
        np.random.set_state(ckpt["numpy_rng"])
        random.setstate(ckpt["python_rng"])
        state = ckpt["state"]
        print(f"Fortgesetzt nach Epoche {state['epoch']} (beste Val-Accuracy bisher {state['best_val_acc']:.4f})")
        if state["finished"]:
            print("Training war schon abgeschlossen, nichts zu tun.")
            return
    elif args.resume:
        print("Kein Checkpoint gefunden, starte neu.")

    for epoch in range(state["epoch"] + 1, args.epochs + 1):
        start = time.time()
        train_loss, train_acc = run_epoch(model, train_loader, criterion, optimizer, scaler, device, args.max_batches)
        scheduler.step()
        val_acc, _, _ = evaluate(model, val_loader, device)

        state["epoch"] = epoch
        state["history"].append({"epoch": epoch, "train_loss": train_loss, "train_acc": train_acc, "val_acc": val_acc})
        print(f"Epoche {epoch:2d}  loss {train_loss:.4f}  train {train_acc:.4f}  val {val_acc:.4f}  "
              f"({time.time() - start:.0f}s)", flush=True)

        if val_acc > state["best_val_acc"]:
            state["best_val_acc"], state["best_epoch"], state["epochs_without_improvement"] = val_acc, epoch, 0
            atomic_save(model.state_dict(), args.weights_out)
        else:
            state["epochs_without_improvement"] += 1

        if state["epochs_without_improvement"] >= args.patience:
            print(f"Early Stopping nach Epoche {epoch}")
            state["finished"] = True
        if epoch == args.epochs:
            state["finished"] = True

        atomic_save({
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "scaler": scaler.state_dict() if scaler is not None else None,
            "loader_generator": g.get_state(),
            "torch_rng": torch.get_rng_state(),
            "cuda_rng": torch.cuda.get_rng_state_all() if device == "cuda" else None,
            "numpy_rng": np.random.get_state(),
            "python_rng": random.getstate(),
            "state": state,
        }, args.checkpoint)
        write_history(history_path, args, state, state["finished"])

        if state["finished"]:
            break

    print(f"\nBestes Modell: Epoche {state['best_epoch']}, Val-Accuracy {state['best_val_acc']:.4f}")
    print(f"Gewichte: {args.weights_out}")
    print("Jetzt einmalig auf dem Testset auswerten mit evaluate_resnet.py")


if __name__ == "__main__":
    main()

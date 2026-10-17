# Emotion Recognition

Projekt aus dem Modul *Grundlagen der Computer Vision* (SoSe 2025, OVGU Magdeburg), zusammen mit Miká und Konstantin.

Mein Teil im Team waren die Deep-Learning-Modelle (eigenes CNN und ResNet34), dazu habe ich am FACS-Ansatz und an der Präsentation mitgearbeitet.

Die Frage war ziemlich einfach: Wie gut kann man Emotionen auf kleinen Gesichtsbildern erkennen, und wie viel bringt Deep Learning gegenüber klassischen Verfahren wirklich? Wir haben dafür zuerst mit Kantenfiltern und HOG-Features gearbeitet, dann einen FACS-Ansatz mit MediaPipe ausprobiert und am Ende zwei CNNs trainiert.

## Datensatz

FER-2013 in der Kaggle-Version mit einem Ordner pro Emotion ([ananthu017/emotion-detection-fer](https://www.kaggle.com/datasets/ananthu017/emotion-detection-fer)):

- 28.709 Trainingsbilder, 7.178 Testbilder
- 48×48 Pixel, Graustufen
- 7 Klassen: angry, disgusted, fearful, happy, neutral, sad, surprised

Der Datensatz ist ziemlich unausgeglichen. *happy* hat über 7.000 Bilder, *disgusted* nur 436. Das sieht man später auch in den Ergebnissen.

## Ergebnisse

Alle Werte auf dem Testset (7.178 Bilder).

| Ansatz | Accuracy | Nachvollziehbar über |
|---|---|---|
| Immer „happy“ raten (Mehrheitsklasse) | 24,7 % | Klassenverteilung |
| Canny + lineare SVM | 25,6 % | `Canny.ipynb` |
| Sobel + lineare SVM | 26,8 % | `Sobel.ipynb` |
| HOG + lineare SVM | 42,1 % | `HOG.ipynb` |
| FACS (MediaPipe-Landmarks + Random Forest) | ca. 54 %¹ | nur Präsentation |
| Eigenes CNN, 30 Epochen | 67 % | nur Präsentation² |
| ResNet34, 50 Epochen | 66,8 % | `evaluate_resnet.py` + Gewichte |

¹ MediaPipe hat nicht in jedem 48×48-Bild ein Gesicht gefunden. Die 54 % beziehen sich nur auf die 6.059 Bilder, bei denen das geklappt hat (aus der Konfusionsmatrix in unserer Abschlusspräsentation berechnet).

² Für das CNN und den FACS-Teil sind weder Code noch Gewichte im Repo, die Zahlen kann man hier also nicht nachprüfen.

Das ResNet34 lässt sich mit `evaluate_resnet.py` auf dem Testset neu auswerten, die Ausgabe liegt in [`results/`](results/). Die Werte pro Klasse weichen leicht von unserer Präsentation ab (dort stand 66 %), vermutlich weil auf der Folie ein anderer Epochenstand gezeigt wurde.

Wichtig zur Einordnung: Der Kaggle-Datensatz hat nur train/ und test/. Wir haben das Testset während des Trainings auch zur Validierung benutzt und darüber das beste Modell gespeichert. Die 66,8 % sind deshalb eher etwas zu optimistisch. Sauberer wäre ein eigener Validierungs-Split aus den Trainingsdaten.

Ein paar Sachen, die uns aufgefallen sind:

- Die reinen Kantenfilter (Canny, Sobel) liegen praktisch auf dem Niveau der Mehrheitsklassen-Baseline (24,7 %). Zufälliges Raten bei 7 Klassen wäre 14 %, die SVM lernt also schon etwas, aber kaum mehr als „meistens happy sagen“. Bei 48×48 Pixeln bleibt mit reinen Kantenbildern einfach zu wenig Struktur übrig.
- HOG ist deutlich besser, weil es die Richtung der Kanten mitnimmt und nicht nur ob da eine Kante ist.
- Bei den SVMs haben wir nur auf 10.000 zufälligen Trainingsbildern trainiert, sonst hat das Training zu lange gedauert.
- Unser kleines CNN und ResNet34 liegen am Ende praktisch gleichauf (67 % vs. 66,8 %), obwohl ResNet34 viel größer ist. Beide overfitten stark (Train-Accuracy fast 100 %, Val bleibt bei ~66 %).
- *happy* und *surprised* klappen bei allen Modellen am besten. Beim ResNet werden *fearful* und *sad* oft verwechselt, genauso *neutral* und *sad* (siehe Konfusionsmatrix in `results/`).

Zum Einordnen: Khaireddin & Chen kommen mit einem einzelnen, gut getunten VGG-Netz auf 73,28 % [1]. Für Menschen wird die Genauigkeit auf FER-2013 mit etwa 65 ± 5 % angegeben [2], viele Bilder sind also selbst für Menschen nicht eindeutig.

## Was im Repo ist

```
HOG.ipynb                     HOG-Features + SVM, inkl. Datenanalyse
Sobel.ipynb                   Sobel-Gradienten + SVM
Canny.ipynb                   Canny-Kanten + SVM
evaluate_resnet.py            Auswertung des ResNet34 auf dem Testset
models/resnet34_fer2013.pth   trainierte Gewichte vom ResNet34
results/                      Klassifikationsbericht + Konfusionsmatrix vom ResNet34
```

Der Trainingscode für das CNN, das ResNet34 und den FACS-Teil ist (noch) nicht hier drin.

## Ausführen

```bash
pip install -r requirements.txt
```

Den Datensatz von Kaggle laden und so ablegen:

```
data/
  train/angry/ ... train/surprised/
  test/angry/  ... test/surprised/
```

Danach die Notebooks einfach von oben nach unten durchlaufen lassen.

Das ResNet34 auswerten:

```bash
python evaluate_resnet.py
```

### ResNet34 laden

Das Modell erwartet ein Graustufenbild (1 Kanal), skaliert auf 96×96 und normalisiert mit Mittelwert 0,5 und Standardabweichung 0,5 (siehe `evaluate_resnet.py`).

```python
import torch
import torch.nn as nn
from torchvision.models import resnet34

class EmotionResNet(nn.Module):
    def __init__(self, num_classes=7):
        super().__init__()
        self.model = resnet34(weights=None)
        self.model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.model.fc = nn.Linear(512, num_classes)

    def forward(self, x):
        return self.model(x)

model = EmotionResNet()
model.load_state_dict(torch.load("models/resnet34_fer2013.pth", map_location="cpu"))
model.eval()

labels = ["angry", "disgusted", "fearful", "happy", "neutral", "sad", "surprised"]
```

## Was wir anders machen würden

- Mit einem ausgeglicheneren Datensatz arbeiten oder *disgusted* gezielt augmentieren
- Einen eigenen Validierungs-Split nehmen und früher mit Early Stopping aufhören, das Overfitting sieht man in den Kurven sehr deutlich
- Andere Architekturen testen, z. B. EfficientNet
- Die klassischen Features (HOG + Landmarks) kombinieren statt einzeln zu testen

## Quellen

[1] Y. Khaireddin, Z. Chen: *Facial Emotion Recognition: State of the Art Performance on FER2013*, 2021. [arXiv:2105.03588](https://arxiv.org/abs/2105.03588)

[2] I. J. Goodfellow et al.: *Challenges in Representation Learning: A report on three machine learning contests*, ICONIP 2013. [arXiv:1307.0414](https://arxiv.org/abs/1307.0414)

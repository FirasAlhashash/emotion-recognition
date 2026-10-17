# Emotion Recognition

Projekt aus dem Modul *Grundlagen der Computer Vision* (SoSe 2025, OVGU Magdeburg), zusammen mit Miká und Konstantin.

Die Frage war ziemlich einfach: Wie gut kann man Emotionen auf kleinen Gesichtsbildern erkennen, und wie viel bringt Deep Learning gegenüber klassischen Verfahren wirklich? Wir haben dafür zuerst mit Kantenfiltern und HOG-Features gearbeitet, dann einen FACS-Ansatz mit MediaPipe ausprobiert und am Ende zwei CNNs trainiert.

## Datensatz

FER-2013 in der Kaggle-Version mit einem Ordner pro Emotion:

- 28.709 Trainingsbilder, 7.178 Testbilder
- 48×48 Pixel, Graustufen
- 7 Klassen: angry, disgusted, fearful, happy, neutral, sad, surprised

Der Datensatz ist ziemlich unausgeglichen. *happy* hat über 7.000 Bilder, *disgusted* nur 436. Das sieht man später auch in den Ergebnissen.

## Ergebnisse

Alle Werte auf dem Testset (7.178 Bilder).

| Ansatz | Accuracy |
|---|---|
| Canny + lineare SVM | 26 % |
| Sobel + lineare SVM | 27 % |
| HOG + lineare SVM | 42 % |
| FACS (MediaPipe-Landmarks + Random Forest) | ca. 54 %* |
| Eigenes CNN, 30 Epochen | 67 % |
| ResNet34, 50 Epochen | 66 % |

\* MediaPipe hat nicht in jedem 48×48-Bild ein Gesicht gefunden. Die 54 % beziehen sich nur auf die ca. 6.000 Bilder, bei denen das geklappt hat.

Ein paar Sachen, die uns aufgefallen sind:

- Die reinen Kantenfilter (Canny, Sobel) sind kaum besser als Raten. Bei 48×48 Pixeln bleibt einfach zu wenig Struktur übrig.
- HOG ist deutlich besser, weil es die Richtung der Kanten mitnimmt und nicht nur ob da eine Kante ist.
- Bei den SVMs haben wir nur auf 10.000 zufälligen Trainingsbildern trainiert, sonst hat das Training zu lange gedauert.
- Unser kleines CNN war am Ende minimal besser als ResNet34. Beide overfitten stark (Train-Accuracy fast 100 %, Val bleibt bei ~66 %).
- *happy* und *surprised* klappen bei allen Modellen am besten. *fearful* und *sad* werden oft verwechselt.

Zum Einordnen: Auf FER-2013 liegen gute Paper bei ca. 73–76 %. Die menschliche Genauigkeit auf dem Datensatz wird auf ungefähr 65 % geschätzt, viele Bilder sind also selbst für Menschen nicht eindeutig.

## Was im Repo ist

```
HOG.ipynb       HOG-Features + SVM, inkl. Datenanalyse
Sobel.ipynb     Sobel-Gradienten + SVM
Canny.ipynb     Canny-Kanten + SVM
models/resnet34_fer2013.pth   trainierte Gewichte vom ResNet34
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

### ResNet34 laden

Das Modell erwartet ein Graustufenbild (1 Kanal). In unserer Vorverarbeitung wurden die Bilder auf 96×96 Pixel skaliert.

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
- Früher mit Early Stopping arbeiten, das Overfitting sieht man in den Kurven sehr deutlich
- Andere Architekturen testen, z. B. EfficientNet
- Die klassischen Features (HOG + Landmarks) kombinieren statt einzeln zu testen

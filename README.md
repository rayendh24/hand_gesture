# Hand Gesture Starter

A small webcam application for detecting hands and recognizing gestures.
This is a starting point for a new project, adapted from existing open-source
work. Attribution is kept in CREDITS.md and the Apache-2.0 license in LICENSE.

## What it does

- Detects up to two hands and draws their landmarks.
- Uses the included models to classify hand signs and finger movements.
- Displays hand position, movement direction and custom gesture rules.
- Keeps the supplied OK, PEACE, POINT DOWN, HORN and two-hand SMILING rules.
  SMILING is a hand gesture label, not facial expression recognition.
- Does not connect to or move a robot. Robot-camera integration is future work.

## Files

| File | Purpose |
| --- | --- |
| app.py | Webcam capture, recognition loop and display |
| gestures.py | Landmark processing and custom gesture rules |
| model/ | Two pretrained TFLite models, labels and inference wrappers |
| utils/ | FPS calculation |

## Run on Ubuntu

Use Python 3.10 or 3.11 with a local desktop and webcam. A fresh environment
keeps this separate from your robot SDK project.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python app.py
```

On Windows use `py -3.11 -m venv .venv`, then
`.venv\Scripts\python.exe -m pip install -r requirements.txt` and
`.venv\Scripts\python.exe app.py`.

Press Esc to exit. Try `python app.py --device 1` if the wrong camera opens.
The webcam image is mirrored. Left/right labels describe that displayed image;
do not map them directly to robot turning without checking orientation.

The code uses the older `mp.solutions.hands` API. The dependency versions below
are a proposed reproducible starting set, not a verified installation lock.
Do not replace a working environment until the new one is tested.
TensorFlow remains a large dependency even though the project ZIP is small.

## Change your gestures

Edit `get_custom_gesture()` in gestures.py to change geometric rules.
The label CSVs name the existing model outputs; renaming a label does not train
a model to recognize a new gesture. Training datasets, notebooks and HDF5
checkpoints were excluded from this starter; keep your original project if you
want to retrain later.

## Validation and limits

Syntax, model-file preservation, preprocessing and argument parsing were checked
offline. Webcam operation, model inference and dependency installation have not
been tested in the delivery environment. Test this copy in simulation/webcam mode
before connecting it to anything that moves. Gesture output is not a safety-rated
control signal and hand identity can change when hands cross or leave the frame.

## Start a new repository

No old Git history or remote is included. After testing and reviewing files:

```bash
git init
git add .
git commit -m "Start hand gesture recognition project"
git branch -M main
```

Create an empty repository on your hosting service, then use its instructions to
add the remote and push. Keep LICENSE and CREDITS.md with redistributed copies.

# Source and attribution

This starter is derived from the project supplied by Rayen. Its README credited:

- Kazuhito Takahashi — original hand-gesture-recognition-using-mediapipe project:
  https://github.com/Kazuhito00/hand-gesture-recognition-using-mediapipe
- Nikita Kiselov — English translation and other improvements:
  https://github.com/kinivi

These credits are carried forward from the supplied README; the precise cloned
repository revision was not provided. The included classifier models, labels,
wrappers and portions of the application/helpers are reused, not newly authored
models. The supplied Apache License 2.0 is preserved verbatim in LICENSE.

Changes on 2026-10-22: removed training-only files and dataset-writing modes;
replaced project documentation; split gesture helpers from the camera loop;
used MediaPipe's landmark renderer; fixed relative asset paths, confidence
argument parsing, zero normalization and camera cleanup. Preserved custom rules
from the uploaded app without claiming which contributor originally wrote them.

MediaPipe, TensorFlow, NumPy and OpenCV are separately installed dependencies
with their own licenses.

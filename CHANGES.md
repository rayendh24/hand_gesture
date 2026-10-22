# Cleanup changes

Removed from this copy:
- Three training notebooks, two training CSV datasets and two HDF5 checkpoints.
- Old README branding, screenshots, tutorial sections and promotional links.
- Dataset logging keyboard shortcuts and their write-to-CSV code.
- Long manual skeleton-drawing function (uses MediaPipe drawing instead).

Kept:
- Both small TFLite classifiers and their label CSVs, unchanged.
- Existing hand-position, direction, rotation and custom-gesture logic.
- Apache license and source credits.

Added or fixed:
- Reusable gestures.py, installation instructions and focused .gitignore.
- Script-relative model and label paths, including launching from other folders.
- Floating-point tracking confidence and zero-valued normalization handling.
- Camera open error and finally-based camera/MediaPipe cleanup.

Not added: Unitree connectivity, robot camera capture or movement commands.
The uploaded original ZIP was not modified. No commit, remote or push was made.

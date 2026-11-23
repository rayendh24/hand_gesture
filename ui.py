"""On-screen HUD for the gesture app. Drawing only: no recognition logic here.

Each frame, main() calls begin_frame(), then add_hand() / add_trail() while
it processes hands, and finally render() once. Everything is drawn in one
pass so translucent panels are blended once and text is rendered once.

Text uses Pillow with a TrueType font when available (Pillow is normally
installed alongside MediaPipe). If not, it falls back to OpenCV's built-in
font, so the app still runs.
"""
from pathlib import Path

import cv2 as cv
import numpy as np

try:
    from PIL import Image, ImageDraw, ImageFont
except ImportError:  # Fallback to OpenCV text
    Image = ImageDraw = ImageFont = None


def _bgr(hex_rgb):
    hex_rgb = hex_rgb.lstrip('#')
    r, g, b = (int(hex_rgb[i:i + 2], 16) for i in (0, 2, 4))
    return (b, g, r)


# Palette (BGR)
PANEL = _bgr('#121A2E')
PANEL_ALPHA = 0.6
TEXT = _bgr('#EEF2FA')
MUTED = _bgr('#93A0BD')
ACCENT = _bgr('#FFD166')
INK = _bgr('#1A1A24')
HAND_COLORS = {'Left': _bgr('#4CC9F0'), 'Right': _bgr('#B388FF')}
DEFAULT_HAND_COLOR = _bgr('#9EE493')

FINGERTIPS = (4, 8, 12, 16, 20)
HAND_CONNECTIONS = (
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20), (0, 17),
)

FONT_CANDIDATES = {
    False: ['/usr/share/fonts/truetype/ubuntu/Ubuntu-R.ttf',
            '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',
            'C:/Windows/Fonts/segoeui.ttf',
            '/System/Library/Fonts/Supplemental/Arial.ttf'],
    True: ['/usr/share/fonts/truetype/ubuntu/Ubuntu-B.ttf',
           '/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf',
           'C:/Windows/Fonts/segoeuib.ttf',
           '/System/Library/Fonts/Supplemental/Arial Bold.ttf'],
}


def hand_color(label):
    return HAND_COLORS.get(label, DEFAULT_HAND_COLOR)


def describe_movement(direction):
    return 'Still' if direction == 'stationary' else f'Moving {direction}'


def describe_position(position):
    return position.replace('-', ' ').replace('middle center', 'center')


class _Text:
    """Measures text and draws queued text in one pass."""

    def __init__(self):
        self._fonts = {}
        self._paths = {}
        if ImageFont is not None:
            for bold, paths in FONT_CANDIDATES.items():
                self._paths[bold] = next(
                    (p for p in paths if Path(p).exists()), None)
        self.truetype = bool(self._paths.get(False))

    def _font(self, size, bold):
        key = (size, bold)
        if key not in self._fonts:
            path = self._paths.get(bold) or self._paths.get(False)
            self._fonts[key] = ImageFont.truetype(path, size)
        return self._fonts[key]

    def width(self, text, size, bold=False):
        if self.truetype:
            return int(self._font(size, bold).getlength(text))
        scale, thick = self._cv_style(size, bold)
        return cv.getTextSize(text, cv.FONT_HERSHEY_SIMPLEX, scale, thick)[0][0]

    @staticmethod
    def _cv_style(size, bold):
        return size / 32.0, 2 if bold else 1

    def draw_all(self, image, items):
        """items: (text, x, y_center, size, bold, color_bgr)."""
        if not items:
            return image
        if self.truetype:
            pil = Image.fromarray(image)  # Channels stay BGR; colors are BGR too.
            draw = ImageDraw.Draw(pil)
            for text, x, y, size, bold, color in items:
                draw.text((x, y), text, font=self._font(size, bold),
                          fill=tuple(color), anchor='lm')
            return np.asarray(pil).copy()
        for text, x, y, size, bold, color in items:
            scale, thick = self._cv_style(size, bold)
            (_, h), _ = cv.getTextSize(text, cv.FONT_HERSHEY_SIMPLEX, scale, thick)
            cv.putText(image, text, (int(x), int(y + h / 2)),
                       cv.FONT_HERSHEY_SIMPLEX, scale, color, thick, cv.LINE_AA)
        return image


def _rounded_rect(image, x1, y1, x2, y2, radius, color):
    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
    r = int(max(0, min(radius, (x2 - x1) // 2, (y2 - y1) // 2)))
    cv.rectangle(image, (x1 + r, y1), (x2 - r, y2), color, -1, cv.LINE_AA)
    cv.rectangle(image, (x1, y1 + r), (x2, y2 - r), color, -1, cv.LINE_AA)
    for cx, cy in ((x1 + r, y1 + r), (x2 - r, y1 + r),
                   (x1 + r, y2 - r), (x2 - r, y2 - r)):
        cv.circle(image, (cx, cy), r, color, -1, cv.LINE_AA)


class Hud:
    def __init__(self, title='Hand gesture recognition'):
        self.title = title
        self.text = _Text()
        self.show_panels = True
        self.begin_frame()

    def toggle_panels(self):
        self.show_panels = not self.show_panels

    # ---- data collection (called from main) ----
    def begin_frame(self):
        self.hands = []
        self.trails = []

    def add_hand(self, label, brect, landmark_list, hand_sign, finger_gesture,
                 movement_direction, hand_position, custom_gesture):
        self.hands.append(dict(
            label=label, brect=brect, landmarks=landmark_list,
            sign=hand_sign, finger=finger_gesture,
            movement=movement_direction, position=hand_position,
            custom=custom_gesture))

    def add_trail(self, label, point_history):
        points = [tuple(p) for p in point_history if p[0] != 0 and p[1] != 0]
        if points:
            self.trails.append((label, points))

    # ---- rendering ----
    def render(self, image, fps, smiling=False):
        h, w = image.shape[:2]
        texts = []

        # Hand drawings first, so the panels stay readable on top of them.
        for label, points in self.trails:
            self._trail(image, points, hand_color(label))
        for hand in self.hands:
            self._skeleton(image, hand['landmarks'], hand_color(hand['label']))
            self._brackets(image, hand['brect'], hand_color(hand['label']))

        if not self.show_panels:
            # Camera view with only the hand drawings on top.
            return image

        panels = image.copy()
        self._header(panels, texts, w, fps)
        if self.hands:
            self._readout(panels, texts, w, h)
        else:
            self._empty_state(panels, texts, w, h)
        image = cv.addWeighted(panels, PANEL_ALPHA, image, 1 - PANEL_ALPHA, 0)

        for hand in self.hands:
            self._tag(image, texts, hand, w)
        if smiling:
            self._banner(image, texts, w, 'Smiling gesture detected')

        return self.text.draw_all(image, texts)

    def _header(self, layer, texts, w, fps):
        _rounded_rect(layer, 12, 12, w - 12, 60, 14, PANEL)
        texts.append((self.title, 30, 36, 19, True, TEXT))

        x = w - 30
        hint = 'H hides panels   Esc to quit'
        x -= self.text.width(hint, 14)
        texts.append((hint, x, 36, 14, False, MUTED))

        count = len(self.hands)
        hands_text = 'No hands' if count == 0 else (
            '1 hand' if count == 1 else f'{count} hands')
        for value, color in ((f'{fps:.0f} fps', TEXT), (hands_text, TEXT)):
            pill_w = self.text.width(value, 14, True) + 26
            x -= pill_w + 14
            _rounded_rect(layer, x, 23, x + pill_w, 49, 13, _bgr('#26314F'))
            texts.append((value, x + 13, 36, 14, True, color))

    def _readout(self, layer, texts, w, h):
        col_w, card_h, gap = 250, 122, 12
        hands = sorted(self.hands, key=lambda d: d['label'] != 'Left')
        total_w = len(hands) * col_w + (len(hands) - 1) * gap
        x0, y0 = 12, h - 12 - card_h
        if total_w > w - 24:
            col_w = (w - 24 - (len(hands) - 1) * gap) // len(hands)
        for i, hand in enumerate(hands):
            x = x0 + i * (col_w + gap)
            _rounded_rect(layer, x, y0, x + col_w, y0 + card_h, 16, PANEL)
            color = hand_color(hand['label'])
            cv.circle(layer, (x + 24, y0 + 24), 5, color, -1, cv.LINE_AA)
            texts.append((f"{hand['label']} hand", x + 36, y0 + 24, 15, True,
                          color))

            main_text = hand['custom'] or hand['sign'] or 'Unknown'
            texts.append((main_text, x + 18, y0 + 58, 30, True,
                          ACCENT if hand['custom'] else TEXT))

            detail = f"Shape {hand['sign']}"
            if hand['finger']:
                detail += f"   Finger {hand['finger']}"
            texts.append((detail, x + 18, y0 + 89, 13, False, MUTED))
            where = (f"{describe_movement(hand['movement'])}, "
                     f"{describe_position(hand['position'])}")
            texts.append((where, x + 18, y0 + 107, 13, False, MUTED))

    def _empty_state(self, layer, texts, w, h):
        line1 = 'Hold a hand up to the camera'
        line2 = 'Up to two hands are tracked at once'
        box_w = max(self.text.width(line1, 20, True),
                    self.text.width(line2, 14)) + 56
        x1, y1 = (w - box_w) // 2, h // 2 - 40
        _rounded_rect(layer, x1, y1, x1 + box_w, y1 + 80, 18, PANEL)
        texts.append((line1, x1 + 28, y1 + 30, 20, True, TEXT))
        texts.append((line2, x1 + 28, y1 + 56, 14, False, MUTED))

    @staticmethod
    def _trail(image, points, color):
        n = len(points)
        for i in range(1, n):
            t = i / max(n - 1, 1)
            faded = tuple(int(c * (0.35 + 0.65 * t)) for c in color)
            cv.line(image, points[i - 1], points[i], faded,
                    1 + int(4 * t), cv.LINE_AA)
        cv.circle(image, points[-1], 7, color, -1, cv.LINE_AA)
        cv.circle(image, points[-1], 7, TEXT, 2, cv.LINE_AA)

    @staticmethod
    def _skeleton(image, landmarks, color):
        for a, b in HAND_CONNECTIONS:
            cv.line(image, tuple(landmarks[a]), tuple(landmarks[b]),
                    color, 2, cv.LINE_AA)
        for i, point in enumerate(landmarks):
            if i in FINGERTIPS:
                cv.circle(image, tuple(point), 6, color, -1, cv.LINE_AA)
                cv.circle(image, tuple(point), 6, TEXT, 2, cv.LINE_AA)
            else:
                cv.circle(image, tuple(point), 3, TEXT, -1, cv.LINE_AA)

    @staticmethod
    def _brackets(image, brect, color, pad=14, arm=20):
        x1, y1 = brect[0] - pad, brect[1] - pad
        x2, y2 = brect[2] + pad, brect[3] + pad
        for cx, cy, dx, dy in ((x1, y1, 1, 1), (x2, y1, -1, 1),
                               (x1, y2, 1, -1), (x2, y2, -1, -1)):
            cv.line(image, (cx, cy), (cx + dx * arm, cy), color, 3, cv.LINE_AA)
            cv.line(image, (cx, cy), (cx, cy + dy * arm), color, 3, cv.LINE_AA)

    def _tag(self, image, texts, hand, w, pad=14):
        label = hand['label']
        value = hand['custom'] or hand['sign']
        text = f'{label}  {value}' if value else label
        tag_w = self.text.width(text, 14, True) + 24
        x = int(min(max(hand['brect'][0] - pad, 12), w - 12 - tag_w))
        y = int(max(hand['brect'][1] - pad - 36, 68))
        _rounded_rect(image, x, y, x + tag_w, y + 26, 13,
                      hand_color(label))
        texts.append((text, x + 12, y + 13, 14, True, INK))

    def _banner(self, image, texts, w, message):
        banner_w = self.text.width(message, 18, True) + 48
        x = (w - banner_w) // 2
        _rounded_rect(image, x, 72, x + banner_w, 110, 19, ACCENT)
        texts.append((message, x + 24, 91, 18, True, INK))
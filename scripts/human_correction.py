import cv2
import numpy as np
import colorsys
import json
import os
from copy import deepcopy
from collections import defaultdict


class HumanCorrectionGUI:
    """
    Human correction GUI — Linux-compatible controls:

      SCROLL WHEEL          : Zoom in / out (0.5x – 5x)
      LEFT BUTTON drag      : Pan image (when not clicking a detection)
      LEFT BUTTON on detect : Classify/delete detection
      MIDDLE BUTTON drag    : Draw new bbox (ADD mode)

    Internal detection format : [cls_id, xc, yc, w, h, conf, is_added]
    Returned format (run())   : [cls_id, xc, yc, w, h, conf]
    """

    # ------------------------------------------------------------------ init
    def __init__(self, image_path, detections, class_map,
                 max_window_size=(1400, 900), state_file=None):

        self.original_image = cv2.imread(image_path)
        if self.original_image is None:
            raise FileNotFoundError(f"Image not found: {image_path}")

        self.orig_h, self.orig_w = self.original_image.shape[:2]
        self.state_file = state_file

        # window layout
        self.display_w           = max_window_size[0]
        self.display_h           = max_window_size[1]
        self.legend_width        = 280
        self.effective_display_w = self.display_w - self.legend_width

        # zoom / pan - calculate initial zoom to fit full image in window
        zoom_w = self.effective_display_w / self.orig_w
        zoom_h = self.display_h / self.orig_h
        self.zoom_level = min(zoom_w, zoom_h, 1.0)  # Don't zoom in if image is smaller than window
        
        # Center the image if it's smaller than the display
        zoomed_w = self.orig_w * self.zoom_level
        zoomed_h = self.orig_h * self.zoom_level
        self.pan_x = 0 if zoomed_w >= self.effective_display_w else (self.effective_display_w - zoomed_w) / 2
        self.pan_y = 0 if zoomed_h >= self.display_h else (self.display_h - zoomed_h) / 2
        # left-button state (panning or detection interaction)
        self._left_down = False
        self._pan_start = None  # Starting position for left-click pan

        self.font = cv2.FONT_HERSHEY_SIMPLEX

        # class maps
        self.id_to_name   = dict(class_map)
        self.name_to_id   = {v: k for k, v in self.id_to_name.items()}
        self.class_ids    = sorted(self.id_to_name.keys())
        self.class_names  = [self.id_to_name[i] for i in self.class_ids]
        self.class_colors = self._generate_class_colors(self.class_ids)

        # interaction state
        self.selected_class    = None
        self.selected_class_id = None
        self.selected_det      = None
        self.mode              = "classify"   # classify | add | delete

        # detections : [cls, xc, yc, w, h, conf, is_added]
        self.detections = []
        for det in detections:
            self.detections.append(list(map(float, det[:6])) + [False])

        # ADD-mode drag (display-space coords, left button only)
        self.drag_start = None
        self.drag_end   = None

        # counters
        self.added_count        = 0
        self.deleted_count      = 0
        self.reclassified_count = 0
        self.added_by_class          = defaultdict(int)
        self.deleted_by_class        = defaultdict(int)
        self.reclassified_from_to    = defaultdict(lambda: defaultdict(int))

        # undo
        self.history = []
        self._save_state()

        # Load state if provided
        if self.state_file and os.path.exists(self.state_file):
            self.load_state_from_file(self.state_file)

        # class buttons rebuilt each frame by draw_menu()
        self.class_buttons = []

        # Create windows
        # Try WINDOW_AUTOSIZE to prevent window manager interference
        win_flag = cv2.WINDOW_AUTOSIZE

        # Main IMAGE window
        cv2.namedWindow("HC Image", win_flag)
        cv2.resizeWindow("HC Image", self.effective_display_w, self.display_h)
        # Try to prevent window movement
        try:
            cv2.setWindowProperty("HC Image", cv2.WND_PROP_AUTOSIZE, 1)
            cv2.setWindowProperty("HC Image", cv2.WND_PROP_ASPECT_RATIO, 1)
        except:
            pass
        cv2.setMouseCallback("HC Image", self._mouse_callback, param="image")

        # CONTROL panel window (no zoom/pan, just clickable buttons)
        cv2.namedWindow("HC Controls", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("HC Controls", self.legend_width, self.display_h)
        cv2.setMouseCallback("HC Controls", self._mouse_callback, param="menu")

        # Statistics window as before
        cv2.namedWindow("Statistics", cv2.WINDOW_NORMAL)



    # --------------------------------------------------------------- undo
    def _save_state(self):
        self.history.append({
            'detections':           deepcopy(self.detections),
            'added_count':          self.added_count,
            'deleted_count':        self.deleted_count,
            'reclassified_count':   self.reclassified_count,
            'added_by_class':       deepcopy(self.added_by_class),
            'deleted_by_class':     deepcopy(self.deleted_by_class),
            'reclassified_from_to': deepcopy(self.reclassified_from_to),
        })
        if len(self.history) > 50:
            self.history.pop(0)

    def undo(self):
        if len(self.history) > 1:
            self.history.pop()
            s = self.history[-1]
            self.detections           = deepcopy(s['detections'])
            self.added_count          = s['added_count']
            self.deleted_count        = s['deleted_count']
            self.reclassified_count   = s['reclassified_count']
            self.added_by_class       = deepcopy(s['added_by_class'])
            self.deleted_by_class     = deepcopy(s['deleted_by_class'])
            self.reclassified_from_to = deepcopy(s['reclassified_from_to'])
            self.selected_det = None
            # Auto-save to file after undo
            if self.state_file:
                self.save_state_to_file(self.state_file)

    # ------------------------------------------------------- save/load state
    def save_state_to_file(self, filepath):
        """Save current annotation state to a JSON file"""
        state = {
            'detections': self.detections,
            'zoom_level': self.zoom_level,
            'pan_x': self.pan_x,
            'pan_y': self.pan_y,
            'selected_class': self.selected_class,
            'selected_class_id': self.selected_class_id,
            'selected_det': self.selected_det,
            'mode': self.mode,
            'added_count': self.added_count,
            'deleted_count': self.deleted_count,
            'reclassified_count': self.reclassified_count,
            'added_by_class': dict(self.added_by_class),
            'deleted_by_class': dict(self.deleted_by_class),
            'reclassified_from_to': {k: dict(v) for k, v in self.reclassified_from_to.items()},
        }
        try:
            with open(filepath, 'w') as f:
                json.dump(state, f, indent=2)
            print(f"✅ Annotation state saved to: {filepath}")
        except Exception as e:
            print(f"❌ Error saving state to {filepath}: {e}")

    def load_state_from_file(self, filepath):
        """Load annotation state from a JSON file"""
        try:
            with open(filepath, 'r') as f:
                state = json.load(f)
            
            # Restore detections
            self.detections = state.get('detections', self.detections)
            
            # Restore view state
            self.zoom_level = state.get('zoom_level', 1.0)
            self.pan_x = state.get('pan_x', 0)
            self.pan_y = state.get('pan_y', 0)
            
            # Restore interaction state
            self.selected_class = state.get('selected_class')
            self.selected_class_id = state.get('selected_class_id')
            self.selected_det = state.get('selected_det')
            self.mode = state.get('mode', 'classify')
            
            # Restore counters
            self.added_count = state.get('added_count', 0)
            self.deleted_count = state.get('deleted_count', 0)
            self.reclassified_count = state.get('reclassified_count', 0)
            self.added_by_class = defaultdict(int, state.get('added_by_class', {}))
            self.deleted_by_class = defaultdict(int, state.get('deleted_by_class', {}))
            self.reclassified_from_to = defaultdict(lambda: defaultdict(int))
            for k, v in state.get('reclassified_from_to', {}).items():
                self.reclassified_from_to[int(k)] = defaultdict(int, {int(kk): vv for kk, vv in v.items()})
            
            # Reset undo history and save current state
            self.history = []
            self._save_state()
            
            print(f"✅ Annotation state loaded from: {filepath}")
            
        except Exception as e:
            print(f"❌ Error loading state from {filepath}: {e}")

    # ----------------------------------------------------------- colours
    def _generate_class_colors(self, class_ids):
        colors = {}
        n = max(len(class_ids), 1)
        for i, cid in enumerate(class_ids):
            r, g, b = colorsys.hsv_to_rgb(i / n, 0.7, 1.0)
            colors[cid] = (int(b * 255), int(g * 255), int(r * 255))
        return colors

    # ------------------------------------------------------- geometry
    def _clamp_pan(self):
        zw = int(self.orig_w * self.zoom_level)
        zh = int(self.orig_h * self.zoom_level)
        self.pan_x = max(0, min(self.pan_x, max(0, zw - self.effective_display_w)))
        self.pan_y = max(0, min(self.pan_y, max(0, zh - self.display_h)))

    def yolo_to_display(self, det):
        """YOLO normalised -> display-space (x1, y1, x2, y2)."""
        _, xc, yc, w, h, _ = det[:6]
        cx = xc * self.orig_w * self.zoom_level - self.pan_x
        cy = yc * self.orig_h * self.zoom_level - self.pan_y
        hw = w  * self.orig_w * self.zoom_level / 2
        hh = h  * self.orig_h * self.zoom_level / 2
        return int(cx - hw), int(cy - hh), int(cx + hw), int(cy + hh)

    def display_to_yolo(self, x1, y1, x2, y2, cls_id):
        """Display-space rectangle -> YOLO normalised detection."""
        x1p = (x1 + self.pan_x) / self.zoom_level
        y1p = (y1 + self.pan_y) / self.zoom_level
        x2p = (x2 + self.pan_x) / self.zoom_level
        y2p = (y2 + self.pan_y) / self.zoom_level
        xc  = ((x1p + x2p) / 2) / self.orig_w
        yc  = ((y1p + y2p) / 2) / self.orig_h
        bw  = abs(x2p - x1p) / self.orig_w
        bh  = abs(y2p - y1p) / self.orig_h
        return [float(cls_id), xc, yc, bw, bh, 1.0]

    @staticmethod
    def _hit(x, y, bbox):
        x1, y1, x2, y2 = bbox
        return x1 <= x <= x2 and y1 <= y <= y2

    # -------------------------------------------------------- draw menu
    def draw_menu(self):
        menu = 240 * np.ones((self.display_h, self.legend_width, 3), dtype=np.uint8)

        def btn(top, label, active):
            col = (0, 150, 255) if active else (180, 180, 180)
            cv2.rectangle(menu, (20, top), (self.legend_width - 20, top + 40), col, -1)
            cv2.putText(menu, label, (30, top + 27), self.font, 0.6, (0, 0, 0), 2)

        btn(20,  "CLASSIFY", self.mode == "classify")
        btn(80,  "ADD",      self.mode == "add")
        btn(140, "DELETE",   self.mode == "delete")

        y0 = 220
        cv2.putText(menu, "CLASSES", (20, y0), self.font, 0.6, (0, 0, 0), 2)

        self.class_buttons = []
        # enlarge hitbox margins for easier clicking: horiz_pad, top/bottom pad
        horiz_pad = 10
        top_pad = 10
        bottom_pad = 15
        for i, cls in enumerate(self.class_names):
            cid    = self.class_ids[i]
            yy     = y0 + 30 + i * 35
            col    = self.class_colors[cid]
            active = (cls == self.selected_class)
            # colored small square / pill
            cv2.rectangle(menu, (20, yy - 8), (32, yy + 4), col, -1)
            if active:
                cv2.rectangle(menu,
                              (15, yy - 12), (self.legend_width - 15, yy + 18),
                              col, 2)
            cv2.putText(menu, cls, (40, yy + 2), self.font, 0.5, (0, 0, 0), 1)
            # store a larger, well-aligned hitbox that covers the pill + text
            x1_hit = 20 - horiz_pad
            y1_hit = yy - 8 - top_pad
            x2_hit = self.legend_width - 20 + horiz_pad
            y2_hit = yy + 4 + bottom_pad
            # clamp within menu bounds
            x1_hit = max(0, x1_hit)
            y1_hit = max(0, y1_hit)
            x2_hit = min(self.legend_width, x2_hit)
            y2_hit = min(self.display_h, y2_hit)
            self.class_buttons.append((cls, cid, x1_hit, y1_hit, x2_hit, y2_hit))

        cv2.putText(menu, f"Zoom: {self.zoom_level:.1f}x",
                    (20, self.display_h - 80), self.font, 0.55, (80, 80, 80), 1)
        cv2.putText(menu, "Left drag: pan",
                    (20, self.display_h - 55), self.font, 0.45, (80, 80, 80), 1)
        cv2.putText(menu, f"Undo: Ctrl-Z ({len(self.history) - 1})",
                    (20, self.display_h - 20), self.font, 0.5,  (80, 80, 80), 1)
        return menu

    # -------------------------------------------------------- draw canvas
    def draw(self):
        zw = int(self.orig_w * self.zoom_level)
        zh = int(self.orig_h * self.zoom_level)
        zoomed = (cv2.resize(self.original_image, (zw, zh))
                  if self.zoom_level != 1.0
                  else self.original_image.copy())

        self._clamp_pan()

        canvas = 240 * np.ones(
            (self.display_h, self.effective_display_w, 3), dtype=np.uint8)

        py, px = int(self.pan_y), int(self.pan_x)
        y2s    = min(zh, py + self.display_h)
        x2s    = min(zw, px + self.effective_display_w)
        sh, sw = y2s - py, x2s - px
        if sw > 0 and sh > 0:
            canvas[0:sh, 0:sw] = zoomed[py:y2s, px:x2s]

        # draw detections
        for i, det in enumerate(self.detections):
            cid  = int(det[0])
            x1, y1, x2, y2 = self.yolo_to_display(det)
            x1c = max(0, min(x1, self.effective_display_w - 1))
            y1c = max(0, min(y1, self.display_h - 1))
            x2c = max(0, min(x2, self.effective_display_w - 1))
            y2c = max(0, min(y2, self.display_h - 1))
            col = ((255, 255, 255) if i == self.selected_det
                   else self.class_colors.get(cid, (0, 255, 0)))
            thickness = 4 if i == self.selected_det else 2
            cv2.rectangle(canvas, (x1c, y1c), (x2c, y2c), col, thickness)
            if det[6]:  # human-added dot
                cv2.circle(canvas, (x2c - 6, y1c + 6), 5, (255, 0, 0), -1)

        # live drag rectangle (ADD mode, middle button)
        if self.mode == "add" and self.drag_start and self.drag_end:
            cv2.rectangle(canvas, self.drag_start, self.drag_end, (255, 0, 0), 2)

        # only the image canvas here
        return canvas


    # ------------------------------------------------------ draw stats
    def draw_stats(self):
        lines, active = 0, 0
        for cid in self.class_ids:
            a = self.added_by_class[cid]
            d = self.deleted_by_class[cid]
            r = sum(self.reclassified_from_to[cid].values())
            if a or d or r:
                active += 1
                lines  += 1 + bool(a) + bool(d) + bool(r)

        h   = max(400, 200 + lines * 35 + active * 10)
        img = 240 * np.ones((h, 500, 3), dtype=np.uint8)
        y = 30
        cv2.putText(img, "STATISTICS", (20, y), self.font, 1.2, (0, 0, 0), 3);  y += 50
        cv2.line(img, (20, y), (480, y), (100, 100, 100), 2);                    y += 30
        cv2.putText(img, f"Total Added:        {self.added_count}",
                    (30, y), self.font, 0.8, (0, 180, 0),   2);  y += 40
        cv2.putText(img, f"Total Deleted:      {self.deleted_count}",
                    (30, y), self.font, 0.8, (0, 0, 180),   2);  y += 40
        cv2.putText(img, f"Total Reclassified: {self.reclassified_count}",
                    (30, y), self.font, 0.8, (180, 100, 0), 2);  y += 40
        cv2.line(img, (20, y), (480, y), (100, 100, 100), 2);    y += 30

        for cid in self.class_ids:
            name = self.id_to_name[cid]
            a    = self.added_by_class[cid]
            d    = self.deleted_by_class[cid]
            r    = sum(self.reclassified_from_to[cid].values())
            if a or d or r:
                col = self.class_colors[cid]
                cv2.putText(img, name.upper(), (30, y), self.font, 0.7, col, 2); y += 35
                if a: cv2.putText(img, f"  +{a}", (50, y), self.font, 0.9, (0, 200, 0),   2); y += 35
                if d: cv2.putText(img, f"  -{d}", (50, y), self.font, 0.9, (0, 0, 200),   2); y += 35
                if r: cv2.putText(img, f"  >{r}", (50, y), self.font, 0.9, (200, 100, 0), 2); y += 35
                y += 10
        return img

    # --------------------------------------------------- mouse callback
    def _mouse_callback(self, event, x, y, flags, param):
        """
        Two windows:

          - 'HC Image'     : param == 'image'
                • Wheel            → zoom (0.5x–5x)
                • Middle click drag → draw bbox (ADD mode)
                • Left click       → classify / delete detections

          - 'HC Controls' : param == 'menu'
                • Left click       → mode buttons + class buttons only
                • No zoom, no pan
        """

        in_menu = (param == "menu")

        # ── A. MENU WINDOW: only handle clicks on buttons ────────────────
        if in_menu:
            if event == cv2.EVENT_LBUTTONDOWN:
                # mode buttons
                if   20 <= y <= 60:   self.mode = "classify"
                elif 80 <= y <= 120:  self.mode = "add"
                elif 140 <= y <= 180: self.mode = "delete"

                # class buttons (coords already in menu space)
                for cls, cid, x1b, y1b, x2b, y2b in self.class_buttons:
                    if x1b <= x <= x2b and y1b <= y <= y2b:
                        self.selected_class    = cls
                        self.selected_class_id = cid
                        break
            return  # nothing else (no zoom/pan) in the controls window

        # From here on, we are in the IMAGE window
        # Consume ALL mouse events to prevent window manager interference
        if event in [cv2.EVENT_LBUTTONDOWN, cv2.EVENT_LBUTTONUP, cv2.EVENT_LBUTTONDBLCLK,
                     cv2.EVENT_RBUTTONDOWN, cv2.EVENT_RBUTTONUP, cv2.EVENT_RBUTTONDBLCLK,
                     cv2.EVENT_MBUTTONDOWN, cv2.EVENT_MBUTTONUP, cv2.EVENT_MBUTTONDBLCLK,
                     cv2.EVENT_MOUSEMOVE, cv2.EVENT_MOUSEWHEEL]:
            pass  # Continue processing below

        # ── B. ZOOM (scroll wheel) ───────────────────────────────────────
        if event == cv2.EVENT_MOUSEWHEEL:
            old = self.zoom_level
            if flags > 0:
                self.zoom_level = min(5.0, self.zoom_level * 1.2)
            else:
                self.zoom_level = max(0.5, self.zoom_level / 1.2)
            if old != self.zoom_level:
                zf = self.zoom_level / old
                # zoom around cursor position
                self.pan_x = int(self.pan_x * zf + x * (zf - 1))
                self.pan_y = int(self.pan_y * zf + y * (zf - 1))
                self._clamp_pan()
            return

        # ── C. MIDDLE BUTTON — ADD mode or pan ──────────────────────────
        if event == cv2.EVENT_MBUTTONDOWN:
            if self.mode == "add":
                self.drag_start = (x, y)
                self.drag_end   = None
                return
            # Could add pan functionality here if needed in future
            return

        if event == cv2.EVENT_MBUTTONUP:
            if self.mode == "add" and self.drag_start is not None:
                xe = self.drag_end[0] if self.drag_end else x
                ye = self.drag_end[1] if self.drag_end else y

                if self.selected_class_id is not None:
                    x1d, x2d = sorted([self.drag_start[0], xe])
                    y1d, y2d = sorted([self.drag_start[1], ye])
                    if abs(x2d - x1d) > 5 and abs(y2d - y1d) > 5:
                        det = self.display_to_yolo(x1d, y1d, x2d, y2d,
                                                   self.selected_class_id)
                        self.detections.append(det + [True])
                        self.added_count += 1
                        self.added_by_class[self.selected_class_id] += 1
                        self._save_state()
                        # Auto-save to file
                        if self.state_file:
                            self.save_state_to_file(self.state_file)

                self.drag_start = None
                self.drag_end   = None
            return

        # ── D. MOUSE MOVE — ADD bbox preview or pan ────────────────────
        if event == cv2.EVENT_MOUSEMOVE:
            # Panning with left button (when no detection was hit)
            if self._left_down and self._pan_start is not None:
                dx = x - self._pan_start[0]
                dy = y - self._pan_start[1]
                self.pan_x -= dx
                self.pan_y -= dy
                self._clamp_pan()
                self._pan_start = (x, y)  # Update pan start for next move
            
            # ADD bbox preview with middle button
            if (self.drag_start is not None
                    and self.mode == "add"):
                self.drag_end = (x, y)
            return

        # ── E. LEFT BUTTON DOWN on IMAGE ────────────────────────────────
        if event == cv2.EVENT_LBUTTONDOWN:
            self._left_down = True

            # CLASSIFY / DELETE: hit-test detections (topmost first)
            detection_hit = False
            for i in range(len(self.detections) - 1, -1, -1):
                det  = self.detections[i]
                bbox = self.yolo_to_display(det)
                if not self._hit(x, y, bbox):
                    continue

                detection_hit = True
                
                if self.mode == "delete":
                    removed     = self.detections.pop(i)
                    removed_cls = int(removed[0])
                    if removed[6]:  # was human-added
                        self.added_count -= 1
                        self.added_by_class[removed_cls] -= 1
                    self.deleted_count += 1
                    self.deleted_by_class[removed_cls] += 1
                    self._save_state()
                    # Auto-save to file
                    if self.state_file:
                        self.save_state_to_file(self.state_file)
                    return

                if self.mode == "classify" and self.selected_class_id is not None:
                    old_cls = int(det[0])
                    if old_cls != self.selected_class_id:
                        det[0] = float(self.selected_class_id)
                        self.reclassified_count += 1
                        self.reclassified_from_to[old_cls][self.selected_class_id] += 1
                        self._save_state()
                        # Auto-save to file
                        if self.state_file:
                            self.save_state_to_file(self.state_file)
                    self.selected_det = i
                    return
            
            # No detection hit: prepare for panning
            if not detection_hit:
                self._pan_start = (x, y)
            
            return

        # ── F. LEFT BUTTON UP — finish panning/selection ─────────────────
        elif event == cv2.EVENT_LBUTTONUP:
            self._left_down = False
            self._pan_start = None
            return
    def print_log(self):
        print("\n" + "=" * 70)
        print("                    HUMAN CORRECTION LOG")
        print("=" * 70)
        print(f"\n{'SUMMARY':^70}")
        print("-" * 70)
        print(f"  Total Added:        {self.added_count:>5}")
        print(f"  Total Deleted:      {self.deleted_count:>5}")
        print(f"  Total Reclassified: {self.reclassified_count:>5}")
        print("-" * 70)

        print(f"\n{'CHANGES BY CLASS':^70}")
        print("=" * 70)
        for cid in self.class_ids:
            name = self.id_to_name.get(cid, f"Class {cid}")
            a    = self.added_by_class[cid]
            d    = self.deleted_by_class[cid]
            rf   = sum(self.reclassified_from_to[cid].values())
            rt   = sum(v[cid] for v in self.reclassified_from_to.values())
            if a or d or rf or rt:
                print(f"\n  {name.upper()}")
                print(f"  {'-' * len(name)}")
                if a:  print(f"    ✓ Added:              {a:>5}")
                if d:  print(f"    ✗ Deleted:            {d:>5}")
                if rf: print(f"    → Reclassified FROM:  {rf:>5}")
                if rt: print(f"    ← Reclassified TO:    {rt:>5}")

        if self.reclassified_from_to:
            print(f"\n{'RECLASSIFICATION DETAILS':^70}")
            print("=" * 70)
            for fc in sorted(self.reclassified_from_to):
                fn = self.id_to_name.get(fc, f"Class {fc}")
                for tc in sorted(self.reclassified_from_to[fc]):
                    cnt = self.reclassified_from_to[fc][tc]
                    if cnt:
                        tn = self.id_to_name.get(tc, f"Class {tc}")
                        print(f"  {fn:>20} → {tn:<20} : {cnt:>5}")

        print("\n" + "=" * 70 + "\n")

    # --------------------------------------------------------------- run
    def run(self):
        print("\n" + "=" * 70)
        print("HUMAN CORRECTION GUI")
        print("=" * 70)
        print("\n🖱️  MOUSE CONTROLS:")
        print("  SCROLL WHEEL          : Zoom in / out around cursor")
        print("  LEFT button drag      : Pan zoomed image (when not on a detection)")
        print("  LEFT button on detect : Classify / delete detection")
        print("  MIDDLE button drag    : Draw new bbox (ADD mode only)")
        print("\n⌨️  KEYBOARD:")
        print("  C        : mode CLASSIFY")
        print("  A        : mode ADD")
        print("  D        : mode DELETE")
        print("  Ctrl+Z   : Undo (auto-saves)")
        print("  Q / ESC  : Quit (auto-saves)")
        print("  Auto-save: After each action")
        print("\n" + "=" * 70 + "\n")

        while True:
            # Check if windows are still open
            if cv2.getWindowProperty("HC Image", cv2.WND_PROP_VISIBLE) < 1:
                break
                
            cv2.imshow("HC Image",     self.draw())
            cv2.imshow("HC Controls",  self.draw_menu())
            cv2.imshow("Statistics",   self.draw_stats())
            key = cv2.waitKey(20) & 0xFF

            if key in (ord('c'), ord('C')): self.mode = "classify"; print("✓ CLASSIFY")
            if key in (ord('a'), ord('A')): self.mode = "add";      print("✓ ADD")
            if key in (ord('d'), ord('D')): self.mode = "delete";   print("✓ DELETE")
            if key in (27, ord('q'), ord('Q')): break
            if key == 26: self.undo(); print("↶ Undo")   # Ctrl+Z

        # Save state when quitting
        if self.state_file:
            self.save_state_to_file(self.state_file)

        cv2.destroyAllWindows()
        self.print_log()
        return [det[:6] for det in self.detections]
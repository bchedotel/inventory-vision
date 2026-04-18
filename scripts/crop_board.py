import cv2
import numpy as np

clicked_points = []
current_mouse_pos = None

def draw_shape(img, points, moving_point=None):
    """
    Draw the evolving shape (point, line, triangle, quadrilateral)
    using the clicked points and optionally the current mouse position.
    - Outline and points in red.
    - Transparent red fill inside the shape.

    Args:
        img (np.ndarray): the image to draw
        points (np.ndarray): the clicked points to draw
        moving_point (np.ndarray): the clicked points to draw
    """
    pts = points.copy()
    if moving_point is not None and len(pts) < 4:
        pts.append(moving_point)

    overlay = img.copy()

    if len(pts) == 1:
        cv2.circle(img, pts[0], 5, (0, 0, 255), -1)  # Red dot
    elif len(pts) >= 2:
        pts_np = np.array(pts, np.int32).reshape((-1, 1, 2))
        if len(pts) > 2:
            cv2.fillPoly(overlay, [pts_np], color=(0, 0, 255))
            alpha = 0.3  # Transparency factor
            cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)
        cv2.polylines(img, [pts_np], isClosed=len(pts) == 4, color=(0, 0, 255), thickness=2)
        for p in pts:
            cv2.circle(img, p, 5, (0, 0, 255), -1)


def mouse_callback(event, x, y, flags, param):
    """
    Single mouse callback function for both clicks and mouse move.

    Args:
        event (int): the mouse event
        x (int): the x coordinate of clicked point
        y (int): the y coordinate of clicked point
        flags (int): the mouse event flags
        param (np.ndarray): the clicked point
    """
    global clicked_points, current_mouse_pos
    if event == cv2.EVENT_LBUTTONDOWN and len(clicked_points) < 4:
        clicked_points.append((x, y))
    elif event == cv2.EVENT_MOUSEMOVE:
        current_mouse_pos = (x, y)

def order_points(pts):
    """
    Orders points as top-left, top-right, bottom-right, bottom-left.

    Args:
        pts (np.ndarray): the clicked points to draw
    """
    pts = np.array(pts, dtype="float32")
    s = pts.sum(axis=1)
    diff = np.diff(pts, axis=1)
    return np.array([
        pts[np.argmin(s)],
        pts[np.argmin(diff)],
        pts[np.argmax(s)],
        pts[np.argmax(diff)]
    ], dtype="float32")

def crop_and_warp(image, pts):
    """
    Warp the quadrilateral area defined by pts into a rectangle.

    Args:
        image (np.ndarray): the image to crop
        pts (np.ndarray): the clicked points to draw
    """
    rect = order_points(pts)
    (tl, tr, br, bl) = rect

    widthA = np.linalg.norm(br - bl)
    widthB = np.linalg.norm(tr - tl)
    maxWidth = int(max(widthA, widthB))

    heightA = np.linalg.norm(tr - br)
    heightB = np.linalg.norm(tl - bl)
    maxHeight = int(max(heightA, heightB))

    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]
    ], dtype="float32")

    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))
    return warped

def main(input_path, output_path, term: bool = False):
    global clicked_points, current_mouse_pos
    
    # RESET GLOBAL VARIABLES FOR EACH NEW IMAGE
    clicked_points = []
    current_mouse_pos = None

    image = cv2.imread(input_path)
    if image is None:
        raise FileNotFoundError(f"Cannot read image: {input_path}")

    if term:
        # Terminal input mode
        print("Enter 4 points manually as x y (top-left, top-right, bottom-right, bottom-left):")
        pts = []
        for i in ["TL", "TR", "BR", "BL"]:
            coords = input(f"{i} (x y): ").split()
            pts.append((float(coords[0]), float(coords[1])))
    else:
        # GUI mode with dynamic shape preview
        clone = image.copy()
        window_name = "Click 4 points (TL, TR, BR, BL)"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        
        # Force window to front
        cv2.setWindowProperty(window_name, cv2.WND_PROP_TOPMOST, 1)

        cv2.setMouseCallback(window_name, mouse_callback)

        print("Click 4 points in order: TL, TR, BR, BL. Press ESC to quit.")

        while True:
            temp_img = clone.copy()
            draw_shape(temp_img, clicked_points, moving_point=current_mouse_pos)

            cv2.imshow(window_name, temp_img)

            key = cv2.waitKey(20) & 0xFF
            if key == 27:  # ESC to quit
                print("Quitting without completing 4 points.")
                cv2.destroyAllWindows()
                return
            if len(clicked_points) == 4:
                break

        cv2.destroyAllWindows()
        pts = clicked_points

    warped = crop_and_warp(image, pts)
    cv2.imwrite(output_path, warped)
    print(f"Saved cropped image to: {output_path}")
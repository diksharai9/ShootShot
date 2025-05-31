from flask import Flask, render_template, Response, request
import cv2
import numpy as np
import os
import atexit
import time  # Added for potential use, though not strictly in the requested initialize_camera

app = Flask(__name__)

# --- Configuration and Resource Loading ---
base_dir = os.path.dirname(os.path.abspath(__file__))
cascade_path = os.path.join(cv2.data.haarcascades, 'haarcascade_frontalface_default.xml')
glasses_path = os.path.join(base_dir, 'static', 'sunglasses.png')
moustache_path = os.path.join(base_dir, 'static', 'moustache.png')

face_cascade = cv2.CascadeClassifier(cascade_path)
if face_cascade.empty():
    raise IOError(f"Unable to load face cascade classifier from {cascade_path}")

glasses_img = cv2.imread(glasses_path, cv2.IMREAD_UNCHANGED)
if glasses_img is None:
    raise IOError(f"Unable to load sunglasses image from {glasses_path}")

moustache_img = cv2.imread(moustache_path, cv2.IMREAD_UNCHANGED)
if moustache_img is None:
    raise IOError(f"Unable to load moustache image from {moustache_path}")

# Global camera object
cap = None


def initialize_camera():
    global cap
    # If camera is already open and working, no need to re-initialize
    if cap is not None and cap.isOpened():
        print("Camera already initialized and open.")
        return True

    camera_index = 1  # Explicitly set to use camera index 1 as requested

    print(f"Attempting to open camera at index {camera_index}...")
    cap = cv2.VideoCapture(camera_index)

    if cap.isOpened():
        print(f"Camera opened successfully at index {camera_index}")
        # Set desired frame width and height
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

        actual_width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        actual_height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        print(f"Camera resolution confirmed: {actual_width}x{actual_height}")
        return True
    else:
        print(f"Error: Failed to open camera at index {camera_index}.")
        print("Please ensure the camera is connected and not in use by another application.")
        # Try to release if it was partially acquired but not opened
        if cap is not None:
            cap.release()
        cap = None  # Ensure cap is None if opening failed
        return False


def release_camera_on_exit():
    global cap
    if cap is not None and cap.isOpened():
        print("Releasing camera resource...")
        cap.release()
    print("Camera released or was not open.")


atexit.register(release_camera_on_exit)


# --- Image Processing Functions ---
def overlay_image_alpha(frame, image_alpha, x_offset, y_offset):
    """
    Overlays a BGRA image_alpha onto a BGR frame at (x_offset, y_offset).
    Handles partial overlays at frame boundaries.
    """
    try:
        # Frame dimensions
        frame_h, frame_w = frame.shape[:2]
        # Image dimensions
        img_h, img_w = image_alpha.shape[:2]

        # Calculate the start and end coordinates for the overlay on the frame
        y1_frame, y2_frame = y_offset, y_offset + img_h
        x1_frame, x2_frame = x_offset, x_offset + img_w

        # Calculate the start and end coordinates for cropping the image_alpha
        # This handles cases where the overlay is partially outside the frame
        img_y1_crop = 0
        img_y2_crop = img_h
        img_x1_crop = 0
        img_x2_crop = img_w

        if y1_frame < 0:
            img_y1_crop = -y1_frame
            y1_frame = 0
        if x1_frame < 0:
            img_x1_crop = -x1_frame
            x1_frame = 0

        if y2_frame > frame_h:
            img_y2_crop = img_h - (y2_frame - frame_h)
            y2_frame = frame_h
        if x2_frame > frame_w:
            img_x2_crop = img_w - (x2_frame - frame_w)
            x2_frame = frame_w

        # If the cropped image has no size, or the frame ROI has no size, return
        if img_y1_crop >= img_y2_crop or img_x1_crop >= img_x2_crop or \
                y1_frame >= y2_frame or x1_frame >= x2_frame:
            return frame

        # Crop the overlay image
        cropped_image = image_alpha[img_y1_crop:img_y2_crop, img_x1_crop:img_x2_crop]

        # Ensure the cropped image is not empty
        if cropped_image.size == 0:
            return frame

        # Get the region of interest (ROI) from the frame
        roi = frame[y1_frame:y2_frame, x1_frame:x2_frame]

        # Ensure ROI and cropped_image have compatible dimensions for blending
        if roi.shape[0] != cropped_image.shape[0] or roi.shape[1] != cropped_image.shape[1]:
            # This can happen if calculations above are slightly off or due to rounding.
            # A common fix is to resize cropped_image to roi dimensions, but this might distort.
            # For now, we'll print a warning and skip overlay if dimensions mismatch.
            # print(f"Warning: ROI shape {roi.shape} and cropped_image shape {cropped_image.shape} mismatch. Skipping overlay.")
            # As a fallback, try to resize the cropped_image to fit the ROI
            if roi.shape[0] > 0 and roi.shape[1] > 0:  # Only resize if ROI is valid
                cropped_image = cv2.resize(cropped_image, (roi.shape[1], roi.shape[0]), interpolation=cv2.INTER_AREA)
            else:
                return frame  # Cannot proceed if ROI is invalid

        if cropped_image.shape[2] < 4:
            print("Warning: Overlay image does not have an alpha channel.")
            return frame

        alpha_mask = cropped_image[:, :, 3] / 255.0
        bgr_image = cropped_image[:, :, :3]

        # Blend the image
        for c in range(0, 3):
            roi[:, :, c] = (alpha_mask * bgr_image[:, :, c] +
                            (1 - alpha_mask) * roi[:, :, c])

        # Put the modified ROI back into the frame
        frame[y1_frame:y2_frame, x1_frame:x2_frame] = roi
    except Exception as e:
        print(f"Error in overlay_image_alpha: {e}")
        # import traceback
        # traceback.print_exc()
    return frame


def apply_glasses(frame, x, y, w, h):
    target_h = int(h * 0.40)
    target_w = w
    if target_h <= 0 or target_w <= 0: return frame

    resized_filter = cv2.resize(glasses_img, (target_w, target_h), interpolation=cv2.INTER_AREA)
    pos_x = x
    pos_y = y + int(h * 0.18)
    return overlay_image_alpha(frame, resized_filter, pos_x, pos_y)


def apply_moustache(frame, x, y, w, h):
    target_w = int(w * 0.55)
    target_h = int(h * 0.20)
    if target_h <= 0 or target_w <= 0: return frame

    resized_filter = cv2.resize(moustache_img, (target_w, target_h), interpolation=cv2.INTER_AREA)
    pos_x = x + int((w - target_w) / 2)
    pos_y = y + int(h * 0.60)
    return overlay_image_alpha(frame, resized_filter, pos_x, pos_y)


def increase_vibrance(frame, strength=1.3):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[:, :, 1] = np.clip(hsv[:, :, 1] * strength, 0, 255)
    hsv = hsv.astype(np.uint8)
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)


# --- Frame Generation ---
def gen_frames(glasses_on, moustache_on):
    global cap
    if not initialize_camera() or cap is None or not cap.isOpened():
        print("Camera initialization failed or camera not open in gen_frames.")
        error_img = np.zeros((480, 640, 3), dtype=np.uint8)
        cv2.putText(error_img, "Camera Error", (int(640 * 0.25), int(480 * 0.5)), cv2.FONT_HERSHEY_SIMPLEX, 1.5,
                    (255, 255, 255), 2)
        while True:
            ret, buffer = cv2.imencode('.jpg', error_img)
            if not ret:
                # Fallback if encoding error_img fails (highly unlikely)
                yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + b'' + b'\r\n')
                time.sleep(1)  # Prevent tight loop on error
                continue
            frame_bytes = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
            time.sleep(0.1)  # Yield error frame periodically

    print("Starting frame generation loop...")
    while True:
        if not cap.isOpened():
            print("Camera became disconnected during streaming. Attempting to re-initialize.")
            if not initialize_camera():
                print("Failed to re-initialize camera. Streaming error frame.")
                error_img_local = np.zeros((480, 640, 3), dtype=np.uint8)
                cv2.putText(error_img_local, "Camera Lost", (int(640 * 0.2), int(480 * 0.5)), cv2.FONT_HERSHEY_SIMPLEX,
                            1.5, (0, 0, 255), 2)
                ret, buffer = cv2.imencode('.jpg', error_img_local)
                if not ret:
                    time.sleep(1)
                    continue
                frame_bytes_err = buffer.tobytes()
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes_err + b'\r\n')
                time.sleep(1)
                continue
            else:
                print("Camera re-initialized successfully.")

        success, frame = cap.read()
        if not success:
            print("Failed to read frame from camera. Skipping frame.")
            time.sleep(0.05)  # Wait a bit if frame read fails
            continue

        processed_frame = frame.copy()
        gray = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2GRAY)
        # Adjusted parameters for potentially better/faster detection
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.15, minNeighbors=5, minSize=(40, 40))

        for (x, y, w, h) in faces:
            if glasses_on:
                processed_frame = apply_glasses(processed_frame, x, y, w, h)
            if moustache_on:
                processed_frame = apply_moustache(processed_frame, x, y, w, h)

        processed_frame = increase_vibrance(processed_frame, strength=1.4)

        ret, buffer = cv2.imencode('.jpg', processed_frame)
        if not ret:
            print("Failed to encode frame to JPEG.")
            continue

        frame_bytes = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

        # Small delay to control frame rate and reduce CPU load.
        # Adjust as needed. 1/30 FPS = ~0.033s. cv2.waitKey(1) is ~1ms.
        # For streaming, often the client request rate dictates this.
        # A small explicit sleep can help if cv2.waitKey isn't sufficient or used.
        time.sleep(0.01)  # approx 10ms, can be adjusted


# --- Flask Routes ---
@app.route('/')
def index():
    # These initial values will be passed to the template.
    # The template's JS will then use these to set the initial state of checkboxes
    # and make the first call to /video_feed
    return render_template('index.html', glasses_initial='0', moustache_initial='0')


@app.route('/video_feed')
def video_feed():
    glasses = request.args.get('glasses', '0') == '1'
    moustache = request.args.get('moustache', '0') == '1'
    # print(f"Video feed request: glasses={glasses}, moustache={moustache}") # For debugging
    return Response(gen_frames(glasses, moustache),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


if __name__ == '__main__':
    print("Starting Flask app...")
    # Run the Flask app with specified host, port, debug, and threaded options
    app.run(debug=True)

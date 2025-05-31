from flask import Flask, render_template, Response, request
import cv2
import numpy as np
import os
import atexit
import time

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
    raise IOError(f"Unable to load sunglasses image from {glasses_path}. Ensure file exists and path is correct.")

moustache_img = cv2.imread(moustache_path, cv2.IMREAD_UNCHANGED)
if moustache_img is None:
    raise IOError(f"Unable to load moustache image from {moustache_path}. Ensure file exists and path is correct.")

# Global camera object
cap = None

# --- Global Variables for Smoother Face Tracking ---
last_detected_face = None
NO_FACE_THRESHOLD = 15
no_face_frame_count = 0


def initialize_camera():
    global cap
    if cap is not None and cap.isOpened():
        print("Camera already initialized and open.")
        return True

    # Modified: Only try camera at index 1
    camera_index = 1
    print(f"Attempting to open camera at index {camera_index}...")
    cap = cv2.VideoCapture(camera_index)
    if cap.isOpened():
        print(f"Camera opened successfully at index {camera_index}")
        # Set desired frame width and height for consistency
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        return True
    else:
        print(f"Failed to open camera at index {camera_index}.")
        print("Camera at index 1 could not be opened. Please check camera connection and permissions.")
        cap = None
        return False


def release_camera_on_exit():
    global cap
    if cap is not None and cap.isOpened():
        print("Releasing camera.")
        cap.release()
        cap = None


atexit.register(release_camera_on_exit)


# --- Image Processing Functions ---
def overlay_image_alpha(frame, image_alpha, x_offset, y_offset):
    """
    Overlays a BGRA image_alpha onto a BGR frame at (x_offset, y_offset).
    Handles partial overlays at frame boundaries.
    """
    try:
        if image_alpha.shape[2] < 4:  # Check if image has an alpha channel
            print("Warning: Overlay image does not have an alpha channel. Skipping overlay.")
            return frame

        img_h, img_w = image_alpha.shape[:2]
        frame_h, frame_w = frame.shape[:2]

        # Calculate region of interest (ROI) coordinates on the frame
        y1_frame = max(0, y_offset)
        y2_frame = min(frame_h, y_offset + img_h)
        x1_frame = max(0, x_offset)
        x2_frame = min(frame_w, x_offset + img_w)

        # Calculate corresponding crop coordinates on the overlay image
        img_y1_crop = max(0, -y_offset)
        img_y2_crop = img_h - max(0, (y_offset + img_h) - frame_h)
        img_x1_crop = max(0, -x_offset)
        img_x2_crop = img_w - max(0, (x_offset + img_w) - frame_w)

        # Check if ROI is valid
        if y1_frame >= y2_frame or x1_frame >= x2_frame:
            return frame

        # Crop the overlay image to fit within frame boundaries
        cropped_image = image_alpha[img_y1_crop:img_y2_crop, img_x1_crop:img_x2_crop]

        # Resize cropped image to match ROI dimensions if necessary
        roi_h = y2_frame - y1_frame
        roi_w = x2_frame - x1_frame
        if cropped_image.shape[0] != roi_h or cropped_image.shape[1] != roi_w:
            cropped_image = cv2.resize(cropped_image, (roi_w, roi_h), interpolation=cv2.INTER_AREA)

        alpha_mask = cropped_image[:, :, 3] / 255.0
        bgr_image = cropped_image[:, :, :3]

        roi = frame[y1_frame:y2_frame, x1_frame:x2_frame]

        # Blend the images
        for c in range(0, 3):
            roi[:, :, c] = (alpha_mask * bgr_image[:, :, c] +
                            (1 - alpha_mask) * roi[:, :, c])

        frame[y1_frame:y2_frame, x1_frame:x2_frame] = roi

    except Exception as e:
        print(f"Error in overlay_image_alpha: {e}")
    return frame


def apply_glasses(frame, x, y, w, h):
    target_w = w
    target_h = int(h * 0.35)
    if target_h <= 0 or target_w <= 0: return frame  # Handle zero dimensions
    resized_filter = cv2.resize(glasses_img, (target_w, target_h), interpolation=cv2.INTER_AREA)
    pos_x = x
    pos_y = y + int(h * 0.25)
    return overlay_image_alpha(frame, resized_filter, pos_x, pos_y)


def apply_moustache(frame, x, y, w, h):
    target_w = int(w * 0.45)
    target_h = int(h * 0.15)
    if target_h <= 0 or target_w <= 0: return frame  # Handle zero dimensions
    resized_filter = cv2.resize(moustache_img, (target_w, target_h), interpolation=cv2.INTER_AREA)
    pos_x = x + int((w - target_w) / 2)
    pos_y = y + int(h * 0.65)
    return overlay_image_alpha(frame, resized_filter, pos_x, pos_y)


def increase_vibrance(frame, strength=1.4):  # Increased strength slightly for more pop
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[:, :, 1] = np.clip(hsv[:, :, 1] * strength, 0, 255)
    hsv = hsv.astype(np.uint8)
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)


def apply_bw_filter(frame):
    # Convert to grayscale, then convert back to BGR for consistency
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return cv2.cvtColor(gray_frame, cv2.COLOR_GRAY2BGR)


def apply_sepia_filter(frame):
    # Sepia matrix for BGR image
    sepia_matrix = np.array([
        [0.131, 0.534, 0.272],  # B channel
        [0.168, 0.686, 0.349],  # G channel
        [0.189, 0.769, 0.393]  # R channel
    ]).T  # Transpose for correct multiplication with (B,G,R) vector

    # Apply the matrix transformation
    transformed_frame = np.dot(frame.astype(np.float32), sepia_matrix)
    # Clip values to 0-255 and convert back to uint8
    transformed_frame = np.clip(transformed_frame, 0, 255).astype(np.uint8)
    return transformed_frame


def apply_invert_filter(frame):
    return 255 - frame  # Invert all pixel values (BGR)


# --- Frame Generation ---
def gen_frames(glasses_on, moustache_on, bw_on, sepia_on, invert_on):
    global cap, last_detected_face, no_face_frame_count

    # Initial camera check and error frame if not available
    if not initialize_camera():
        error_img = np.zeros((480, 640, 3), dtype=np.uint8)
        cv2.putText(error_img, "Camera not available", (int(640 * 0.1), int(480 * 0.5)), cv2.FONT_HERSHEY_SIMPLEX, 1.5,
                    (255, 255, 255), 2)
        ret, buffer = cv2.imencode('.jpg', error_img)
        frame_bytes = buffer.tobytes()
        while True:  # Keep yielding error frame
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\r\n')  # Added extra \r for compatibility
            time.sleep(1)  # Reduce CPU usage while waiting for camera

    while True:
        # Re-check camera status within the loop
        if not cap.isOpened():
            print("Camera disconnected during stream. Attempting to re-initialize.")
            if not initialize_camera():  # Try to re-initialize
                print("Failed to re-initialize camera. Streaming error frame.")
                error_img_local = np.zeros((480, 640, 3), dtype=np.uint8)
                cv2.putText(error_img_local, "Camera lost", (int(640 * 0.2), int(480 * 0.5)), cv2.FONT_HERSHEY_SIMPLEX,
                            1.5, (0, 0, 255), 2)
                ret, buffer = cv2.imencode('.jpg', error_img_local)
                frame_bytes_err = buffer.tobytes()
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes_err + b'\r\r\n')  # Added extra \r for compatibility
                time.sleep(1)  # Wait before retrying
                continue  # Skip to next loop iteration

        success, frame = cap.read()
        if not success:
            print("Failed to read frame from camera. Retrying...")
            time.sleep(0.1)  # Small delay to avoid busy-waiting
            continue

        processed_frame = frame.copy()

        # Apply global color filters first
        if invert_on:
            processed_frame = apply_invert_filter(processed_frame)
        if sepia_on:
            processed_frame = apply_sepia_filter(processed_frame)
        if bw_on:
            processed_frame = apply_bw_filter(processed_frame)

        # Face detection and overlay (applies to the filtered frame)
        gray = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2GRAY)
        current_faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(50, 50))

        if len(current_faces) > 0:
            largest_face = max(current_faces, key=lambda rect: rect[2] * rect[3])

            if last_detected_face is None:
                last_detected_face = largest_face
            else:
                alpha = 0.7  # Smoothing factor for face tracking
                last_detected_face = (
                    int(last_detected_face[0] * (1 - alpha) + largest_face[0] * alpha),
                    int(last_detected_face[1] * (1 - alpha) + largest_face[1] * alpha),
                    int(last_detected_face[2] * (1 - alpha) + largest_face[2] * alpha),
                    int(last_detected_face[3] * (1 - alpha) + largest_face[3] * alpha)
                )
            no_face_frame_count = 0
        else:
            no_face_frame_count += 1
            if no_face_frame_count >= NO_FACE_THRESHOLD:
                last_detected_face = None

        if last_detected_face is not None:
            x, y, w, h = last_detected_face
            if glasses_on:
                processed_frame = apply_glasses(processed_frame, x, y, w, h)
            if moustache_on:
                processed_frame = apply_moustache(processed_frame, x, y, w, h)

        # Apply vibrance last
        processed_frame = increase_vibrance(processed_frame, strength=1.4)

        ret, buffer = cv2.imencode('.jpg', processed_frame)
        if not ret:
            print("Failed to encode frame to JPEG. Skipping frame.")
            continue

        frame_bytes = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\r\n')  # Added extra \r for compatibility
        cv2.waitKey(1)  # Small delay to yield CPU, adjust if needed for higher FPS


# --- Flask Routes ---
@app.route('/')
def index():
    return render_template('index.html')  # No initial filter states needed here


@app.route('/video_feed')
def video_feed():
    # Get filter states from URL parameters
    glasses = request.args.get('glasses', '0') == '1'
    moustache = request.args.get('moustache', '0') == '1'
    bw = request.args.get('bw', '0') == '1'
    sepia = request.args.get('sepia', '0') == '1'
    invert = request.args.get('invert', '0') == '1'

    return Response(gen_frames(glasses, moustache, bw, sepia, invert),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


if __name__ == '__main__':
    release_camera_on_exit()  # Ensure camera is released on app exit
    import os
    port = int(os.environ.get("PORT", 5000))  # Get port from environment variable, default to 5000
    app.run(host='0.0.0.0', port=port, debug=True)
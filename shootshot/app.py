import cv2
import numpy as np
import os
import atexit
import time
from flask import Flask, render_template, Response, request

app = Flask(__name__)

# --- Configuration and Resource Loading ---
# Determine the base directory of the script
base_dir = os.path.dirname(os.path.abspath(__file__))

# Construct paths to cascade and image files
# Use cv2.data.haarcascades for reliable path to OpenCV data
cascade_path = os.path.join(cv2.data.haarcascades, 'haarcascade_frontalface_default.xml')
glasses_path = os.path.join(base_dir, 'static', 'sunglasses.png')
moustache_path = os.path.join(base_dir, 'static', 'moustache.png')

# Load face cascade classifier
face_cascade = cv2.CascadeClassifier(cascade_path)
if face_cascade.empty():
    # Raise an error if the cascade file cannot be loaded
    raise IOError(f"Unable to load face cascade classifier from {cascade_path}")

# Load overlay images
# cv2.IMREAD_UNCHANGED ensures the alpha channel is read for transparency
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
NO_FACE_THRESHOLD = 15  # Number of frames to wait before clearing last_detected_face
no_face_frame_count = 0


# --- Camera Initialization and Release ---
def initialize_camera():
    global cap
    # If camera is already initialized and open, return True
    if cap is not None and cap.isOpened():
        print("Camera already initialized and open.")
        return True

    # Try common camera indices to find an available camera
    # We'll try from 0 up to a reasonable number (e.g., 5)
    # This loop improves compatibility across different systems
    for camera_index in range(6):  # Tries indices 0, 1, 2, 3, 4, 5
        print(f"Attempting to open camera at index {camera_index}...")
        cap = cv2.VideoCapture(camera_index)  # Try to open camera
        if cap.isOpened():
            print(f"Camera opened successfully at index {camera_index}")
            # Set desired frame width and height for consistency
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            return True
        else:
            print(f"Failed to open camera at index {camera_index}.")
            # If camera not opened, release the cap object to clear any resources
            # This is important if an attempt partially acquires a camera resource
            if cap is not None:
                cap.release()
                cap = None  # Reset cap to None for the next attempt

    # If no camera is found after trying all indices
    print("No working camera found after trying multiple indices. Please check camera connection and permissions.")
    cap = None  # Ensure cap is None if no camera was found
    return False


def release_camera_on_exit():
    """
    Releases the camera resource when the application exits.
    Registered with atexit to ensure it's called reliably.
    """
    global cap
    if cap is not None and cap.isOpened():
        print("Releasing camera.")
        cap.release()
        cap = None


# Register the camera release function to run automatically on application exit
atexit.register(release_camera_on_exit)


# --- Image Processing Functions ---

def overlay_image_alpha(frame, image_alpha, x_offset, y_offset):
    """
    Overlays a BGRA image_alpha (with alpha channel) onto a BGR frame
    at the specified x, y offset. Handles transparency and boundary conditions.
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

        # Check if ROI is valid (i.e., if overlay is within or partially within the frame)
        if y1_frame >= y2_frame or x1_frame >= x2_frame:
            return frame

        # Crop the overlay image to fit within frame boundaries
        cropped_image = image_alpha[img_y1_crop:img_y2_crop, img_x1_crop:img_x2_crop]

        # Resize cropped image to match ROI dimensions if necessary
        roi_h = y2_frame - y1_frame
        roi_w = x2_frame - x1_frame
        if cropped_image.shape[0] != roi_h or cropped_image.shape[1] != roi_w:
            cropped_image = cv2.resize(cropped_image, (roi_w, roi_h), interpolation=cv2.INTER_AREA)

        # Separate alpha channel and BGR channels
        alpha_mask = cropped_image[:, :, 3] / 255.0
        bgr_image = cropped_image[:, :, :3]

        # Get the region of interest from the original frame
        roi = frame[y1_frame:y2_frame, x1_frame:x2_frame]

        # Blend the images using the alpha mask
        # roi = (alpha_mask * bgr_image) + ((1 - alpha_mask) * roi) is the general formula
        for c in range(0, 3):  # Iterate over B, G, R channels
            roi[:, :, c] = (alpha_mask * bgr_image[:, :, c] +
                            (1 - alpha_mask) * roi[:, :, c])

        # Place the blended ROI back into the frame
        frame[y1_frame:y2_frame, x1_frame:x2_frame] = roi

    except Exception as e:
        print(f"Error in overlay_image_alpha: {e}")
    return frame


def apply_glasses(frame, x, y, w, h):
    """Applies sunglasses overlay to the detected face."""
    # Scale glasses to face width; height is proportional
    target_w = w
    target_h = int(h * 0.35)  # Adjusted proportion for glasses
    if target_h <= 0 or target_w <= 0: return frame  # Handle zero dimensions

    # Resize glasses image
    resized_filter = cv2.resize(glasses_img, (target_w, target_h), interpolation=cv2.INTER_AREA)

    # Position glasses over the eyes (adjust y_offset for precise placement)
    pos_x = x
    pos_y = y + int(h * 0.25)  # Place glasses around the upper part of the face

    return overlay_image_alpha(frame, resized_filter, pos_x, pos_y)


def apply_moustache(frame, x, y, w, h):
    """Applies moustache overlay to the detected face."""
    # Scale moustache to a portion of face width
    target_w = int(w * 0.45)  # Moustache width is less than full face width
    target_h = int(h * 0.15)  # Moustache height
    if target_h <= 0 or target_w <= 0: return frame  # Handle zero dimensions

    # Resize moustache image
    resized_filter = cv2.resize(moustache_img, (target_w, target_h), interpolation=cv2.INTER_AREA)

    # Position moustache (adjust x_offset to center, y_offset for mouth area)
    pos_x = x + int((w - target_w) / 2)  # Center horizontally on the face
    pos_y = y + int(h * 0.65)  # Place around the mouth area

    return overlay_image_alpha(frame, resized_filter, pos_x, pos_y)


def increase_vibrance(frame, strength=1.4):
    """Increases the vibrance/saturation of the frame."""
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV).astype(np.float32)
    # Increase saturation channel (index 1)
    hsv[:, :, 1] = np.clip(hsv[:, :, 1] * strength, 0, 255)
    hsv = hsv.astype(np.uint8)
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)


def apply_bw_filter(frame):
    """Applies a black and white (grayscale) filter."""
    # Convert to grayscale, then convert back to BGR for consistency
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return cv2.cvtColor(gray_frame, cv2.COLOR_GRAY2BGR)


def apply_sepia_filter(frame):
    """Applies a sepia tone filter."""
    # Sepia transformation matrix for BGR images
    sepia_matrix = np.array([
        [0.131, 0.534, 0.272],  # Blue channel transformation
        [0.168, 0.686, 0.349],  # Green channel transformation
        [0.189, 0.769, 0.393]  # Red channel transformation
    ]).T  # Transpose the matrix for correct multiplication with (B,G,R) pixel vectors

    # Apply the matrix transformation
    transformed_frame = np.dot(frame.astype(np.float32), sepia_matrix)
    # Clip values to 0-255 and convert back to uint8
    transformed_frame = np.clip(transformed_frame, 0, 255).astype(np.uint8)
    return transformed_frame


def apply_invert_filter(frame):
    """Inverts the colors of the frame."""
    return 255 - frame  # Simple inversion of all pixel values (BGR)


# --- Frame Generation Loop ---
def gen_frames(glasses_on, moustache_on, bw_on, sepia_on, invert_on):
    global cap, last_detected_face, no_face_frame_count

    # Initial camera check. If not available, yield an error frame.
    if not initialize_camera():
        print("Camera not available at start. Streaming error frame.")
        error_img = np.zeros((480, 640, 3), dtype=np.uint8)
        cv2.putText(error_img, "Camera Not Available", (int(640 * 0.1), int(480 * 0.5)), cv2.FONT_HERSHEY_SIMPLEX, 1.5,
                    (255, 255, 255), 2)
        ret, buffer = cv2.imencode('.jpg', error_img)
        frame_bytes = buffer.tobytes()
        while True:  # Keep yielding error frame if camera never becomes available
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\r\n')
            time.sleep(1)  # Reduce CPU usage while waiting for camera

    # Main loop for continuous frame processing
    while True:
        # Re-check camera status within the loop (in case it disconnects)
        if not cap.isOpened():
            print("Camera disconnected during stream. Attempting to re-initialize.")
            if not initialize_camera():  # Try to re-initialize the camera
                print("Failed to re-initialize camera. Streaming error frame.")
                error_img_local = np.zeros((480, 640, 3), dtype=np.uint8)
                cv2.putText(error_img_local, "Camera Lost", (int(640 * 0.2), int(480 * 0.5)), cv2.FONT_HERSHEY_SIMPLEX,
                            1.5, (0, 0, 255), 2)
                ret, buffer = cv2.imencode('.jpg', error_img_local)
                frame_bytes_err = buffer.tobytes()
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes_err + b'\r\r\n')
                time.sleep(1)  # Wait before retrying to avoid rapid failure loop
                continue  # Skip to next loop iteration

        success, frame = cap.read()  # Read a new frame from the camera
        if not success:
            print("Failed to read frame from camera. Retrying...")
            time.sleep(0.1)  # Small delay to avoid busy-waiting
            continue

        processed_frame = frame.copy()  # Work on a copy of the frame

        # Apply global color filters first (order matters)
        if invert_on:
            processed_frame = apply_invert_filter(processed_frame)
        if sepia_on:
            processed_frame = apply_sepia_filter(processed_frame)
        if bw_on:
            processed_frame = apply_bw_filter(processed_frame)

        # Face detection and overlay (applies to the current filtered frame)
        gray = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2GRAY)
        current_faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(50, 50))

        if len(current_faces) > 0:
            # Get the largest face detected
            largest_face = max(current_faces, key=lambda rect: rect[2] * rect[3])

            # Smooth face tracking: use detected face if first, otherwise average with previous
            if last_detected_face is None:
                last_detected_face = largest_face
            else:
                alpha = 0.7  # Smoothing factor (0.0 to 1.0)
                last_detected_face = (
                    int(last_detected_face[0] * (1 - alpha) + largest_face[0] * alpha),
                    int(last_detected_face[1] * (1 - alpha) + largest_face[1] * alpha),
                    int(last_detected_face[2] * (1 - alpha) + largest_face[2] * alpha),
                    int(last_detected_face[3] * (1 - alpha) + largest_face[3] * alpha)
                )
            no_face_frame_count = 0  # Reset counter if face detected
        else:
            no_face_frame_count += 1
            # If no face detected for several frames, clear the last detected face
            if no_face_frame_count >= NO_FACE_THRESHOLD:
                last_detected_face = None

        # Apply face overlays if a face is currently being tracked
        if last_detected_face is not None:
            x, y, w, h = last_detected_face
            if glasses_on:
                processed_frame = apply_glasses(processed_frame, x, y, w, h)
            if moustache_on:
                processed_frame = apply_moustache(processed_frame, x, y, w, h)

        # Apply vibrance last for a pop effect on the final image
        processed_frame = increase_vibrance(processed_frame, strength=1.4)

        # Encode the processed frame to JPEG format for streaming
        ret, buffer = cv2.imencode('.jpg', processed_frame)
        if not ret:
            print("Failed to encode frame to JPEG. Skipping frame.")
            continue

        frame_bytes = buffer.tobytes()  # Convert to bytes

        # Yield the frame in multipart/x-mixed-replace format
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\r\n')

        cv2.waitKey(1)  # Small delay to yield CPU, adjust if needed for higher FPS


# --- Flask Routes ---

@app.route('/')
def index():
    """Renders the main HTML page."""
    return render_template('index.html')


@app.route('/video_feed')
def video_feed():
    """
    Streams the video feed with applied filters.
    Receives filter states as URL parameters.
    """
    # Get filter states from URL query parameters (default to '0' if not present)
    glasses = request.args.get('glasses', '0') == '1'
    moustache = request.args.get('moustache', '0') == '1'
    bw = request.args.get('bw', '0') == '1'
    sepia = request.args.get('sepia', '0') == '1'
    invert = request.args.get('invert', '0') == '1'

    # Return a streaming response with the generated frames
    return Response(gen_frames(glasses, moustache, bw, sepia, invert),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


# --- Main Application Run Block ---
if __name__ == '__main__':
    import os
    port = int(os.environ.get("PORT", 5000))

    app.run(host='0.0.0.0', port=port, debug=False)


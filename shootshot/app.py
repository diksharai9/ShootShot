from flask import Flask, render_template, Response
import cv2
import numpy as np

app = Flask(__name__)

# Load Haar cascade for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Load filter image (PNG with transparency)
filter_img = cv2.imread('static/sunglasses.png', cv2.IMREAD_UNCHANGED)

def apply_filter(frame, x, y, w, h):
    # Resize the filter to match face size
    resized_filter = cv2.resize(filter_img, (w, h))

    # Split filter into channels
    b, g, r, a = cv2.split(resized_filter)
    mask = a / 255.0
    inv_mask = 1.0 - mask

    # Get region of interest
    roi = frame[y:y + h, x:x + w]

    # Blend filter with ROI
    for c in range(3):
        roi[:, :, c] = (mask * resized_filter[:, :, c] + inv_mask * roi[:, :, c])

    frame[y:y + h, x:x + w] = roi
    return frame

def gen_frames():
    cap = cv2.VideoCapture(1)

    while True:
        success, frame = cap.read()
        if not success:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        for (x, y, w, h) in faces:
            # Adjust position if sunglasses should be higher or wider
            frame = apply_filter(frame, x, y, w, h)

        ret, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(debug=True)

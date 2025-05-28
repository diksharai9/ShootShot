from flask import Flask, render_template, Response, request
import cv2
import numpy as np

app = Flask(__name__)

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
filter_img = cv2.imread('static/sunglasses.png', cv2.IMREAD_UNCHANGED)
moustache_img = cv2.imread('static/moustache.png', cv2.IMREAD_UNCHANGED)


def apply_glasses(frame, x, y, w, h):
    resized_filter = cv2.resize(filter_img, (w, h))
    b, g, r, a = cv2.split(resized_filter)
    mask = a / 255.0
    inv_mask = 1.0 - mask
    roi = frame[y:y + h, x:x + w]
    for c in range(3):
        roi[:, :, c] = (mask * resized_filter[:, :, c] + inv_mask * roi[:, :, c])
    frame[y:y + h, x:x + w] = roi
    return frame


def apply_moustache(frame, x, y, w, h):
    mw = int(w * 0.6)
    mh = int(h * 0.2)
    mx = x + int((w - mw) / 2)
    my = y + int(h * 0.65)

    resized_moustache = cv2.resize(moustache_img, (mw, mh))
    b, g, r, a = cv2.split(resized_moustache)
    mask = a / 255.0
    inv_mask = 1 - mask

    roi = frame[my:my + mh, mx:mx + mw]
    for c in range(3):
        roi[:, :, c] = (mask * resized_moustache[:, :, c] + inv_mask * roi[:, :, c])
    frame[my:my + mh, mx:mx + mw] = roi
    return frame


def gen_frames(glasses_on, moustache_on):
    cap = cv2.VideoCapture(1)

    while True:
        success, frame = cap.read()
        if not success:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        for (x, y, w, h) in faces:
            if glasses_on:
                frame = apply_glasses(frame, x, y, w, h)
            if moustache_on:
                frame = apply_moustache(frame, x, y, w, h)

        ret, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/video_feed')
def video_feed():
    glasses = request.args.get('glasses', '0') == '1'
    moustache = request.args.get('moustache', '0') == '1'

    return Response(gen_frames(glasses, moustache),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


if __name__ == '__main__':
    app.run(debug=True)

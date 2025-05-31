# ShootShot

![Project Screenshot Placeholder](d![img.png](img.png)ocs/screenshot_placeholder.png)
*(Replace this with a screenshot or GIF of your application in action! See "Adding Images" below.)*

## Overview

Aesthetic Autotune Cam Studio is a fun, real-time webcam application that transforms your live video and audio feed with a variety of visual filters and an autotune-like vocal effect. Built with Flask for backend video processing and a modern JavaScript frontend, it offers a creative way to capture unique moments.

The application combines server-side OpenCV processing for face-based filters and global color adjustments, with client-side Canvas effects for aesthetic touches. It also includes real-time audio processing to give your voice an autotuned feel, all while allowing you to record your enhanced performance.

## Features

* **Real-time Video Stream:** Captures your webcam feed.
* **Autotune-like Audio Effect:** Processes your microphone input with a unique vocal effect using the Web Audio API.
* **Face Tracking & Overlays (Server-side - OpenCV):**
    * Add **Sunglasses**
    * Add a **Moustache**
* **Global Visual Filters (Server-side - OpenCV):**
    * **Black & White**
    * **Sepia Tone**
    * **Invert Colors**
    * **Vibrance Boost**
* **Aesthetic Visual Filters (Client-side - HTML Canvas):**
    * **Vignette:** Adds a classic darkened edge effect.
    * **Color Tint:** Apply a customizable semi-transparent color overlay.
* **Video Recording:** Record your filtered video and processed audio directly from the browser (saved as a `.webm` file).

## Technologies Used

* **Backend:**
    * Python 3
    * Flask
    * OpenCV (`opencv-python-headless`) - For server-side image processing and face detection.
    * NumPy - For efficient array operations with OpenCV.
* **Frontend:**
    * HTML5
    * CSS (Tailwind CSS for styling)
    * JavaScript (Web Audio API for audio processing, Canvas API for drawing and client-side effects, MediaRecorder API for recording).

## Local Setup Instructions

Follow these steps to get the project running on your local machine.

### Prerequisites

* Python 3.8+
* `git` (for cloning the repository)
* A webcam connected to your computer.
* A microphone (built-in or external).

### 1. Clone the Repository

```bash
git clone [https://github.com/diksharai9/Shootshot.git](https://github.com/diksharai9/Shootshot.git)
cd Shootshot/shootshot # Navigate into the inner shootshot directory where app.py resides
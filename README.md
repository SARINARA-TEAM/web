# Smoking Detection System

## Description

This project detects smoking activities in real-time using computer vision. The ESP32-CAM streams live video feed to Raspberry Pi 5, where a trained model performs inference to detect smoking actions. When a smoking event is detected, an audio alert is triggered through a connected speaker. All detections are logged and can be monitored in real-time via a web interface.

### System Flow
<img width="431" height="673" alt="system flow" src="https://github.com/user-attachments/assets/c320a252-2d71-4309-a4c4-e6b6b4339e62" />

## Features

- Real-time smoking detection
- Live video stream from ESP32-CAM
- Audio alert via speaker
- Web-based dashboard for monitoring
- Detection logging into SQLite database

## Tech Stack

### Backend:
- Python
- Flask

### Frontend:
- HTML
- CSS
- JavaScript

### Hardware:
- ESP32-CAM – For video streaming
- Raspberry Pi 5 – Main processing unit
- Speaker – Audio alert output

### Tools & Libraries:
- CNN model
- Pandas
- Numpy
- Flask
- NLTK
- Scikit-learn
- opencv-python
- TensorFlow
- pyttsx3
- Matplotlib
- LIME
- SQLAlchemy

## How to Run

### 1. Hardware Setup
- Flash firmware onto ESP32-CAM (Check this : https://github.com/SARINARA-TEAM/esp32cam )
- Connect speaker to Raspberry Pi via GPIO or USB or Bluetooth etc
- Ensure all devices are on the same network

### 2. Setup Environment (on Raspberry Pi 5 in this case)
Clone this repo, then follow this step.
```bash
cd <your_path>
python3 -m venv venv
source venv/bin/activate

pip install -r requirements.txt
```

### 3. Run the App
```bash
python app.py
```

### 4. Access Web Interface
Open browser and go to:
```
http://<raspberry_pi_ip>:5000
```

## Future Improvements

- Improve or change Model
- Email alerts
- Implement user authentication
- Build analytics dashboard
- Add support for multiple cameras

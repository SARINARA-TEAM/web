from flask import Flask, render_template, jsonify, Response, request
from chatbot_engine import get_answer
import cv2
from detect import process_frame
from utils.tts import speak
import time
import threading
import numpy as np

app = Flask(__name__)


# Status global
stop_detect = False
latest_frame = None
latest_status = {"label": None}
last_tts_time = 0

def detection_loop():
    global latest_status, last_tts_time, latest_frame, stop_detect
    cap = cv2.VideoCapture(0)
    # cap = cv2.VideoCapture(r"http://192.168.1.7:81/stream", cv2.CAP_FFMPEG)  # URL kamera ESP32-CAM http://192.168.1.1:81/stream

    if not cap.isOpened():
        print("Kamera tidak bisa dibuka!")
        return
        
    print("Kamera berhasil dibuka")

    while not stop_detect:
        success, frame = cap.read()
        if not success:
            print("Gagal baca frame dari kamera")
            break
            
        try: 
            detected_frame, label = process_frame(frame)
            latest_status["label"] = label
            
            current_time = time.time()
            if label == "smoking" and (current_time - last_tts_time) > 10:
                threading.Thread(target=speak, args=("smoking detected, don't smoking in this area!",)).start()
                last_tts_time = current_time
            else:
                pass

            ret, buffer = cv2.imencode('.jpg', detected_frame)
            latest_frame = buffer.tobytes()

        except Exception as e:
            print(f"Error detection_loop {e}")
    cap.release()
    print("camera released and detection stopped.")

detection_thread = None

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/start')
def start_detection():
    global detection_thread, stop_detect
    if detection_thread is None or not detection_thread.is_alive():
        stop_detect = False
        detection_thread = threading.Thread(target=detection_loop)
        detection_thread.daemon = True
        detection_thread.start()
        return "Detection Started"
    else:
        return "detection already running"
    
@app.route('/stop')
def stop_detection():
    global stop_detect
    stop_detect = True
    return "request stop detection..."

@app.route('/video_feed')
def video_feed():
    def generate():
        placeholder = cv2.imencode('.jpg', np.zeros((480, 640, 3), dtype=np.uint8))[1].tobytes()
        while True:
            frame_to_send = latest_frame if latest_frame is not None else placeholder
            yield (
                b'--frame\r\n'
                b'Content-Type: image/jpeg\r\n\r\n' +
                frame_to_send + 
                b'\r\n'
            )
            time.sleep(0.1)
    return Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame' )

@app.route('/detection')
def detection():
    return render_template('detection.html')

@app.route('/kluster')
def kluster():
    return render_template('kluster.html')

@app.route('/team')
def team():
    return render_template('team.html')

@app.route('/get_response/<message>', methods=['GET'])
def get_response(message):
    answer = get_answer(message)
    return jsonify(response=answer)

@app.route('/speak', methods=['POST'])
def trigger_speak():
    data = request.get_json()
    text = data.get('text', '')
    if text.strip() == '':
        return jsonify({'error': 'no text provided'}), 400
    threading.Thread(target=speak, args=(text,)).start()
    return jsonify({'status': 'Speaking Started'})

@app.route('/detail_zat/<int:zat_id>')
def detail_zat(zat_id):
    details = {
        0: {
            "name": "Nikotin",
            "description": (
                "Nikotin adalah senyawa kimia yang sangat adiktif dan terdapat secara alami dalam tanaman tembakau. "
                "Zat ini bekerja langsung pada otak dan sistem saraf, memicu pelepasan dopamin yang memberikan sensasi senang sementara. "
                "Namun, nikotin juga dapat meningkatkan tekanan darah, denyut jantung, dan risiko gangguan jantung serta ketergantungan."
            )
        },
        1: {
            "name": "Tar",
            "description": (
                "Tar adalah zat residu lengket berwarna coklat gelap hingga hitam yang dihasilkan dari pembakaran rokok. "
                "Zat ini mengandung ribuan bahan kimia berbahaya, termasuk karsinogen (pemicu kanker) seperti benzena dan arsenik. "
                "Tar dapat menempel di paru-paru, merusak jaringan paru, menyebabkan iritasi saluran pernapasan, dan meningkatkan risiko kanker paru-paru serta penyakit pernapasan kronis."
            )
        },
        2: {
            "name": "Carbon Monoxide (CO)",
            "description": (
                "Carbon Monoxide (CO) adalah gas beracun tanpa warna dan bau yang dihasilkan dari pembakaran tembakau. "
                "Gas ini mengikat hemoglobin dalam darah lebih kuat daripada oksigen, sehingga mengurangi kemampuan darah mengangkut oksigen ke seluruh tubuh. "
                "Akibatnya, organ vital seperti jantung dan otak bisa kekurangan oksigen, meningkatkan risiko penyakit jantung, stroke, dan gangguan pernapasan."
            )
        },
        3: {
            "name": "Formaldehyde",
            "description": (
                "Formaldehyde adalah senyawa kimia beracun yang biasa digunakan sebagai bahan pengawet dalam industri medis dan kosmetik. "
                "Dalam asap rokok, formaldehyde terbentuk dari pembakaran zat organik. Zat ini bersifat iritan dan karsinogenik, dapat menyebabkan iritasi pada mata, hidung, tenggorokan, dan saluran pernapasan. "
                "Paparan jangka panjang formaldehyde dapat meningkatkan risiko kanker hidung, tenggorokan, dan saluran pernapasan bagian atas."
            )
        },
    }
    zat = details.get(zat_id, {"name": "Unknown", "description": "No details available."})
    return render_template('detail_zat.html', zat=zat)

if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True, threaded=True)

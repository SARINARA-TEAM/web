from flask import Flask, render_template, jsonify, Response, request
from chatbot_engine import get_answer
import cv2
from detect import process_frame
from utils.tts import speak
import time
import threading

app = Flask(__name__)


# Status global
latest_status = {"label": None}
last_tts_time = 0

def generate_frames():
    global latest_status, last_tts_time
    cap = cv2.VideoCapture(0)
    # cap = cv2.VideoCapture(r"http://192.168.1.7:81/stream", cv2.CAP_FFMPEG)  # URL kamera ESP32-CAM http://192.168.1.1:81/stream

    if not cap.isOpened():
        print("Kamera tidak bisa dibuka!")
        return
    
    while True:
        success, frame = cap.read()
        if not success:
            print("Gagal baca frame dari kamera")
            break
        
        detected_frame, label = process_frame(frame)
        latest_status["label"] = label
        
        current_time = time.time()
        if label == "smoking" and (current_time - last_tts_time) > 10:
            threading.Thread(target=speak, args=("smoking detected, don't smoking in this area!",)).start()
            last_tts_time = current_time
        else:
            pass

        ret, buffer = cv2.imencode('.jpg', detected_frame)
        frame_bytes = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

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

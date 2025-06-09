from flask import Flask, render_template, jsonify, Response, request, redirect
from chatbot_engine import get_answer
import cv2
from detect import process_frame
from utils.tts import speak
import time
import threading
import numpy as np
from flask_sqlalchemy import SQLAlchemy
from datetime import datetime
import os

app = Flask(__name__)

app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///detections.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db = SQLAlchemy(app)

UPLOAD_FOLDER = 'static/detections'
os.makedirs(UPLOAD_FOLDER,exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

class Detection(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    image_path = db.Column(db.String(255), nullable=False)
    date = db.Column(db.Date, nullable=False)
    time = db.Column(db.Time, nullable=False)
    label = db.Column(db.String(50), nullable=False)

with app.app_context():
    db.create_all()

# Status global
stop_detect = False
latest_frame = None
latest_status = {"label": None}
last_tts_time = 0
last_save_time = 0

def detection_loop():
    global latest_status, last_tts_time, latest_frame, stop_detect, last_save_time

    with app.app_context():

        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        # cap = cv2.VideoCapture(r"http://192.168.1.7:81/stream", cv2.CAP_FFMPEG)  # URL kamera ESP32-CAM http://192.168.1.1:81/stream

        if not cap.isOpened():
            print("Kamera tidak bisa dibuka!")
            return
            
        print("Kamera berhasil dibuka")
        frame_counter = 0

        while not stop_detect:
            success, frame = cap.read()
            if not success:
                print("Gagal baca frame dari kamera")
                break
                
            try: 
                frameSkipFactor = 3 # use every 3 frame
                frame_counter += 1
                if frame_counter % frameSkipFactor != 0:
                    continue
                detected_frame, label = process_frame(frame)
                latest_status["label"] = label
                
                if label == "smoking":
                    current_time = time.time()
                    
                    #trigger sound
                    if current_time - last_tts_time > 10:
                        threading.Thread(target=speak, args=("smoking detected, don't smoking in this area!",)).start()
                        last_tts_time = current_time

                    #save detections
                    if current_time - last_save_time > 30:
                        threading.Thread(target=save_detection, args=(detected_frame, db, app.config['UPLOAD_FOLDER'])).start()
                        last_save_time = current_time
                    
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
            time.sleep(0.05)
    return Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame' )

@app.route('/detection')
def detection():
    return render_template('detection.html')

def save_detection(frame, db_instance, upload_folder):
    try:
        timestamp = datetime.now()
        filename = f"{timestamp.strftime('%Y%m%d_%H%M%S')}.jpg"
        filepath = os.path.join(upload_folder, filename)
        cv2.imwrite(filepath, frame)

        with app.app_context():
            new_detection = Detection(
                image_path=filename,
                date=timestamp.date(),
                time=timestamp.time(),
                label="smoking"
            )
            db_instance.session.add(new_detection)
            db_instance.session.commit()
    except Exception as e:
        print(f"Error saving detection: {e}")

@app.route('/history')
def detection_history():
    page = request.args.get('page', 1, type=int)
    per_page = 5
    paginated = Detection.query.filter_by(label="smoking").order_by(Detection.id.desc()).paginate(page=page, per_page=per_page)
    return render_template('history.html', detections=paginated)

@app.route('/history/delete/<int:detection_id>', methods=['POST'])
def delete_detection(detection_id):
    detection = Detection.query.get_or_404(detection_id)
    
    # Hapus file gambar dari direktori
    image_path = os.path.join(app.config['UPLOAD_FOLDER'], detection.image_path)
    if os.path.exists(image_path):
        os.remove(image_path)
    
    # Hapus dari database
    db.session.delete(detection)
    db.session.commit()

    return redirect('/history')

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

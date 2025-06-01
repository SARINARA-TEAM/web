import pyttsx3
import threading
import time

last_speak_time = 0
lock = threading.Lock()

def speak(text):
    global last_speak_time
    current_time = time.time()
    
    if current_time - last_speak_time < 10:
        return

    def run():
        engine = pyttsx3.init()
        engine.setProperty('rate', 130)
        engine.setProperty('volume', 0.9)
        with lock:
            try:
                engine.say(text)
                engine.runAndWait()
            finally:
                engine.stop()
        
    threading.Thread(target=run).start()
    last_speak_time = current_time
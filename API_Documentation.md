# Sistem Smoking Detection – API Documentation

This system is designed to detect smoking activity in real-time using a camera, enhanced with educational features via a chatbot, text-to-speech (TTS), and detection history logging

---

## General Information

- **Host**: `http://localhost:5000`

---

## API Endpoints

### 1. Start Detection

Starts the smoking detection thread using camera input.

- **URL**: `/start`
- **Method**: `GET`
- **Response**:
  ```json
  { "message": "Detection Started" }
  ```

### 2. Stop Detection

Stops the currently running detection process.

- **URL**: `/stop`
- **Method**: `GET`
- **Response**:
  ```json
  { "message": "request stop detection..." }
  ```

### 3. Video Feed (Real-time Stream)

Provides an MJPEG video stream displaying real-time detection results.

- **URL**: `/video_feed`
- **Method**: `GET`
- **Response**: Stream video multipart/x-mixed-replace.

### 4. Chatbot

Returns an educational response from the chatbot based on the user’s message (limited to smoking and health-related topics).

- **URL**: `/get_response/<message>`
- **Method**: `GET`
- **Parameter**: `<message>` (user input text)
- **Response**:
  ```json
  {
    "response": "Merokok dapat menyebabkan kanker paru-paru, penyakit jantung, stroke, dan gangguan pernapasan kronis."
  }
  ```

### 5. Text-to-Speech (TTS)

Plays audio output based on the provided text using a local TTS engine.

- **URL**: `/speak`
- **Method**: `POST`
- **Request Body**: { "text": "Merokok sangat berbahaya bagi kesehatan." }
- **Response**:
  - success response
  ```json
  { "status": "speaking Started" }
  ```
  - failed response
  ```json
  { "error": "no text provided", "code": 400 }
  ```

### 6. Detection History (Paginated)

Displays a paginated list of all recorded smoking detection events.

- **URL**: `/history`
- **Method**: `GET`
- **Params**: `page` as `int` // optional
- **Response**: HTML page rendering the detection history list

### 7. Delete Detection History Entry

Deletes a specific detection record by ID, including associated files on the filesystem.

- **URL**: `/history/delete/<int:detection_id>`
- **Method**: `POST`
- **Parameter**: `detection_id` as `int`
- **Response**: Automatically redirects to /history after successful deletion

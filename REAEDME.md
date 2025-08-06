Project Overview
Purpose
The software is a real-time assistive perception system designed to run on a server and interact with one or more client devices over WebSockets. It ingests a live video stream, performs multiple vision-based tasks in parallel, and returns actionable results to the client:

Face registration and recognition

Text extraction (OCR) with medicine-name matching

Obstacle detection that combines object detection (YOLO) with monocular depth estimation (MiDaS)

Location reporting via public IP lookup

Spoken warnings through text-to-speech (optional)

The goal is to help visually-impaired users or autonomous-navigation prototypes by identifying nearby hazards, known people, and medicine packages in real time. The JAAD (Joint Attention for Autonomous Driving) dataset supplies realistic pedestrian-traffic video that can be streamed into the pipeline for testing obstacle detection and depth estimation.

Key Components & Data Flow
1. Video Input
Clients send JPEG frames (Base-64 or raw bytes) to /video.
The server:

Decodes and rotates frames for upright orientation.

Stores the latest frame in pending_frame so that all modules can access it without duplication.

(Optional) Shows the live feed for debugging.

2. Command Channel
Clients connect to /command and issue JSON commands such as:

start_obstacle_detection, stop_obstacle_detection

start_face_reg, register_face, start_face_det

start_ocr

send my current location

Each command triggers a handler that grabs the latest video frame and invokes the corresponding module.

3. Face Module
Database: face_db.json stores 128-D face embeddings created by face_recognition.

Registration: captures a frame, verifies exactly one face, stores its embedding by user-supplied name.

Recognition: compares current embedding with every stored one; reports recognized/not-recognized based on 0.6 cosine-distance threshold.

4. OCR & Medicine Matching
Uses EasyOCR to pull text strings from the frame.

Embeds both the extracted text and ∼15 k drug names from an Excel sheet into a 768-D SBERT vector space (pre-computed with FAISS for O(1) lookup).

Returns the closest brand/composition name to the client.

5. Obstacle Detection
YOLOv8 model detects objects relevant to safe navigation (person, vehicles, benches, fire hydrants, etc.).

MiDaS depth network converts a single RGB frame into a dense depth map, rescaled to the frame size.

For each detected object, the code samples the depth at its center and labels proximity as FAR / NEARBY / VERY CLOSE.

Results are streamed back every processed frame (~25 fps / 20 = 1.25 fps) for 60 s or until cancelled.

The depth map is overlayed in the corner for visual debugging; an MP4 writer optionally records the session.

6. Location & TTS
Queries ipinfo.io to approximate geographic location when requested.

PyTTSx3 can announce warnings (“Obstacle very close!”) locally on the server.

Use of JAAD Dataset
JAAD provides dash-cam style pedestrian-rich sequences that mimic a blind user’s street-level view. In this project the dataset can be:

Offline benchmark – feed prerecorded JAAD frames through the pipeline to measure detection accuracy, depth-error, and false-alert rate.

Live simulation – stream a JAAD video over WebSockets from a client script, letting the server treat it as a real user feed.

Because JAAD includes bounding-box and behavioral annotations, it is ideal for verifying that the YOLO-MiDaS fusion correctly flags pedestrians at dangerous distances, refining the proximity thresholds, and tuning model confidence.
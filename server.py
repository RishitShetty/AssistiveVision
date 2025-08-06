import asyncio
import websockets
import cv2
import numpy as np
import json
import base64
import face_recognition
import os
import requests
import time
import easyocr
import pandas as pd
import faiss
from sentence_transformers import SentenceTransformer
import torch
from ultralytics import YOLO
import pyttsx3


# Global variables for face processing, OCR, and obstacle detection
FACE_DB_FILE = "face_db.json"
pending_face_frame = None
obstacle_detection_task = None  # Global task handle for obstacle detection


def load_face_db():
    if os.path.exists(FACE_DB_FILE):
        with open(FACE_DB_FILE, "r") as f:
            data = json.load(f)
            return {name: np.array(embedding) for name, embedding in data.items()}
    return {}


def save_face_db(face_db):
    with open(FACE_DB_FILE, "w") as f:
        json.dump({name: embedding.tolist() for name, embedding in face_db.items()}, f)


face_db = load_face_db()

# OCR and medicine database initialization (if needed)
DB_FILE = "OCR_pharmaData/data1.xlsx"
CACHE_FILE = "medicine_embeddings.npy"


def load_database(file_path):
    try:
        df = pd.read_excel(file_path, usecols=['name', 'short_composition1'], engine='openpyxl')
        return df
    except Exception as e:
        print(f"Error loading database: {e}")
        return None


def precompute_embeddings(df, cache_file):
    model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')
    if os.path.exists(cache_file):
        print("Loading cached embeddings...")
        embeddings = np.load(cache_file)
    else:
        print("Computing embeddings...")
        medicine_names = df['name'].tolist()
        embeddings = np.array(model.encode(medicine_names, convert_to_tensor=False))
        np.save(cache_file, embeddings)
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)
    return model, index


database = load_database(DB_FILE)
model_sbert, index = precompute_embeddings(database, CACHE_FILE) if database is not None else (None, None)


def extract_text_from_frame(frame):
    reader = easyocr.Reader(['en'])
    result = reader.readtext(frame)
    return ' '.join([text[1] for text in result]).strip()


def find_closest_match(extracted_text, model, index, df):
    if not extracted_text:
        return "No text detected"
    query_embedding = np.array(model.encode([extracted_text], convert_to_tensor=False))
    _, nearest_idx = index.search(query_embedding, 1)
    closest_match = df.iloc[nearest_idx[0][0]]['name']
    return closest_match


# Initialize text-to-speech engine
tts_engine = pyttsx3.init()
tts_engine.setProperty('rate', 150)

# Load YOLO model for obstacle detection
yolo_model = YOLO("yolo11x.pt")

# Load MiDaS depth estimation model
print("Loading MiDaS model...")
model_type = "DPT_Hybrid"  # Options: "DPT_Large", "DPT_Hybrid", "MiDaS_small"
midas = torch.hub.load("intel-isl/MiDaS", model_type)
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
print(f"Using device: {device}")
midas.to(device)
midas.eval()
midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
if model_type in ["DPT_Large", "DPT_Hybrid"]:
    transform_midas = midas_transforms.dpt_transform
else:
    transform_midas = midas_transforms.small_transform

# Define obstacle classes of interest and class names dictionary
obstacle_classes = {'person', 'bicycle', 'car', 'motorcycle', 'truck', 'bus', 'stop sign', 'bench', 'fire hydrant'}
class_names = yolo_model.model.names if hasattr(yolo_model.model, 'names') else {}

# Video output object (optional)
out = cv2.VideoWriter("output_with_depth.mp4", cv2.VideoWriter_fourcc(*"mp4v"), 25, (640, 480))


#########################
# Video Streaming Handler
#########################
async def video_handler(websocket, path):
    global pending_face_frame
    print("Video client connected. Receiving video stream...")
    while True:
        try:
            frame_bytes = await websocket.recv()
        except websockets.ConnectionClosed:
            print("Video client disconnected.")
            break

        # Ensure frame_bytes are bytes (if string, decode using base64)
        if isinstance(frame_bytes, str):
            try:
                frame_bytes = base64.b64decode(frame_bytes)
            except Exception as e:
                print("Error decoding frame:", e)
                continue

        nparr = np.frombuffer(frame_bytes, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if frame is None:
            continue

        frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)

        pending_face_frame = frame.copy()

        # (Optional) Display the live video feed on the server
        cv2.imshow("Live Video Feed", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
        await asyncio.sleep(0.001)
    cv2.destroyAllWindows()


#########################
# Obstacle Detection Loop
#########################
async def obstacle_detection_loop(websocket):
    frame_counter = 0
    start_time = time.time()  # Record the start time
    detection_duration = 60  # Run for 60 seconds (1 minute)

    while True:
        try:
            # Check if one minute has elapsed
            current_time = time.time()
            if current_time - start_time >= detection_duration:
                print("[INFO] Obstacle detection automatically stopped after one minute")
                await websocket.send(json.dumps({
                    "status": "obstacle_detection_stopped",
                    "message": "Obstacle detection automatically stopped after one minute"
                }))
                break  # Exit the loop after one minute
            if pending_face_frame is None:
                print("[DEBUG] No new frame available, skipping iteration...")
                await asyncio.sleep(0.5)
                continue  # Ensures the loop doesn't stop if no frame is available

            frame_counter += 1

            # Process only every 20th frame
            if frame_counter % 20 != 0:
                await asyncio.sleep(0.05)  # Small delay to avoid CPU overuse
                continue

            print(f"[DEBUG] Processing Frame {frame_counter}")

            height, width = pending_face_frame.shape[:2]

            # Convert image for depth processing
            img_rgb = cv2.cvtColor(pending_face_frame, cv2.COLOR_BGR2RGB)
            input_batch = transform_midas(img_rgb).to(device)

            with torch.no_grad():
                prediction = midas(input_batch)
                prediction = torch.nn.functional.interpolate(
                    prediction.unsqueeze(1), size=(height, width),
                    mode="bicubic", align_corners=False
                ).squeeze()
                depth_map = prediction.cpu().numpy()

            depth_colored = cv2.normalize(depth_map, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
            depth_colored = cv2.applyColorMap(depth_colored, cv2.COLORMAP_PLASMA)

            # YOLO Object Detection
            print("[DEBUG] Running YOLO object detection...")
            results = yolo_model.predict(pending_face_frame, conf=0.25)

            detected_objects = []
            close_obstacle_detected = False
            nearby_obstacle_detected = False

            for result in results:
                for box in result.boxes:
                    cls_id = int(box.cls.cpu().numpy()[0])
                    cls_name = class_names.get(cls_id, "unknown")

                    if cls_name in obstacle_classes:
                        x1, y1, x2, y2 = map(int, box.xyxy.cpu().numpy()[0])  # Convert to standard int
                        center_x, center_y = (x1 + x2) // 2, (y1 + y2) // 2

                        if 0 <= center_y < height and 0 <= center_x < width:
                            object_depth = float(depth_map[center_y, center_x])  # Convert to float

                            print(f"[INFO] Detected {cls_name} at depth {object_depth}")

                            if object_depth > 1500:
                                proximity = "VERY CLOSE"
                                close_obstacle_detected = True
                            elif object_depth > 2000:
                                proximity = "NEARBY"
                                nearby_obstacle_detected = True
                            else:
                                proximity = "FAR"

                            detected_objects.append({
                                "class": cls_name,
                                "proximity": proximity,
                                "bounding_box": [x1, y1, x2, y2]
                            })

                            # Draw bounding box
                            cv2.rectangle(pending_face_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                            cv2.putText(pending_face_frame, f"{cls_name}: {proximity}",
                                        (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            # Overlay Depth Map on Frame
            depth_vis_size = (width // 4, height // 4)
            depth_vis = cv2.resize(depth_colored, depth_vis_size)
            pending_face_frame[height - depth_vis_size[1]:height, width - depth_vis_size[0]:width] = depth_vis

            # Send detection results in every frame
            response_msg = {
                "status": "obstacle_detection",
                "objects": [{"class": obj["class"], "proximity": obj["proximity"]} for obj in detected_objects]
                 }

            if detected_objects:  # Only send data if objects are detected
                response_msg = {
                    "status": "obstacle_detection",
                    "objects": [{"class": obj["class"], "proximity": obj["proximity"]} for obj in detected_objects]
                }

            try:
                await websocket.send(json.dumps(response_msg))
                print("[DEBUG] Sent detection results to client.")
            except Exception as e:
                print(f"[ERROR] WebSocket send failed: {e}")
                break  # Exit loop if WebSocket is disconnected

            # Play Alert Only When an Obstacle is Close
            if close_obstacle_detected:
                print("[ALERT] Warning: Obstacle very close!")
                # tts_engine.say("Warning: Obstacle very close!")
                # tts_engine.runAndWait()
            elif nearby_obstacle_detected:
                print("[ALERT] Caution: Obstacle nearby!")
                # tts_engine.say("Caution: Obstacle nearby!")
                # tts_engine.runAndWait()

        except Exception as e:
            print(f"[ERROR] Exception in obstacle detection loop: {e}")
            await asyncio.sleep(0.5)  # Prevent crash loop





#########################
# Command Handler
#########################
async def command_handler(websocket, path):
    global pending_face_frame, face_db, obstacle_detection_task
    print("Command client connected.")
    async for message in websocket:
        print("Received command:", message)
        try:
            data = json.loads(message)
        except json.JSONDecodeError as e:
            print("JSON decode error:", e)
            data = {"command": message}
        cmd = data.get("command", "").lower()

        if cmd == "send my current location":
            try:
                response = requests.get("https://ipinfo.io/json")
                loc_data = response.json()
                location_str = f"{loc_data.get('city', 'Unknown')}, {loc_data.get('region', 'Unknown')}, {loc_data.get('country', 'Unknown')}"
                print("Sending location:", location_str)
                await websocket.send(json.dumps({"status": "location_sent", "location": location_str}))
            except Exception as e:
                await websocket.send(json.dumps({"status": "error", "message": "Error fetching location."}))

        elif cmd == "start_ocr":
            if pending_face_frame is None:
                await websocket.send(json.dumps({"status": "no_frame_available"}))
                continue
            extracted_text = extract_text_from_frame(pending_face_frame)
            closest_match = find_closest_match(extracted_text, model_sbert, index,
                                               database) if extracted_text else "No text detected"
            response_msg = {"status": "ocr_complete", "extracted_text": extracted_text, "closest_match": closest_match}
            print("OCR Result:", response_msg)
            await websocket.send(json.dumps(response_msg))

        elif cmd == "start_face_reg":
            duration = data.get("duration", 10)
            print(f"Starting face registration mode for {duration} seconds.")
            start_time = asyncio.get_event_loop().time()
            detected_face_b64 = None
            while asyncio.get_event_loop().time() - start_time < duration:
                if pending_face_frame is not None:
                    rgb_frame = cv2.cvtColor(pending_face_frame, cv2.COLOR_BGR2RGB)
                    face_locs = face_recognition.face_locations(rgb_frame)
                    if face_locs:
                        ret, buffer = cv2.imencode('.jpg', pending_face_frame)
                        if ret:
                            detected_face_b64 = base64.b64encode(buffer.tobytes()).decode('utf-8')
                            break
                await asyncio.sleep(0.5)
            response_msg = {"status": "face_detected", "frame": detected_face_b64} if detected_face_b64 else {
                "status": "no_face_detected"}
            await websocket.send(json.dumps(response_msg))

        elif cmd == "register_face":
            name = data.get("name")
            frame_b64 = data.get("frame")
            if not name or not frame_b64:
                await websocket.send(
                    json.dumps({"status": "error", "message": "Missing name or frame data for registration"}))
                continue
            try:
                frame_data = base64.b64decode(frame_b64)
                nparr = np.frombuffer(frame_data, np.uint8)
                frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                if frame is None:
                    await websocket.send(
                        json.dumps({"status": "error", "message": "Failed to decode frame for registration"}))
                    continue
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                embeddings = face_recognition.face_encodings(rgb_frame)
                if not embeddings:
                    await websocket.send(
                        json.dumps({"status": "error", "message": "No face detected in the provided frame"}))
                    continue
                face_db[name] = embeddings[0]
                save_face_db(face_db)
                await websocket.send(
                    json.dumps({"status": "success", "message": f"Face registered successfully for {name}"}))
            except Exception as e:
                await websocket.send(
                    json.dumps({"status": "error", "message": f"Error during face registration: {str(e)}"}))

        elif cmd == "start_face_det":
            duration = data.get("duration", 10)
            print(f"Starting face detection mode for {duration} seconds.")
            start_time = asyncio.get_event_loop().time()
            detected_result = None
            while asyncio.get_event_loop().time() - start_time < duration:
                if pending_face_frame is not None:
                    rgb_frame = cv2.cvtColor(pending_face_frame, cv2.COLOR_BGR2RGB)
                    embeddings = face_recognition.face_encodings(rgb_frame)
                    if embeddings and face_db:
                        current_embedding = embeddings[0]
                        names = list(face_db.keys())
                        registered_embeddings = list(face_db.values())
                        distances = face_recognition.face_distance(registered_embeddings, current_embedding)
                        best_match_index = int(np.argmin(distances))
                        threshold = 0.6
                        detected_result = {"status": "recognized", "name": names[best_match_index]} if distances[
                                                                                                           best_match_index] < threshold else {
                            "status": "not_recognized"}
                        break
                await asyncio.sleep(0.5)
            await websocket.send(json.dumps(detected_result or {"status": "no_face_detected"}))

        elif cmd == "start_obstacle_detection":
            # If an obstacle detection task is already running, ignore the command
            global obstacle_detection_task
            if obstacle_detection_task is None or obstacle_detection_task.done():
                obstacle_detection_task = asyncio.create_task(obstacle_detection_loop(websocket))
                # Add a callback to clear the task reference when it completes
                obstacle_detection_task.add_done_callback(lambda t: globals().update({'obstacle_detection_task': None}))

                await websocket.send(json.dumps({"status": "obstacle_detection_started"}))

                await websocket.send(json.dumps({"status": "obstacle_detection_started"}))

            else:
                await websocket.send(json.dumps({"status": "obstacle_detection_already_running"}))

        elif cmd == "stop_obstacle_detection":
            if obstacle_detection_task is not None and not obstacle_detection_task.done():
                obstacle_detection_task.cancel()
                await websocket.send(json.dumps({"status": "obstacle_detection_stopped"}))
            else:
                await websocket.send(json.dumps({"status": "no_obstacle_detection_task"}))

        else:
            await websocket.send(json.dumps({"status": "error", "message": "Command not recognized."}))


async def handler(websocket, path):
    if path == "/video":
        await video_handler(websocket, path)
    elif path == "/command":
        await command_handler(websocket, path)
    else:
        await websocket.send("Unknown endpoint")
        await websocket.close()


async def main():
    server = await websockets.serve(handler, "0.0.0.0", 8765)
    print("Server is listening on port 8765...")
    await server.wait_closed()


if __name__ == '__main__':
    asyncio.run(main())

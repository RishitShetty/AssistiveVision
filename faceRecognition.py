# face_recognition_module.py
import cv2
import numpy as np
import json
import base64
import face_recognition
import os
from config import FACE_DB_FILE, FACE_RECOGNITION_THRESHOLD


class FaceRecognitionManager:
    def __init__(self):
        self.face_db = self.load_face_db()

    def load_face_db(self):
        """Load face database from file"""
        if os.path.exists(FACE_DB_FILE):
            with open(FACE_DB_FILE, "r") as f:
                data = json.load(f)
                return {name: np.array(embedding) for name, embedding in data.items()}
        return {}

    def save_face_db(self):
        """Save face database to file"""
        with open(FACE_DB_FILE, "w") as f:
            json.dump({name: embedding.tolist() for name, embedding in self.face_db.items()}, f)

    def register_face(self, name, frame):
        """Register a new face with given name"""
        try:
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            embeddings = face_recognition.face_encodings(rgb_frame)
            if not embeddings:
                return {"status": "error", "message": "No face detected in the provided frame"}

            self.face_db[name] = embeddings[0]
            self.save_face_db()
            return {"status": "success", "message": f"Face registered successfully for {name}"}
        except Exception as e:
            return {"status": "error", "message": f"Error during face registration: {str(e)}"}

    def detect_face(self, frame):
        """Detect and recognize faces in frame"""
        try:
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            embeddings = face_recognition.face_encodings(rgb_frame)

            if embeddings and self.face_db:
                current_embedding = embeddings[0]
                names = list(self.face_db.keys())
                registered_embeddings = list(self.face_db.values())
                distances = face_recognition.face_distance(registered_embeddings, current_embedding)
                best_match_index = int(np.argmin(distances))

                if distances[best_match_index] < FACE_RECOGNITION_THRESHOLD:
                    return {"status": "recognized", "name": names[best_match_index]}
                else:
                    return {"status": "not_recognized"}

            return {"status": "no_face_detected"}
        except Exception as e:
            return {"status": "error", "message": str(e)}

    def find_face_in_frame(self, frame):
        """Find face locations in frame"""
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        face_locs = face_recognition.face_locations(rgb_frame)
        return len(face_locs) > 0

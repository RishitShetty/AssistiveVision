# obstacle_detection.py
import cv2
import numpy as np
import torch
from ultralytics import YOLO
import time
import asyncio
import json
from config import OBSTACLE_CLASSES, DETECTION_DURATION


class ObstacleDetector:
    def __init__(self):
        # Load YOLO model
        self.yolo_model = YOLO("yolo11x.pt")
        self.class_names = self.yolo_model.model.names if hasattr(self.yolo_model.model, 'names') else {}

        # Load MiDaS depth estimation model
        print("Loading MiDaS model...")
        model_type = "DPT_Hybrid"
        self.midas = torch.hub.load("intel-isl/MiDaS", model_type)
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        print(f"Using device: {self.device}")
        self.midas.to(self.device)
        self.midas.eval()

        midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
        if model_type in ["DPT_Large", "DPT_Hybrid"]:
            self.transform_midas = midas_transforms.dpt_transform
        else:
            self.transform_midas = midas_transforms.small_transform

        # Video output (optional)
        self.out = cv2.VideoWriter("output_with_depth.mp4", cv2.VideoWriter_fourcc(*"mp4v"), 25, (640, 480))

    def get_depth_map(self, frame):
        """Generate depth map for frame"""
        height, width = frame.shape[:2]
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        input_batch = self.transform_midas(img_rgb).to(self.device)

        with torch.no_grad():
            prediction = self.midas(input_batch)
            prediction = torch.nn.functional.interpolate(
                prediction.unsqueeze(1), size=(height, width),
                mode="bicubic", align_corners=False
            ).squeeze()
            depth_map = prediction.cpu().numpy()

        return depth_map

    def detect_objects(self, frame, depth_map):
        """Detect objects and determine their proximity"""
        height, width = frame.shape[:2]
        results = self.yolo_model.predict(frame, conf=0.25)

        detected_objects = []
        close_obstacle_detected = False
        nearby_obstacle_detected = False

        for result in results:
            for box in result.boxes:
                cls_id = int(box.cls.cpu().numpy()[0])
                cls_name = self.class_names.get(cls_id, "unknown")

                if cls_name in OBSTACLE_CLASSES:
                    x1, y1, x2, y2 = map(int, box.xyxy.cpu().numpy()[0])
                    center_x, center_y = (x1 + x2) // 2, (y1 + y2) // 2

                    if 0 <= center_y < height and 0 <= center_x < width:
                        object_depth = float(depth_map[center_y, center_x])

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
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        cv2.putText(frame, f"{cls_name}: {proximity}",
                                    (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        return detected_objects, close_obstacle_detected, nearby_obstacle_detected

    def add_depth_overlay(self, frame, depth_map):
        """Add depth map overlay to frame"""
        height, width = frame.shape[:2]
        depth_colored = cv2.normalize(depth_map, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        depth_colored = cv2.applyColorMap(depth_colored, cv2.COLORMAP_PLASMA)

        depth_vis_size = (width // 4, height // 4)
        depth_vis = cv2.resize(depth_colored, depth_vis_size)
        frame[height - depth_vis_size[1]:height, width - depth_vis_size[0]:width] = depth_vis

        return frame

    async def detection_loop(self, get_frame_func, websocket):
        """Main detection loop"""
        frame_counter = 0
        start_time = time.time()

        while True:
            try:
                current_time = time.time()
                if current_time - start_time >= DETECTION_DURATION:
                    print("[INFO] Obstacle detection automatically stopped after one minute")
                    await websocket.send(json.dumps({
                        "status": "obstacle_detection_stopped",
                        "message": "Obstacle detection automatically stopped after one minute"
                    }))
                    break

                frame = get_frame_func()
                if frame is None:
                    await asyncio.sleep(0.5)
                    continue

                frame_counter += 1
                if frame_counter % 20 != 0:
                    await asyncio.sleep(0.05)
                    continue

                # Get depth map and detect objects
                depth_map = self.get_depth_map(frame)
                detected_objects, close_detected, nearby_detected = self.detect_objects(frame, depth_map)

                # Add depth overlay
                frame = self.add_depth_overlay(frame, depth_map)

                # Send results
                response_msg = {
                    "status": "obstacle_detection",
                    "objects": [{"class": obj["class"], "proximity": obj["proximity"]} for obj in detected_objects]
                }

                await websocket.send(json.dumps(response_msg))

                # Alert for close obstacles
                if close_detected:
                    print("[ALERT] Warning: Obstacle very close!")
                elif nearby_detected:
                    print("[ALERT] Caution: Obstacle nearby!")

            except Exception as e:
                print(f"[ERROR] Exception in obstacle detection loop: {e}")
                await asyncio.sleep(0.5)

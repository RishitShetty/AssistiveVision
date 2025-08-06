import asyncio
import websockets
import json
import base64
import cv2
import numpy as np
import pyttsx3
import socket


# List of possible server IPs
SERVER_URIS = [
    #"ws://172.30.55.158:8765/video",
    #"ws://172.20.10.4:8765/video",
    "ws://192.168.137.142:8765/command",
    #"ws://172.30.44.117:8765/video",
    #"ws://172.30.32.1:8765/video",
    "ws://192.168.137.1:8765/command"
]

def check_server(ip):
    """Check if a server is reachable on port 8765"""
    host = ip.replace("ws://", "").split(":")[0]  # Extract just the IP
    port = 8765  # WebSocket server port

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(1)  # 1-second timeout
        result = sock.connect_ex((host, port))
        if result == 0:
            print(f"✅ Connection successful: {host}:{port}")
        else:
            print(f"❌ Connection failed: {host}:{port}")
        return result == 0  # Returns True if server is reachable


async def face_registration(uri):
    async with websockets.connect(uri) as websocket:
        # Send command to start face registration mode for 10 seconds
        start_cmd = {"command": "start_face_reg", "duration": 10}
        await websocket.send(json.dumps(start_cmd))
        print("Sent start_face_reg command; waiting for response...")
        
        response = await websocket.recv()
        response_data = json.loads(response)

        if response_data.get("status") != "face_detected":
            engine = pyttsx3.init()
            engine.say(f"No face detected")
            engine.runAndWait()
            print("No face detected for registration.")
            return
        response_data = json.loads(response)
        detected_face_b64 = response_data.get("frame")
        # Optionally, display the received face image (decode and show using OpenCV)
        face_data = base64.b64decode(detected_face_b64)
        nparr = np.frombuffer(face_data, np.uint8)
        face_img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if face_img is not None:
            cv2.imshow("Detected Face for Registration", face_img)
            cv2.waitKey(500)
            cv2.destroyWindow("Detected Face for Registration")
        
        # Prompt user for a name
        name = input("Enter name for registration: ").strip()
        engine = pyttsx3.init()
        engine.say(f"face name is {name}")
        engine.runAndWait()
        if not name:
            print("No name entered; aborting registration.")
            return

        # Send registration command with the name and the detected face frame (re-use the same face image)
        reg_cmd = {"command": "register_face", "name": name, "frame": detected_face_b64}
        await websocket.send(json.dumps(reg_cmd))
        reg_response = await websocket.recv()
        print("Server response:", reg_response)
        
        
if __name__ == '__main__':
    # Automatically select the working server
    selected_uri = None
    for uri in SERVER_URIS:
        if check_server(uri):
            selected_uri = uri
            break

    if selected_uri:
        print(f"Using server: {selected_uri}")
        asyncio.run(face_registration(selected_uri))
    else:
        print("No available servers found. Please check your connections.")


import asyncio
import websockets
import cv2
import numpy as np
import os
import socket
from picamera2 import Picamera2

# List of possible server IPs
SERVER_URIS = [
    #"ws://172.30.55.158:8765/video",
    "ws://172.20.10.4:8765/video",
    "ws://192.168.137.142:8765/video",
    #"ws://172.30.44.117:8765/video",
    #"ws://172.30.32.1:8765/video",
    "ws://192.168.137.1:8765/video"
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

async def stream_video(uri):
    picam2 = None
    async with websockets.connect(uri) as websocket:
        print(f"Connected to video server at {uri}. Streaming video...")
        
        while True:
            # Capture and send frames
            if picam2 is None:
                try:
                    picam2 = Picamera2()
                    config = picam2.create_preview_configuration(main={"size": (640, 480)})
                    picam2.configure(config)
                    picam2.start()
                    print("Camera initialized for video streaming.")
                except Exception as e:
                    print("Error initializing Picamera2:", e)
                    await asyncio.sleep(1)
                    continue

            # Capture a frame
            frame = picam2.capture_array()
            if frame is None:
                print("Failed to capture frame.")
                await asyncio.sleep(0.033)
                continue

            ret, buffer = cv2.imencode('.jpg', frame)
            if not ret:
                print("Failed to encode frame.")
                await asyncio.sleep(0.033)
                continue

            try:
                await websocket.send(buffer.tobytes())
            except websockets.ConnectionClosed:
                print("Video connection closed by server.")
                break

            await asyncio.sleep(0.033)  # ~30 FPS

    if picam2 is not None:
        picam2.stop()
    print("Video streaming ended.")


if __name__ == '__main__':
    # Automatically select the working server
    selected_uri = None
    for uri in SERVER_URIS:
        if check_server(uri):
            selected_uri = uri
            break

    if selected_uri:
        print(f"Using video server: {selected_uri}")
        asyncio.run(stream_video(selected_uri))
    else:
        print("No available video servers found. Please check your connections.")
        
        
        
        


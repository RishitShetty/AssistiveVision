import RPi.GPIO as GPIO
import subprocess
import time
import os
import logging
from datetime import datetime

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("/home/pi/button_log.txt"),
        logging.StreamHandler()
    ]
)

# Define GPIO pins for buttons
BUTTON_FACE = 17      # Face detection & registration
BUTTON_OBJ_DET = 27   # Object detection
BUTTON_OCR = 22       # OCR
BUTTON_LOC_SRC = 4   # Location source

# Virtual environment path
VENV_PYTHON = "/home/pi/Assistive_Vision/myenv/bin/python3"

# Store process references
processes = {}

# Button state tracking
button_states = {
    BUTTON_FACE: {"last_press": 0, "pressed": False},
    BUTTON_OBJ_DET: {"last_press": 0, "pressed": False, "is_obstacle_running": False},
    BUTTON_OCR: {"last_press": 0, "pressed": False},
    BUTTON_LOC_SRC: {"last_press": 0, "pressed": False}
}

# Dictionary mapping buttons to their respective scripts
SCRIPT_MAPPING = {
    BUTTON_OCR: "/home/pi/Assistive_Vision/ocr.py",
    BUTTON_LOC_SRC: "/home/pi/Assistive_Vision/tts.py",
}

# Special script paths
OBSTACLE_SCRIPT = "/home/pi/Assistive_Vision/obstacle_det.py"
STOP_SCRIPT = "/home/pi/Assistive_Vision/stop.py"

# Constants
DEBOUNCE_TIME = 200  # milliseconds
DOUBLE_PRESS_THRESHOLD = 0.5  # seconds


def face_button_callback(channel):
    """Improved double press detection for face button."""
    global button_states
    current_time = time.time()
    
    # Calculate time since last press
    time_since_last_press = current_time - button_states[channel]["last_press"]
    
    # Update the last press time
    button_states[channel]["last_press"] = current_time
    
    # Check if this is a double press (between 0.1s and 0.5s since last press)
    # The lower bound helps avoid bounce issues being detected as double presses
    if 0.1 < time_since_last_press < 0.5:
        print("Double Press Detected! Running face_reg.py")
        toggle_script("/home/pi/Assistive_Vision/face_reg.py", BUTTON_FACE)
    else:
        # Start a timer to wait for potential second press
        # This is handled in the main loop
        button_states[channel]["waiting_for_double"] = True
        button_states[channel]["double_press_timeout"] = current_time + 0.5  # 500ms window


def toggle_script(script_name, button_pin):
    """Start or stop a script within the virtual environment."""
    global processes
    
    try:
        # Check if process exists and is still running
        if button_pin in processes:
            proc = processes[button_pin]
            if proc.poll() is None:  # Process is still running
                logging.info(f"Stopping {script_name}")
                proc.terminate()
                try:
                    proc.wait(timeout=3)  # Wait up to 3 seconds for graceful termination
                except subprocess.TimeoutExpired:
                    logging.warning(f"Process for {script_name} didn't terminate, killing it")
                    proc.kill()
                    proc.wait()
                del processes[button_pin]
                return
            else:
                # Process has ended but wasn't cleaned up
                del processes[button_pin]
        
        # Start the script
        logging.info(f"Starting {script_name} in venv")
        processes[button_pin] = subprocess.Popen(
            [VENV_PYTHON, script_name],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
    except Exception as e:
        logging.error(f"Error toggling script {script_name}: {e}")

def handle_obstacle_detection_button(channel):
    """Special handler for obstacle detection button that toggles between start and stop scripts."""
    global button_states
    
    # Toggle the state
    button_states[channel]["is_obstacle_running"] = not button_states[channel]["is_obstacle_running"]
    
    if button_states[channel]["is_obstacle_running"]:
        # Start obstacle detection
        logging.info(f"Starting obstacle detection")
        toggle_script(OBSTACLE_SCRIPT, channel)
    else:
        # Stop obstacle detection by running the stop script
        logging.info(f"Stopping obstacle detection via stop.py")
        # First terminate the running obstacle detection process if it exists
        if channel in processes and processes[channel].poll() is None:
            processes[channel].terminate()
            try:
                processes[channel].wait(timeout=3)
            except subprocess.TimeoutExpired:
                processes[channel].kill()
            del processes[channel]
        
        # Run the stop script
        subprocess.run([VENV_PYTHON, STOP_SCRIPT])
        logging.info("Stop script executed")

def check_button_press(channel):
    """Handles button press with improved debouncing."""
    current_time = time.time()
    
    # Ignore if it's been less than debounce time since last press
    if (current_time - button_states[channel]["last_press"]) < (DEBOUNCE_TIME / 1000):
        return
    
    button_states[channel]["last_press"] = current_time
    
    # Special handling for face button (double press detection)
    if channel == BUTTON_FACE:
        # Check for double press
        if (current_time - button_states[channel]["last_press"]) < DOUBLE_PRESS_THRESHOLD:
            logging.info("Double Press Detected! Running face_reg.py")
            toggle_script("/home/pi/Assistive_Vision/face_reg.py", BUTTON_FACE)
        else:
            logging.info("Single Press Detected! Running face_det.py")
            toggle_script("/home/pi/Assistive_Vision/face_det.py", BUTTON_FACE)
    # Special handling for obstacle detection button
    elif channel == BUTTON_OBJ_DET:
        handle_obstacle_detection_button(channel)
    # Handle other buttons
    elif channel in SCRIPT_MAPPING:
        script_path = SCRIPT_MAPPING[channel]
        logging.info(f"Button {channel} Press Detected! Running {script_path}")
        toggle_script(script_path, channel)

def button_callback(channel):
    """Main callback for all button events with improved debouncing."""
    # Read the current state of the button
    button_state = not GPIO.input(channel)  # Inverted because of pull-up
    
    if button_state:  # Button is pressed (LOW)
        if not button_states[channel]["pressed"]:
            button_states[channel]["pressed"] = True
            check_button_press(channel)
    else:  # Button is released (HIGH)
        button_states[channel]["pressed"] = False

# GPIO setup with improved configuration
def setup_gpio():
    GPIO.setmode(GPIO.BCM)
    GPIO.setwarnings(False)
    
    # Setup all buttons with pull-up resistors
    for pin in [BUTTON_FACE, BUTTON_OBJ_DET, BUTTON_OCR, BUTTON_LOC_SRC]:
        GPIO.setup(pin, GPIO.IN, pull_up_down=GPIO.PUD_UP)
        # Use both edges for better detection
        GPIO.add_event_detect(pin, GPIO.BOTH, callback=button_callback, bouncetime=DEBOUNCE_TIME)
    
    logging.info("GPIO setup complete")

def cleanup_processes():
    """Clean up any running processes before exiting."""
    for pin, process in processes.items():
        if process.poll() is None:
            logging.info(f"Terminating process on pin {pin}")
            process.terminate()
            try:
                process.wait(timeout=2)
            except subprocess.TimeoutExpired:
                process.kill()


if __name__ == "__main__":
    try:
        setup_gpio()
        logging.info("Button script started. Waiting for button presses...")
        
        # Main loop with periodic process checking
        while True:
            # Check for pending double press timeouts
            for pin in button_states:
                if button_states[pin].get("waiting_for_double", False):
                    if time.time() > button_states[pin].get("double_press_timeout", 0):
                        # Double press timeout expired, treat as single press
                        button_states[pin]["waiting_for_double"] = False
                        if pin == BUTTON_FACE:
                            logging.info("Single Press Confirmed! Running face_det.py")
                            toggle_script("/home/pi/Assistive_Vision/face_det.py", BUTTON_FACE)
            
            # Periodically check if processes are still running
            for pin in list(processes.keys()):
                if processes[pin].poll() is not None:
                    logging.info(f"Process on pin {pin} has ended")
                    # If obstacle detection process ended on its own, update the state
                    if pin == BUTTON_OBJ_DET:
                        button_states[pin]["is_obstacle_running"] = False
                    del processes[pin]
            
            time.sleep(0.1)
    
    except KeyboardInterrupt:
        logging.info("Exiting due to keyboard interrupt...")
    except Exception as e:
        logging.error(f"Unexpected error: {e}")
    finally:
        cleanup_processes()
        GPIO.cleanup()
        logging.info("GPIO cleaned up")


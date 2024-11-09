print("Booting up...")
# Import necessary packages
import sys
sys.path.append('/root/jetracer')
from jetracer.nvidia_racecar import NvidiaRacecar
from jetcam.csi_camera import CSICamera
import cv2
import torch
from stable_baselines3 import PPO
from IPython.display import display, clear_output, Image
import time
import subprocess
try:
    import gymnasium as gym
except ImportError:
    print("gymnasium is not installed. Installing now...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "gymnasium"])
    # Try the import again after installation
    try:
        import gymnasium as gym
        print("gymnasium installed successfully.")
    except ImportError:
        print("Failed to install gymnasium. Please install it manually.")
        exit()
print("Dependencies loaded!")
 
print("Initializing car...")
# Initialize the car
car = NvidiaRacecar()
print("Car initialized!")
 
print("Loading model...")
# Load the stable_baselines3 model
model_path = "model_60000.zip"
model = PPO.load(model_path)
print("Model loaded!")
 
# Define image preprocessing function if needed
def preprocess_image(image, target_size=(160, 120)):
    """Resize and normalize the input image for the model."""
    image = cv2.resize(image, target_size)  # Resize to 160x120
    image = image / 255.0  # Normalize pixel values
    # Convert to numpy array with channels first (CHW format)
    image = image.transpose(2, 0, 1)  # Move channels to first dimension
    return image  # Return numpy array instead of tensor
 
print("Booting up camera...")
# Initialize camera (replace '0' with the correct camera index or path)
camera = CSICamera(width=224, height=224, capture_fps=65)
 
print("Running...")
try:
    while True:
        # Capture a frame from the camera
        frame = camera.read()
        
        # Clear previous output and display the current frame
        clear_output(wait=True)
        display(Image(data=frame))
 
        # Preprocess the frame
        input_tensor = preprocess_image(frame)
 
        # Use the model to predict action
        with torch.no_grad():
            action, _states = model.predict(input_tensor)
 
        # Get steering and throttle from action output (customize based on model)
        # Here we assume action[0] is steering and action[1] is throttle
        steering = float(action[0])  # Scale to [-1, 1]
        throttle = float(action[1])  # Scale to [0, 1]
 
        # Apply controls to the car
        car.steering = steering
        car.throttle = -throttle  # Throttle controls are inverted
 
except KeyboardInterrupt:
    # Stop the car on interrupt
    car.throttle = 0
    car.steering = 0
    print("Control interrupted, stopping the car.")
 
finally:
    # Release camera resources
    camera.release()

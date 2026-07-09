import cv2
import numpy as np
import threading
import time
import torch
import os
from torchvision import transforms
from Model import UNet1280x720

RTSP_URL = "rtsp://admin:Trumpf@1030@192.168.1.64:554/cam/realmonitor?channel=1&subtype=0"


class RTSPStream:
    def __init__(self, url):
        self.cap = cv2.VideoCapture(url)
        self.ret = False
        self.frame = None
        self.stopped = False
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        if self.cap.isOpened():
            self.ret, self.frame = self.cap.read()
        else:
            print("Error: Could not open RTSP stream.")
            exit()

        self.thread = threading.Thread(target=self.update, args=())
        self.thread.daemon = True
        self.thread.start()

    def update(self):
        while not self.stopped:
            if self.cap.isOpened():
                self.ret, self.frame = self.cap.read()
            time.sleep(0.01)

    def read(self):
        return self.ret, self.frame

    def stop(self):
        self.stopped = True
        self.thread.join()
        self.cap.release()

xml_path = "Camera.xml"
if not os.path.exists(xml_path):
    print(f"Error: {xml_path} not found!")
    exit()

cv_file = cv2.FileStorage(xml_path, cv2.FILE_STORAGE_READ)
camera_matrix = cv_file.getNode("camera_matrix").mat()
dist_coeffs = cv_file.getNode("dist_coeffs").mat()
cv_file.release()
print("Loaded camera calibration data successfully.")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

model = UNet1280x720(in_channels=3, out_channels=1).to(device)
weights_path = "Unet1280x720.pth"

if os.path.exists(weights_path):
    model.load_state_dict(torch.load(weights_path, map_location=device))
    print(f"Successfully loaded model weights from {weights_path}")
else:
    print(f"Warning: Weights file '{weights_path}' not found.")
model.eval()

# Transform for inference
transform = transforms.ToTensor()

# ==========================================
# 3. Main Application Logic
# ==========================================
print(f"Connecting to RTSP Stream: {RTSP_URL}...")
stream = RTSPStream(RTSP_URL)
time.sleep(1.0)

cv2.namedWindow('Camera Feed')

print("\n=== Application Active ===")
print("Press 'p' to run inference and apply the green/red mask overlay.")
print("Press 'c' to clear the mask.")
print("Press 'q' to quit.")

current_mask = None

while True:
    ret, frame = stream.read()
    if not ret or frame is None:
        continue

    undistorted_frame = cv2.undistort(frame, camera_matrix, dist_coeffs)
    display_frame = undistorted_frame.copy()

    if current_mask is not None:
        color_overlay = np.zeros_like(display_frame)

        color_overlay[current_mask == 1] = [0, 255, 0]
        color_overlay[current_mask == 0] = [0, 0, 255]

        alpha = 0.4
        display_frame = cv2.addWeighted(display_frame, 1 - alpha, color_overlay, alpha, 0)

        cv2.putText(display_frame, "Inference Overlay Active (Press 'c' to clear)",
                    (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    cv2.imshow('Camera Feed', display_frame)

    key = cv2.waitKey(1) & 0xFF

    if key == ord('q'):
        print("\nExiting...")
        break

    elif key == ord('c'):
        print("Clearing mask...")
        current_mask = None

    elif key == ord('p'):
        print("Running inference...")

        rgb_frame = cv2.cvtColor(undistorted_frame, cv2.COLOR_BGR2RGB)

        if rgb_frame.shape[:2] != (720, 1280):
            rgb_frame = cv2.resize(rgb_frame, (1280, 720))

        img_tensor = transform(rgb_frame).unsqueeze(0).to(device)

        with torch.no_grad():
            output = model(img_tensor)
            probs = torch.sigmoid(output)
            mask = (probs > 0.5).float()

        current_mask = mask.squeeze().cpu().numpy()
        print("Inference complete. Mask applied.")

stream.stop()
cv2.destroyAllWindows()
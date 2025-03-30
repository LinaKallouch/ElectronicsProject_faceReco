# ElectronicsProject_faceReco
About This Project:

This project is a facial recognition system that uses an interactive game to identify players and track scores. The code includes some comments and variable names written in Dutch. It utilizes an Intel RealSense camera and the face_recognition library to detect and compare faces with pre-stored assistant images. Based on the similarity scores, a mechanism controlled via a serial connection moves a physical tower upwards.
Additionally, there is an alternative version of the main script that does not require serial input. Instead, it uses a hardcoded list of selected assistants to simulate serial input for testing purposes (under main_code_w_cam_serialtest.py).

FEATURES

Face Recognition: Detects and compares faces in real-time with a database of assistants.
RealSense Camera Integration: Uses an Intel RealSense camera for image capture.
Serial Communication: Sends commands via a serial port to control an external mechanism (e.g., a rising tower).
Game Mechanics: Players identify themselves, and the best matches increase the tower height.
Automatic Score Tracking: Tracks scores and adjusts the tower position based on performance.

REQUIRED LIBRARIES

This project requires the following Python libraries:
pip install: numpy, opencv-python, face-recognition, pyserial , pyrealsense2

HOW IT WORKS

1. Face Recognition:
  The camera captures an image and detects faces.
  A deep learning model (CNN) processes the image and extracts a 128-dimensional vector for each face.
  This vector is compared with stored assistant images.

2. Game Flow:
  Four players participate and look at the camera.
  The software compares their faces with known assistants and assigns a score.
  If the match is correct, the tower height increases.
  After four rounds, the final score is calculated, and the game ends.

3. Serial Communication:
  If a correct match is found, the program sends a signal to the serial port (COM4) to raise the tower.
  At the end of the game, the tower resets to the starting position.

USAGE

- Ensure that the reference images of assistants are stored in the All Pictures/ folder.
- Connect the RealSense camera and make sure COM4 is available.

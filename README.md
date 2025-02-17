# Detectify-Deepfake Video Detection System
This project focuses on detecting deepfake videos by analyzing them frame by frame. The process involves splitting the video into individual frames, training a machine learning model, and classifying each frame as either real or fake. Below is a breakdown of the methodology and tools used.

Methodology
1. Video to Frame Conversion
Using OpenCV, we extract frames from the video file. Each video is divided into a series of still images for frame-by-frame analysis.
2. Data Preparation and Model Training
The extracted frames are preprocessed using NumPy for numerical computations and Pillow for image manipulation.
A neural network model is designed and trained using Keras (high-level API) and TensorFlow (deep learning framework).
The model is trained on a dataset containing real and fake frames to differentiate between the two.
3. Frame Classification
After training, the model is used to analyze new frames.
Each frame is passed through the trained model to determine whether it is real or fake.
The results are compiled to provide a final classification of the video.
4. Visualization and User Interface
Matplotlib is used to visualize the results, such as accuracy graphs and confusion matrices during training and testing.
A user-friendly interface for uploading videos and viewing results is created using Tkinter.
Libraries and Tools Used
1. NumPy
For high-performance numerical computations, including array manipulations.
2. OpenCV
For extracting frames from videos and basic image processing tasks.
3. Keras
A high-level API for neural network design, used for training the deepfake detection model.
4. Pillow
For preprocessing and manipulating image data.
5. Matplotlib
For visualizing training results, such as accuracy, loss, and frame classification outcomes.
6. TensorFlow
The backend framework for building and training the deep learning model.
7. Tkinter
To develop the user interface for tasks like video upload, login forms, and result display.

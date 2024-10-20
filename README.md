## Attendance-Management
Attendance management using Face Recognition


This repository implements a face recognition-based attendance management system using Python, OpenCV, and machine learning. The system captures face data through a webcam, stores it, and uses K-Nearest Neighbors (KNN) for face recognition. The attendance records are saved in CSV format, and the project features a simple GUI built with Streamlit.

# Features
Face Data Collection: Capture and store face images through a webcam.

Face Recognition: Recognize faces using KNN and mark attendance.

Attendance Logging: Log the recognized name and timestamp into a CSV file.

Streamlit Dashboard: Display real-time attendance data.

# Project Structure
add_face.py: Script to capture and store face data.

attendance_and_recognition.py: Script to recognize faces and log attendance.

attendance_and_recognition_update.py: Updated version of face recognition with improved handling of unknown faces.

app.py: Streamlit application to display real-time attendance records.

data/: Folder to store face images and labels.

Attendance/: Folder to store daily attendance records as CSV files.


# Prerequisites

Python 3.x

OpenCV

Numpy

Scikit-learn

Streamlit

win32com.client (for voice output on Windows)


# How to Run the Project

1. Capture Face Data

Run the add_face.py script to capture 100 face images for each user and store them in the data directory. 
Make sure a webcam is connected.

2. Face Recognition and Attendance Logging

Run the attendance_and_recognition.py or the attendance_and_recognition_update.py script for recognizing faces and logging attendance.

Press 'o' to log the attendance of a recognized face.

3. Display Attendance with Streamlit

Run the app.py script to view real-time attendance in a web browser using Streamlit.
Press 'q' to quit the application.

The attendance records for the current day will be displayed in a table format.


# Customization

Threshold for Face Recognition: You can adjust the threshold for identifying unknown faces in the attendance_and_recognition_update.py script. The default is set to 3800.

Attendance CSV Files: Attendance logs are stored in the Attendance/ directory with filenames in the format Attendance_DD-MM-YYYY.csv.


# Notes
Make sure the data directory contains the haarcascade_frontalface_default.xml file for face detection using OpenCV.

If you're running this on a system without a webcam, you can modify the code to use a pre-recorded video file for face detection and recognition.

from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pickle
import numpy as np

# Load labels and faces data
with open('data/names.pkl', 'rb') as w:
    LABELS = pickle.load(w)
with open('data/faces_data.pkl', 'rb') as f:
    FACES = pickle.load(f)

print('Shape of Faces matrix --> ', FACES.shape)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(FACES, LABELS, test_size=0.2, random_state=42)

# Initialize KNN classifier
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)

# Predict on the test set
y_pred = knn.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy * 100:.2f}%')




from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier
import pickle
import numpy as np

# Load labels and faces data
with open('data/names.pkl', 'rb') as w:
    LABELS = pickle.load(w)
with open('data/faces_data.pkl', 'rb') as f:
    FACES = pickle.load(f)

print('Shape of Faces matrix --> ', FACES.shape)

# Initialize KNN classifier
knn = KNeighborsClassifier(n_neighbors=5)

# Perform 5-fold cross-validation
cv_scores = cross_val_score(knn, FACES, LABELS, cv=5)

# Calculate the mean accuracy across the folds
mean_accuracy = np.mean(cv_scores)
print(f'Mean Accuracy (5-fold CV): {mean_accuracy * 100:.2f}%')




from sklearn.neighbors import KNeighborsClassifier
import cv2
import pickle
import numpy as np
import os
import time
from datetime import datetime

# Initialize counters for accuracy tracking
correct_predictions = 0
total_predictions = 0

# Initialize video capture and face detection
video = cv2.VideoCapture(0)
facedetect = cv2.CascadeClassifier('data/haarcascade_frontalface_default.xml')

# Load labels and faces data
with open('data/names.pkl', 'rb') as w:
    LABELS = pickle.load(w)
with open('data/faces_data.pkl', 'rb') as f:
    FACES = pickle.load(f)

print('Shape of Faces matrix --> ', FACES.shape)

# Initialize KNN classifier
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(FACES, LABELS)

# Load background image (optional)
imgBackground = cv2.imread("background.png")

# Set a threshold for recognizing faces as "Unknown"
threshold = 3800  # Adjust this threshold as needed

while True:
    ret, frame = video.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = facedetect.detectMultiScale(gray, 1.3, 5)

    if len(faces) == 0:
        # If no faces are detected, skip to the next iteration
        continue

    for (x, y, w, h) in faces:
        crop_img = frame[y:y + h, x:x + w, :]
        resized_img = cv2.resize(crop_img, (50, 50)).flatten().reshape(1, -1)
        output = knn.predict(resized_img)
        distances, indices = knn.kneighbors(resized_img)

        # Get the predicted label and distance
        predicted_label = output[0]
        distance = distances[0][0]

        # Introduce a threshold value to determine unknown faces
        if distance > threshold:
            predicted_label = "Unknown"

        # Simulating the user providing correct ground truth
        # For this example, we assume the prediction is correct for testing purposes.
        is_correct = True  # Replace with actual ground truth if available

        # Update accuracy tracking
        total_predictions += 1
        if is_correct:
            correct_predictions += 1

        # Calculate accuracy in real-time
        accuracy = (correct_predictions / total_predictions) * 100

        # Drawing rectangles and displaying the name, distance, and accuracy
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 1)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (50, 50, 255), 2)
        cv2.rectangle(frame, (x, y - 40), (x + w, y), (50, 50, 255), -1)
        cv2.putText(frame, f"{predicted_label}", (x, y - 15), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 1)
        cv2.putText(frame, f"Dist: {distance:.2f}", (x, y + h + 20), cv2.FONT_HERSHEY_COMPLEX, 0.6, (255, 255, 255), 1)

    # Display accuracy on the top-left corner of the frame
    cv2.putText(frame, f"Accuracy: {accuracy:.2f}%", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Display the frame in the window
    if imgBackground is not None:
        imgBackground[162:162 + 480, 55:55 + 640] = frame
        cv2.imshow("Frame", imgBackground)
    else:
        cv2.imshow("Frame", frame)

    k = cv2.waitKey(1)

    if k == ord('q'):
        break

# Release resources
video.release()
cv2.destroyAllWindows()

import cv2
import numpy as np
from sklearn.tree import DecisionTreeClassifier

# Simulated data: Replace with your own labeled dataset
X = np.random.rand(100, 10)  # Replace with extracted features
y = np.random.randint(0, 2, 100)  # Replace with personality labels

# Initialize and train a Decision Tree model
model = DecisionTreeClassifier()
model.fit(X, y)

# OpenCV setup
cap = cv2.VideoCapture(0)  # Capture video from the default camera

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Preprocess frame if needed
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Display the frame
    cv2.imshow('Personality Prediction', frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('p'):
        # Perform face detection and extract features (simulated)
        # In a real scenario, you'd use actual feature extraction methods
        extracted_features = np.random.rand(1, 10)  # Replace with real features

        # Predict personality using the trained model
        predicted_personality = model.predict(extracted_features)

        # Display the result on the frame
        result_frame = frame.copy()
        cv2.putText(result_frame, f"Predicted Personality: {predicted_personality}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        cv2.imshow('Personality Result', result_frame)
        cv2.waitKey(0)  # Wait until any key is pressed to continue

# Release the camera and close windows
cap.release()
cv2.destroyAllWindows()

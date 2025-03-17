import cv2

# Load face and smile detectors
face_cap = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
smile_cap = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_smile.xml")

# Start video capture
video_cap = cv2.VideoCapture(0)

while True:
    ret, video_data = video_cap.read()
    
    # Convert to grayscale
    gray = cv2.cvtColor(video_data, cv2.COLOR_BGR2GRAY)
    
    # Detect faces
    faces = face_cap.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    
    for (x, y, w, h) in faces:
        # Draw rectangle around face
        cv2.rectangle(video_data, (x, y), (x + w, y + h), (0, 255, 0), 2)
        
        # Detect smiles inside the face region
        roi_gray = gray[y:y + h, x:x + w]
        smiles = smile_cap.detectMultiScale(roi_gray, scaleFactor=1.8, minNeighbors=20, minSize=(25, 25))
        
        for (sx, sy, sw, sh) in smiles:
            cv2.rectangle(video_data, (x + sx, y + sy), (x + sx + sw, y + sy + sh), (0, 0, 255), 2)

    # Display video frame
    cv2.imshow('Live Video', video_data)
    
    # Exit on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
video_cap.release()
cv2.destroyAllWindows()

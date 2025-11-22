import cv2
from deepface import DeepFace

# Configuration flag
display_dominant_emotion = True  # Set to True to show only dominant emotion, False to show all emotions

# Load face cascade classifier
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Start capturing video
cap = cv2.VideoCapture(0)

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()

    # Convert frame to grayscale
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Convert grayscale frame to RGB format
    rgb_frame = cv2.cvtColor(gray_frame, cv2.COLOR_GRAY2RGB)

    # Detect faces in the frame
    faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    for (x, y, w, h) in faces:
        # Extract the face ROI (Region of Interest)
        face_roi = rgb_frame[y:y + h, x:x + w]

        
        # Perform emotion analysis on the face ROI
        result = DeepFace.analyze(face_roi, actions=['emotion'], enforce_detection=False)

        # Get all emotions from the result
        emotions = result[0]['emotion']
        dominant_emotion = result[0]['dominant_emotion']
        
        # Draw rectangle around face (teal color, thick)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (192, 192, 0), 5)
        
        teal_color = (192, 192, 0)
        
        if display_dominant_emotion:
            # Display only the dominant emotion
            font_scale = 1.5
            font_thickness = 3
            text = dominant_emotion.capitalize()
            
            # Get text size to center it
            (text_width, text_height), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness)
            text_x = x + (w - text_width) // 2
            text_y = max(30, y - 10)
            
            cv2.putText(frame, text, (text_x, text_y), 
                       cv2.FONT_HERSHEY_SIMPLEX, font_scale, teal_color, font_thickness)
        else:
            # Display all emotions with their scores
            # Calculate box dimensions for emotion display
            box_x = x
            box_y = max(0, y - 200)  # Position box above face, but not off screen
            box_width = w + 20
            box_height = 180
            
            # Draw background box for emotions
            cv2.rectangle(frame, (box_x, box_y), (box_x + box_width, box_y + box_height), (0, 0, 0), -1)
            cv2.rectangle(frame, (box_x, box_y), (box_x + box_width, box_y + box_height), (192, 192, 0), 3)
            
            # Display all emotions with their scores
            y_offset = box_y + 25
            line_height = 25
            font_scale = 0.6
            font_thickness = 2
            
            # Sort emotions by score (highest first)
            sorted_emotions = sorted(emotions.items(), key=lambda x: x[1], reverse=True)
            
            for emotion_name, score in sorted_emotions:
                text = f"{emotion_name.capitalize()}: {score:.1f}%"
                cv2.putText(frame, text, (box_x + 10, y_offset), 
                           cv2.FONT_HERSHEY_SIMPLEX, font_scale, teal_color, font_thickness)
                y_offset += line_height

    # Display the resulting frame
    cv2.imshow('Real-time Emotion Detection', frame)

    # Press 'q' to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the capture and close all windows
cap.release()
cv2.destroyAllWindows()
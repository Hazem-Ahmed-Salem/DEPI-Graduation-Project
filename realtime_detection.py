# Real-time Hand Gesture Recognition
# This Code loads a pre-trained model and runs real-time hand gesture recognition using your webcam.


# Import Required Libraries
import mediapipe as mp
import cv2
import numpy as np
from keras.models import load_model
import pickle


# Load Pre-trained Model and Label Encoder

# Load the trained model
model = load_model('best_model.h5')

# Load the label encoder
with open('hand_gesture_model.pkl', 'rb') as f:
    _, le = pickle.load(f)  # We only need the label encoder, not the model


# Setup MediaPipe Configuration

# Initialize MediaPipe
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.7)


# Prediction Smoothing Function

# Use dictionary to store predictions for each hand
prediction_history = {}
QUEUE_LENGTH = 5

def get_smooth_prediction(pred_array, hand_id):

    '''
    To reduce jitter in predictions, we'll implement a smoothing function that averages predictions over time.
    '''
    # Initialize history for new hand
    if hand_id not in prediction_history:
        prediction_history[hand_id] = []
    
    # Add new prediction
    prediction_history[hand_id].append(pred_array)
    
    # Keep only recent predictions
    if len(prediction_history[hand_id]) > QUEUE_LENGTH:
        prediction_history[hand_id].pop(0)
    
    # Calculate average
    avg_pred = np.mean(prediction_history[hand_id], axis=0)
    return avg_pred


# Real-time Hand Gesture Recognition

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    frame = cv2.flip(frame, 1)
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(img_rgb)
    
    # Clear background for text
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (300, 200), (0, 0, 0), -1)
    cv2.rectangle(overlay, (frame.shape[1]-300, 0), (frame.shape[1], 200), (0, 0, 0), -1)
    frame = cv2.addWeighted(overlay, 0.3, frame, 0.7, 0)
    
    if results.multi_hand_landmarks:
        handedness = results.multi_handedness
        
        for idx, (hand_landmarks, hand_info) in enumerate(zip(results.multi_hand_landmarks, handedness)):
            # Draw landmarks
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            
            # Get hand side and ID
            hand_side = hand_info.classification[0].label
            hand_id = f"{hand_side}_{idx}"
            
            # Extract landmarks
            pts = []
            for lm in hand_landmarks.landmark:
                pts.extend([lm.x, lm.y, lm.z])
            pts = np.array(pts)
            
            # Normalize by wrist position
            wrist_x, wrist_y = pts[0], pts[1]
            pts[0::3] -= wrist_x
            pts[1::3] -= wrist_y
            
            # Get prediction
            pts = pts.reshape(1, -1)
            raw_pred = model.predict(pts, verbose=0)
            
            # Smooth predictions
            smooth_pred = get_smooth_prediction(raw_pred[0], hand_id)
            pred_idx = np.argmax(smooth_pred)
            confidence = smooth_pred[pred_idx] * 100
            
            # Get prediction label
            gesture = le.inverse_transform([pred_idx])[0]
            
            # Position text based on hand side
            if hand_side == "Right":  # Actually left side of screen due to flip
                x_pos = 10
            else:  # Left hand will show on right side
                x_pos = frame.shape[1] - 290
            
            # Only show high confidence predictions
            if confidence > 70:
                # Main prediction with larger font
                text = f"{hand_side}: {gesture}"
                cv2.putText(frame, text, (x_pos, 40), 
                          cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                cv2.putText(frame, f"{confidence:.1f}%", (x_pos, 70), 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                
                # Show top 3 predictions with smaller font
                sorted_idx = np.argsort(smooth_pred)[-3:][::-1]
                y_offset = 100
                for pred_i in sorted_idx:
                    conf = smooth_pred[pred_i] * 100
                    if conf > 10:  # Only show if confidence > 10%
                        pred_text = f"{le.inverse_transform([pred_i])[0]}: {conf:.1f}%"
                        cv2.putText(frame, pred_text, (x_pos, y_offset), 
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
                        y_offset += 25
    
    # Clean up old hand histories
    if results.multi_hand_landmarks is None:
        prediction_history.clear()
    
    cv2.imshow("Multi-Hand Gesture Recognition (ESC to quit)", frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
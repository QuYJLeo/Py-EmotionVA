#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Real-time emotion recognition from webcam using PyTorch model with MediaPipe face detection.
Processes video frames, detects faces, and predicts emotion, valence, and arousal.
"""
import numpy as np
import torch
import cv2
import mediapipe as mp
from torchvision import transforms

# Emotion class names mapping
class8_names = {0:'Neutral', 1:'Happy', 2:'Sad', 3:'Surprise', 4:'Fear', 5:'Disgust', 6:'Angry', 7:'Contempt'}


def calc_bounding_rect(image, landmarks):
    """
    Calculate bounding rectangle from face landmarks.

    Args:
        image: Input image (numpy array)
        landmarks: MediaPipe face landmarks

    Returns:
        Bounding rectangle coordinates [x1, y1, x2, y2]
    """
    image_width, image_height = image.shape[1], image.shape[0]

    landmark_array = np.empty((0, 2), int)

    # Extract landmark coordinates
    for _, landmark in enumerate(landmarks.landmark):
        landmark_x = min(int(landmark.x * image_width), image_width - 1)
        landmark_y = min(int(landmark.y * image_height), image_height - 1)

        landmark_point = [np.array((landmark_x, landmark_y))]
        landmark_array = np.append(landmark_array, landmark_point, axis=0)

    # Calculate bounding rectangle
    x, y, w, h = cv2.boundingRect(landmark_array)

    return [x, y, x + w, y + h]


def run_test():
    """
    Main test function for real-time emotion recognition from webcam.
    Loads model, initializes face detector, and processes video stream.
    """
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Input size for MixedFeatureNet: 112x112
    input_size = (112, 112)

    # Import and load DDAMNet model
    from models.DDAM import DDAMNet
    model = DDAMNet(num_class=8, num_head=2, pretrained=False)
    checkpoint = torch.load("./checkpoints/mp_MFN_epoch24.pth", weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()

    # Define image preprocessing transforms
    data_transforms = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(input_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225])])

    # Initialize MediaPipe face mesh detector
    face_mesh = mp.solutions.face_mesh.FaceMesh(max_num_faces=1, min_detection_confidence=0.8, min_tracking_confidence=0.5)

    # Open webcam
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    while cap.isOpened():
        ret, image = cap.read()
        if not ret:
            break

        # Convert color space for face detection
        results = face_mesh.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

        if results.multi_face_landmarks is not None:
            for face_landmarks in results.multi_face_landmarks:
                # Calculate and draw face bounding box
                x1, y1, x2, y2 = calc_bounding_rect(image, face_landmarks)
                cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)

                # Crop face region
                crop = image[y1:y2, x1:x2]
                input = data_transforms(crop)
                input = torch.unsqueeze(input, dim=0)

                # Model inference
                out, feat, heads, out2 = model(input.to(device))

                # Get emotion probabilities
                probably = torch.softmax(torch.squeeze(out, 0), 0).detach().cpu()
                probably = [round(_, 3) for _ in probably.numpy().tolist()]

                # Get valence and arousal predictions
                out2 = torch.squeeze(out2, 0).detach().cpu().numpy().tolist()
                out2 = [round(x, 3) for x in out2]
                valence, arousal = out2[0], out2[1]

                # Draw emotion probability bars
                h = 480
                gap = int((h - 7 * 10) / 7)
                for (i, (emotion, prob)) in enumerate(zip(class8_names.values(), probably)):
                    text = " {} : {:.3f}%".format(emotion, prob * 100)
                    w = int(prob * 300)
                    cv2.rectangle(image, (7, (i * gap) + 10), (w, (i * gap) + gap), (0, 0, 255), -1)
                    cv2.putText(image, text, (10, (i * gap) + 40), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

                # Draw valence-arousal indicators
                box_width = x2 - x1
                valence_center_begin, arousal_center_begin = (x1 + int(box_width / 2), y2), (
                    x1 + int(box_width / 2), y2 + 20)
                valence_length, arousal_length = int(box_width / 2 * valence), int(box_width / 2 * arousal)
                valence_center_end, arousal_center_end = (x1 + int(box_width / 2) + valence_length, y2 + 20), (
                    x1 + int(box_width / 2) + arousal_length, y2 + 20 + 20)
                cv2.putText(image, "valence", (x1 - 60, y2 + 20), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 1)
                cv2.rectangle(image, valence_center_begin, valence_center_end, (0, 0, 255), -1)
                cv2.putText(image, "arousal", (x1 - 60, y2 + 40), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 0, 0), 1)
                cv2.rectangle(image, arousal_center_begin, arousal_center_end, (255, 0, 0), -1)

        cv2.imshow("", image)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    run_test()

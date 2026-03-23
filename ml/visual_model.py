
import cv2
import os
import torch
from torchvision import models, transforms



def extract_frames(video_path, frame_interval = 1):
    """
    Extract frames from a video.
    
    :param video_path: Path to the video file
    :param frame_interval: Capture one frame every N seconds
    :return: List of frames
    """

    if not os.path.exists(video_path) :
        raise FileNotFounfError(f"Video not found at {video_path}")
    
    cap = cv2.VideoCapture(video_path)


    fps = cap.get(cv2.CAP_PROP_FPS)
    frames = []

    frame_count = 0
    success, frame = cap.read()

    while success:
        if int(frame_count % (fps * frame_interval)) == 0:
            frames.append(frame)

        
        success, frame = cap.read()
        frame_count += 1

    
    cap.release()
    return frames

# Load pretrained ResNet model (once)
_resnet_model = models.resnet18(pretrained=True)
_resnet_model.eval()

_feature_extractor = torch.nn.Sequential(*list(_resnet_model.children())[:-1])


_preprocess = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224,224)),
    transforms.ToTensor()
])

def extract_visual_features(frames):
    """
    Extract visual features from frames using a pretrained ResNet model.
    
    :param frames: List of frames (images)
    :return: List of feature tensors
    """

    features = []

    with torch.no_grad():
        for frame in frames:
            img = _preprocess(frame).unsqueeze(0)
            embedding = _feature_extractor(img)
            embedding = embedding.squeeze().numpy()
            features.append(embedding)

    return features


import numpy as np



def predict_genre_from_features(features):
    """
    Convert visual feature vectors into genre probabilities.
    (Simple heuristic-based ML for beginners)
    
    :param features: List of feature vectors
    :return: Dictionary of genre probabilities
    """

    if len(features) == 0:
        raise ValueError("No features provided")
    
    feature_matrix = np.array(features)


    avg_features = feature_matrix.mean(axis=0)


    action_score = float(np.linalg.norm(avg_features[:170]))
    comedy_score = float(np.linalg.norm(avg_features[170:340]))
    horror_score = float(np.linalg.norm(avg_features[340]))


    total = action_score + comedy_score + horror_score


    return {
        "Action": round(action_score / total, 2),
        "Comedy": round(comedy_score / total, 2),
        "Horror": round(horror_score / total, 2)
    }
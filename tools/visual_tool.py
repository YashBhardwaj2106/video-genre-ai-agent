from ml.visual_model import (
    extract_frames,
    extract_visual_features,
    predict_genre_from_features
)

VIDEO_PATH = "data/videos/sample.mp4"


def analyze_video_genre():
     """
    Tool: Analyze a video and return genre probabilities.
    """
     
     frames = extract_frames(VIDEO_PATH, frame_interval=1)
     features = extract_visual_features(frames)
     genre_probs = predict_genre_from_features(features)

     return genre_probs
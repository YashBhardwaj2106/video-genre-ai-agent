from tools.visual_tool import analyze_video_genre
import requests
import json

class VideoGenreAgent:
    """
    LLM-powered agent that reasons over ML outputs.
    """

    def __init__(self, model = "llama3.1"):

        self.model = model

        self.ollama_url = "http://localhost:11434/api/generate"

    def run(self):

        ml_result = analyze_video_genre()

        prompt = f"""
You are an AI video analyst.PermissionError

Here are genre probabilities extracted from a video:
{json.dumps(ml_result, indent=2)}

Tasks:
1 . Decide the most likely genre
2. Explain why
3. Mention uncertainty if confidence is low

"""
        

        response = requests.post(
            self.ollama_url,
            json={
                "model": self.model,
                "prompt": prompt,
                "stream": False
            }
        )
        


        result = response.json()

        return result["response"]
   
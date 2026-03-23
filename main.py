from agent.agent import VideoGenreAgent


def main():
    agent = VideoGenreAgent()
    result = agent.run()

    print("LLM AI Agent Output")
    print(result)
    
if __name__ == "__main__":
    main()
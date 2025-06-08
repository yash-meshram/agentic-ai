import streamlit as st
# from agno.agent import Agent
from phi.agent import Agent
from phi.model.google import Gemini
from phi.tools.duckduckgo import DuckDuckGo
from google.generativeai import upload_file, get_file
import google.generativeai as genai

import time
from pathlib import Path
import tempfile
from dotenv import load_dotenv
import os

# import debugpy
# debugpy.listen(("localhost", 5678))
# print("⚡ Waiting for debugger attach...")
# debugpy.wait_for_client()

import debugpy
if not debugpy.is_client_connected():
    try:
        debugpy.listen(("localhost", 5678))
        print("⚡ Waiting for debugger attach...")
        debugpy.wait_for_client()
    except RuntimeError:
        pass  # Already listening



load_dotenv("../.env")

google_api_key = os.getenv("GOOGLE_API_KEY")

if google_api_key:
    genai.configure(api_key = google_api_key)
    
# Page Configuration
st.set_page_config(
    page_title = "Multimodel AI Agnet - Video Summarizer",
    layout = "wide",
    page_icon = "🎥"
)

st.title("Video AI Summarizer Agent 🎥...")
st.header("Power by Gemini")

@st.cache_resource
def initialize_agent():
    return Agent(
        name = "Video AI Summarizer",
        model = Gemini(id = "gemini-2.0-flash-exp"),
        tools = [DuckDuckGo()],
        markdown = True
    )
    
# Initializing the agent
multimodel_agent = initialize_agent()

# File Uploader
video_file = st.file_uploader(
    label="Upload a video file",
    type=["mp4", "mov", "avi"],
    help="Upload a video for AI analysis"
)

if video_file:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as temp_video:
        temp_video.write(video_file.read())
        video_path = temp_video.name
        
    st.video(video_path, format="video/mp4", start_time=0)
        
    # upload and process video file
    with st.spinner("Processing video and grathering insights..."):
        processed_video = upload_file(video_path)
        while processed_video.state.name == "PROCESSING":
            time.sleep(1)
            processed_video = get_file(processed_video.name)
    
    # user query/question/instruction
    user_query = st.text_area(
        "What insights are you seeking from the video?",
        placeholder="Ask anything about the video content. Ai Agent will anayze and grather information and provide you with response",
        help = "Provide specific questions or insights you want from the video"
    )
    
    if st.button("Analyze Video", key="analyze_video_button"):
        # if no question or query asked
        if not user_query:
            st.warning("PLase enter a question or insights to analyze the video.")
        else:
            try:
                with st.spinner("generating response..."):
                    # # upload and process video file
                    # processed_video = upload_file(video_path)
                    # while processed_video.state.name == "PROCESSING":
                    #     time.sleep(1)
                    #     processed_video = get_file(processed_video.name)
                        
                    # Prompt generation for analysis
                    analysis_prompt = (
                        f"""
                        User's query: {user_query}
                        
                        Analyze the uploaded video for content and context.
                        Respond to the following query using video insights and supplementary web research.
                        
                        Provide a detailed, user-friendly and actionable response.
                        """
                    )
                    
                    # AI agent processing
                    response = multimodel_agent.run(analysis_prompt, videos=[processed_video])
                
                # Display result
                st.subheader("Analysis Result")
                st.markdown(response.content)
                
            except Exception as error:
                st.error(f"An error occure during analysis: {error}")
                
            finally:
                # Clean up temporary video file
                Path(video_path).unlink(missing_ok=True)
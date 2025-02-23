import gradio as gr
import os

def load_video():
    video_path = "./shared_data/videos/video1/space_station.mp4"
    if os.path.exists(video_path):
        return video_path
    else:
        return "Video not found."

def chat_response(question):
    # Placeholder for actual chat response logic
    return f"You asked: {question}"

with gr.Blocks() as demo:
    gr.Markdown("# Welcome to the Video Chat Demo")
    
    video = gr.Video(load_video)
    chatbox = gr.Textbox(label="What question do you want to ask?")
    response = gr.Textbox(label="Response", interactive=False)
    
    chatbox.change(fn=chat_response, inputs=chatbox, outputs=response)

demo.launch()
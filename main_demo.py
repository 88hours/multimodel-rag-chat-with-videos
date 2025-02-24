from pathlib import Path
import gradio as gr
import os
from PIL import Image
import ollama
from utility import download_video, get_transcript_vtt, extract_meta_data
vid_dir = "./shared_data/videos/yt_video"
Path(vid_dir).mkdir(parents=True, exist_ok=True)
metadatas=[]

def get_metadata_of_yt_video_with_captions(vid_url):  
    vid_filepath = download_video(vid_url, vid_dir)
    vid_transcript_filepath = get_transcript_vtt(vid_url, vid_dir)
    global metadatas
    metadatas = extract_meta_data(vid_dir, vid_filepath, vid_transcript_filepath) #should return lowercase file name without spaces
    return vid_filepath

def init_chat_ui(vid_filepath):
    with gr.Blocks():        
        video = gr.Video(vid_filepath)
        chatbox = gr.Textbox(label="What question do you want to ask?")
        response = gr.Textbox(label="Response", interactive=False)
        def chat_response_llvm(instruction):
            file_path = metadatas['video_path']
            result = ollama.generate(
                model='llava',
                prompt=instruction,
                images=[file_path],
                stream=True
            )['response']
            print(result)
            return result
        
        chatbox.change(fn=chat_response_llvm, inputs=chatbox, outputs=response)

    
def process_url_and_init(youtube_url):
    vid_filepath = get_metadata_of_yt_video_with_captions(youtube_url)
    return init_chat_ui(vid_filepath)

def init_ui():
    with gr.Blocks() as demo:
        url_input = gr.Textbox(label="Enter YouTube URL", value="https://www.youtube.com/watch?v=7Hcg-rLYwdM")
        submit_btn = gr.Button("Process Video")
        submit_btn.click(fn=process_url_and_init, inputs=url_input, outputs=None)
    return demo

if __name__ == '__main__':
    demo = init_ui()
    demo.launch()

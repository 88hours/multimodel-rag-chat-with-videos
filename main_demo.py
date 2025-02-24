from pathlib import Path
import gradio as gr
import os
from PIL import Image
import ollama
from utility import download_video, get_transcript_vtt, extract_meta_data
vid_dir = "./shared_data/videos/yt_video"
Path(vid_dir).mkdir(parents=True, exist_ok=True)
the_metadatas=[]

def get_metadata_of_yt_video_with_captions(vid_url):  
    vid_filepath = download_video(vid_url, vid_dir)
    vid_transcript_filepath = get_transcript_vtt(vid_url, vid_dir)
    metadatas = extract_meta_data(vid_dir, vid_filepath, vid_transcript_filepath) #should return lowercase file name without spaces
    return vid_filepath, metadatas

def chat_response_llvm(instruction):
    print("Metadatas: ", the_metadatas)
    #file_path = the_metadatas[0]
    file_path = 'shared_data/videos/yt_video/extracted_frame/'
    result = ollama.generate(
        model='llava',
        prompt=instruction,
        images=[file_path],
        stream=True
    )['response']
    return result
    
def process_url_and_init(youtube_url):
    vid_filepath, metadatas = get_metadata_of_yt_video_with_captions(youtube_url)
    the_metadatas = metadatas
    return vid_filepath, vid_filepath

def init_ui():
    with gr.Blocks() as demo:
        url_input = gr.Textbox(label="Enter YouTube URL", value="https://www.youtube.com/watch?v=7Hcg-rLYwdM")
        video_path = gr.Textbox(interactive=False)
        submit_btn = gr.Button("Process Video")
        #vid_filepath = 'shared_data/videos/yt_video/Welcome_back_to_Planet_Earth.mp4'
        chatbox = gr.Textbox(label="What question do you want to ask?", value="What is the astronaut doing?")
        response = gr.Textbox(label="Response", interactive=False)
        video = gr.Video()
        submit_btn2 = gr.Button("ASK")
        
        submit_btn.click(fn=process_url_and_init, inputs=url_input, outputs=[video_path,video])
        submit_btn2.click(fn=chat_response_llvm, inputs=[chatbox], outputs=[response])        
    return demo

if __name__ == '__main__':
    demo = init_ui()
    demo.launch()

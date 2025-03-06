from pathlib import Path
import os
from os import path as osp
import whisper
from moviepy import VideoFileClip
from PIL import Image
from utility import download_video, extract_meta_data, get_transcript_vtt, getSubs
from urllib.request import urlretrieve
from IPython.display import display
import ollama

def demp_video_input_that_has_transcript():  
    # first video's url
    vid_url = "https://www.youtube.com/watch?v=7Hcg-rLYwdM"

    # download Youtube video to ./shared_data/videos/video1
    vid_dir = "./shared_data/videos/video1"
    vid_filepath = download_video(vid_url, vid_dir)

    # download Youtube video's subtitle to ./shared_data/videos/video1
    vid_transcript_filepath = get_transcript_vtt(vid_url, vid_dir)

    return extract_meta_data(vid_dir, vid_filepath, vid_transcript_filepath)

def demp_video_input_that_has_no_transcript():  
        # second video's url
    vid_url=(
        "https://multimedia-commons.s3-us-west-2.amazonaws.com/" 
        "data/videos/mp4/010/a07/010a074acb1975c4d6d6e43c1faeb8.mp4"
    )
    vid_dir = "./shared_data/videos/video2"
    vid_name = "toddler_in_playground.mp4"

    # create folder to which video2 will be downloaded 
    Path(vid_dir).mkdir(parents=True, exist_ok=True)
    vid_filepath = urlretrieve(
                            vid_url, 
                            osp.join(vid_dir, vid_name)
                        )[0]
    
    path_to_video_no_transcript = vid_filepath

    # declare where to save .mp3 audio
    path_to_extracted_audio_file = os.path.join(vid_dir, 'audio.mp3')

    # extract mp3 audio file from mp4 video video file
    clip = VideoFileClip(path_to_video_no_transcript)
    clip.audio.write_audiofile(path_to_extracted_audio_file)

    model = whisper.load_model("small")
    options = dict(task="translate", best_of=1, language='en')
    results = model.transcribe(path_to_extracted_audio_file, **options)

    vtt = getSubs(results["segments"], "vtt")

    # path to save generated transcript of video1
    path_to_generated_trans = osp.join(vid_dir, 'generated_video1.vtt')
    # write transcription to file
    with open(path_to_generated_trans, 'w') as f:
        f.write(vtt)

    return extract_meta_data(vid_dir, vid_filepath, path_to_generated_trans)



def ask_llvm(instruction, file_path):
    result = ollama.generate(
        model='llava',
        prompt=instruction,
        images=[file_path],
        stream=False
    )['response']
    img=Image.open(file_path, mode='r')
    img = img.resize([int(i/1.2) for i in img.size])
    display(img) 
    for i in result.split('.'):
        print(i, end='', flush=True)
if __name__ == "__main__":
    meta_data = demp_video_input_that_has_transcript()
    
    meta_data1 = demp_video_input_that_has_no_transcript()
    data = meta_data1[1]
    caption = data['transcript']
    print(f'Generated caption is: "{caption}"')
    frame = Image.open(data['extracted_frame_path'])
    display(frame)
    instruction = "Can you describe the image?"
    ask_llvm(instruction, data['extracted_frame_path'])
    #print(meta_data)
    
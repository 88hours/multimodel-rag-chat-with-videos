# Add your utilities or helper functions to this file.

import os
from pathlib import Path
from dotenv import load_dotenv, find_dotenv
from io import StringIO, BytesIO
import textwrap
from typing import Iterator, TextIO, List, Dict, Any, Optional, Sequence, Union
from enum import auto, Enum
import base64
import glob
from moviepy import VideoFileClip
import requests
from tqdm import tqdm
from pytubefix import YouTube, Stream
import webvtt
import whisper
from youtube_transcript_api import YouTubeTranscriptApi
from youtube_transcript_api.formatters import WebVTTFormatter
from predictionguard import PredictionGuard
import cv2
import re
import json
import PIL
from ollama import chat
from ollama import ChatResponse
from PIL import Image
import dataclasses
import random
from datasets import load_dataset
from os import path as osp
from IPython.display import display
from langchain_core.prompt_values import PromptValue
from langchain_core.messages import (
    MessageLikeRepresentation,
)
from transformers import pipeline

MultimodalModelInput = Union[PromptValue, str, Sequence[MessageLikeRepresentation], Dict[str, Any]]

def get_from_dict_or_env(
    data: Dict[str, Any], key: str, env_key: str, default: Optional[str] = None
) -> str:
    """Get a value from a dictionary or an environment variable."""
    if key in data and data[key]:
        return data[key]
    else:
        return get_from_env(key, env_key, default=default)

def get_from_env(key: str, env_key: str, default: Optional[str] = None) -> str:
    """Get a value from a dictionary or an environment variable."""
    if env_key in os.environ and os.environ[env_key]:
        return os.environ[env_key]
    else:
        return default
        
def load_env():
    _ = load_dotenv(find_dotenv())

def get_openai_api_key():
    load_env()
    openai_api_key = os.getenv("OPENAI_API_KEY")
    return openai_api_key

def get_prediction_guard_api_key():
    load_env()
    PREDICTION_GUARD_API_KEY = os.getenv("PREDICTION_GUARD_API_KEY", None)
    if PREDICTION_GUARD_API_KEY is None:
        PREDICTION_GUARD_API_KEY = input("Please enter your Prediction Guard API Key: ")
    return PREDICTION_GUARD_API_KEY
    
PREDICTION_GUARD_URL_ENDPOINT = os.getenv("DLAI_PREDICTION_GUARD_URL_ENDPOINT", "https://dl-itdc.predictionguard.com") ###"https://proxy-dl-itdc.predictionguard.com"

# prompt templates
templates = [
    'a picture of {}',
    'an image of {}',
    'a nice {}',
    'a beautiful {}',
]

# function helps to prepare list image-text pairs from the first [test_size] data of a Huggingface dataset
def prepare_dataset_for_umap_visualization(hf_dataset, class_name, templates=templates, test_size=1000):
    # load Huggingface dataset (download if needed)
    dataset = load_dataset(hf_dataset, trust_remote_code=True)
    # split dataset with specific test_size
    train_test_dataset = dataset['train'].train_test_split(test_size=test_size)
    # get the test dataset
    test_dataset = train_test_dataset['test']
    img_txt_pairs = []
    for i in range(len(test_dataset)):
        img_txt_pairs.append({
            'caption' : templates[random.randint(0, len(templates)-1)].format(class_name),
            'pil_img' : test_dataset[i]['image']
        })
    return img_txt_pairs
    

def download_video(video_url, path):
    print(f'Getting video information for {video_url}')

    def progress_callback(stream: Stream, data_chunk: bytes, bytes_remaining: int) -> None:
        pbar.update(len(data_chunk))
    stream = None
    try:
        yt = YouTube(video_url, on_progress_callback=progress_callback)
        stream = yt.streams.filter(progressive=True, file_extension='mp4', res='480p').desc().first()
        if stream is None:
            stream = yt.streams.filter(progressive=True, file_extension='mp4').order_by('resolution').desc().first()
    except Exception as e:
        print(f"Youtube Exception Occured.Loading from local resource: {e}")

    uncleaned_filename = stream.default_filename.replace(' ', '').lower() if stream else "blackholes101nationalgeographic.mp4"
    print(f'Uncleaned filename: {uncleaned_filename}')
    filename= re.sub(r'[^a-zA-Z0-9]', '', uncleaned_filename).replace('mp4', '')
    filename_without_extension = os.path.splitext(filename)[0]
    filename_with_extension = filename+'.mp4'
    folder_path = os.path.join(path, filename_without_extension)
    print(f'Checking the folder path {folder_path}')
    full_file_path = os.path.join(folder_path, filename_with_extension)

    if not os.path.exists(folder_path):
        os.makedirs(folder_path, exist_ok=True)
    
    if os.path.exists(full_file_path):
        print('Video already downloaded at the folder path', full_file_path)
        is_downloaded = False
        return full_file_path, folder_path, is_downloaded

    
    is_downloaded = True

    print('Downloading video from YouTube...')
    pbar = tqdm(desc='Downloading video from YouTube', total=stream.filesize, unit="bytes")
    stream.download(folder_path, filename=filename_with_extension)
    pbar.close()
    return full_file_path, folder_path, is_downloaded

def get_video_id_from_url(video_url):
    """
    Examples:
    - http://youtu.be/SA2iWivDJiE
    - http://www.youtube.com/watch?v=_oPAwA_Udwc&feature=feedu
    - http://www.youtube.com/embed/SA2iWivDJiE
    - http://www.youtube.com/v/SA2iWivDJiE?version=3&amp;hl=en_US
    """
    import urllib.parse
    url = urllib.parse.urlparse(video_url)
    if url.hostname == 'youtu.be':
        return url.path[1:]
    if url.hostname in ('www.youtube.com', 'youtube.com'):
        if url.path == '/watch':
            p = urllib.parse.parse_qs(url.query)
            return p['v'][0]
        if url.path[:7] == '/embed/':
            return url.path.split('/')[2]
        if url.path[:3] == '/v/':
            return url.path.split('/')[2]

    return video_url

def generate_transcript_vtt(vid_dir, vid_filepath):
    print("Generating transcript for video ", vid_filepath)
     # declare where to save .mp3 audio
    path_to_extracted_audio_file = os.path.join(vid_dir, 'audio.mp3')

    # extract mp3 audio file from mp4 video video file
    path_to_video_no_transcript = vid_filepath
    clip = VideoFileClip(path_to_video_no_transcript)
    clip.audio.write_audiofile(path_to_extracted_audio_file)

    model = whisper.load_model("small")
    options = dict(task="translate", best_of=1, language='en')
    results = model.transcribe(path_to_extracted_audio_file, **options)

    vtt = getSubs(results["segments"], "vtt")

    # path to save generated transcript of video1
    path_to_generated_trans = osp.join(vid_dir, 'captions.vtt')
    # write transcription to file
    with open(path_to_generated_trans, 'w') as f:
        f.write(vtt)
    return path_to_generated_trans


# if this has transcript then download
def get_transcript_vtt(path, video_url, vid_file_path, from_gen=False):
    if from_gen:
        return generate_transcript_vtt(path,vid_file_path)
    video_id = get_video_id_from_url(video_url)
    filepath = os.path.join(path,'captions.vtt')
    if os.path.exists(filepath):
        print('Transcript already exists')
        return filepath
    
    print('Downloading Transcript...')

    transcript = YouTubeTranscriptApi.get_transcript(video_id, languages=['en-GB', 'en'])
    formatter = WebVTTFormatter()
    webvtt_formatted = formatter.format_transcript(transcript)
    
    with open(filepath, 'w', encoding='utf-8') as webvtt_file:
        webvtt_file.write(webvtt_formatted)
    webvtt_file.close()

    return filepath
    

# helper function for convert time in second to time format for .vtt or .srt file
def format_timestamp(seconds: float, always_include_hours: bool = False, fractionalSeperator: str = '.'):
    assert seconds >= 0, "non-negative timestamp expected"
    milliseconds = round(seconds * 1000.0)

    hours = milliseconds // 3_600_000
    milliseconds -= hours * 3_600_000

    minutes = milliseconds // 60_000
    milliseconds -= minutes * 60_000

    seconds = milliseconds // 1_000
    milliseconds -= seconds * 1_000

    hours_marker = f"{hours:02d}:" if always_include_hours or hours > 0 else ""
    return f"{hours_marker}{minutes:02d}:{seconds:02d}{fractionalSeperator}{milliseconds:03d}"

# a help function that helps to convert a specific time written as a string in format `webvtt` into a time in miliseconds
def str2time(strtime):
    # strip character " if exists
    strtime = strtime.strip('"')
    # get hour, minute, second from time string
    hrs, mins, seconds = [float(c) for c in strtime.split(':')]
    # get the corresponding time as total seconds 
    total_seconds = hrs * 60**2 + mins * 60 + seconds
    total_miliseconds = total_seconds * 1000
    return total_miliseconds
    
def _processText(text: str, maxLineWidth=None):
    if (maxLineWidth is None or maxLineWidth < 0):
        return text

    lines = textwrap.wrap(text, width=maxLineWidth, tabsize=4)
    return '\n'.join(lines)

# Resizes a image and maintains aspect ratio
def maintain_aspect_ratio_resize(image, width=None, height=None, inter=cv2.INTER_AREA):
    # Grab the image size and initialize dimensions
    dim = None
    (h, w) = image.shape[:2]

    # Return original image if no need to resize
    if width is None and height is None:
        return image

    # We are resizing height if width is none
    if width is None:
        # Calculate the ratio of the height and construct the dimensions
        r = height / float(h)
        dim = (int(w * r), height)
    # We are resizing width if height is none
    else:
        # Calculate the ratio of the width and construct the dimensions
        r = width / float(w)
        dim = (width, int(h * r))

    # Return the resized image
    return cv2.resize(image, dim, interpolation=inter)
    
# helper function to convert transcripts generated by whisper to .vtt file
def write_vtt(transcript: Iterator[dict], file: TextIO, maxLineWidth=None):
    print("WEBVTT\n", file=file)
    for segment in transcript:
        text = _processText(segment['text'], maxLineWidth).replace('-->', '->')

        print(
            f"{format_timestamp(segment['start'])} --> {format_timestamp(segment['end'])}\n"
            f"{text}\n",
            file=file,
            flush=True,
        )

# helper function to convert transcripts generated by whisper to .srt file
def write_srt(transcript: Iterator[dict], file: TextIO, maxLineWidth=None):
    """
    Write a transcript to a file in SRT format.
    Example usage:
        from pathlib import Path
        from whisper.utils import write_srt
import requests
        result = transcribe(model, audio_path, temperature=temperature, **args)
        # save SRT
        audio_basename = Path(audio_path).stem
        with open(Path(output_dir) / (audio_basename + ".srt"), "w", encoding="utf-8") as srt:
            write_srt(result["segments"], file=srt)
    """
    for i, segment in enumerate(transcript, start=1):
        text = _processText(segment['text'].strip(), maxLineWidth).replace('-->', '->')

        # write srt lines
        print(
            f"{i}\n"
            f"{format_timestamp(segment['start'], always_include_hours=True, fractionalSeperator=',')} --> "
            f"{format_timestamp(segment['end'], always_include_hours=True, fractionalSeperator=',')}\n"
            f"{text}\n",
            file=file,
            flush=True,
        )

def getSubs(segments: Iterator[dict], format: str, maxLineWidth: int=-1) -> str:
    segmentStream = StringIO()

    if format == 'vtt':
        write_vtt(segments, file=segmentStream, maxLineWidth=maxLineWidth)
    elif format == 'srt':
        write_srt(segments, file=segmentStream, maxLineWidth=maxLineWidth)
    else:
        raise Exception("Unknown format " + format)

    segmentStream.seek(0)
    return segmentStream.read()

# encoding image at given path or PIL Image using base64
def encode_image(image_path_or_PIL_img):
    if isinstance(image_path_or_PIL_img, PIL.Image.Image):
        # this is a PIL image
        buffered = BytesIO()
        image_path_or_PIL_img.save(buffered, format="JPEG")
        return base64.b64encode(buffered.getvalue()).decode('utf-8')
    else:
        # this is a image_path
        with open(image_path_or_PIL_img, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')

# checking whether the given string is base64 or not
def isBase64(sb):
    try:
        if isinstance(sb, str):
                # If there's any unicode here, an exception will be thrown and the function will return false
                sb_bytes = bytes(sb, 'ascii')
        elif isinstance(sb, bytes):
                sb_bytes = sb
        else:
                raise ValueError("Argument must be string or bytes")
        return base64.b64encode(base64.b64decode(sb_bytes)) == sb_bytes
    except Exception:
            return False

def encode_image_from_path_or_url(image_path_or_url):
    try:
        # try to open the url to check valid url
        f = urlopen(image_path_or_url)
        # if this is an url
        return base64.b64encode(requests.get(image_path_or_url).content).decode('utf-8')
    except:
        # this is a path to image
        with open(image_path_or_url, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')

# helper function to compute the joint embedding of a prompt and a base64-encoded image through PredictionGuard
def bt_embedding_from_prediction_guard(prompt, base64_image):
    # get PredictionGuard client
    client = _getPredictionGuardClient()
    message = {"text": prompt,}
    if base64_image is not None and base64_image != "":
        if not isBase64(base64_image): 
            raise TypeError("image input must be in base64 encoding!")
        message['image'] = base64_image
    response = client.embeddings.create(
        model="bridgetower-large-itm-mlm-itc",
        input=[message]
    )
    return response['data'][0]['embedding']

    
def load_json_file(file_path):
    # Open the JSON file in read mode
    with open(file_path, 'r') as file:
        data = json.load(file)
    return data

def display_retrieved_results(results):
    print(f'There is/are {len(results)} retrieved result(s)')
    print()
    for i, res in enumerate(results):
        print(f'The caption of the {str(i+1)}-th retrieved result is:\n"{results[i].page_content}"')
        print()
        print(results[i])
        #display(Image.open(results[i].metadata['metadata']['extracted_frame_path']))
        print("------------------------------------------------------------")

class SeparatorStyle(Enum):
    """Different separator style."""
    SINGLE = auto()
    
@dataclasses.dataclass
class Conversation:
    """A class that keeps all conversation history"""
    system: str
    roles: List[str]
    messages: List[List[str]]
    map_roles: Dict[str, str]
    version: str = "Unknown"
    sep_style: SeparatorStyle = SeparatorStyle.SINGLE
    sep: str = "\n"    

    def _get_prompt_role(self, role):
        if self.map_roles is not None and role in self.map_roles.keys():
            return self.map_roles[role]
        else:
            return role
            
    def _build_content_for_first_message_in_conversation(self, first_message: List[str]):
        content = []
        if len(first_message) != 2:
            raise TypeError("First message in Conversation needs to include a prompt and a base64-enconded image!")
        
        prompt, b64_image = first_message[0], first_message[1]
        
        # handling prompt
        if prompt is None:
            raise TypeError("API does not support None prompt yet")
        content.append({
            "type": "text",
            "text": prompt
        })
        if b64_image is None:
            raise TypeError("API does not support text only conversation yet")
            
        # handling image
        if not isBase64(b64_image):
            raise TypeError("Image in Conversation's first message must be stored under base64 encoding!")
        
        content.append({
            "type": "image_url",
            "image_url": {
                "url": b64_image,
            }
        })
        return content

    def _build_content_for_follow_up_messages_in_conversation(self, follow_up_message: List[str]):

        if follow_up_message is not None and len(follow_up_message) > 1:
            raise TypeError("Follow-up message in Conversation must not include an image!")
        
        # handling text prompt
        if follow_up_message is None or follow_up_message[0] is None:
            raise TypeError("Follow-up message in Conversation must include exactly one text message")

        text = follow_up_message[0]
        return text
        
    def get_message(self):
        messages = self.messages
        api_messages = []
        for i, msg in enumerate(messages):
            role, message_content = msg
            if i == 0:                
                # get content for very first message in conversation
                content = self._build_content_for_first_message_in_conversation(message_content)
            else:
                # get content for follow-up message in conversation
                content = self._build_content_for_follow_up_messages_in_conversation(message_content)
                
            api_messages.append({
                "role": role,
                "content": content,
            })
        return api_messages

    # this method helps represent a multi-turn chat into as a single turn chat format
    def serialize_messages(self):
        messages = self.messages
        ret = ""
        if self.sep_style == SeparatorStyle.SINGLE:
            if self.system is not None and self.system != "":
                ret = self.system + self.sep
            for i, (role, message) in enumerate(messages):
                role = self._get_prompt_role(role)
                if message:
                    if isinstance(message, List):
                        # get prompt only
                        message = message[0]
                    if i == 0:
                        # do not include role at the beginning
                        ret += message
                    else:
                        ret += role + ": " + message
                    if i < len(messages) - 1:
                        # avoid including sep at the end of serialized message
                        ret += self.sep
                else:
                    ret += role + ":"
        else:
            raise ValueError(f"Invalid style: {self.sep_style}")

        return ret
    
    def append_message(self, role, message):
        if len(self.messages) == 0:
            # data verification for the very first message
            assert role == self.roles[0], f"the very first message in conversation must be from role {self.roles[0]}"
            assert len(message) == 2, f"the very first message in conversation must include both prompt and an image"
            prompt, image = message[0], message[1]
            assert prompt is not None, f"prompt must be not None"
            assert isBase64(image), f"image must be under base64 encoding"
        else:
            # data verification for follow-up message
            assert role in self.roles, f"the follow-up message must be from one of the roles {self.roles}"
            assert len(message) == 1, f"the follow-up message must consist of one text message only, no image"
            
        self.messages.append([role, message])

    def copy(self):
        return Conversation(
            system=self.system,
            roles=self.roles,
            messages=[[x,y] for x, y in self.messages],
            version=self.version,
            map_roles=self.map_roles,
        )

    def dict(self):
        return {
            "system": self.system,
            "roles": self.roles,
            "messages": [[x, y[0] if len(y) == 1 else y] for x, y in self.messages],
            "version": self.version,
        }

prediction_guard_llava_conv = Conversation(
    system="",
    roles=("user", "assistant"),
    messages=[],
    version="Prediction Guard LLaVA enpoint Conversation v0",
    sep_style=SeparatorStyle.SINGLE,
    map_roles={
        "user": "USER", 
        "assistant": "ASSISTANT"
    }
)

# get PredictionGuard Client
def _getPredictionGuardClient():
    PREDICTION_GUARD_API_KEY = get_prediction_guard_api_key()
    client = PredictionGuard(
        api_key=PREDICTION_GUARD_API_KEY,
        url=PREDICTION_GUARD_URL_ENDPOINT,
    )
    return client

# helper function to call chat completion endpoint of PredictionGuard given a prompt and an image
def lvlm_inference(prompt, image, max_tokens: int = 200, temperature: float = 0.95, top_p: float = 0.1, top_k: int = 10):
    # prepare conversation
    conversation = prediction_guard_llava_conv.copy()
    conversation.append_message(conversation.roles[0], [prompt, image])
    return lvlm_inference_with_conversation(conversation, max_tokens=max_tokens, temperature=temperature, top_p=top_p, top_k=top_k)
    
    

def lvlm_inference_with_conversation(conversation, max_tokens: int = 200, temperature: float = 0.95, top_p: float = 0.1, top_k: int = 10):
    # get PredictionGuard client
    client = _getPredictionGuardClient()
    # get message from conversation
    messages = conversation.get_message()
    # call chat completion endpoint at Grediction Guard
    response = client.chat.completions.create(
        model="llava-1.5-7b-hf",
        messages=messages,
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
    )
    return response['choices'][-1]['message']['content']

def lvlm_inference_with_local(prompt):
    # Remove temperature or use correct parameters for Ollama client
    response = chat(
        model="phi3",  # or your chosen model
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ]
    )
    return response['message']['content']

def lvlm_inference_with_phi(prompt):

    response = chat(
        model="phi3",  # or your chosen model
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ]
    )
    return response['message']['content']

def lvlm_inference_with_tiny_model(prompt):
    classifier = pipeline(
        "text-generation",
        model="microsoft/phi-2",  # Only ~2.7GB
        device_map="auto",
        torch_dtype="auto",
    )
    
    response = classifier(
        prompt,
        max_new_tokens=512,  # Remove max_length and use only max_new_tokens
        temperature=0.7,
        do_sample=True,
        num_return_sequences=1,
        truncation=True,     # Add explicit truncation
        pad_token_id=classifier.tokenizer.eos_token_id,
        eos_token_id=classifier.tokenizer.eos_token_id,
    )[0]['generated_text']
    
    # Remove the input prompt from the response and clean up
    return response.replace(prompt, "").strip()

# function `extract_and_save_frames_and_metadata``:
#   receives as input a video and its transcript
#   does extracting and saving frames and their metadatas
#   returns the extracted metadatas
def extract_and_save_frames_and_metadata(
        path_to_video, 
        path_to_transcript, 
        path_to_save_extracted_frames,
        path_to_save_metadatas):
    
    # metadatas will store the metadata of all extracted frames
    metadatas = []

    # load video using cv2
    print(f"Loading video from {path_to_video}")
    video = cv2.VideoCapture(path_to_video)
    # load transcript using webvtt
    print(f"Loading transcript from {path_to_transcript}")
    trans = webvtt.read(path_to_transcript)
    
    # iterate transcript file
    # for each video segment specified in the transcript file
    for idx, transcript in enumerate(trans):
        # get the start time and end time in seconds
        start_time_ms = str2time(transcript.start)
        end_time_ms = str2time(transcript.end)
        # get the time in ms exactly 
        # in the middle of start time and end time
        mid_time_ms = (end_time_ms + start_time_ms) / 2
        # get the transcript, remove the next-line symbol
        text = transcript.text.replace("\n", ' ')
        # get frame at the middle time
        video.set(cv2.CAP_PROP_POS_MSEC, mid_time_ms)
        print(f"Extracting frame at {mid_time_ms} ms")
        success, frame = video.read()
        if success:
            # if the frame is extracted successfully, resize it
            image = maintain_aspect_ratio_resize(frame, height=350)
            # save frame as JPEG file
            img_fname = f'frame_{idx}.jpg'
            img_fpath = osp.join(
                path_to_save_extracted_frames, img_fname
            )
            cv2.imwrite(img_fpath, image)

            # prepare the metadata
            metadata = {
                'extracted_frame_path': img_fpath,
                'transcript': text,
                'video_segment_id': idx,
                'video_path': path_to_video,
                'mid_time_ms': mid_time_ms,
            }
            metadatas.append(metadata)

        else:
            print(f"ERROR! Cannot extract frame: idx = {idx}")

    # save metadata of all extracted frames
    fn = osp.join(path_to_save_metadatas, 'metadatas.json')
    with open(fn, 'w') as outfile:
        json.dump(metadatas, outfile)
    return metadatas

def extract_meta_data(vid_dir, vid_filepath, vid_transcript_filepath):
    # output paths to save extracted frames and their metadata 
    extracted_frames_path = osp.join(vid_dir, 'extracted_frame')
    metadatas_path = vid_dir

    # create these output folders if not existing
    print(f"Creating folders {extracted_frames_path} and {metadatas_path}")
    Path(extracted_frames_path).mkdir(parents=True, exist_ok=True)
    Path(metadatas_path).mkdir(parents=True, exist_ok=True)
    print("Extracting frames the video path ", vid_filepath)

    # call the function to extract frames and metadatas
    metadatas = extract_and_save_frames_and_metadata(
                    vid_filepath, 
                    vid_transcript_filepath,
                    extracted_frames_path,
                    metadatas_path,
                )
    return metadatas

# function extract_and_save_frames_and_metadata_with_fps
#   receives as input a video 
#   does extracting and saving frames and their metadatas
#   returns the extracted metadatas
def extract_and_save_frames_and_metadata_with_fps(
        lvlm_prompt,
        path_to_video,  
        path_to_save_extracted_frames,
        path_to_save_metadatas,
        num_of_extracted_frames_per_second=1):
    
    # metadatas will store the metadata of all extracted frames
    metadatas = []

    # load video using cv2
    video = cv2.VideoCapture(path_to_video)
    
    # Get the frames per second
    fps = video.get(cv2.CAP_PROP_FPS)
    # Get hop = the number of frames pass before a frame is extracted
    hop = round(fps / num_of_extracted_frames_per_second) 
    curr_frame = 0
    idx = -1
    while(True):
        # iterate all frames
        ret, frame = video.read()
        if not ret: 
            break
        if curr_frame % hop == 0:
            idx = idx + 1
        
            # if the frame is extracted successfully, resize it
            image = maintain_aspect_ratio_resize(frame, height=350)
            # save frame as JPEG file
            img_fname = f'frame_{idx}.jpg'
            img_fpath = osp.join(
                            path_to_save_extracted_frames, 
                            img_fname
                        )
            cv2.imwrite(img_fpath, image)

            # generate caption using lvlm_inference
            b64_image = encode_image(img_fpath)
            caption = lvlm_inference(lvlm_prompt, b64_image)
                
            # prepare the metadata
            metadata = {
                'extracted_frame_path': img_fpath,
                'transcript': caption,
                'video_segment_id': idx,
                'video_path': path_to_video,
            }
            metadatas.append(metadata)
        curr_frame += 1
        
    # save metadata of all extracted frames
    metadatas_path = osp.join(path_to_save_metadatas,'metadatas.json')
    with open(metadatas_path, 'w') as outfile:
        json.dump(metadatas, outfile)
    return metadatas

if __name__ == "__main__":
    res = lvlm_inference_with_ollama("Tell me a story")
    print(res)
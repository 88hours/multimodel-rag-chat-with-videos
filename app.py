from pathlib import Path
import gradio as gr
import os
from PIL import Image
import ollama
from utility import download_video, get_transcript_vtt, extract_meta_data, lvlm_inference_with_phi, lvlm_inference_with_tiny_model, lvlm_inference_with_tiny_model
from mm_rag.embeddings.bridgetower_embeddings import (
    BridgeTowerEmbeddings
)
from mm_rag.vectorstores.multimodal_lancedb import MultimodalLanceDB
import lancedb
import json
import os
from PIL import Image
from utility import load_json_file, display_retrieved_results
import pyarrow as pa

# declare host file
LANCEDB_HOST_FILE = "./shared_data/.lancedb"
# declare table name
# initialize vectorstore
db = lancedb.connect(LANCEDB_HOST_FILE)
# initialize an BridgeTower embedder
embedder = BridgeTowerEmbeddings()
video_processed = False
base_dir = "./shared_data/videos/yt_video"
Path(base_dir).mkdir(parents=True, exist_ok=True)


def open_table(table_name):
    # open a connection to table TBL_NAME
    tbl = db.open_table(table_name)

    print(f"There are {tbl.to_pandas().shape[0]} rows in the table")
    # display the first 3 rows of the table
    tbl.to_pandas()[['text', 'image_path']].head(3)


def check_if_table_exists(table_name):
    return table_name in db.table_names()


def store_in_rag(vid_table_name, vid_metadata_path):

    # load metadata files

    vid_metadata = load_json_file(vid_metadata_path)

    vid_subs = [vid['transcript'] for vid in vid_metadata]
    vid_img_path = [vid['extracted_frame_path'] for vid in vid_metadata]

    # for video1, we pick n = 7
    n = 7
    updated_vid_subs = [
        ' '.join(vid_subs[i-int(n/2): i+int(n/2)]) if i-int(n/2) >= 0 else
        ' '.join(vid_subs[0: i + int(n/2)]) for i in range(len(vid_subs))
    ]

    # also need to update the updated transcripts in metadata
    for i in range(len(updated_vid_subs)):
        vid_metadata[i]['transcript'] = updated_vid_subs[i]

    # you can pass in mode="append"
    # to add more entries to the vector store
    # in case you want to start with a fresh vector store,
    # you can pass in mode="overwrite" instead

    print("Creating vid_table_name ", vid_table_name)
    _ = MultimodalLanceDB.from_text_image_pairs(
        texts=updated_vid_subs,
        image_paths=vid_img_path,
        embedding=embedder,
        metadatas=vid_metadata,
        connection=db,
        table_name=vid_table_name,
        mode="overwrite",
    )
    open_table(vid_table_name)

    return vid_table_name


def get_metadata_of_yt_video_with_captions(vid_url, from_gen=False):
    vid_filepath, vid_folder_path, is_downloaded = download_video(
        vid_url, base_dir)
    if is_downloaded:
        print("Video downloaded at ", vid_filepath)
    if from_gen:
        # Delete existing caption and metadata files if they exist
        caption_file = f"{vid_folder_path}/captions.vtt"
        metadata_file = f"{vid_folder_path}/metadatas.json"
        if os.path.exists(caption_file):
            os.remove(caption_file)
            print(f"Deleted existing caption file: {caption_file}")
        if os.path.exists(metadata_file):
            os.remove(metadata_file)
            print(f"Deleted existing metadata file: {metadata_file}")

    print("checking transcript")
    vid_transcript_filepath = get_transcript_vtt(
        vid_folder_path, vid_url, vid_filepath, from_gen)
    vid_metadata_path = f"{vid_folder_path}/metadatas.json"
    print("checking metadatas at", vid_metadata_path)
    if os.path.exists(vid_metadata_path):
        print('Metadatas already exists')
    else:
        print("Downloading metadatas for the video ", vid_filepath)
        # should return lowercase file name without spaces
        extract_meta_data(vid_folder_path, vid_filepath,
                          vid_transcript_filepath)

    parent_dir_name = os.path.basename(os.path.dirname(vid_metadata_path))
    vid_table_name = f"{parent_dir_name}_table"
    print("Checking db and Table name ", vid_table_name)
    if not check_if_table_exists(vid_table_name):
        print("Table does not exists Storing in RAG")
    else:
        print("Table exists")

        def delete_table(table_name):
            db.drop_table(table_name)
            print(f"Deleted table {table_name}")
        delete_table(vid_table_name)

    store_in_rag(vid_table_name, vid_metadata_path)
    return vid_filepath, vid_table_name


def return_top_k_most_similar_docs(vid_table_name, query, use_llm=False):
    if not video_processed:
        gr.Error("Please process the video first in Step 1")
    # Initialize results variable outside the if condition
    max_docs = 2
    print("Querying ", vid_table_name)
    vectorstore = MultimodalLanceDB(
        uri=LANCEDB_HOST_FILE,
        embedding=embedder,
        table_name=vid_table_name
    )

    retriever = vectorstore.as_retriever(
        search_type='similarity',
        search_kwargs={"k": max_docs}
    )

    # Get results first
    results = retriever.invoke(query)

    if use_llm:
        # Read captions.vtt file
        def read_vtt_file(file_path):
            with open(file_path, 'r', encoding='utf-8') as f:
                return f.read()

        vid_table_name = vid_table_name.split('_table')[0]
        caption_file = 'shared_data/videos/yt_video/' + vid_table_name + '/captions.vtt'
        print("Caption file path ", caption_file)
        captions = read_vtt_file(caption_file)
        prompt = "Answer this query : " + query + " from the content " + captions
        print("Prompt ", prompt)
        all_page_content = lvlm_inference_with_phi(prompt)
    else:
        all_page_content = "\n\n".join(
            [result.page_content for result in results])

    page_content = gr.Textbox(all_page_content, label="Response",
                              elem_id='chat-response', visible=True, interactive=False)
    image1 = Image.open(results[0].metadata['extracted_frame_path'])
    image2_path = results[1].metadata['extracted_frame_path']

    if results[0].metadata['extracted_frame_path'] == image2_path:
        image2 = gr.update(visible=False)
    else:
        image2 = Image.open(image2_path)
        image2 = gr.update(value=image2, visible=True)

    return page_content, image1, image2


def process_url_and_init(youtube_url, from_gen=False):
    video_processed = True
    url_input = gr.update(visible=False)
    submit_btn = gr.update(visible=True)
    chatbox = gr.update(visible=True)
    submit_btn2 = gr.update(visible=True)
    frame1 = gr.update(visible=True)
    frame2 = gr.update(visible=False)
    chatbox_llm, submit_btn_chat = gr.update(
        visible=True), gr.update(visible=True)
    vid_filepath, vid_table_name = get_metadata_of_yt_video_with_captions(
        youtube_url, from_gen)
    video = gr.Video(vid_filepath, render=True)
    return url_input, submit_btn, video, vid_table_name, chatbox, submit_btn2, frame1, frame2, chatbox_llm, submit_btn_chat


def test_btn():
    text = "hi"
    res = lvlm_inference_with_phi(text)
    response = gr.Textbox(res, visible=True, interactive=False)
    return response


def init_ui():
    with gr.Blocks() as demo:

        gr.Markdown("Welcome to video chat demo - Initial processing can take up to 2 minutes, and responses may be slow. Please be patient and avoid clicking repeatedly.")
        url_input = gr.Textbox(label="Enter YouTube URL", visible=False, elem_id='url-inp',
                               value="https://www.youtube.com/watch?v=kOEDG3j1bjs", interactive=True)
        vid_table_name = gr.Textbox(
            label="Enter Table Name", visible=False, interactive=False)
        video = gr.Video()
        with gr.Row():
            submit_btn = gr.Button("Process Video By Download Subtitles")
            submit_btn_gen = gr.Button("Process Video By Generating Subtitles")

        with gr.Row():
            chatbox = gr.Textbox(label="Enter the keyword/s and AI will get related captions and images",
                                 visible=False, value="event horizan", scale=4)
            submit_btn_whisper = gr.Button(
                "Submit", elem_id='chat-submit', visible=False, scale=1)
        with gr.Row():
            chatbox_llm = gr.Textbox(
                label="Ask a Question", visible=False, value="what this video is about?", scale=4)
            submit_btn_chat = gr.Button("Ask", visible=False, scale=1)

        response = gr.Textbox(
            label="Response", elem_id='chat-response',  visible=False, interactive=False)

        with gr.Row():
            frame1 = gr.Image(visible=False, interactive=False, scale=2)
            frame2 = gr.Image(visible=False, interactive=False, scale=2)
        submit_btn.click(fn=process_url_and_init, inputs=[url_input], outputs=[
                         url_input, submit_btn, video, vid_table_name, chatbox, submit_btn_whisper, frame1, frame2, chatbox_llm, submit_btn_chat])
        submit_btn_gen.click(fn=lambda x: process_url_and_init(x, from_gen=True), inputs=[url_input], outputs=[
                             url_input, submit_btn, video, vid_table_name, chatbox, submit_btn_whisper, frame1, frame2, chatbox_llm, submit_btn_chat])
        submit_btn_whisper.click(fn=return_top_k_most_similar_docs, inputs=[
                                 vid_table_name, chatbox], outputs=[response, frame1, frame2])

        submit_btn_chat.click(
            fn=lambda table_name, query: return_top_k_most_similar_docs(
                vid_table_name=table_name,
                query=query,
                use_llm=True
            ),
            inputs=[vid_table_name, chatbox_llm],
            outputs=[response, frame1, frame2]
        )
        reset_btn = gr.Button("Reload Page")
        reset_btn.click(None, js="() => { location.reload(); }")

        test_llama = gr.Button("Test Llama")
        test_llama.click(test_btn, None, outputs=[response])
    return demo


def init_improved_ui():

    with gr.Blocks(theme=gr.themes.Soft()) as demo:
        # Header Section with Introduction
        with gr.Accordion(label=" # 🎬 Video Analysis Assistant", open=True):
            gr.Markdown("""
            ## How it Works:
            1. 📥 Provide a YouTube URL.
            2. 🔄 Choose a processing method:
               - Download the video and its captions/subtitles from YouTube.
               - Download the video and generate captions using Whisper AI.
                The system will load the video in video player for preview and process the video and extract frames from it. 
                It will then pass the captions and images to the RAG model to store them in the database.
                The RAG (Lance DB) uses a pre-trained BridgeTower model to generate embeddings that provide pairs of captions and related images.
            3. 🤖 Analyze video content through:
               - Keyword Search - Use this functionality to search for keywords in the video. Our RAG model will return the most relevant captions and images.
               - AI-powered Q&A - Use this functionality to ask questions about the video content. Our system will use the Meta/LLaMA model to analyze the captions and images and provide detailed answers.
            4. 📊 Results will be displayed in the response section with related images.
            
            > **Note**: Initial processing takes several minutes. Please be patient and monitor the logs for progress updates.
            """)

        # Video Input Section
        with gr.Group():
            url_input = gr.Textbox(
                label="YouTube URL",
                value="https://www.youtube.com/watch?v=kOEDG3j1bjs",
                visible=True,
                interactive=False
            )
            vid_table_name = gr.Textbox(label="Table Name", visible=False)
            video = gr.Video(label="Video Preview")

            with gr.Row():
                submit_btn = gr.Button(
                    "📥 Step 1: Process with Existing Subtitles", variant="primary", size='md')
                submit_btn_gen = gr.Button(
                    "🎯 Generate New Subtitles", variant="secondary", visible=False)

        # Analysis Tools Section
        with gr.Group():
            gr.Markdown("### 🔍 Step 2: Chat AI about the video")

            with gr.Row():
                chatbox = gr.Textbox(
                    label="Step 2: Search Keywords",
                    value="event horizon, black holes, space",
                    visible=False
                )
                submit_btn_whisper = gr.Button(
                    "🔎 Search",
                    visible=False,
                    variant="primary"
                )

            with gr.Row():
                chatbox_llm = gr.Textbox(
                    label="",
                    value="What is this video about?",
                    visible=True
                )
                submit_btn_chat = gr.Button(
                    "🤖 Ask",
                    visible=True,
                    scale=1
                )

        # Results Display Section
        with gr.Group():
            gr.Markdown("### 📊 AI Response")
            response = gr.Textbox(
                label="AI Response",
                visible=True,
                interactive=False
            )

            with gr.Row():
                frame1 = gr.Image(
                    visible=False, label="Related Frame 1", scale=1)
                frame2 = gr.Image(
                    visible=False, label="Related Frame 2", scale=2)

        # Control Buttons
        with gr.Row():
            reset_btn = gr.Button("🔄 Start Over", variant="secondary")
            test_llama = gr.Button("🧪 Say Hi to Llama",
                                   visible=False, variant="secondary")

        # Event Handlers
        submit_btn.click(
            fn=process_url_and_init,
            inputs=[url_input],
            outputs=[url_input, submit_btn, video, vid_table_name,
                     chatbox, submit_btn_whisper, frame1, frame2,
                     chatbox_llm, submit_btn_chat]
        )

        submit_btn_gen.click(
            fn=lambda x: process_url_and_init(x, from_gen=True),
            inputs=[url_input],
            outputs=[url_input, submit_btn, video, vid_table_name,
                     chatbox, submit_btn_whisper, frame1, frame2,
                     chatbox_llm, submit_btn_chat]
        )

        submit_btn_whisper.click(
            fn=return_top_k_most_similar_docs,
            inputs=[vid_table_name, chatbox],
            outputs=[response, frame1, frame2]
        )

        submit_btn_chat.click(
            fn=lambda table_name, query: return_top_k_most_similar_docs(
                vid_table_name=table_name,
                query=query,
                use_llm=True
            ),
            inputs=[vid_table_name, chatbox_llm],
            outputs=[response, frame1, frame2]
        )

        reset_btn.click(None, js="() => { location.reload(); }")
        test_llama.click(test_btn, None, outputs=[response])

    return demo


if __name__ == '__main__':
    demo = init_improved_ui()  # Updated function name here
    demo.launch(share=True, debug=True)

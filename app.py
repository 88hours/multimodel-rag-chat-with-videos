from pathlib import Path
import gradio as gr
import os
from PIL import Image
import ollama
from utility import download_video, get_transcript_vtt, extract_meta_data
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

base_dir = "./shared_data/videos/yt_video"
Path(base_dir).mkdir(parents=True, exist_ok=True)


def open_table(table_name):
    # open a connection to table TBL_NAME
    tbl = db.open_table(table_name)

    print(f"There are {tbl.to_pandas().shape[0]} rows in the table")
    # display the first 3 rows of the table
    tbl.to_pandas()[['text', 'image_path']].head(3)

def store_in_rag(vid_metadata_path):

    # load metadata files
    
    vid_metadata = load_json_file(vid_metadata_path)


    vid_subs = [vid['transcript'] for vid in vid_metadata]
    vid_img_path = [vid['extracted_frame_path'] for vid in vid_metadata]


    # for video1, we pick n = 7
    n = 7
    updated_vid_subs = [
    ' '.join(vid_subs[i-int(n/2) : i+int(n/2)]) if i-int(n/2) >= 0 else
    ' '.join(vid_subs[0 : i + int(n/2)]) for i in range(len(vid_subs))
    ]

    # also need to update the updated transcripts in metadata
    for i in range(len(updated_vid_subs)):
        vid_metadata[i]['transcript'] = updated_vid_subs[i]

    # you can pass in mode="append" 
    # to add more entries to the vector store
    # in case you want to start with a fresh vector store,
    # you can pass in mode="overwrite" instead 

    parent_dir_name = os.path.basename(os.path.dirname(vid_metadata_path))

    vid_table_name = f"{parent_dir_name}_table"
    print("TABLE NAME ", vid_table_name)
    _ = MultimodalLanceDB.from_text_image_pairs(
        texts=updated_vid_subs,
        image_paths=vid_img_path,
        embedding=embedder,
        metadatas=vid_metadata,
        connection=db,
        table_name=vid_table_name,
        mode="overwrite", 
    )
    return vid_table_name

def get_metadata_of_yt_video_with_captions(vid_url):  
    vid_filepath, vid_folder_path, is_downloaded = download_video(vid_url, base_dir)
    if is_downloaded:
        print("Video downloaded at ", vid_filepath)
    
    print("checking transcript")
    vid_transcript_filepath = get_transcript_vtt(vid_folder_path, vid_url, vid_filepath)
    vid_metadata_path = f"{vid_folder_path}/metadatas.json"
    print("checking metadatas at", vid_metadata_path)
    if os.path.exists(vid_metadata_path):
        print('Metadatas already exists')
    else
        extract_meta_data(vid_folder_path, vid_filepath, vid_transcript_filepath) #should return lowercase file name without spaces
    
    vid_table_name= store_in_rag(vid_metadata_path)
    print("Table name ", vid_table_name)
    open_table(vid_table_name)
    return vid_filepath

""" 
def chat_response_llvm(instruction):
    #file_path = the_metadatas[0]
    file_path = 'shared_data/videos/yt_video/extracted_frame/'
    result = ollama.generate(
        model='llava',
        prompt=instruction,
        images=[file_path],
        stream=True
    )['response']
    return result
     """

def return_top_k_most_similar_docs(vid_metadata_path, query="show me a group of astronauts", max_docs=1):
    # ask to return top 3 most similar documents
        # Creating a LanceDB vector store 
    table_name = os.path.dirname(vid_metadata_path)
    vectorstore = MultimodalLanceDB(
        uri=LANCEDB_HOST_FILE, 
        embedding=embedder, 
        table_name=table_name)

    # creating a retriever for the vector store
    # search_type="similarity" 
    #  declares that the type of search that the Retriever should perform 
    #  is similarity search
    # search_kwargs={"k": 1} means returning top-1 most similar document
    
    
    retriever = vectorstore.as_retriever(
    search_type='similarity', 
    search_kwargs={"k": max_docs})
    
    results = retriever.invoke(query)
    return results[0].page_content, Image.open(results[0].metadata['extracted_frame_path'])


def process_url_and_init(youtube_url):
    vid_filepath = get_metadata_of_yt_video_with_captions(youtube_url)
    return vid_filepath

def init_ui():
    with gr.Blocks() as demo:
        url_input = gr.Textbox(label="Enter YouTube URL", value="https://www.youtube.com/watch?v=7Hcg-rLYwdM", interactive=True)
        print(url_input)
        submit_btn = gr.Button("Process Video")
        #vid_filepath = 'shared_data/videos/yt_video/Welcome_back_to_Planet_Earth.mp4'
        chatbox = gr.Textbox(label="What question do you want to ask?", value="show me a group of astronauts")
        response = gr.Textbox(label="Response", interactive=False)
        video = gr.Video()
        frame = gr.Image()
        submit_btn2 = gr.Button("ASK")
        
        submit_btn.click(fn=process_url_and_init, inputs=url_input, outputs=[video])
        submit_btn2.click(fn=return_top_k_most_similar_docs, inputs=[chatbox], outputs=[response, frame])        
    return demo

if __name__ == '__main__':
    demo = init_ui()
    demo.launch(True)
    

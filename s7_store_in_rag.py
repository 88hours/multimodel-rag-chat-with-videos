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
TBL_NAME = "test_tbl"
# initialize vectorstore
db = lancedb.connect(LANCEDB_HOST_FILE)
# initialize an BridgeTower embedder 
embedder = BridgeTowerEmbeddings()


def return_top_k_most_similar_docs(max_docs=3):
    # ask to return top 3 most similar documents
        # Creating a LanceDB vector store 
    vectorstore = MultimodalLanceDB(
        uri=LANCEDB_HOST_FILE, 
        embedding=embedder, 
        table_name=TBL_NAME)

    # creating a retriever for the vector store
    # search_type="similarity" 
    #  declares that the type of search that the Retriever should perform 
    #  is similarity search
    # search_kwargs={"k": 1} means returning top-1 most similar document
    
    
    retriever = vectorstore.as_retriever(
    search_type='similarity', 
    search_kwargs={"k": max_docs})
    query2 = (
            "an astronaut's spacewalk "
            "with an amazing view of the earth from space behind"
    )
    results2 = retriever.invoke(query2)
    display_retrieved_results(results2)
    query3 = "a group of astronauts"
    results3 = retriever.invoke(query3)
    display_retrieved_results(results3) 


def open_table(TBL_NAME):
    # open a connection to table TBL_NAME
    tbl = db.open_table()

    print(f"There are {tbl.to_pandas().shape[0]} rows in the table")
    # display the first 3 rows of the table
    tbl.to_pandas()[['text', 'image_path']].head(3)

def store_in_rag():

    # load metadata files
    vid1_metadata_path = './shared_data/videos/video1/metadatas.json'
    vid2_metadata_path = './shared_data/videos/video2/metadatas.json'
    vid1_metadata = load_json_file(vid1_metadata_path)
    vid2_metadata = load_json_file(vid2_metadata_path)

    # collect transcripts and image paths
    vid1_trans = [vid['transcript'] for vid in vid1_metadata]
    vid1_img_path = [vid['extracted_frame_path'] for vid in vid1_metadata]

    vid2_trans = [vid['transcript'] for vid in vid2_metadata]
    vid2_img_path = [vid['extracted_frame_path'] for vid in vid2_metadata]


    # for video1, we pick n = 7
    n = 7
    updated_vid1_trans = [
    ' '.join(vid1_trans[i-int(n/2) : i+int(n/2)]) if i-int(n/2) >= 0 else
    ' '.join(vid1_trans[0 : i + int(n/2)]) for i in range(len(vid1_trans))
    ]

    # also need to update the updated transcripts in metadata
    for i in range(len(updated_vid1_trans)):
        vid1_metadata[i]['transcript'] = updated_vid1_trans[i]

    # you can pass in mode="append" 
    # to add more entries to the vector store
    # in case you want to start with a fresh vector store,
    # you can pass in mode="overwrite" instead 

    _ = MultimodalLanceDB.from_text_image_pairs(
        texts=updated_vid1_trans+vid2_trans,
        image_paths=vid1_img_path+vid2_img_path,
        embedding=embedder,
        metadatas=vid1_metadata+vid2_metadata,
        connection=db,
        table_name=TBL_NAME,
        mode="overwrite", 
    )

if __name__ == "__main__":
    tbl  = db.open_table(TBL_NAME)
    print(f"There are {tbl.to_pandas().shape[0]} rows in the table")
    #display the first 3 rows of the table
    return_top_k_most_similar_docs()
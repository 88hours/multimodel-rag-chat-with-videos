# Journey into Learning/Disecting - 4:00 PM  

[**Interactive Demo and Multimodal RAG System Architecture**](https://learn.deeplearning.ai/courses/multimodal-rag-chat-with-videos/lesson/2/interactive-demo-and-multimodal-rag-system-architecture)  

### A multimodal AI system should be able to understand both text and video content.  

---

## Step 1 - Learn Gradio (UI) (30 mins)  

Gradio is a powerful Python library for quickly building browser-based UIs. It supports hot reloading for fast development.  

### Key Concepts:  
- **fn**: The function wrapped by the UI.  
- **inputs**: The Gradio components used for input (should match function arguments).  
- **outputs**: The Gradio components used for output (should match return values).  

📖 [**Gradio Documentation**](https://www.gradio.app/docs/gradio/introduction)  

Gradio includes **30+ built-in components**.  

💡 **Tip**: For `inputs` and `outputs`, you can pass either:  
- The **component name** as a string (e.g., `"textbox"`)  
- An **instance of the component class** (e.g., `gr.Textbox()`)  

### Sharing Your Demo  
```python
demo.launch(share=True)  # Share your demo with just one extra parameter.
```

## Gradio Advanced Features  

### **Gradio.Blocks**  
Gradio provides `gr.Blocks`, a flexible way to design web apps with **custom layouts and complex interactions**:  
- Arrange components freely on the page.  
- Handle multiple data flows.  
- Use outputs as inputs for other components.  
- Dynamically update components based on user interaction.  

### **Gradio.ChatInterface**  
- Always set `type="messages"` in `gr.ChatInterface`.  
- The default (`type="tuples"`) is **deprecated** and will be removed in future versions.  
- For more UI flexibility, use `gr.ChatBot`.  
- `gr.ChatInterface` supports **Markdown** (not tested yet).  

---

## Step 2 - Learn Bridge Tower Embedding Model (Multimodal Learning) (15 mins)  

Developed in collaboration with Intel, this model maps image-caption pairs into **512-dimensional vectors**.  

### Measuring Similarity  
- **Cosine Similarity** → Measures how close images are in vector space (**efficient & commonly used**).  
- **Euclidean Distance** → Uses `cv2.NORM_L2` to compute similarity between two images.  

### Converting to 2D for Visualization  
- **UMAP** reduces 512D embeddings to **2D for display purposes**.  

## Preprocessing Videos for Multimodal RAG  

### **Case 1: WEBVTT → Extracting Text Segments from Video**  
    - Converts video + text into structured metadata.  
    - Splits content into multiple segments.  

### **Case 2: Whisper (Small) → Video Only**  
    - Extracts **audio** → `model.transcribe()`.  
    - Applies `getSubs()` helper function to retrieve **WEBVTT** subtitles.  
    - Uses **Case 1** processing.  

### **Case 3: LvLM → Video + Silent/Music Extraction**  
    - Uses **Llava (LvLM model)** for **frame-based captioning**.  
    - Encodes each frame as a **Base64 image**.  
    - Extracts context and captions from video frames.  
    - Uses **Case 1** processing.  

# Step 4 - What is LLaVA?
LLaVA (Large Language-and-Vision Assistant), a large multimodal model that connects a vision encoder that doesn't just see images but understands them, reads the text embedded in them, and reasons about their context—all.

# Step 5 - what is a vector Store?
A vector store is a specialized database designed to:

- Store and manage high-dimensional vector data efficiently
- Perform similarity-based searches where K=1 returns the most similar result

- In LanceDB specifically, store multiple data types:
    . Text content (captions)
    . Image file paths
    . Metadata
    . Vector embeddings

```python
_ = MultimodalLanceDB.from_text_image_pairs(
    texts=updated_vid1_trans+vid2_trans,
    image_paths=vid1_img_path+vid2_img_path,
    embedding=BridgeTowerEmbeddings(),
    metadatas=vid1_metadata+vid2_metadata,
    connection=db,
    table_name=TBL_NAME,
    mode="overwrite", 
)
```
# Gotchas and Solutions
    Image Processing: When working with base64 encoded images, convert them to PIL.Image format before processing with BridgeTower
    Model Selection: Using BridgeTowerForContrastiveLearning instead of PredictionGuard due to API access limitations
    Model Size: BridgeTower model requires ~3.5GB download
    Image Downloads: Some Flickr images may be unavailable; implement robust error handling
    Token Decoding: BridgeTower contrastive learning model works with embeddings, not token predictions
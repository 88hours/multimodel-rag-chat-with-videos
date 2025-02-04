# Journey into learning - 4:00 pm
https://learn.deeplearning.ai/courses/multimodal-rag-chat-with-videos/lesson/2/interactive-demo-and-multimodal-rag-system-architecture
- A MULTIMODEL AI SYSTEM SHOULD BE ABLE TO UNDERSTAND TEXT, VIDEO CONTENT

## Step 1 - learn Gradio (UI): 30 mins
Great for making quick UI in python, that will run in browser. It also has hot reloading.

fn: The function to wrap a user interface (UI) around
inputs: the Gradio component(s) to use for the input. The number of components should match the number of arguments in your function.
outputs: the Gradio component(s) to use for the output. The number of components should match the number of return values from your function.

https://www.gradio.app/docs/gradio/introduction

Gradio includes more than 30 built-in components 

Tip:
 For the `inputs` and `outputs` arguments, you can pass in the name of these components as a string (`"textbox"`) or an instance of the class (`gr.Textbox()`).

## Sharing Demo
 demo.launch(share=True)  # Share your demo with just 1 extra parameter.
 ### Why did not hot reloading worked?



## Gradio.Blocks
Gradio offers a low-level approach for designing web apps with more customizable layouts and data flows with the gr.Blocks class. Blocks supports things like controlling where components appear on the page, handling multiple data flows and more complex interactions (e.g. outputs can serve as inputs to other functions), and updating properties/visibility of components based on user interaction — still all in Python.

## Gradio.ChatInterface
 Always set type="messages" in gr.ChatInterface. The default value (type="tuples") is deprecated and will be removed in a future version of Gradio.
 Better to use gr.ChatBot for more ui options
 gr.ChatInterface also supports markdown (I did not try it thou)
## Step 2 Learn Bridge Tower Embedding Model (MultiModel Learning Space): 15 mins
 Made by Collab at Intel
 512- Dimenstions, so mean each image,caption pair has been coverted into a 512 dimension vector
### Measuing distance
 Cosine Similarity to see how much images are far and near to each other - Cheap Computing, Commanly used
 Euclidean Distance using CV vision call by passing two images with cv2.NORM_L2
 ### Conversion to 2D for display purposes
 UMAP to transfer 512 vector dimension embeddings to 2 dimension embeddings 

## Preprocessing Videos for Multimodal RAG¶
### Case 1: WEBVVT =>  Text Segment from video 
    - Convert video and text into metadata, by dividing into multiple segment 

### Case 2: Whisper(small) => Video Only
    - Extract Audio -> model.transcribe
    - Apply getSubs() helper function to get WEBVVT
    - Apply Case 1

### Case 3: Lvlm => Video + silent/music (extract from video )
    - Lava 
    - Encode each frame as a base64 image 
    - Pass it to Lava (LvLM model) to get captions and get the context

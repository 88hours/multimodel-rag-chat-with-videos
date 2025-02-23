import gradio as gr

def greet(name, intensity): #Number of inputs should match the number of input components
    return "Hello, " + name + "!" * int(intensity)


basicDemo = gr.Interface(
    fn=greet,
    inputs=["text", "slider"],
    outputs=["text"],
)


with gr.Blocks() as blockDemo:
    gr.Markdown("Enter your Name and Intensity.")
    with gr.Row():
        inp1 = gr.Textbox(placeholder="What is your name?")
        inp2 = gr.Slider(minimum=1, maximum=100)
        out = gr.Textbox()
    btn = gr.Button("Run")
    btn.click(fn=greet, inputs=[inp1,inp2], outputs=out)


def random_response(messages, history):
    return "I am a bot. I don't understand human language. I can only say Hello. 🤖"

with gr.Blocks() as chatInterfaceDemo:
    with gr.Row():
        with gr.Column(scale=4):
            video_input = gr.Video(height=512, width=512, elem_id="video", interactive=True)
            video_url = gr.Textbox(placeholder="Enter YouTube URL")
            load_button = gr.Button("Load Video")
        with gr.Column(scale=7):
            gr.ChatInterface(
                fn=random_response, 
                type="messages"
            )

    def load_video(url):
        # Here you can add logic to download the video from YouTube and return the file path
        # For now, we will just return the URL
        return url

    load_button.click(fn=load_video, inputs=video_url, outputs=video_input)

chatInterfaceDemo.launch(share=False)  # Share your demo with just 1 extra parameter 🚀


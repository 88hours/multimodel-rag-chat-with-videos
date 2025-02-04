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
               gr.Video(height=512, width=512, elem_id="video", interactive=False )
            with gr.Column(scale=7):
                gr.ChatInterface(
                    fn=random_response, 
                    type="messages"                )

chatInterfaceDemo.launch(share=False)  # Share your demo with just 1 extra parameter 🚀


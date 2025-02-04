## Start the gradio server
from gradio_utils import get_demo

#You will need to restart the kernel each time you rerun this cell;
#otherwise, the port will not be available.

debug = False # change this to True if you want to debug

demo = get_demo()
demo.launch(server_name="0.0.0.0", server_port=9999, debug=debug)
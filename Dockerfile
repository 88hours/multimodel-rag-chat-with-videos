FROM mcr.microsoft.com/devcontainers/python:3.11

# Install system dependencies
RUN apt-get update && export DEBIAN_FRONTEND=noninteractive \
    && apt-get -y install --no-install-recommends \
    ffmpeg \
    && apt-get clean -y && rm -rf /var/lib/apt/lists/*


# Install the required system libraries for OpenCV
RUN apt-get update && apt-get install -y libgl1-mesa-glx

# Install PyTorch and other dependencies
RUN pip install -r requirements.txt

# Run the application
CMD ["python", "app.py"]
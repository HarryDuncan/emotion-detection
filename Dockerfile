# 1. Use the NVIDIA TensorFlow base image (Optimized for Blackwell/RTX 50-series)
FROM nvcr.io/nvidia/tensorflow:25.02-tf2-py3

# 2. Prevent interactive prompts during apt-get
ENV DEBIAN_FRONTEND=noninteractive

# 3. Install system dependencies for OpenCV and Video4Linux
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    v4l-utils \
    libv4l-dev \
    && rm -rf /var/lib/apt/lists/*

# 4. Set the working directory inside the container
WORKDIR /workspace

# 5. Copy your requirements file first to leverage Docker caching
COPY requirements.txt .

# 6. Install Python dependencies
# We use --no-cache-dir to keep the image slim
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# 7. Copy the rest of your application code
COPY . .

# 8. Expose the port your app runs on
EXPOSE 5005
ENV PORT=5005

# 9. Run the app (bind to 0.0.0.0 so it accepts connections from host)
CMD ["python3", "appv2.py"]
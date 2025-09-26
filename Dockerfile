# Choose a NVIDIA Triton Server. If needed, choose a later version.
FROM nvcr.io/nvidia/tritonserver:24.01-py3
RUN pip install opencv-python ultralytics "tensorrt>=10.2,<10.3"
# This step is important to avoid the following error: ImportError: libGL.so.1
# To follow this issue, refer to https://stackoverflow.com/questions/55313610/importerror-libgl-so-1-cannot-open-shared-object-file-no-such-file-or-directo
RUN apt-get update && apt-get install ffmpeg libsm6 libxext6  -y

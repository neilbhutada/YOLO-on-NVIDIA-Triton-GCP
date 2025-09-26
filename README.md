# YOLO-on-NVIDIA-Triton-GCP
Repository containing code to implement a YOLO model on NVIDIA Triton and host it on Google Cloud Platform (Vertex AI)

## Pre-requisites

1. NVIDIA GPU(s) - for example :NVIDIA L4 
2. Docker 
3. Python 3
4. Ideally, a linux machine. For example, a g2 instance on Vertex AI Workbench. 
5. **Most importantly - clone this repo**

## Steps

1. ### Convert your YOLO Pytorch model to a TensorRT format

    i. Download/Train a YOLO Pytorch model of your choice. 

    ii. Build the docker image from Dockerfile of this repo with `docker build -t <image name> .` Makesure you run this command at the root of this repo and please change the `<image name>` to something more appropriate.  

    iii. Copy your Pytorch model in the `convert-to-tensorRT` directory. Change the model name in `yolo-pt-to-tensorrt.py`. 

    iv. Run the docker image built in step (ii) in interactive mode and mount the current directory with `docker run -ti --rm --gpus all -p 8000:8000 -v $(pwd):/workspace <image name> bash` 

    v. Navigate to `/workspace/convert-to-tensorRT` within the docker container. Then run the following commands:
    ```
    pip install -r tensorRT-conversion.txt
    python3 yolo-pt-to-tensorrt.py
    ```
    After this command you should see a .engine file with your YOLO model name. Please move/copy this `.engine` file under `/workspace/models/yolo-model/1/`. 
    
    Note, you can run the copy/move command outside the docker container as we have mounted the current working directory.

    vi. Shut down your container, if you want to test/run the Tritonserver locally with the model repository on Google Cloud Storage (GCS). Otherwise, you don't need to shutdown the container. 

2. ### Test the Triton Server locally

    As mentioned in the last step, there are two ways to test the Triton Server locally: with and without GCS integration.

    #### Without GCS integration

        i. If your docker container is not running from the previous section, please restart by running the command in step (ii.) of the previous section.

        ii. Run the follow command within the container to start the Triton Server:

        `tritonserver --model-repository=/workspace/models` 

        


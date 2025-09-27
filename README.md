# YOLO-on-NVIDIA-Triton-GCP
Repository containing code to implement a YOLO model on NVIDIA Triton and host it on Google Cloud Platform (Vertex AI)

## Pre-requisites

1. NVIDIA GPU(s) - for example: NVIDIA L4 
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
    After this command you should see a TensorRT (`.engine`) file with your YOLO model name. Please move/copy this TensorRT file under `/workspace/models/yolo-model/1/`. 
    
    Note, you can run the copy/move command outside the docker container as we have mounted the current working directory.

    vi. Edit `models/yolo-model/1/input_config.json` to include the name of the TensorRT file.
  
2. ### Test the Triton Server locally

    There are two ways to test the Triton Server locally: with and without GCS integration.

    #### Without GCS integration

    i. If your docker container is not running, please start it by running the command in step (ii.) of the previous section.

    ii. Run the follow command within the container to start the Triton Server:

    `tritonserver --model-repository=/workspace/models`

    #### With GCS Integration

    i. Make sure the docker container is not running anymore. Enter `exit` to terminate the docker container.

    ii. Copy the `models` folder to GCS. For simplicity sake, you can use `gsutil`. The destination GCS URL for `models` should look something like this: `gs://<your gcs bucket>/<an optional directory in your bucket>/models`

    iii. Run the follow command to almost replicate how Triton Server will be behave on a Vertex AI Endpoint:
    ```
    docker run --rm --gpus all -p 8000:8000 -e  AIP_STORAGE_URI=<gcs url of your model repository from the previous step> -e AIP_MODE=True <docker image name> 
    ```

    #### Testing Inferencing Locally

    i. Open another terminal and leave the container running the Triton Server from the previous sections undistrubuted.

    ii. Edit `payload_generation.py` to include the file names/paths of the image(s) you want to inference.

    iii. Run the follow command to install appropriate packages and `payload_generation.py`:

    ```
    pip install opencv-python
    python payload_generation.py
    ```

    The output will be a json file called `payload.json`

    iv. Use the following curl command and get predictions for the corresponding image(s):

    ```
    curl -s -X POST http://localhost:8000/v2/models/yolo-model/infer \
  -H "Content-Type: application/json" \
  --data @payload.json
    ```
3. ### Deploy to Vertex AI Endpoints

  Please refer to `upload-to-vertex_ai.ipynb` for detailed instructions.

## References

- <a href="https://docs.ultralytics.com/modes/predict/#inference-sources"> YOLO model predictions guide </a>
- <a href="https://docs.ultralytics.com/integrations/tensorrt/"> Convert YOLO Pytorch models to a TensorRT format </a>
- <a href="https://catalog.ngc.nvidia.com/orgs/nvidia/containers/tritonserver"> NVIDIA Triton Server Docker images </a>
- <a href="https://docs.nvidia.com/deeplearning/triton-inference-server/user-guide/docs/python_backend/README.html"> NVIDIA Trition Server's Python Backend guide </a>
- <a href="https://cloud.google.com/vertex-ai/docs/predictions/using-nvidia-triton"> GCP x NVIDIA Trition Server Guide </a>





 








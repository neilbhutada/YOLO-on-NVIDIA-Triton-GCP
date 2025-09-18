# YOLO with Python Backend on Triton Server

## Pre-requisities

- Clone this repo
- Consider running this project on a machine that uses GPUs 
- Have docker installed
- Use an py3 based Triton base image.


## BYOM (Bring our own YOLO model?)
- Bring a YOLO pytorch model that you would want to use and store it under models/yolo11v/1. 
- Convert the pytorch model into a tensorRT model by doing the following:
    -  Build Docker image with `docker build -t <image_name> .`
    -  Run the docker image in interactive model and mount current working directory: `docker run -ti --rm --gpus all -p 8000:8000 -v $(pwd):/workspace <docker image name> bash`
    - Look at the packages in pyproject.toml and install them in the docker container after running the previous step. 
    - Navigate to /workspace and copy `yolo-to-tensorRT.py' in the same directory you have the YOLO pytorch file in.
    - Run `python3 yolo-to-tensorRT.py`
- Copy the generated .engine tensorRT under models/yolo11v/1. 
- Change the filename of the tensorRT file in the initialize function in models/yolo11v/1/model.py on line 69.
    
    

## Running Triton Server

- Build Docker image with `docker build -t <image_name> .`
- Optionally, please store the model.py file and other artifacts (basically the 'models' directory) on GCS
- If you have stored the model artifacts on GCS, then you run the following command to spin your Triton Server 
```
docker run --rm --gpus all -p 8000:8000 -e  AIP_STORAGE_URI=<gcs path where you stored model folder> -e AIP_MODE=True <docker image name> 
```

- If you haven't saved your models directory on GCS, consider running the docker container in interactive mode and mount the current directory the contains the models folder

```
docker run -ti --rm --gpus all -p 8000:8000 -v $(pwd):/workspace <docker image name> bash
tritonserver --model-repository=/workspace/models
```

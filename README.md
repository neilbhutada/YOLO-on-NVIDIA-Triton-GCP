# YOLO on NVIDIA Triton (GCP / Vertex AI)

Code and instructions to deploy a YOLO model with **NVIDIA Triton Inference Server** and host it on **Google Cloud Platform (Vertex AI Endpoints)**.

## Table of Contents
- [Prerequisites](#prerequisites)
- [Quick Overview](#quick-overview)
- [1) Convert your YOLO PyTorch model to TensorRT](#1-convert-your-yolo-pytorch-model-to-tensorrt)
- [2) Run and test Triton Server locally](#2-run-and-test-triton-server-locally)
  - [2.1 Without GCS integration](#21-without-gcs-integration)
  - [2.2 With GCS integration](#22-with-gcs-integration)
  - [2.3 Local inference test](#23-local-inference-test)
- [3) Deploy to Vertex AI Endpoints](#3-deploy-to-vertex-ai-endpoints)
- [References](#references)

## Prerequisites
1. **NVIDIA GPU** (e.g., NVIDIA L4)
2. **Docker**
3. **Python 3**
4. **Linux** machine recommended (e.g., a **G2** GPU instance or Vertex AI Workbench VM)
5. **Clone this repository**
   ```bash
   git clone <this-repo-url>
   cd YOLO-on-NVIDIA-Triton-GCP
   ```

## Quick Overview
- Convert a YOLO **PyTorch** model to **TensorRT** (`.engine`).
- Organize the **Triton model repository** under `models/`.
- Run **Triton Server** locally (optionally backed by **GCS** for parity with Vertex AI).
- Verify inference using a generated JSON payload and a `curl` request.
- Deploy the same artifact layout to **Vertex AI Endpoints**.

---

## 1) Convert your YOLO PyTorch model to TensorRT

i) Download or train a YOLO **PyTorch** model of your choice.

ii) Build the Docker image from this repo’s `Dockerfile` (run at the repo root). Replace `<IMAGE_NAME>` with something meaningful.
```bash
docker build -t <IMAGE_NAME> .
```

iii) Copy your **PyTorch** model into the `convert-to-tensorRT/` directory and update the model path/name in `convert-to-tensorRT/yolo-pt-to-tensorrt.py`.

iv) Run the image in interactive mode, mounting the current directory:
```bash
docker run -ti --rm --gpus all   -p 8000:8000   -v "$(pwd)":/workspace   <IMAGE_NAME> bash
```

v) Inside the container, navigate and run the conversion:
```bash
cd /workspace/convert-to-tensorRT
pip install -r tensorRT-conversion.txt
python3 yolo-pt-to-tensorrt.py
```

After the script completes, you should see a **TensorRT `.engine`** file for your model.  
**Move or copy** the `.engine` file to:
```
/workspace/models/yolo-model/1/
```

> Note: You can also run the copy/move from your host—since the current directory is mounted to `/workspace`.

vi) Update `models/yolo-model/1/input_config.json` to reference the generated TensorRT engine filename.

---

## 2) Run and test Triton Server locally

There are two ways to test locally: **without** and **with** **GCS** integration.

### 2.1 Without GCS integration

i) If your container isn’t running, (re)start it as in step **1.iv**.

ii) Launch Triton inside the container:
```bash
tritonserver --model-repository=/workspace/models
```
> Default ports: HTTP `8000`, gRPC `8001`, Metrics `8002`.

### 2.2 With GCS integration

i) If a container is still running, type `exit` to stop it.

ii) Copy the `models/` folder to GCS. For simplicity’s sake, you can use `gsutil`.  
The destination should look like:  
`gs://<YOUR_BUCKET>/<optional/subdir>/models`

iii) Start a new container that points Triton to the GCS model repository (this mimics Vertex AI behavior):
```bash
docker run --rm --gpus all -p 8000:8000   -e AIP_STORAGE_URI="gs://<YOUR_BUCKET>/<optional/subdir>/models"   -e AIP_MODE=True   <IMAGE_NAME>
```

### 2.3 Local inference test

i) Open another terminal; leave the Triton container running **undisturbed**.

ii) Edit `payload_generation.py` and set the file names/paths of the image(s) you want to run inference on.

iii) Install dependencies and generate the payload:
```bash
pip install opencv-python
python payload_generation.py
```
This produces a `payload.json`.

iv) Send the request to Triton:
```bash
curl -s -X POST http://localhost:8000/v2/models/yolo-model/infer   -H "Content-Type: application/json"   --data @payload.json
```
You should receive JSON predictions corresponding to your input image(s).

---

## 3) Deploy to Vertex AI Endpoints

See `upload-to-vertex_ai.ipynb` for step-by-step deployment instructions to **Vertex AI Endpoints** using the same model repository layout.

---

## References
- <a href="https://docs.ultralytics.com/modes/predict/#inference-sources">YOLO model predictions guide</a>  
- <a href="https://docs.ultralytics.com/integrations/tensorrt/">Convert YOLO PyTorch models to TensorRT</a>  
- <a href="https://catalog.ngc.nvidia.com/orgs/nvidia/containers/tritonserver">NVIDIA Triton Server Docker images</a>  
- <a href="https://docs.nvidia.com/deeplearning/triton-inference-server/user-guide/docs/python_backend/README.html">NVIDIA Triton Python Backend guide</a>  
- <a href="https://cloud.google.com/vertex-ai/docs/predictions/using-nvidia-triton">GCP × NVIDIA Triton Server guide</a>

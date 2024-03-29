import time
import subprocess
import mimetypes
import uuid

import runpod
import requests
from requests.adapters import HTTPAdapter, Retry
from firebase_admin import credentials, initialize_app, storage, firestore

import torch
from diffusers import PixArtAlphaPipeline
from runpod.serverless.modules.rp_logger import RunPodLogger


logger = RunPodLogger()

LOCAL_URL = "http://127.0.0.1:5000"

INPUT_SCHEMA = {
    "seed": {
        "type": int,
        "title": "Seed",
        "required": False,
        "description": "Random seed. Leave blank to randomize the seed"
    },
    "prompt": {
        "type": str,
        "title": "Prompt",
        "default": "With the style of van gogh, A young couple dances under the moonlight by the lake.",
        "required": True,
        "description": "Prompt for video generation."
    },
    "save_fps": {
        "type": int,
        "title": "Save Fps",
        "default": 10,
        "required": False,
        "description": "Frame per second for the generated video."
    },
    "ddim_steps": {
        "type": int,
        "title": "Ddim Steps",
        "default": 50,
        "required": False,
        "description": "Number of denoising steps."
    },
    "unconditional_guidance_scale": {
        "type": float,
        "title": "Unconditional Guidance Scale",
        "default": 12,
        "required": False,
        "description": "Classifier-free guidance scale."
    }
}


def get_extension_from_mime(mime_type):
    extension = mimetypes.guess_extension(mime_type)
    return extension

def upload_pix(pil_obj, filename):
    destination_blob_name = f'pixart/{filename}'
    bucket = storage.bucket()
    blob = bucket.blob(destination_blob_name)

    pil_obj.save(filename)
    blob.upload_from_filename(filename)

    # Opt : if you want to make public access from the URL
    blob.make_public()

    logger.info("File uploaded to firebase...")
    return blob.public_url



def run_pixart():
    pipe = PixArtAlphaPipeline.from_pretrained("PixArt-alpha/PixArt-XL-2-1024-MS", torch_dtype=torch.float16)
    pipe = pipe.to('cuda')

    prompt = "A alpaca made of colorful building blocks, cyberpunk"
    result = pipe(prompt)
    # print(result)
    return result.images[0] # .save("image.png")


# cog_session = requests.Session()
# retries = Retry(total=10, backoff_factor=0.1, status_forcelist=[502, 503, 504])
# cog_session.mount('http://', HTTPAdapter(max_retries=retries))

# ----------------------------- Start API Service ---------------------------- #
# Call "python -m cog.server.http" in a subprocess to start the API service.
# subprocess.Popen(["python", "-m", "cog.server.http"])


# ---------------------------------------------------------------------------- #
#                              Automatic Functions                             #
# ---------------------------------------------------------------------------- #
def wait_for_service(url):
    '''
    Check if the service is ready to receive requests.
    '''
    while True:
        try:
            health = requests.get(url, timeout=120)
            status = health.json()["status"]

            if status == "READY":
                time.sleep(1)
                return

        except requests.exceptions.RequestException:
            print("Service not ready yet. Retrying...")
        except Exception as err:
            print("Error: ", err)

        time.sleep(0.2)


# def run_inference(inference_request):
#     '''
#     Run inference on a request.
#     '''
#     response = cog_session.post(url=f'{LOCAL_URL}/predictions',
#                                 json=inference_request, timeout=600)
#     return response.json()


# ---------------------------------------------------------------------------- #
#                                RunPod Handler                                #
# ---------------------------------------------------------------------------- #
def handler(event):
    '''
    This is the handler function that will be called by the serverless.
    '''

    # json = run_inference({"input": event["input"]})

    op = run_pixart()
    url = upload_pix(op, f'{uuid.uuid4()}.png')

    return url


if __name__ == "__main__":
    # wait_for_service(url=f'{LOCAL_URL}/health-check')

    print("Cog API Service is ready. Starting RunPod serverless handler...")

    runpod.serverless.start({"handler": handler})

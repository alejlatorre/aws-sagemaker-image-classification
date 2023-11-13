import io
import json
import logging
import os
import sys

import requests
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logger.addHandler(logging.StreamHandler(sys.stdout))
JSON_CONTENT_TYPE = "application/json"
JPEG_CONTENT_TYPE = "image/jpeg"


def Net():
    model = models.resnet50(pretrained=True)
    for param in model.parameters():
        param.requires_grad = False
    model.fc = nn.Sequential(
        nn.Linear(2048, 256), nn.ReLU(inplace=True), nn.Linear(256, 133)
    )
    return model


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def model_fn(model_dir):
    print("In model_fn. Model directory is -")
    print(model_dir)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Net().to(device)

    with open(os.path.join(model_dir, "model.pth"), "rb") as f:
        print("Loading the dog-classifier model")
        checkpoint = torch.load(f, map_location=device)
        model.load_state_dict(checkpoint)
        print("MODEL-LOADED")
        logger.info("model loaded successfully")
    model.eval()
    return model


def input_fn(request_body, content_type=JPEG_CONTENT_TYPE):
    logger.info("Deserializing the input data.")
    logger.debug(f"Request body CONTENT-TYPE is: {content_type}")
    logger.debug(f"Request body TYPE is: {type(request_body)}")
    if content_type == JPEG_CONTENT_TYPE:
        logger.debug("Loading image from request body")
        return Image.open(io.BytesIO(request_body))

    if content_type == JSON_CONTENT_TYPE:
        logger.debug(f"Request body is: {request_body}")
        request = json.loads(request_body)
        logger.debug(f"Loaded JSON object: {request}")
        url = request["url"]
        img_content = requests.get(url).content
        return Image.open(io.BytesIO(img_content))

    raise Exception(
        "Requested unsupported ContentType in content_type: {}".format(content_type)
    )


def predict_fn(input_object, model):
    logger.info("In predict fn")
    test_transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ]
    )
    logger.info("transforming input")
    input_object = test_transform(input_object)

    with torch.no_grad():
        logger.info("Calling model")
        prediction = model(input_object.unsqueeze(0))
    return prediction

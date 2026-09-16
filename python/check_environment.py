import cv2
import fastapi
import numpy
import torch
import torchvision
import transformers
import ultralytics


print("torch", torch.__version__)
print("cuda_runtime", torch.version.cuda)
print("cuda_available", torch.cuda.is_available())
print("torchvision", torchvision.__version__)
print("numpy", numpy.__version__)
print("opencv", cv2.__version__)
print("transformers", transformers.__version__)
print("ultralytics", ultralytics.__version__)
print("fastapi", fastapi.__version__)

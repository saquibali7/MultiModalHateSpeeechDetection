import os
import numpy as np
import matplotlib.pyplot as plt
import random
from PIL import Image
import cv2
import tqdm
import torch.nn.functional as F
import albumentations as A
import pytesseract
import shutil
import random
from pytesseract import pytesseract
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset,  random_split
from torchvision import transforms
from PIL import Image
import torchvision.models as models
import torchtext
import transformers
from transformers import AutoModel, BertTokenizerFast
from transformers import BertTokenizerFast as BertTokenizer, BertModel, AdamW, get_linear_schedule_with_warmup
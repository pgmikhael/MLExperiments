import torch
import torchvision
from PIL import Image, ImageFile
import os
import sys
import os.path
import warnings
from transformers.transforms import ComposeTrans
ImageFile.LOAD_TRUNCATED_IMAGES = True
import pdb 

class image_loader():
    def __init__(self, img_transformers, tnsr_transformers, args):
        self.img_transformers = img_transformers
        self.tnsr_transformers = tnsr_transformers
        self.composed_all_transformers = ComposeTrans(img_transformers, tnsr_transformers, args)

    def get_image(self, path):
        '''
        Returns a transformed image by its absolute path.
        If cache is used - transformed image will be loaded if available,
        and saved to cache if not.
        '''
        
        image = Image.open(path)
        return self.composed_all_transformers(image)




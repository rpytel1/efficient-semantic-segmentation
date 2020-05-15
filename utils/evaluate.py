from PIL import Image
import os
from fastai import *
from fastai.vision import *


means = imagenet_stats[0]
stds = imagenet_stats[1]

def saveF(self, fn:PathOrStr):
    "Save the image segment to `fn`."
    x = image2np(self.data).astype(np.uint8)
    img = PIL.Image.fromarray(x)
    img = img.resize((2048,1024), resample=PIL.Image.NEAREST)
    img.save(fn)
    
ImageSegment.save = saveF

def save_to_eval(learn, test_databunch, to_send = False):
    if not os.path.isdir('results'):
        os.makedirs('results')
    for (input, target), file_path in zip(test_databunch.valid_ds, test_databunch.valid_ds.items):  
        filename = os.path.basename(file_path)
        pred = learn.predict(input)
        if to_send:
            filename = filename.split(".")[0]
            filename+="_labelsId.png"
        pred[0].save('results/{}'.format(filename)) 


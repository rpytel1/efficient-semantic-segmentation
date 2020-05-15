import os
import shutil
from glob import glob

def sanitize_data(directory):
    images = glob(directory + '/leftImg8bit/*/*.png')

    for img in images:
        new = img.replace("_leftImg8bit", "")
        os.rename(img,new)

    gt = glob(directory + '/gtFine/*/*.png')

    for img in gt:
        new = img.replace("_gtFine_labelIds", "")
        os.rename(img,new)

## Flatten cityscapes
for gfolder in ['gtFine','leftImg8bit']:
    for folder in ['train', 'val','test']:
        dir = os.path.join("data/cityspaces/", gfolder, folder)
        directories=[d for d in os.listdir(dir) if os.path.isdir(os.path.join(dir,d))]

        for d in directories:
            files = os.listdir(os.path.join(dir,d))
            for f in files:
                if f.endswith(".png") or f.endswith(".json"):
                    what = os.path.join(dir,d,f)
                    to = os.path.join(dir)
                    shutil.move(what, to)
                    
    for folder in ['train', 'val','test']:
        dir = os.path.join("data/cityspaces/", gfolder, folder)
        files =  files = os.listdir(dir)
        for f in files:
            if f.endswith("_gtFine_color.png") or f.endswith("_instanceIds.png") or f.endswith("gtFine_polygons.json"):
                os.remove(os.path.join(dir,f))

sanitize_data("data/minicity")
sanitize_data("data/cityspaces")
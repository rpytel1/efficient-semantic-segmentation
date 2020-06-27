import os
import shutil
import random


import argparse
parser = argparse.ArgumentParser(description='Train keypoints network')
    # general
parser.add_argument('--fraction',
                        help='what fraction should be used (0.1, 0.2, 0.5)',
                        required=True,
                        type=float)

args = parser.parse_args()

fract = args.fraction
name = str(fract).replace(".","")
name = "cityscapes" + name
os.mkdir(os.path.join("data",name))
os.mkdir(os.path.join("data",name,"leftImg8bit"))
os.mkdir(os.path.join("data",name,"leftImg8bit","train"))
val_path = os.path.join("data",name,"leftImg8bit","val")
# os.mkdir(val_path)
test_path = os.path.join("data",name,"leftImg8bit","test") 
# os.mkdir(test_path)

os.mkdir(os.path.join("data",name,"gtFine"))
os.mkdir(os.path.join("data",name,"gtFine","train"))
val_path_gt = os.path.join("data",name,"gtFine","val")
# os.mkdir(val_path_gt)
test_path_gt = os.path.join("data",name,"gtFine","test") 
# os.mkdir(test_path_gt)

for city in os.listdir("data/cityscapes/leftImg8bit/train"):
    # TRAIN
    files = os.listdir(os.path.join("data/cityscapes/leftImg8bit/train",city))
    num_samples = int(len(files) * fract)
    chosen_imgs = random.sample(files, num_samples)
    
    #Imgs
    path = os.path.join("data/cityscapes/leftImg8bit/train",city)
    dest_path =  os.path.join("data",name,"leftImg8bit","train",city)
    os.mkdir(os.path.join("data",name,"leftImg8bit","train", city))
         
    for img in chosen_imgs:
         shutil.copy(os.path.join(path,img),os.path.join(dest_path, img))
    
    #GT
    path = os.path.join("data/cityscapes/gtFine/train",city)
    dest_path =  os.path.join("data",name,"gtFine","train",city)
    os.mkdir(os.path.join("data",name,"gtFine","train", city))
         
    chosen_imgs = [elem.replace("leftImg8bit.png","gtFine_color.png") for elem in chosen_imgs]
    for img in chosen_imgs:
         shutil.copy(os.path.join(path,img),os.path.join(dest_path, img))
            
    chosen_imgs = [elem.replace("gtFine_color.png","gtFine_labelIds.png") for elem in chosen_imgs]
    for img in chosen_imgs:
         shutil.copy(os.path.join(path,img),os.path.join(dest_path, img))
         
# VAL
shutil.copytree("data/cityscapes/leftImg8bit/val",val_path)
shutil.copytree("data/cityscapes/gtFine/val",val_path_gt)       
# TEST
shutil.copytree("data/cityscapes/leftImg8bit/test",test_path)       
shutil.copytree("data/cityscapes/gtFine/test",test_path_gt)       
   

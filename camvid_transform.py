import os
import numpy as np
from multiprocessing import Process
from collections import namedtuple
from PIL import Image

# Run this script for train, val and test
LABELS_FILE_PATH = 'data/CamVid/val_labels'
OUTPUT_PATH = 'data/CamVid/val_classes'
NUM_PROCESSES = 8


Label_CV = namedtuple( 'Label' , [

    'name'        , # The identifier of this label, e.g. 'car', 'person', ... .
                    # We use them to uniquely name a class

    'id'          , # An integer ID that is associated with this label.
                    # The IDs are used to represent the label in ground truth images
                    # An ID of -1 means that this label does not have an ID and thus
                    # is ignored when creating ground truth images (e.g. license plate).
                    # Do not modify these IDs, since exactly these IDs are expected by the
                    # evaluation server.

    'color'       , # The color of this label
    ])


labels = [
    #       name          id     color
    Label_CV('Void',              0 ,  (0, 0, 0)),
    Label_CV('Animal',            1 ,  (64, 128, 64)),
    Label_CV('Archway',           2 ,  (192, 0, 128)),
    Label_CV('Bicyclist',         3 ,  (0, 128, 192)),
    Label_CV('Bridge',            4 ,  (0, 128, 64)),
    Label_CV('Building',          5 ,  (128, 0, 0)),
    Label_CV('Car',               6 ,  (64, 0, 128)),
    Label_CV('CartLuggagePram',   7 ,  (64, 0, 192)),
    Label_CV('Child',             8 ,  (192, 128, 64)),
    Label_CV('Column_Pole',       9 ,  (192, 192, 128)),
    Label_CV('Fence',             10 ,  (64, 64, 128)),
    Label_CV('LaneMkgsDriv',      11 , (128, 0, 192)),
    Label_CV('LaneMkgsNonDriv',   12 , (192, 0, 64)),
    Label_CV('Misc_Text',         13 , (128, 128, 64)),
    Label_CV('MotorcycleScooter', 14 , (192, 0, 192)),
    Label_CV('OtherMoving',       15 , (128, 64, 64)),
    Label_CV('ParkingBlock',      16 , (64, 192, 128)),
    Label_CV('Pedestrian',        17 , (64, 64, 0)),
    Label_CV('Road',              18 , (128, 64, 128)),
    Label_CV('RoadShoulder',      19 , (128, 128, 192)),
    Label_CV('Sidewalk',          20 , (0, 0, 192)),
    Label_CV('SignSymbol',        21 , (192, 128, 128)),
    Label_CV('Sky',               22 , (128, 128, 128)),
    Label_CV('SUVPickupTruck',    23 , (64, 128, 192)),
    Label_CV('TrafficCone',       24 , (0, 0, 64)),
    Label_CV('TrafficLight',      25 , (0, 64, 64)),
    Label_CV('Train',             26 , (192, 64, 128)),
    Label_CV('Tree',              27 , (128, 128, 0)),
    Label_CV('Truck_Bus',         28 , (192, 128, 192)),
    Label_CV('Tunnel',            29 , (64, 0, 64)),
    Label_CV('VegetationMisc',    30 , (192, 192, 0)),
    Label_CV('Wall',              31 , (64, 192, 0)),
]

color2id = {label.color: label.id for label in labels}


def process_data():
    """ Spawns processes the process the data in parallel. """
    path = list(os.walk(LABELS_FILE_PATH))
    data = split_paths(path)
    processes = []

    for x in range(NUM_PROCESSES):
        files = data[x]
        p = Process(target=process_thread, args=(str(x), files,))
        processes.append(p)
        p.start()

    for x in range(NUM_PROCESSES):
        processes[x].join()


def split_paths(path):
    """ Splits the list of files for each process."""
    splitted = []
    for i in range(NUM_PROCESSES):
        splitted.append([])

    for i in range(len(path[0][2])):
        file = path[0][2][i]
        splitted[i % NUM_PROCESSES].append(file)

    return splitted


def colors2labelIds(labelColors):
    shape = labelColors.shape[0:2]
    target = np.zeros(shape=shape)
    for i in range(shape[0]):
        for j in range(shape[1]):
            t = tuple(labelColors[i][j])
            if t in color2id:
                target[i][j] = color2id[t]

    return target


def process_thread(name, files):
    """ Reads the image with colors as lables and transforms it to image with ids. """
    i = 0
    for file in files:
        if i % 20 == 0:
            print(f'Process {name} finished {i} images.')

        colors = np.array(Image.open(LABELS_FILE_PATH + '/' + file))
        labels = colors2labelIds(colors)
        im = Image.fromarray(np.uint8(labels))
        im.save(OUTPUT_PATH + '/' + file)
        i += 1


if __name__ == '__main__':
    process_data()

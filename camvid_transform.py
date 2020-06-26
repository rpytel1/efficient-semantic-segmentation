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
    Label_CV('CartLuggagePram',   6 ,  (64, 0, 192)),
    Label_CV('Child',             7 ,  (192, 128, 64)),
    Label_CV('Column_Pole',       8 ,  (192, 192, 128)),
    Label_CV('Fence',             9 ,  (64, 64, 128)),
    Label_CV('LaneMkgsDriv',      10 , (128, 0, 192)),
    Label_CV('LaneMkgsNonDriv',   11 , (192, 0, 64)),
    Label_CV('Misc_Text',         12 , (128, 128, 64)),
    Label_CV('MotorcycleScooter', 13 , (192, 0, 192)),
    Label_CV('OtherMoving',       14 , (128, 64, 64)),
    Label_CV('ParkingBlock',      15 , (64, 192, 128)),
    Label_CV('Pedestrian',        16 , (64, 64, 0)),
    Label_CV('Road',              17 , (128, 64, 128)),
    Label_CV('RoadShoulder',      18 , (128, 128, 192)),
    Label_CV('Sidewalk',          19 , (0, 0, 192)),
    Label_CV('SignSymbol',        20 , (192, 128, 128)),
    Label_CV('Sky',               21 , (128, 128, 128)),
    Label_CV('SUVPickupTruck',    22 , (64, 128, 192)),
    Label_CV('TrafficCone',       23 , (0, 0, 64)),
    Label_CV('TrafficLight',      24 , (0, 64, 64)),
    Label_CV('Train',             25 , (192, 64, 128)),
    Label_CV('Tree',              26 , (128, 128, 0)),
    Label_CV('Truck_Bus',         27 , (192, 128, 192)),
    Label_CV('Tunnel',            28 , (64, 0, 64)),
    Label_CV('VegetationMisc',    29 , (192, 192, 0)),
    Label_CV('Wall',              30 , (64, 192, 0)),
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

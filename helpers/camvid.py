import os
import numpy as np
import torch.utils.data as data
from .helpers import pil_loader, get_files
from collections import namedtuple

# a label and all meta information
Label_CV = namedtuple( 'Label' , [

    'name'        , # The identifier of this label, e.g. 'car', 'person', ... .
                    # We use them to uniquely name a class

    'id'          , # An integer ID that is associated with this label.
                    # The IDs are used to represent the label in ground truth images

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


class CamVid(data.Dataset):
    """CamVid dataset loader.
    Keyword arguments:
    - root_dir (``string``): Root directory path.
    - mode (``string``): The type of dataset: 'train' for training set, 'val'
    for validation set, and 'test' for test set.
    - transform (``callable``, optional): A function/transform that takes in
    an PIL image and labels and returns a transformed version of both of them. Default: None.
    - loader (``callable``, optional): A function to load an image given its
    path. By default ``default_loader`` is used.
    """
    # Training dataset root folders
    train_folder = 'train'
    train_lbl_folder = 'train_classes'

    # Validation dataset root folders
    val_folder = 'val'
    val_lbl_folder = 'val_classes'

    # Test dataset root folders
    test_folder = 'test'
    test_lbl_folder = 'test_classes'

    # Images extension
    img_extension = '.png'

    color2id = {label.color: label.id for label in labels}

    # Convert ids to colors
    mask_colors = [list(label.color) for label in labels if label.id >= 0]
    mask_colors = np.array(mask_colors)

    # List of valid class ids
    validClasses = np.unique([label.id for label in labels if label.id >= 0])
    validClasses = list(validClasses)

    # Create list of class names
    classLabels = [label.name for label in labels]

    voidClass = 0

    def __init__(self,
                 root_dir,
                 mode='train',
                 transforms=None,
                 loader=pil_loader):
        self.root_dir = root_dir
        self.mode = mode
        self.transforms = transforms
        self.loader = loader

        if self.mode.lower() == 'train':
            # Get the training data and labels filepaths
            self.train_data = get_files(
                os.path.join(root_dir, self.train_folder),
                extension_filter=self.img_extension)

            self.train_labels = get_files(
                os.path.join(root_dir, self.train_lbl_folder),
                extension_filter=self.img_extension)
        elif self.mode.lower() == 'val':
            # Get the validation data and labels filepaths
            self.val_data = get_files(
                os.path.join(root_dir, self.val_folder),
                extension_filter=self.img_extension)

            self.val_labels = get_files(
                os.path.join(root_dir, self.val_lbl_folder),
                extension_filter=self.img_extension)
        elif self.mode.lower() == 'test':
            # Get the test data and labels filepaths
            self.test_data = get_files(
                os.path.join(root_dir, self.test_folder),
                extension_filter=self.img_extension)

            self.test_labels = get_files(
                os.path.join(root_dir, self.test_lbl_folder),
                extension_filter=self.img_extension)
        else:
            raise RuntimeError("Unexpected dataset mode. "
                               "Supported modes are: train, val and test")

    def __getitem__(self, index):
        """
        Args:
        - index (``int``): index of the item in the dataset
        Returns:
        A tuple of ``PIL.Image`` (image, label) where label is the ground-truth
        of the image.
        """
        if self.mode.lower() == 'train':
            data_path, label_path = self.train_data[index], self.train_labels[
                index]
        elif self.mode.lower() == 'val':
            data_path, label_path = self.val_data[index], self.val_labels[
                index]
        elif self.mode.lower() == 'test':
            data_path, label_path = self.test_data[index], self.test_labels[
                index]
        else:
            raise RuntimeError("Unexpected dataset mode. "
                               "Supported modes are: train, val and test")

        img, label = self.loader(data_path, label_path)

        if self.transforms is not None:
            if self.mode.lower() != 'test':
                img, label = self.transforms(img, mask=label)

                return img, label, data_path
            else:
                img = self.transforms(img)

                return img, data_path

    def __len__(self):
        """Returns the length of the dataset."""
        if self.mode.lower() == 'train':
            return len(self.train_data)
        elif self.mode.lower() == 'val':
            return len(self.val_data)
        elif self.mode.lower() == 'test':
            return len(self.test_data)
        else:
            raise RuntimeError("Unexpected dataset mode. "
                               "Supported modes are: train, val and test")


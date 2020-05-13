# Efficient semantic segmentation
## Data preparation
In premise of the project we use Minicity dataset for training, which is subset of Cityscapes. It can be created using 
instructions [here.](https://github.com/VIPriors/vipriors-challenges-toolkit) 
For evaluating performance we use validation set of CityScapes.

To prepare for training put both datasets at ``data`` and run:
```bash
python prepare_data.py 
```
## Training 
Check notebooks.
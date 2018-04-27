# FBPConvNet

Includes basic example script to train and visualize

## Dependencies
Heavily recommended to use anaconda
- `Python 3.6`
- [Pytorch](http://pytorch.org/)
- `numpy`
- `matplotlib.pyplot`
- `h5py` (might be hdf5 in conda)

## Contents
- `FBPConvNet.ipynb` - iPython notebook with basic train, test and visualization
- `main.py` - Same thing but in python script

## How to run main.py
Download data [here](https://gtvault-my.sharepoint.com/:u:/r/personal/jjin77_gatech_edu/Documents/RombergSP/FBPConvNet/data/LineEllipses20.hdf5?csf=1&e=GDeQrC). Adjust the script in main.py to point to it. Currently is set to 
```
pathtodata = '../data/LineEllipses20.hdf5'
```

To train net from scratch:
```
python main.py -m train [-bs BATCH_SIZE]
```

To use already trained weights(included)
```
python main.py
```

# Freehand ultrasound without external trackers

This repository contains algorithms to train deep neural networks, using scans of freehand ultrasound image frames acquired with ground-truth frame locations from external spatial trackers. The aim is to reconstruct the spatial frame locations or relative transformation between them, on the newly acquired scans.

The most up-to-date code is in the `dev0` branch, where the `train.py` and `test.py` under the `scripts` folder can be adapted with local data path. The conda environment required to run the code is detailed in [requirements](/doc/requirements.md).

The link to the data used in paper ``Qi. et al. Trackerless freehand ultrasound with sequence modelling and auxiliary transformation over past and future frames'' will be made available here upon publication (due to local regulatory requirement).

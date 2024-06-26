# Freehand ultrasound without external trackers

This repository contains algorithms to train deep neural networks, using scans of freehand ultrasound image frames acquired with ground-truth frame locations from external spatial trackers. The aim is to reconstruct the spatial frame locations or relative transformation between them, on the newly acquired scans.

The most up-to-date code is in the `dev0` branch, where the `train.py` and `test.py` under the `scripts` folder can be adapted with local data path. The conda environment required to run the code is detailed in [requirements](/doc/requirements.md).

The data used in the following papers can be downloaded [here](https://doi.org/10.5281/zenodo.7740734).
We have collected a new large freehand ultrasound dataset and are organising a MICCAI2024 Challenge [TUS-REC Challenge](https://github-pages.ucl.ac.uk/tus-rec-challenge/). Check [Part 1](https://zenodo.org/records/11178509) and [Part 2](https://zenodo.org/records/11180795) of the training dataset. 

If you find this code or data set useful for your research, please consider citing the following works:

- Qi Li, Ziyi Shen, Qian Li, Dean C. Barratt, Thomas Dowrick, Matthew J. Clarkson, Tom Vercauteren, and Yipeng Hu. "Trackerless freehand ultrasound with sequence modelling and auxiliary transformation over past and future frames." In 2023 IEEE 20th International Symposium on Biomedical Imaging (ISBI), pp. 1-5. IEEE, 2023. doi: [10.1109/ISBI53787.2023.10230773](https://doi.org/10.1109/ISBI53787.2023.10230773)

- Qi Li, Ziyi Shen, Qian Li, Dean C. Barratt, Thomas Dowrick, Matthew J. Clarkson, Tom Vercauteren, and Yipeng Hu. "Long-term Dependency for 3D Reconstruction of Freehand Ultrasound Without External Tracker." IEEE Transactions on Biomedical Engineering, vol. 71, no. 3, pp. 1033-1042, 2024. doi: [10.1109/TBME.2023.3325551](https://ieeexplore.ieee.org/abstract/document/10288201).

- Qi Li, Ziyi Shen, Qian Li, Dean C. Barratt, Thomas Dowrick, Matthew J. Clarkson, Tom Vercauteren, and Yipeng Hu. "Privileged Anatomical and Protocol Discrimination in Trackerless 3D Ultrasound Reconstruction." In International Workshop on Advances in Simplifying Medical Ultrasound, pp. 142-151. Cham: Springer Nature Switzerland, 2023. doi: [https://doi.org/10.1007/978-3-031-44521-7_14](https://doi.org/10.1007/978-3-031-44521-7_14)
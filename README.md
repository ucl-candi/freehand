# Freehand ultrasound without external trackers

This repository contains algorithms to train deep neural networks, using scans of freehand ultrasound image frames acquired with ground-truth frame locations from external spatial trackers. The aim is to reconstruct the spatial frame locations or relative transformation between them, on the newly acquired scans.

The most up-to-date code is in the `dev0` branch, where the `train.py` and `test.py` under the `scripts` folder can be adapted with local data path. The conda environment required to run the code is detailed in [requirements](/doc/requirements.md).

The data used in the following papers can be downloaded [here](https://doi.org/10.5281/zenodo.7740734).

If you find this code or data set useful for your research, please consider citing the following works:

Qi Li, Ziyi Shen, Qian Li, Dean C. Barratt, Thomas Dowrick, Matthew J. Clarkson, Tom Vercauteren, and Yipeng Hu. "Trackerless freehand ultrasound with sequence modelling and auxiliary transformation over past and future frames." In 2023 IEEE 20th International Symposium on Biomedical Imaging (ISBI), pp. 1-5. IEEE, 2023.

Qi Li, Ziyi Shen, Qian Li, Dean C. Barratt, Thomas Dowrick, Matthew J. Clarkson, Tom Vercauteren, and Yipeng Hu. "Long-term Dependency for 3D Reconstruction of Freehand Ultrasound Without External Tracker." IEEE Transactions on Biomedical Engineering (2023).

Qi Li, Ziyi Shen, Qian Li, Dean C. Barratt, Thomas Dowrick, Matthew J. Clarkson, Tom Vercauteren, and Yipeng Hu. "Privileged Anatomical and Protocol Discrimination in Trackerless 3D Ultrasound Reconstruction." In International Workshop on Advances in Simplifying Medical Ultrasound, pp. 142-151. Cham: Springer Nature Switzerland, 2023.
# Freehand ultrasound without external trackers

This repository contains algorithms to train deep neural networks, using scans of freehand ultrasound image frames acquired with ground-truth frame locations from external spatial trackers. The aim is to reconstruct the spatial frame locations or relative transformation between them, on the newly acquired scans.

<!-- The most up-to-date code is in the `dev1` branch, where the `train.py` and `test.py` under the `scripts` folder can be adapted with local data path.  -->


The data can be downloaded [here](https://doi.org/10.5281/zenodo.7740734).
We have collected a new large freehand ultrasound dataset and are organising a MICCAI2024 Challenge [TUS-REC Challenge](https://github-pages.ucl.ac.uk/tus-rec-challenge/). Check [Part 1](https://zenodo.org/records/11178509) and [Part 2](https://zenodo.org/records/11180795) of the training dataset. 


## Steps to run the code
### 1. Clone the repository.
```
git clone https://github.com/ucl-candi/freehand.git
```

### 2. Navigate to the root directory.
```
cd freehand
```

<!-- ### 3. Switch to dev1.
```
git checkout dev1
``` -->

### 3. Install conda environment

``` bash
conda create -n FUS python=3.9.13
conda activate FUS
pip install -r requirements.txt
```

<!-- ### 5. Create directories.
```
mkdir -p data/Freehand_US_data
``` -->


### 4. Download data and put `Freehand_US_data.zip` into `./data` directory. (You may need to install `zenodo_get`)

```
pip3 install zenodo_get
zenodo_get 7740734
mv Freehand_US_data.zip ./data
```

### 5. Unzip.
Unzip `Freehand_US_data.zip` into `./data/Freehand_US_data` directory.

```
unzip data/Freehand_US_data.zip -d ./data
```
### 6. Make sure the data folder structure is the same as follows.
```bash
├── data/ 
│ ├── Freehand_US_data/ 
│  ├── 000/
│    ├── *.mha
│    ├── ...
│  ├── ...
│  ├── 018/ 
```

### 7. Data processing (Generate one `.h5` file, using downloaded `.mha` files)

```
python data/prep.py
```

### 8. Train model

```
python scripts/train.py
```


### 9. Test model

```
python scripts/test.py
```


If you find this code or data set useful for your research, please consider citing some of the following works:

* Qi Li, Ziyi Shen, Qian Li, Dean C. Barratt, Thomas Dowrick, Matthew J. Clarkson, Tom Vercauteren, and Yipeng Hu. "Trackerless freehand ultrasound with sequence modelling and auxiliary transformation over past and future frames." In 2023 IEEE 20th International Symposium on Biomedical Imaging (ISBI), pp. 1-5. IEEE, 2023. doi: [10.1109/ISBI53787.2023.10230773](https://doi.org/10.1109/ISBI53787.2023.10230773).
* Qi Li, Ziyi Shen, Qianye Yang, Dean C. Barratt, Matthew J. Clarkson, Tom Vercauteren, and Yipeng Hu. "Nonrigid Reconstruction of Freehand Ultrasound without a Tracker." In International Conference on Medical Image Computing and Computer-Assisted Intervention, pp. 689-699. Cham: Springer Nature Switzerland, 2024. doi: [10.1007/978-3-031-72083-3_64](https://doi.org/10.1007/978-3-031-72083-3_64).
* Qi Li, Ziyi Shen, Qian Li, Dean C. Barratt, Thomas Dowrick, Matthew J. Clarkson, Tom Vercauteren, and Yipeng Hu. "Long-term Dependency for 3D Reconstruction of Freehand Ultrasound Without External Tracker." IEEE Transactions on Biomedical Engineering, vol. 71, no. 3, pp. 1033-1042, 2024. doi: [10.1109/TBME.2023.3325551](https://ieeexplore.ieee.org/abstract/document/10288201).
* Qi Li, Ziyi Shen, Qian Li, Dean C. Barratt, Thomas Dowrick, Matthew J. Clarkson, Tom Vercauteren, and Yipeng Hu. "Privileged Anatomical and Protocol Discrimination in Trackerless 3D Ultrasound Reconstruction." In International Workshop on Advances in Simplifying Medical Ultrasound, pp. 142-151. Cham: Springer Nature Switzerland, 2023. doi: [https://doi.org/10.1007/978-3-031-44521-7_14](https://doi.org/10.1007/978-3-031-44521-7_14).

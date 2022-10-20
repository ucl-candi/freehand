# Tracked Ultrasound


# for running the train.py script on GPU and CPU, respectively:
conda create -n tracked-train pytorch torchvision h5py cudatoolkit=10.2 -c pytorch  

conda create -n tracked-train pytorch torchvision h5py cpuonly -c pytorch


# for running the test.py script, add matplotlib 
conda create -n tracked-test matplotlib pytorch torchvision h5py cudatoolkit=10.2 -c pytorch  

conda create -n tracked-test matplotlib pytorch torchvision h5py cpuonly -c pytorch


# all other dependencies are added here

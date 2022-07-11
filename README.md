# GraphEvo
data and code for the paper 'Evolution-strengthened knowledge graph enables predicting the targetability and druggability of genes'

![AppVeyor](https://img.shields.io/badge/tensorflow-1.13.1-brightgreen)
![AppVeyor](https://img.shields.io/badge/python-3.7.10-blue)
![AppVeyor](https://img.shields.io/badge/numpy-1.19.1-red)

## Data list
**target-disease.txt**:The interaction between targets and diseases \
**train-num-TransE_l2.txt**:The knowledge graph embedding of targets in train dataset\
**test-num-TransE_l2.txt**:The knowledge graph embedding of targets in test dataset 

You can replace the above data files with your own data

## Run code
If you directly use the target-disease association feature of targets we provide in file **targetf.npy**:
```
python GraphEvo.py
```

If you want to train the target-disease association feature of targets yourself, you can use the following command:
```
python gcn.py
```
**gcn.py**is able to automatically generate target and disease lists from the provided target-disease associations in **target-disease.txt**, then you can use the following command to train the model.
```
python GraphEvo.py
```

# Task 3: Pattern deviation probability map prediction

The official pytorch baseline for Task 3 of STAGE. 

The pipeline of this simple baseline is as follows: We use Resnet to encode OCT features and concatenate them with learnable data_info embeddings. Subsequently, the concatenation are fed into MLP for ordinal regression.

# Usage
- Please download the STAGE datasets by yourself and put them under `.data/`.
- Please set the configuration parameters in file `lib/config.py`.

## Train
```
python baseline.py
```
## Evaluation
```
python infer.py
```
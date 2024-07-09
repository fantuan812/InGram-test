# InGram: Inductive Knowledge Graph Embedding via Relation Graphs
This code is the official implementation of the following [paper](https://proceedings.mlr.press/v202/lee23c.html):

> Jaejun Lee, Chanyoung Chung, and Joyce Jiyoung Whang, InGram: Inductive Knowledge Graph Embedding via Relation Graphs, The 40th International Conference on Machine Learning (ICML), 2023.


## Requirements

We used Python 3.8 and PyTorch 1.12.1 with cudatoolkit 11.3.

You can install all requirements with:

```shell
pip install -r requirements.txt
```

## Reproducing the Reported Results

The command to reproduce the results in our paper:

```python
python test.py --data_name [dataset_name] -d_e [dimension_entity] -d_r [dimension_relation] --best
```
the independent results
```python
python multitest.py --data_name [dataset_name] -d_e [dimension_entity] -d_r [dimension_relation] --single --best
```
the federated results
```python
python multitest.py --data_name [dataset_name] -d_e [dimension_entity] -d_r [dimension_relation] --best
```

## Training from Scratch

To train InGram from scratch, run `train.py` with arguments. Please refer to `my_parser.py` for the examples of the arguments. Please tune the hyperparameters of our model using the range provided in Appendix C of the paper because the best hyperparameters may be different due to randomness.

independent train
```python
python single_train.py --data_name [dataset_name] -d_e [dimension_entity] -d_r [dimension_relation]
```

federated train
```python
python aggtest.py --data_name [dataset_name] -d_e [dimension_entity] -d_r [dimension_relation]
```

if want train only one client
```python
python train.py --data_name [dataset_name] -d_e [dimension_entity] -d_r [dimension_relation]
```

The list of arguments of `train.py`:
- `--data_name`: name of the dataset
- `--exp`: experiment name
- `-m, --margin`: $\gamma$
- `-lr, --learning_rate`: learning rate
- `-nle, --num_layer_ent`: $\widehat{L}$
- `-nlr, --num_layer_rel`: $L$
- `-d_e, --dimension_entity`: $\widehat{d}$
- `-d_r, --dimension_relation`: $d$
- `-hdr_e, --hidden_dimension_ratio_entity`: $\widehat{d'}/\widehat{d}$
- `-hdr_r, --hidden_dimension_ratio_relation`: $d'/d$
- `-b, --num_bin`: $B$
- `-e, --num_epoch`: number of epochs to run
- `--target_epoch`: the epoch to run test (only used for test.py)
- `-v, --validation_epoch`: duration for the validation
- `--num_head`: $\widehat{K}=K$
- `--num_neg`: number of negative triplets per triplet
- `--best`: use the best checkpoints 
- `--no_write`: don't save the checkpoints (only used for train.py)
- `--client_num`: client number
- `--single`: use Independent checkpoints (only used for multitest.py)

# InGram

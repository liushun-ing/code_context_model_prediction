# GNN based Code Context Prediction

This repository provides the code for xxx. 
The environment can be built following the commands in the Singularity file. 
To start, clone the repository and navigate to the root directory.

## Directory Structure

```
code context model prediction
│
├── ( dataset collection )
│         ├── data_count
│         ├── data_extraction
│         ├── params_validation
│         ├── xmlparser
│         ├── model_expansiondataset_split_util
│         ├── dataset_replication
│         └── dataset_split_util
├── ( our prediction approach )
│         ├── ( GNN model )
│         │         └── GNN_Node_Classification
│         └── ( code embedding approach )
│                   ├── astnn_embedding
│                   ├── codebert_embedding
│                   ├── glove_embedding
│                   └── my_embedding
└── (RQ and baseline)
          ├── RQ_1
          ├── RQ_2
          ├── RQ_4
          └── baseline
```







# GNN based Code Context Model Prediction

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



Our experiments were all conducted using the **PyCharm** development tool.



## Dataset

### Data Collection

Run the `main.py` file under the `data_extraction/` directory to fetch bug records from the Eclipse Bugzilla data website. This program will create a `bug_dataset` directory in the project's root directory and download the bugs into this directory. Our program includes an automatic retry mechanism for failures, but some bugs may still fail to download. If this occurs, you can set `index = xxx` in the code to skip processing the bug at that index, as indicated by the console output. After the data collection is complete, you will see a directory structure similar to the following example:

```
bug_dataset
└── mylyn_zip
     ├── Mylyn
     │     └── 102663
     │						└── 102263_42671.zip
     ├── ECF
     ├── PDE
     └── Platfrom
```



### Data Cleaning

> **Decompress**

Run the `01_zip_out.py` file under the `data_count/` directory to decompress the zip files in the directory structure mentioned above. For example, `102263_42671.zip` will be decompressed to `102263_42671_zip`.

> **Split working periods**

Run the `2_periods_break.py` file under the `params_validation/working_periods/` directory. This script reads each `bug` in the `bug_dataset` directory and splits the working periods according to different time intervals. Each working period will generate an XML file, resulting in a directory structure similar to the following example within the `params_validation/working_periods` directory:

```bug_dataset
working_periods
└── periods
     ├── 00
     │		├── ...
     │    └── Mylyn
     │					└── 1.xml
     ├── ...
     └── 09
```

> **Filter working periods**

Run the `4_periods_filter.py` file under the `params_validation/working_periods/` directory to filter the working periods split in the previous step. You will see the following output upon execution.

```
working_periods
└── code_elements
     ├── 00
     │		├── ...
     │    └── Mylyn
     │					└── 1.xml
     ├── ...
     └── 09
```

> **Extract code elements**

Run the `1_extract_code_elements_and_timestamp.py` file under the `params_validation/repo_vs_commit_order/` directory to extract code elements and timestamps from the filtered working periods. You will see the following output upon execution:

```
repo_vs_commit_order
└── code_timestamp
     └── 05
     		 ├── ...
         └── Mylyn
     					└── 1.xml
```

> **Calculate IQR and filter outliers**

Run the `2_quartile_IQR.py` file under the `params_validation/repo_vs_commit_order/` directory to calculate the values of `Q1 - 3 * IQR` and `Q3 + 3 * IQR` for each project's output result. Update these values in the `3_IQR_filter.py` file (modify the parameters passed to the `main_func` method call) and then execute the `3_IQR_filter.py` file. You will see the following output upon execution:

```
repo_vs_commit_order
└── IQR_code_timestamp
     └── 05
     		 ├── ...
         └── Mylyn
     					└── 1.xml
```

> **Clone github repository**

According to the instructions in `params_validation/git_repos.txt`, clone the corresponding repositories to your local machine. After cloning, the directory structure should resemble the following:

```
params_validation
└── git_repo_code
     ├── mylyn
     │		├── org.eclipse.mylyn
     │		├── org.eclipse.mylyn.builds
     │		├── org.eclipse.mylyn.commons
     │		├── org.eclipse.mylyn.context
     │		├── org.eclipse.mylyn.context.mft
     │		├── org.eclipse.mylyn.docs
     │		├── org.eclipse.mylyn.incubator
     │		├── org.eclipse.mylyn.reviews
     │		├── org.eclipse.mylyn.tasks
     │		└── org.eclipse.mylyn.versions
     ├── platform
     │		├── eclipse.platform
     │		├── eclipse.platform.swt
     │		├── eclipse.platform.ui
     │		└── eclipse.platform.releng.buildtools
     ├── ecf
     │		└── ecf
     └── pde
     			└── eclipse.pde
```

> **Construct code context model**

Run the `4_extract_model_repo_first.py` file under the `params_validation/repo_vs_commit_order/` directory. After execution, you will find the generated `code context model` dataset files in the `git_repo_code` directory.

```
params_validation
└── git_repo_code
     ├── my_mylyn
     │		└── 42
     │				 ├── doxygen (Doxygen Parsing File)
     │				 			├── org.eclipse.mylyn.tasks.tests
     │				 			└── org.eclipse.mylyn.tasks.ui
     │				 ├── org.eclipse.mylyn.tasks.tests (Source File)
     │				 ├── org.eclipse.mylyn.tasks.ui (Source File)
     │				 └── code_context_model.xml (Code Context Model File)
     ├── my_platform
     ├── my_ecf
     └── my_pde
```

We have organized the source code files and the code context model files at the following address.



### Dataset Details



## Research Questions



### RQ1: Compare to baseline

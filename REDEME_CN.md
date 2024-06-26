# 基于图神经网络的代码上下文模型预测

This repository provides the code for xxx. 
The environment can be built following the commands in the Singularity file. 
To start, clone the repository and navigate to the root directory.



## 目录结构

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



我们的实验均在 Pycharm 开发者工具中进行。



## 数据

### 数据采集

1.1 说明数据采集的步骤，相关的代码文件，运行方法，输入/输出文件夹

运行 `data_extraction/` 下的 `main.py` 文件， 从 Eclipse bugzilla_data 网站拉取 bug 记录，该程序会在项目的根目录下创建`bug_dataset`目录，并将 bug 下载到该目录下。我们的程序设置了失败自动重试机制，但仍然可能遇到某些 bug 无法下载的情况，如果出现这种情况，可以根据控制台输出，设置代码中的`index = xxx`，跳过该序号的 bug 的处理。数据收集完后，你将看到如下示例的目录结构：

```
bug_dataset
└── mylyn_zip
     ├── Mylyn
     │     └── 102663
     │           └── 102263_42671.zip
     ├── ECF
     ├── PDE
     └── Platfrom
```

### 数据清洗

1.2 说明数据清洗的步骤，相关的代码文件，运行方法，输入/输出文件夹

> 解压缩

运行`data_count/` 下的 `01_zip_out.py`文件，会将上述目录结构中的 zip 文件解压缩，例如：`102263_42671.zip` --> `102263_42671_zip`.

> 拆分working periods

运行`params_validation/working_periods/` 下的 `2_periods_break.py`文件，读取上述 `bug_dataset` 中的每一个 `bug`，按照不同的时间间隔进行 working periods 的划分，每个 working periods 将得到一个 xml 文件，将会在 `params_validation/working_periods` 目录生成如下示例的目录结构：

```bug_dataset
working_periods
└── periods
     ├── 00
     │	  ├── ...
     │    └── Mylyn
     │          └── 1.xml
     ├── ...
     └── 09
```

> 过滤 working periods

运行 `params_validation/working_periods/` 下的 `4_periods_filter.py` 文件，对上一步拆分的 working periods 进行过滤，将看到如下的运行结果。

```
working_periods
└── code_elements
     ├── 00
     │	  ├── ...
     │    └── Mylyn
     │          └── 1.xml
     ├── ...
     └── 09
```

> 提取 `code elements` 和 `timestamp`

运行 `params_validation/repo_vs_commit_order/` 下的`1_extract_code_elements_and_timestamp.py` 文件，对上一步过滤后的 working periods 进行 code elements 的提取，将看到如下运行结果：

```
repo_vs_commit_order
└── code_timestamp
     └── 05
     	 ├── ...
         └── Mylyn
     	        └── 1.xml
```

> 计算 IQR，进一步过滤

运行 `params_validation/repo_vs_commit_order/` 下的 `2_quartile_IQR.py`，将各个项目输出结果的 `Q1 - 3 * IQR 和 
Q3 + 3 * IQR]`值，更新到 `3_IQR_filter.py` 文件中（修改 `main_func` 方法调用时的参数），并运行该文件，将看到如下运行结果：

```
repo_vs_commit_order
└── IQR_code_timestamp
     └── 05
     	 ├── ...
         └── Mylyn
     	        └── 1.xml
```

> 拉取 `GitHub repository`

根据 `params_validation/git_repos.txt` 中的提示，克隆对应的仓库到本地，克隆结束应当形成如下目录结构：

```
params_validation
└── git_repo_code
     ├── mylyn
     │      ├── org.eclipse.mylyn
     │      ├── org.eclipse.mylyn.builds
     │      ├── org.eclipse.mylyn.commons
     │      ├── org.eclipse.mylyn.context
     │      ├── org.eclipse.mylyn.context.mft
     │      ├── org.eclipse.mylyn.docs
     │      ├── org.eclipse.mylyn.incubator
     │      ├── org.eclipse.mylyn.reviews
     │      ├── org.eclipse.mylyn.tasks
     │      └── org.eclipse.mylyn.versions
     ├── platform
     │      ├── eclipse.platform
     │      ├── eclipse.platform.swt
     │      ├── eclipse.platform.ui
     │	    └── eclipse.platform.releng.buildtools
     ├── ecf
     │	  └── ecf
     └── pde
     	  └── eclipse.pde
```

> 构建 `code context model`

运行`params_validation/repo_vs_commit_order/` 下的 `4_extract_model_repo_first.py` 文件，将会在`git_repo_code`下看到生层的 `code context model` 数据集文件:

```
params_validation
└── git_repo_code
     ├── my_mylyn
     │      └── 42
     │          ├── doxygen (Doxygen Parsing File)
     │          │       ├── org.eclipse.mylyn.tasks.tests
     │          │       └── org.eclipse.mylyn.tasks.ui
     │          ├── org.eclipse.mylyn.tasks.tests (Source File)
     │          ├── org.eclipse.mylyn.tasks.ui (Source File)
     │          └── code_context_model.xml (Code Context Model File)
     ├── my_platform
     ├── my_ecf
     └── my_pde
```

我们已经将源代码文件和 `code context model` 文件整理到了如下地址。



### 数据信息

代码上下文模型规模数据

<img src="./REDEME_CN.assets/statistics_dataset_1.png" alt="screenshot2024-05-23 19.35.57" style="zoom: 67%;" />

代码上下文模型节点类型数据

<img src="./REDEME_CN.assets/code_context_model_node.png" alt="screenshot2024-05-23 21.03.30" style="zoom:67%;" />

代码上下文模型边类型数据

<img src="./REDEME_CN.assets/code_context_model_edge.png" alt="screenshot2024-05-23 21.05.46" style="zoom:67%;" />



## RQ

#### 实验数据集准备

> 扩展code context model

运行`model_expansion/` 下的 `_01_expand_model.py` 文件，得到扩展数据集，该文件将会读取每个前面数据处理部分得到的每个 code context model 进行扩展（1-step, 2-step 和 3-step），并输出三个 xml 文件（`1_step_enpanded_model.xml`, `2_step_expanded_model.xml` 和 `3_step_expanded_model.xml` ）

> 节点编码

ASTNN：运行`astnn_embedding/` 下的 `astnn_entry.py` 文件，将会为在每个 code context model 的目录下，生成相应的节点编码`pkl`文件。

CodeBERT：运行`codebert_embedding/` 下的 `embedding.py` 文件。

#### RQ1

RQ1结果复现：相关的代码文件（在Mylyn数据集上，与SOTA比较），运行方法，输入文件夹，论文结果输出

> baseline

运行 `rq1/baseline/` 下的 `assign_stereotype.py`  文件分配原型，运行 `origin_pattern_match.py` 文件进行子图匹配，得到结果。

> our approach

运行`rq1/our/` 下的 `our_astnn_mylyn.py` 文件，进行 GNN 图构建和训练，并在测试集上测试训练结果。

两个的运行结果均输出在控制台。
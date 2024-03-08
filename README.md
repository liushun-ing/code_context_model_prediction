# 2023 Code Context Model

2023代码上下文模型数据收集项目

## 项目考量

实验共考虑五个项目:PDE,Mylyn,Platform,ECF,MDT

其中MDT项目经过爬虫后，没有采集到数据，后面的数据处理不再考虑，实际只有四个项目，即`PDE`,`Mylyn`,`Platform`,`ECF`



## 程序介绍

### /2023_dataset

用来保存2023收集到的bug的数据，包含`mylyn_zip`和`patch_txt`两个文件夹，分别保存bug interactionHistory的xml文件和附加的代码说明，主要用到的是`mylyn_zip`中的xml文件



### /data_extraction

- `/bugzilla_data`: 用来记录四个项目爬虫的地址，和网页bug数目和实际采集数目等信息

使用bugzilla采集数据，分为step1-step4，其中第三步和第四步可以合并，得到`step_merge.py`文件，运行该文件夹下的`main.py`的`main`函数，即可爬取数据，保存爬取的数据将保存到`2023_dataset`文件夹下。

这里需要针对五个项目分别运行，取消需要运行项目的注释，运行程序即可



### /data_count

用来进行数据的统计等，初步分析数据构成

- `01_zip_out` 解压zip文件，得到xml文件，每个xml文件解压到他所在的压缩文件的同级目录，
- `02_traces_and_events` 统计四个项目的各个traces的开始时间和event个数等、导出到四个项目的对应的xls文件，即该目录下的`Mylyn.xls`，`ECF.xls`，`MDT.xls`，`PDE.xls`，`Platform.xls`，实际上`MDT`项目是没有意义的，因为他没有bug数据。
- `03_event_filter` 针对所有traces过滤java相关的event，提取各种字段，以xls文件形式保存到四个项目对应的文件夹下，即该目录下的`ECF`，`Mylyn`，`PDE`，`Platform`文件夹。
- `04_xls_event_count` 统计有效event数目，将数据结果导出到该目录下的`filter_event_data.xls`文件
- `time_util` UTC时间转换工具函数

### /model_formation

Code Context Model Formation，由于采用xls格式保存数据，有行数上限，所以接下来的数据处理，均采用xml格式保存。

- `01_break_periods` 根据时间间隔分割working periods，得到分割后的时间片数据，和两个起止时间戳，保存在xml文件中，即`/periods`下的01-08文件夹，01-08分别表示不同的时间间隔，从0.5h到4h
- `02_filter_resource` 对`event_structure_kind=resource`事件也进行数据的收集工作，可能存在访问代码元素事件，输出到`resource`目录，经过分析，该部分数据暂时不采用; 
- `03_filter_working_periods` 对working period中一些event的kind和structureHandle进行了有效性的过滤，留下的event导出到该目录下的`/code_elements`文件夹下
- `04_extract_code_elements_and_timestamps` 在03的基础上，进一步的进行event的element进行去重以及两个时间戳的记录，并导出为新格式的xml文件，保存在当前目录的`/code_timestamp`中
- `05_quartile_IQR` 针对04的结果，进行element个数的分位线的计算，该结果直接输出在窗口中，手动复制保存到了`IQR.txt`文件中
- `06_IQR_filter` 针对05的分位线计算结果，对04的结果进行异常值过滤，将范围内的数据保留导出到`IQR_code_timestamp`中
- `07_search_gitrepo_and_doxygen` 针对06的数据结果，进行git仓库的提取，运行`doxygen`，得到doxygen输出文件等，进一步找出各个节点之间的关系，并生成上下文模型，保存到xml文件中，文件夹`D:/git_code`中
  - `07`这一步用到了doxygen官方提供的doxygen xml输出文件的解析包，然后自己编写了相关的关系解析工具包，放在`/xmlparser`文件夹下
  - `/xmlparser/doxmlparser`为官方提供工具包，`/xmlparser/doxygen_main`为自己编写工具包，入口文件为`get_relations.main_func()`
  - 该目录下的`/utils_07`文件夹封装了07步骤的一些工具函数
- `08_period_interval_time` 用来统计每个working periods的时间间隔大小，放在该目录下的`/period_duration`下
- `09_model_info` 统计code context model的节点，边，连通分量等数据，并输出到`/model_info`对应的xml文件下
- `09_01_output_model_info` 读取09步统计的数据，计算最小值，最大值，中位数，平均值，标准差，并绘制相关分布直方图
- `IQR.txt` 保存了项目的IQR阈值信息


### model_expansion

Code Context Model扩展，得到d-step的扩展model

- `01_expand_model` 扩展逻辑
- `model_loader` utils: load model and save new model


### params_validation

在这里主要进行方法重要参数的取证工作

**working_periods**

Choose time interval threshold of interaction histories breaking to get working periods.

- `1_duration_count` count the interaction events' duration information of bugs(interaction histories)
- `2_periods_break` break interaction histories into working periods based a series of time interval threshold(0.5h-5h)
- `3_show_period_result` count the working periods of each threshold and show them by figures
- `4_periods_filter` filter out invalid events to get final working periods
- `5_show_event_result` count the interaction events in working periods and show them by figures

**repo_vs_commit_order**

Compare the six strategies of code context model extraction on the quantity and quality of code context models.

- `1_extract_code_elements_and_timestamps` 对event的element进行去重以及两个时间戳的记录，并导出为新格式的xml文件，保存在当前目录的`/code_timestamp`中
- `2_quartile_IQR` 针对1的结果，进行element个数的分位线的计算，该结果直接输出在窗口中，手动复制保存到了`IQR.txt`文件中
- `3_IQR_filter` 针对2的分位线计算结果，对1的结果进行异常值过滤，将范围内的数据保留导出到`IQR_code_timestamp`中
- `4` 针对3的结果，匹配repo仓库，运行doxygen提取结构关系，构建code context model，保存到`git_repo_code`的各目录下
  - `4_extract_model_commit_first` commit_first strategy on 1,2,3 threshold
  - `4_extract_model_repo_first` repo_first strategy on 1,2,3 threshold
- `5_model_info` 统计code context model的节点，边，连通分量等数据，并输出到`/model_info`对应的xml文件下
- `5_1_output_model_info` 读取5统计的数据，计算最小值，最大值，中位数，平均值，标准差，并绘制相关分布直方图
- `IQR.txt` 保存了项目的IQR阈值信息
- `git_repo_code` 保存的项目repo数据，和model数据等


### /xmlparser

**doxmlparser**

这里是[doxygen](https://www.doxygen.nl/index.html)官方提供的API函数库，用于解析其输出的xml文件


**doxygen_main**

这里面是基于doxygen的输出文件，解析elements对应的代码元素，以及他们的之间的结构关系，构建关系图的API库。

- `ClassEntity, FieldEntity, MethodEntity, Metrics`：这些是用于存储解析doxygen输出的xml文件的结果，一个repo将构建一个`Metrics`，里面有一系列的`ClassEntity`组成，每个`ClassEntity`又包含一组`FieldEntity`和`MethodEntity`
- `Graph, Edge, Vertex`：这些是构建图的数据结构，一个repo将构建一个`Graph`，每个`Graph`由一组`Edge`和`Vertex`组成
- `get_standard_elements`：这个文件的作用是提供API函数，将working periods中elements的表现形式转换为标准形式，如`=org.eclipse.mylyn.tasks.tests/src&lt;org.eclipse.mylyn.tasks.tests{AllTasksTests.java`将会转换为`org.eclipse.mylyn.tasks.tests.AllTasksTests`
- `solve_graph`：根据repo矩阵，以及标准的element name，解析图关系，将输出最终的图集合。
- `get_relations`：**主入口**，在这里将调用API解析doxygen输出文件，得到repo metrics，也将调用API对elements进行转换得到标准形式，最后调用API，解析出最终的图集合，并返回。
- `expand_graph`：这里封装了基于step对code context model进行扩展的相关APIs。



## 数据介绍

### 2023_dataset

这里保存了四个项目爬取到的bug数据包，包含原始的zip文件，也包含程序解压缩之后的xml文件，这些解压后的xml文件是后续程序的基础文件数据集。


### params_validation/git_repo_code

这里保存了项目相关的repositories, 以及收集到的code context models，以及抽取的java code
然后后面进行ast解析以及code embedding等的结果都会保存到各model文件夹下


### data_count

这个文件下包含四个数据文件夹，分别是`ECF`, `Mylyn`, `PDE`, `Platform`文件夹，这四个文件夹内分别保存了四个项目的每个bug记录中关于java的event的kind，structureKind，StructureHandle和StartDate元数据


### model_formation

- `periods`：根据八种不同的时间间隔对bug进行分割后的working_periods的数据，这里是全局数据，是没有经过任何过滤规则过滤的events，按照`/time_petiod/project/bug`格式记录
- `resource`：该文件下保存的是kind=resource的类型的数据，之所以采集他是因为该类型的event也有可能包含对java元素的访问，但是经过人工随机检查，该部分数据可用性较差，暂时不采用
- `code_elements`：针对`periods`文件夹下的working_period数据，经过系列过滤规则过滤不合理的events数据，留下需要的，组织形式和`periods`一致，需要注意这里只考虑了2h和3h的时间间隔
- `code_timestamp`：进一步的，对`code_elements`中的event数据进行去重处理，以及两个时间戳的统计，并调整了xml文件的格式，整体组织形式和上面保持一致
- `IQR_code_timestamp`：对`code_timestamp`的数据进行异常值的过滤的结果
- `period_duration`：该目录的文件记录了每个working_period的最早和最晚events的时间差数据
- `model_info`：该目录下的文件记录了每个项目最终解析的所有模型的节点，边等数据信息








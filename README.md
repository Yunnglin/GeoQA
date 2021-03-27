# GeoQA

> 高考地理问答系统，题目背景关键词抽取部分代码

**注：[github](https://github.com/Yunnglin/GeoQA)中不包括BERT模型、训练数据和训练完成的模型完整项目位置在 `/home/data_ti5_d/maoyl/GeoQA`**

## 项目主要文件简述

```txt
GeoQA
│  config.py # argparse 项目的主要配置文件
│  data_process.py # 将BIO文本数据处理成BERT需要的形式
│  evaluate.py # 加载model，测试dev或test数据的效果
│  metrics.py # 计算模型效果指标: precision, recall, f1
│  predict.py # 加载model，从question和background文本中抽取关键词
│  raw_data_process.py # 处理原始json文件为BIO格式，并按一定比例分为train dev test集合
│  test.py # 指定测试文件路径和模型，进行预测并写入文件
│  train.py # 使用train数据，训练数据
│      
├─bert # 下载的各种BERT模型
│  ├─albert_chinese_large
│  ├─bert_base_chinese
│  ├─chinese_bert_wwm_ext
│  └─chinese_roberta_wwm_ext_large
├─data # 保存数据
│  ├─processed # 处理过的BIO数据
│  │  ├─data_all # 全部数据，包括有图和无图
│  │  │  ├─cut # 使用jieba分词
│  │  │  │  └─redundant # 保留重复词
│  │  │  └─no_cut # 不分词，进行最长匹配
│  │  │      └─redundant
│  │  └─data_no_graph # 无图数据
│  │      ├─cut
│  │      │  ├─no_redundant # 不保留重复词
│  │      │  └─redundant
│  │      └─no_cut
│  │          ├─no_redundant
│  │          └─redundant
│  ├─raw # 原始json数据
│  │  ├─data_all
│  │  └─data_no_graph
│  └─test_data # 测试数据
├─model # 项目关键模型py文件
├─result # 保存test结果
├─save_model # 保存训练模型结果
└─utils # logger等工具
```

## 项目依赖

`./requirements.txt`

```
numpy==1.20.0
torch==1.7.1
jieba==0.42.1
transformers==4.3.1
matplotlib==3.3.2
seqeval==1.2.2
PyYAML==5.4.1
```

## 主要文件运行与配置

### 处理原始数据

```bash
python raw_data_process.py
```

- 配置：

``` python
# 处理数据类型，全部数据/无图数据
data_process_types = ['data_all','data_no_graph']
# 是否使用jieba分词，分词/不分词
cuts = ['cut', 'no_cut']
# 是否保留重复词，保留/不保留
redundants = ['redundant','no_redundant']
# 数据分割比例
data_split_dict={'train': 0.7, 'dev': 0.2, 'test': 0.1}
```
根据这些配置可以生成`2*2*2=8`种处理后的数据。

- 输入：`./data/raw/`下的json文件，需包含`question`, `background/scenario_text`, `explanation`字段。

示例：

``` json
[
  {
    "topic": "16",
    "graph_text": "",
    "background": "考点一区域农业发展贵州喀斯特山区山多坡陡，坡耕地面积所占比重大，生态环境脆弱，水土流失和石漠化严重，是国家退耕还林的重要地区。据此完成下面两题。",
    "question": "该区域农业可持续发展的基础是",
    "optionA": "保护和恢复生态环境",
    "optionB": "解决当地就业问题",
    "optionC": "扩大粮食生产面积",
    "optionD": "扩大农业产业化规模",
    "answer": "A",
    "explanation": "材料显示贵州喀斯特山区生态环境脆弱，则可持续发展的基础是保护和恢复生态环境",
    "other_exp": "易错分析扩大粮食生产面积和扩大农业产业化规模，会加剧生态环境破坏"
  },
]
```



- 输出：按照一定比例分割的BIO格式的数据集，放在`./data/processed/`下，包括`train.txt`, `dev.txt`, `test.txt`。处理过程的日志在`./logs/data_info.log`中。

示例：

```txt
-DOC_START- # 标志example的开始
该 O
区 O
域 O
农 O
业 O
可 B
持 I
续 I
发 I
展 I
的 I
基 I
础 I
是 I
		# 空行，分割question与background
考 O
点 O
一 O
区 O
域 O
.......
```

## 训练模型

```bash
python train.py
```

- 配置

主要见`config.py`文件

```python
# 使用的BERT模型
bert_names = ['chinese_bert_wwm_ext']
# 同上
data_process_types = ['data_no_graph']
cuts = ['cut', 'no_cut']
redundants = ['redundant']
# 以下参数在运行过程中改变，无需指定
args.bert_path # BERT路径
args.store_name # 训练模型名称
args.data_dir # 训练数据路径
```

- 输入：经`data_process.py`处理过的数据，若未缓存过则会经`torch.save()`保存，否则直接读取。
- 输出：训练完毕的模型保存在`./save_model`下，训练日志在`./logs/info.log`中。

## 预测

```bash
python predict.py
```

- 输入：question和background字符串
- 输出：python字典类型

示例：见测试输出示例的keywords字段

## 测试

对predict的进一步封装

```bash
python test.py
```

- 配置

```python
# 测试数据路径
data_paths = ['data/test_data/53_no_graph_test_data_95.json', 'data/test_data/53_graph_test_data_133.json']
# 测试用model
models = [ModelInfo(bert_type='bert_base_chinese',
                     name='bert_base_chinese-lstm-crf-cut-redundant-epoch_14.bin',
                     path='/home/data_ti5_d/maoyl/GeoQA/save_model/data_no_graph'),]
```

- 输入：同原始输入数据
- 输出：每道题目的关键词，保存在`./result`下，格式如下：

```python
[
  {
    "id": -6943376012003136193, # 题目有id字段则为id，否则为uuid(question)
    "keywords": {# 关键词
      "in_question": [# 问题中的关键词
        {
          "text": "热电站",  # 关键词内容
          "start": 22,	# 开始位置
          "end": 25	# 结束位置
        }
      ],
      "in_background": [ # 背景中的关键词
        {
          "text": "太阳",
          "start": 3,
          "end": 5
        },
        {
          "text": "太阳能",
          "start": 12,
          "end": 15
        }
      ]
    }
  },
]
```



## BERT

[bert-base-chinese](https://huggingface.co/bert-base-chinese)

[albert_chinese_large](https://huggingface.co/voidful/albert_chinese_large)

[chinese-bert-wwm-ext](https://huggingface.co/hfl/chinese-bert-wwm-ext)

[chinese-roberta-wwm-ext-large](https://huggingface.co/hfl/chinese-roberta-wwm-ext-large)

## 一些中间结果

**question与background长度和 频率累计直方图**

128的长度可以覆盖48%的数据，而256的长度可以覆盖97%的数据。
![](https://i.loli.net/2021/03/20/MFWrILdujNPg4t9.png)

**抽取关键词词频 对数直方图**

出现词频在7及以下的有90%，根据信息量的定义，其包含的信息量较大。
![](https://i.loli.net/2021/03/20/LpXJgtFmSOUj49Q.png)

**4种BERT与是否使用CRF**

8个模型实验，训练数据5k+，`max_seq_len=128`。

除了ALBERT，模型效果相似。

![loss](https://i.loli.net/2021/03/20/uSgPEthFLZ18pV3.png)
![f1](https://i.loli.net/2021/03/20/z7T6bnfHdeDpUqj.png)

**2种BERT与是否使用jieba分词，是否保留重复词**

8个模型实验，训练数据9k+，`max_seq_len=256`，同时去掉一定噪声。

是否分词对模型效果影响较大，其他条件相同情况下，模型效果好坏基本为：
`cut >> no_cut ; redundant > no_redundant`

![loss](https://i.loli.net/2021/03/20/iMmYVGjHxaJzcr6.png)
![f1](https://i.loli.net/2021/03/20/EIU6lBWjDQrLkvA.png)





## Reference

1. [停用词](https://github.com/fighting41love/Chinese_from_dongxiexidian/blob/master/dict/%E4%B8%AD%E6%96%87%E5%81%9C%E7%94%A8%E8%AF%8D%E5%BA%93.txt)
2. 苏剑林. (Feb. 07, 2020). 《你的CRF层的学习率可能不够大 》[Blog post]. Retrieved from https://kexue.fm/archives/7196
3. 苏剑林. (Oct. 29, 2020). 《用ALBERT和ELECTRA之前，请确认你真的了解它们 》[Blog post]. Retrieved from https://kexue.fm/archives/7846
4. [AdamW优化器理解](https://www.lizenghai.com/archives/64931.html)
5. [Weight decay](https://blog.csdn.net/program_developer/article/details/80867468)
6. [BERT的下接结构调参](https://zhuanlan.zhihu.com/p/107378382)
7. [BERT原理及应用](https://zhuanlan.zhihu.com/p/101570806)
8. [写作框架参考](https://zhuanlan.zhihu.com/p/100884995)
9. [TF-IDF与余弦相似性](http://www.ruanyifeng.com/blog/2013/03/tf-idf.html)
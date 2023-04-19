# MM-DFN
Source code for ICASSP 2022 paper "[MM-DFN: Multimodal Dynamic Fusion Network For Emotion Recognition in Conversations](https://arxiv.org/pdf/2203.02385.pdf)".

In this work, we focus on emotion recognition in multimodal conversations (multimodal ERC). If you are interested in textual ERC, you can refer to a related work [DialogueCRN](https://arxiv.org/pdf/2106.01978.pdf) ([code](https://github.com/zerohd4869/DialogueCRN)).

## Quick Start

### Requirements
* python 3.6.10          
* torch 1.4.0            
* torch-geometric 1.4.3
* torch-scatter 2.0.4
* scikit-learn 0.21.2
* CUDA 10.1


Install related dependencies:
```bash
pip install -r requirements.txt
```

### Dataset

The original datasets can be found at [IEMOCAP](https://sail.usc.edu/iemocap/) and [MELD](https://github.com/SenticNet/MELD).


Following previous works (DialogueRNN, MMGCN, et al.), raw utterance-level features of textual, acoustic, and visual modality are extracted by **TextCNN with Glove embedding**, **OpenSmile**, and **DenseNet**, respectively.
The processed features can be found by the [link](https://github.com/zerohd4869/MM-DFN/tree/main/data).

Besides, another alternative is to use BERT/RoBERTa to process text features, which will achieve better performance in most cases. You can find the code and processed textual features with RoBERTa embedding in [COSMIC](https://github.com/declare-lab/conv-emotion/tree/master/COSMIC/feature-extraction).


### Run examples

For training model on IEMOCAP and MELD datasets, you can refer to the following:

```bash
# IEMOCAP dataset
bash ./script/run_train_ie.sh
# MELD dataset
bash ./script/run_train_me.sh
```

Note: The optimal hyper-parameters (i.e., the number of gcn layers) are slight differences under different experimental configurations (i.e., the version of CUDA and PyTorch). To facilitate further research by interested parties, we retain the complete code including ablation and control experiments.

## Results

Results (F1-score) of MM-DFN on IEMOCAP dataset:

| **IEMOCAP**| | | | | | | | |
|:-----:|:-----:|:-----:|:-----:|:-----:|:-----:|:-----:|:-----:|:-----:|
|Happy|Sad|Neutral|Angry|Excited|Frustrated|Acc|Macro-F1|Weighted-F1|
|42.22|78.98|66.42|69.77|75.56|66.33|68.21|66.54|68.18|

Results (F1-score) of MM-DFN on MELD dataset:

| **MELD** | | | | | | | | |
|:-----:|:-----:|:-----:|:-----:|:-----:|:-----:|:-----:|:-----:|:-----:|
|Neutral|Surprise|Sadness|Happy|Anger|Fear/Disgust|Acc|Macro-F1|Weighted-F1|
|77.76|50.69|22.93|54.78|47.82|-|62.49|36.28|59.46|


# Citation
```
@inproceedings{DBLP:conf/icassp/HuHWJM22,
  author    = {Dou Hu and
               Xiaolong Hou and
               Lingwei Wei and
               Lian{-}Xin Jiang and
               Yang Mo},
  title     = {{MM-DFN:} Multimodal Dynamic Fusion Network for Emotion Recognition
               in Conversations},
  booktitle = {{ICASSP}},
  pages     = {7037--7041},
  publisher = {{IEEE}},
  year      = {2022}
}
```




# MM-DFN
Source code for ICASSP 2022 paper "MM-DFN: Multimodal Dynamic Fusion Network For Emotion Recognition in Conversations".


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

In this work, we focus on ERC in multimodal settings. 
Following MMGCN, raw utterance-level features of textual, acoustic, and visual modality are extracted by TextCNN, OpenSmile, and DenseNet, respectively.
The processed features can be found by the [link](https://github.com/hujingwen6666/MMGCN).



### Run examples
```bash
bash ./script/run_train_ie.sh
bash ./script/run_train_me.sh
```


## Results

Reproduced results of MM-DFN on the IEMOCAP datasets:

| **IEMOCAP**| | | | | | | | |
|:-----:|:-----:|:-----:|:-----:|:-----:|:-----:|:-----:|:-----:|:-----:|
|Happy|Sad|Neutral|Angry|Excited|Frustrated|Acc|Macro-F1|Weighted-F1|
|42.22|78.98|66.42|69.77|75.56|66.33|68.21|66.54|68.18|

Reproduced results of MM-DFN on the MELD datasets:

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



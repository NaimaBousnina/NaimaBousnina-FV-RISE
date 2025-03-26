<table><td align="center" width="9999">

  <h1 align="center"> FV-RISE Explainability Approach </h1>
  This sub-repository contains the source code for the FV-RISE explainability approach [A RISE based explainability method for genuine and impostor face verification](https://www.it.pt/Publications/DownloadPaperConference/40796) developed by IT and presented at the 23rd International Conference of the Biometrics Special Interest Group (BIOSIG 2023).


  
</td></table>
<table>

 ## 1. Introduction

 They key idea behind the RISE-based FV explainability method (FV-RISE) is to generate Similarity Heat Maps (S-HMs) and Dissimilarity Heat Maps (D-HMs) to explain the FV decisions according to the type of decision  performed, notably acceptance or rejection, for both genuine and impostor verification attempts. More precisely, S-HMs and D-HMs are used to explain the acceptance and rejection cases, respectively, regardless of true or false FV decisions being performed. The FV-RISE method is inspired by the random-masking approach used by [RISE](https://arxiv.org/pdf/1806.07421.pdf) to explain object classification, with the novelty that FV-RISE applies that approach to the FV task. 

 ## 2. Requirements
- Install Python 3.x

     `$ sudo apt-get update` \
     `$ sudo apt-get install python3.x`
     
- Install cuda toolkit

     For cuda Toolkit instalation, please refer to [CUDA Toolkit Download](https://developer.nvidia.com/cuda-toolkit-archive).
  
- Install PyTorch >= 1.6
   
     For PyTorch installation, please refer to [PyTorch Installation](https://gitlab.eurecom.fr/xaiface_project/xaiface_private/xaiface_face_recognition_pipelines/-/blob/master/Face_processing_tools/Recognition/ArcFace/Pytorch_install.md?ref_type=heads).

- Install `tensorboard`, `easydict`, `sklearn`.

## 3. Scripts execution
### 3.1  Explainability heat maps generation

To generate the Similarity Heat Maps (S-HMs) and Dissimilarity Heat Maps (D-HMs):

1- Clone the current repository to your local machine.

2- Download the LFW dataset from [here](https://drive.google.com/drive/folders/1QAZEFkM7iADo5FAC8Z3kepDdLJ_sRovm) and add it to the cloned repository.

3- Execute the FV-RISE script using the following command line-call:

`python FV-RISE.py --model-prefix backbones/backbone_resnet100.pth --network 'r100'`

The command line-call requires arguments, notably:

`--model-prefix`: The path to the pretrained model. There exist two pretrained models in the backbones folder, including backbone_resnet50.pth and backbone_resnet100.pth.

`--network`: The ArcFace backbone to be considered. There exist backbones including, 'r100' and 'r50'.

### 3.2  FV-RISE performace assessment
To assess the FV-RISE performance, the face verification performance - in terms of the recall metric - is assessed through the so-called deletion and insertion processes.

To perform the deletion and insetion processes, excute the `Inset_delete_metric` using the following command line-call:


`python Inset_delete_metric.py --model-prefix backbones/backbone_resnet100.pth --network 'r100'`


 ## 4. Example of results

### 4.1  Genuine case: true acceptances and false aejections


<table>
  

<p align="center">
<img src="https://drive.google.com/uc?export=view&id=1JFl9lKw6Rj09yFzs9Y_Bu5b05yvr3vMB">

</p>
<figcaption align="center">Figure 4.1: Heat map explanations (red rectangle) for genuine pairs: True Acceptance (left) and False Rejections (right) face verification decision examples.
</figcaption>
</table>


<table>

### 4.2   Impostors case: true rejections and false acceptances

<p align="center">
<img src="https://drive.google.com/uc?export=view&id=1QVgWLRAEXswlvhZnaTuhJjrxLGaXFkFg">
</p>
<figcaption align="center">Figure 4.2: Heat map explanations (red rectangle) for impostor pairs: True Rejections (left) and False Acceptance (right) face verification decision examples.

</figcaption>
</table>

<table>

### 4.3   Qualitative comparison of the FV-RISE method with four explainability benchmarks. 

<p align="center">
<img src="https://drive.google.com/uc?export=view&id=1owIjMExizyZh_k4GqyelRSE004G7OKAS">
</p>

<figcaption align="center">Figure 4.3: Facial S-HMs comparison for five face verification explainability approaches including four benchmarks.
</table>

### 4.4   Quantitative comparison of the FV-RISE method with four explainability benchmarks. 

<p align="center">
<img src="https://drive.google.com/uc?export=view&id=1uQFss-7dTgVYIZJxK3rSWMONTxvDvq_W">
</p>

<figcaption align="center">Figure 4.4: Recall versus percentage of manipulated pixels: deleted (left) and inserted pixels (right).

</table>






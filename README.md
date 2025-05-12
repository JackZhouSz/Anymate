
<h1 align="center">🪄Anymate: A Dataset and Baselines for Learning 3D Object Rigging</h1>
  <p align="center">
    <a href="https://yfde.cc/"><strong>Yufan Deng*</strong></a>
    &nbsp;&nbsp;
    <a href="https://yzhanglp.com/"><strong>Yuhao Zhang*</strong></a>
    &nbsp;&nbsp;
    <a href="https://chen-geng.com/"><strong>Chen Geng</strong></a>
    &nbsp;&nbsp;
    <a href="https://elliottwu.com/"><strong>Shangzhe Wu†</strong></a>
    &nbsp;&nbsp;
    <a href="https://jiajunwu.com/"><strong>Jiajun Wu†</strong></a>
  </p>
<p align="center"><a href=https://www.arxiv.org/abs/2505.06227><img src='https://img.shields.io/badge/arXiv-Paper-red?logo=arxiv&logoColor=white' alt='arXiv'></a>
<a href=https://anymate3d.github.io><img src='https://img.shields.io/badge/Project_Page-Website-green?logo=googlechrome&logoColor=white' alt='Project Page'></a>
<a href=https://huggingface.co/spaces/yfdeng/Anymate><img src='https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Demo-blue'></a>

</p>
<p align="center"><img src="assets/teaser.gif" width="100%"></p>


We present the <span style="font-size: 16px; font-weight: 600;">Anymate</span> Dataset, a large-scale dataset of 230K 3D assets paired with expert-crafted rigging and skinning information around 70 times larger than existing datasets. Using this dataset, we develop a scalable learning-based auto-rigging framework with three sequential modules for joint, connectivity, and skinning weight prediction. We experiment with various architectures for each module and conduct comprehensive evaluations on our dataset to compare their performance.

***Check out our [Project Page](https://anymate3d.github.io/) for more videos and demos!***
## ⏩ Update
- 2024.5: 🔥 Paper & Code available! 
## 🌐 Web Demo
<a href=https://huggingface.co/spaces/yfdeng/Anymate><img src='https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Demo-blue'></a>
## 💡Quick Start
1. Setup environment
```bash
conda env create -f environment.yaml
conda activate anymate
pip install -e ./ThirdParty/PointLLM
```

2. Download weights
```bash
bash Anymate/get_checkpoints.sh
# if needs to use shade encoder from michelangelo
# bash ThirdParty/michelangelo/get_checkpoints.sh 
```

3. Start the UI
```bash
python Anymate_ui.py
```

## 🏋️‍♂️Train

1. Download the dataset
```bash
bash Anymate/get_datasets.sh
```

2. Train the Model (After downloading the datset)  
You can modify the training configuration through config file at `Anymate/configs`

```bash
# The --split argument is used for distributed training, where the dataset is divided into *n* partitions (one per GPU). 
# By default, it is configured for 8-GPU training. 
# If you want to train on N gpu, you need to split the Anymate_train into N partitions and rename them as Anymate_train_{i}, where i range from 1 to N.

#Train Joints Prediction Model
python Train.py --config joints --split

#Train Diffusion-based Joints Weight Prediction Model
python Train.py --config diffusion --split

#Train Connectivity Prediction Model 
python Train.py --config conn --split

#Train Skinning Weight Prediction Model
python Train.py --config skin --split

```

3. Evaluate the Model (After downloading the weight and the dataset)
```bash
# You can evaluate different model by changing checkpoints' path in Evaluate.py
python Evaluate.py
```

## 📚Anymate Dataset Documentation

### Download the dataset
```bash
bash Anymate/get_datasets.sh
```

The `Anymate_test.pt` is the test set.
The `Anymate_train_0.pt` to `Anymate_train_7.pt` are the splited train set.

They can be loaded by `dataset = torch.load('Anymate_xxx.pt')`. The dataset is a list of data asset, with each element being a dictionary. Each dictionary contains the following keys:

### Details of the processed dataset dictionary for Anymate dataset:

|   key  |   type  |  shape  |  description  |
|---|---|---|---|
|   name  |   str  |   1  |  unique id of the asset  |
|   pc |   float32  |   8196x6  |  points cloud sampled from the 3D mesh: [position, normal] for 8196 points  |
|   vox  |   bool |   64^3  |  voxelized version of the 3D mesh: resolusion is 64  |
|     |     |     |     |
| joints  |   float32  |   96x3  |  joints position: position of each joint. 96 is the maximum number of joints. the matrix is padded with -3  |
| joints_num |   int  |   1  |  number of joints  |
| joints_mask |   bool |   (<96)  |  mask for padded joints: 1 for valid joints and 0 for padded joints  |
|   conns  |   int8  | 96x(<96) |  connectivity matrix: 1 indicates joint i is connected with joint j. 96 is the maximum number of joints. the matrix is padded with zeros  |
|     |     |     |     |
|   bones  |   float32  |   64x6  |  start and end position of bones: [head position, tail position] for each bone. 64 is the maximum number of bones. the matrix is padded with -3  |
|   bones_num  |   int  |   1  |  number of bones  |
|   bones_mask  |   bool |   (<64)  |  mask for padded bones: 1 for valid bones and 0 for padded bones  |
|   skins  |   float16 |   8192x(<64) |  skinning weights: 8192 points' skinning weights w.r.t. at most 64 bones  |

### Extra elements in testset for evaluation that require original mesh:

|   key  |   type  |  shape  |  description  |
|---|---|---|---|
|   mesh_skins  | float16 |   Vx(<64)  |  skinning weights for each vertex w.r.t. at most 64 bones  |
|   mesh_pc  |   float32  |   Vx6  |  [position, normal] of each vertex  |
|   mesh_face  |   int |   Fx3  |  face index of each triangle  |

### Dataset Processing Script

The script that process the object id in Objaverse-XL into the downloaded pytorch tensor dataset can be run by:
```bash
python Dataset_process.py
```
download blender at https://download.blender.org/release/Blender4.0/blender-4.0.0-linux-x64.tar.xz and unzip it to `ThirdParty/blender-4.0.0-linux-x64`


## 📜Acknowledgement
Third party codebases include [Objaverse XL](https://github.com/allenai/objaverse-xl), [Michelangelo](https://github.com/NeuralCarver/Michelangelo), [PointLLM](https://github.com/OpenRobotLab/PointLLM), [EG3D](https://github.com/NVlabs/eg3d.git), [RigNet](https://github.com/zhan-xu/RigNet)

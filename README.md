# <div align="center">Classification and Detection Fusion Model</div>

Here, we provide a PyTorch implementation of our series of work on automatic crack detection. The code is based on [YOLOv5](https://github.com/ultralytics/yolov5) and the main contributions are as follows:
 - Contribution 1: [A Grid-based Classification and Box-based Detection Fusion Model for Asphalt Pavement Crack](https://onlinelibrary.wiley.com/doi/abs/10.1111/mice.12962) and follow-up work.
 - Contribution 2: We employ Vision Transformer as the backbone of our model.
 - Contribution 3: Self-supervised learning(e.g. [SAIM](https://github.com/qiy20/SAIM)) is applied to improve the performance of our model. 

# Main Results

| Model | Box_ap | Crack_ap | Flops | Speed |
| :---: | :---: | :---: | :---: | :---: |

# Getting Started
## Train Classification and Detection Fusion Model
The main code is in the `patch_classify` folder. The training process is as follows:
### Install
```bash
pip install -r requirements.txt  # install
```
### Train
```bash
python -m torch.distributed.run --nproc_per_node 3 patch_classify/train.py 
        --batch 192 
        --data data/crack_box_grid.yaml 
        --cfg models/patch_classify/yolov5l-pc-panet.yaml
        --weights yolov5l.pt
        --hyp data/hyps/hyp.scratch-patch-classify-high.yaml
        --cos-lr
        --epochs 300
        --cache
```
### Valadation
```bash
python patch_classify/val.py 
       --weights runs_pc/train/exp/weights/best.pt
       --data data/crack_box_grid.yaml
       --img 640
       --half
```
### Export
```bash
python patch_classify/export.py 
        --weights runs_pc/train/exp/weights/best.pt
        --data data/crack_box_grid.yaml
        --dynamic
        --include onnx
        --simplify
```
### Detect
```bash
python patch_classify/detect.py 
        --weights runs_pc/train/exp/weights/best.pt
        --source /home/qiyu/datasets/pavement_paper/pavement_box_grid/detect.txt
```
## Pretrain on Unlabeled Data
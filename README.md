#  🌠 MonoTTA
This is the official project repository for the ECCV 2024 paper [Fully Test-Time Adaptation for Monocular 3D Object Detection]🔗(https://arxiv.org/abs/2405.19682v1) by
Hongbin Lin, Yifan Zhang, Shuaicheng Niu, Shuguang Cui, Zhen Li

Monocular 3D object detection (Mono 3Det) aims to identify 3D objects from a single RGB image. However, existing methods often assume training and test data follow the same distribution, which may not hold in real-world test scenarios. To address the out-of-distribution (OOD) problems, we explore a new adaptation paradigm for Mono 3Det, termed Fully Test-time Adaptation. It aims to adapt a well-trained model to unlabeled test data by handling potential data distribution shifts at test time without access to training data and test labels. However, applying this paradigm in Mono 3Det poses significant challenges due to OOD test data causing a remarkable decline in object detection scores. This decline conflicts with the pre-defined score thresholds of existing detection methods, leading to severe object omissions (i.e., rare positive detections and many false negatives). Consequently, the limited positive detection and plenty of noisy predictions cause test-time adaptation to fail in Mono 3Det. 

MonoTTA consists of:
- 1️⃣ Reliability-driven adaptation: we empirically find that high-score objects are still reliable and the optimization of high-score objects can enhance confidence across all detections. Thus, we devise a self-adaptive strategy to identify reliable objects for model adaptation, which discovers potential objects and alleviates omissions.
- 2️⃣ Noise-guard adaptation: since high-score objects may be scarce, we develop a negative regularization term to exploit the numerous low-score objects via negative learning, preventing overfitting to noise and trivial solutions.

## Data Preparation
Tianyi Cloud: https://cloud.189.cn/t/Jn2INvaQFJBr (Password: q7st)

Google Drive: TODO

## Installation
We recommend reproducing experiments of MonoFlex (https://github.com/zhangyp15/MonoFlex) due to its relatively stable performance
1. We adopt torch1.7.1+cu110, by
 ```
 pip install torch==1.7.1+cu110 torchvision==0.8.2+cu110 torchaudio==0.7.2 -f https://download.pytorch.org/whl/torch_stable.html

 pip install -r requirements.txt (of MonoFlex)
 ```
2. Then
 ```
cd model/backbone/DCNv2

. make.sh

cd ../../..

python setup develop
 ```

3. For the source-only setting (with the official checkpoint [https://drive.google.com/drive/folders/1U60gUYp4JFOkG0VMefc4aVEMxtGM-AMu?usp=sharing]). Don't forget to config the data path in './config/paths_catalog.py'
```
CUDA_VISIBLE_DEVICES=0 python tools/plain_train_net.py --config runs/monoflex.yaml --ckpt YOUR_CKPT  --eval
```

## Usage
You can run MonoTTA by
 ```
CUDA_VISIBLE_DEVICES=0 python tools/tta_monotta.py --config runs/monoflex.yaml --ckpt model_moderate_best_soft.pth --eval --output kitti-c/gaussian1
 ```

## Results
|     MonoFlex(Gaussian_Noise_1)      | Car | Pedestrian | Cyclist | Avg. |
| :---------- | :--------------: | :-----------------------: | :-----------------------: | :-----------------------: | 
| No adapt   | 4.82            | 0.23                    | 0.34    | 1.8 | 
| MonoTTA (ours) | **21.15** | **6.54** | **3.01** | **10.27** | 

Note: The small difference between this repo and the main paper is due to the randomness in generating perturbed data. 

## Correspondence 

Please contact Hongbin Lin by [linhongbinanthem@gmail.com] if you have any questions.  📬


## Citation
If our MonoTTA method or fully TTA for Monocular 3D Object Detection settings are helpful in your research, please consider citing our paper:
```
@inproceedings{lin2024fully,
  title={Fully Test-Time Adaptation for Monocular 3D Object Detection},
  author={Lin, Hongbin and Zhang, Yifan and Niu, Shuaicheng and Cui, Shuguang and Li, Zhen},
  booktitle = {European Conference on Computer Vision},
  year = {2024}
}
```

## Acknowledgment
The code is inspired by the [MonoFlex 🔗](https://github.com/zhangyp15/MonoFlex) [Tent 🔗](https://github.com/DequanWang/tent), [EATA 🔗](https://github.com/mr-eggplant/EATA).

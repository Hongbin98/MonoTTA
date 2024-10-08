#  🌠 MonoTTA
This is the official project repository for *[MonoTTA: Fully Test-Time Adaptation for Monocular 3D Object Detection](https://arxiv.org/abs/2405.19682v1)*

## News
- We have released the full KITTI-C dataset in huggingface! Please check the link for more: [https://huggingface.co/datasets/anthemlin/kitti-c]

## Demo
https://github.com/Hongbin98/MonoTTA/assets/23754668/1abee3cc-8620-4974-bdf5-d7b83f05512b



https://github.com/Hongbin98/MonoTTA/assets/23754668/6fa98b3c-5772-4466-8aa7-f4fba1435158




https://github.com/Hongbin98/MonoTTA/assets/23754668/4f675ff8-7abf-42da-96b3-fe73d0fde758

**Note**
- 1️⃣ With a single NVIDIA RTX 4090, MonoTTA only requires **~45ms** to adapt one test image (1280X384) on KITTI-C, i.e., fps >= **15**.
- 2️⃣ MonoTTA conducts reliable adaptation, thereby improving TP and reducing FN. Meanwhile, MonoTTA prevents the model from overfitting to noisy predictions and falling into trivial solutions via negative regularization, thus reducing FP.

## Abstract
Monocular 3D object detection (Mono 3Det) aims to identify 3D objects from a single RGB image. However, existing methods often assume training and test data follow the same distribution, which may not hold in real-world test scenarios. To address the out-of-distribution (OOD) problems, we explore a new adaptation paradigm for Mono 3Det, termed Fully Test-time Adaptation. It aims to adapt a well-trained model to unlabeled test data by handling potential data distribution shifts at test time without access to training data and test labels. However, applying this paradigm in Mono 3Det poses significant challenges due to OOD test data causing a remarkable decline in object detection scores. This decline conflicts with the pre-defined score thresholds of existing detection methods, leading to severe object omissions (i.e., rare positive detections and many false negatives). Consequently, the limited positive detection and plenty of noisy predictions cause test-time adaptation to fail in Mono 3Det. 

![image](https://github.com/Hongbin98/MonoTTA/assets/23754668/454414d4-d4e5-46e9-9636-b8b3f4372dc4)


MonoTTA consists of:
- 1️⃣ Reliability-driven adaptation: we empirically find that high-score objects are still reliable and the optimization of high-score objects can enhance confidence across all detections. Thus, we devise a self-adaptive strategy to identify reliable objects for model adaptation, which discovers potential objects and alleviates omissions.
- 2️⃣ Noise-guard adaptation: since high-score objects may be scarce, we develop a negative regularization term to exploit the numerous low-score objects via negative learning, preventing overfitting to noise and trivial solutions.

## Data Preparation
Tianyi Cloud: [https://cloud.189.cn/t/aYZzumiUZrY3 （password：s9dc）]

Google Drive: https://drive.google.com/file/d/1Se_0wpGCV4-pEEmLVY59J_H1VkkjfWTX/view?usp=sharing

## Installation
We recommend reproducing experiments of MonoFlex (https://github.com/zhangyp15/MonoFlex) due to its relatively stable performance
1. We adopt torch1.7.1+cu110, by
 ```
 pip install torch==1.7.1+cu110 torchvision==0.8.2+cu110 torchaudio==0.7.2 -f https://download.pytorch.org/whl/torch_stable.html

 pip install -r requirements.txt 
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

### --Docker Env--

If you find that the environment above conflicts with your GPU server (e.g., conflicting cuda or gcc versions), it is recommended to use Docker to build the environment:

```
cd docker/pytorch1.7.1_cu110
docker build -t monotta:pytorch1.7.1-cuda11.0 .
```
Then, you can initilize a docker container with the built image "monotta:pytorch1.7.1-cuda11.0" by
```
docker run --gpus all -it --shm-size=64g monotta:pytorch1.7.1-cuda11.0
```
or (with directory mounting)
```
docker run --gpus all -it --shm-size=64g -v /dir_local:/dir_docker monotta:pytorch1.7.1-cuda11.0
```

Next, you can continue configuring the environment in the Docker container mentioned above.


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

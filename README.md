# STIG: Spectrum Translation for refinement of Image Generation
STIG is for boosting the quality of image generation by reducing spectral discrepancies of current generative models including GANs and Diffusion Models. The algorithm is elaborated in our paper ```paper``` that has been published in AAAI 2024.


## Quick Overview

### STIG Framework
STIG mitigates spectral discrepancies of the generated images based on GAN-based image-to-image translation architecture and patch-wise contrastive learning. It manipulates the frequency components to address the spectral discrepancy components effectively in the frequency domain. Auxiliary regularizations prevent the potential corruption of the image during spectral translation.
![figure_3_camera_ready_version](https://github.com/ykykyk112/STIG/assets/59644868/33fc02a5-c95f-43fb-a74a-c49486aa65b1)

### Effectiveness of STIG


## Installing dependency
```
conda create -n stig python=3.9.2
conda activate stig
pip install torch==1.8.0+cu111 torchvision==0.9.0+cu111 torchaudio==0.8.0 -f https://download.pytorch.org/whl/torch_stable.html
pip install -r requirements.txt
```

Especially, our version of pytorch is as follows.
```
pytorch==1.8.0
torchvision==0.9.0
cudatoolkit==1.11.1
```

## Preparing datasets
Put real and fake datasets in the folder ```datasets/```.

The real images should be located in ```datasets/{dataset_name}/real/``` and generated images should be in ```datasets/{dataset_name}/fake/```. We suppose the type of image file is ```.png```, so you should consider it.

## Training STIG
Enter the command below. You can change the GPU device by modifying the option ```--device```.

The sampled results are visualized in the ```results/{experiment_name}/sample/``` during the training. After training, the results image and magnitude spectrum will be saved at each folder in ```results/{experiment_name}/eval/```.

```python train.py --size {size} --data {dataset_name} --epoch 10 --batch_size 1 --lr 0.00008 --device {gpu_ids} --dst {experiment_name}```

We also provide a training log using the ```tensorboard``` library. If you want to use it, you can access the training log using the code below.

```tensorboard --logdir='./results/{experiment_name}/tensorboard'```

## Evaluation STIG
We provide three evaluation metric, FID and log frequency distance. To evaluate your experiment result, enter the command below. You can choose the evaluation metric in ```[image_fid / magnitude_fid / lfd]```. The default option is set to ```magnitude_fid```.

```python eval.py --eval_root {experiment_name} --eval_mode {metric} --device {gpu_ids}```

## Deepfake detection
We provide sample codes to train and evaluate the detectors in the paper. To train the detector, choose the classifier from ```[cnn / vit]``` and enter the command below.

```python detect.py --is_train True --classifier {classifier} --lr 0.0002 --size {size} --device {gpu_ids} --class_epoch 20 --class_batch_size 32 --dst {experiment_name}```

The training results of the detector are saved in the folder ```results/{experiment_name}/{classifier}_classifier/```.

For evaluation a classifier using generated images, enter the command below.

```python detect.py --is_train True --classifier {classifier} --lr 0.0002 --size {size} --device {gpu_ids} --class_batch_size 32 --dst {experiment_name} --eval_root {folder_of_generated_images}```

## Non-deterministic behavior of the upsampling layer
We use the bi-linear upsampling layer (```nn.Upsample```) in the STIG.

It is well known that ```nn.Upsample``` behaves non-deterministically because the internal layer ```F.interpolate``` of the PyTorch implementation.

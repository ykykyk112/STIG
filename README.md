# STIG: Spectrum Translation for refinement of Image Generation
STIG is for boosting the quality of image generation by reducing spectral discrepancies of current generative models including GANs and Diffusion Models. The algorithm is elaborated in our paper ["Spectrum Translation for Refinement of Image Generation (STIG) Based on Contrastive Learning and Spectral Filter Profile"](https://ojs.aaai.org/index.php/AAAI/article/view/28074/28154) that has been published in AAAI 2024.


## Quick Overview

### STIG Framework
STIG mitigates spectral discrepancies of the generated images based on GAN-based image-to-image translation architecture and patch-wise contrastive learning. It manipulates the frequency components to address the spectral discrepancy components effectively in the frequency domain. Auxiliary regularizations prevent the potential corruption of the image during spectral translation.

![figure_3_camera_ready_version](https://github.com/ykykyk112/STIG/assets/59644868/33fc02a5-c95f-43fb-a74a-c49486aa65b1)

### Effectiveness of STIG
STIG can reduce the spectral disparity of various generative models including GANs and diffusion models. It erased the disparity pattern (checkerboard artifacts) on the spectrum of GAN models and enriched high-frequency details for images from diffusion models.

![effectiveness_figure](https://github.com/ykykyk112/STIG/assets/59644868/bf19856b-5dfe-4d5d-b728-db7e10ee867a)

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

```python train.py --size {size} --data {dataset_name} --epoch 10 --batch_size 1 --lr 0.00008 --device {gpu_ids} --dst {experiment_name}```

Enter the command. You can change the GPU device by modifying the option ```--device```.

The sampled results are visualized in the ```results/{experiment_name}/sample/``` during the training. After training, the results image and magnitude spectrum will be saved at each folder in ```results/{experiment_name}/eval/```.

We also provide a training log using the ```tensorboard``` library. If you want to use it, you can access the training log using the code below.

```tensorboard --logdir='./results/{experiment_name}/tensorboard'```

## Inference STIG

```python inference.py --size {size} --inference_data {folder_path_of_images} --device {gpu_ids} --dst {experiment_name}```

Enter the command. You can change the GPU device by modifying the option ```--device```.

The inference results are saved at ```results/{experiment_name}/inference/```. Put the folder path of the inference data into ```--inference_data```, and also put the path of model parameters onto ```--inference_param```.

## Evaluation STIG

```python eval.py --eval_root {experiment_name} --eval_mode {metric} --device {gpu_ids}```

We provide three evaluation metric, FID and log frequency distance. To evaluate your experiment result, enter the command. You can choose the evaluation metric in ```[image_fid / magnitude_fid / lfd]```. The default option is set to ```magnitude_fid```.


## Deepfake detection

```python detect.py --is_train True --classifier {classifier} --lr 0.0002 --size {size} --device {gpu_ids} --class_epoch 20 --class_batch_size 32 --dst {experiment_name}```

We provide sample codes to train and evaluate the detectors in the paper. To train the detector, choose the classifier from ```[cnn / vit]``` and enter the command.

The training results of the detector are saved in the folder ```results/{experiment_name}/{classifier}_classifier/```.

For evaluation a classifier using generated images, enter the command below.

```python detect.py --is_train True --classifier {classifier} --lr 0.0002 --size {size} --device {gpu_ids} --class_batch_size 32 --dst {experiment_name} --eval_root {folder_of_generated_images}```

## Non-deterministic behavior of the upsampling layer
We use the bi-linear upsampling layer (```nn.Upsample```) in the STIG.

It is well known that ```nn.Upsample``` behaves non-deterministically because the internal layer ```F.interpolate``` of the PyTorch implementation.


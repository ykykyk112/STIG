# Code for STIG

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

```python train.py --size 256 --data {dataset_name} --epoch 10 --batch_size 1 --lr 0.00008 --device 0 --dst {experiment_name}```

We also provide an training log using the ```tensorboard```. If you want to use it, you can access the training log using the code below.

```tensorboard --logdir='./results/{experiment_name}/tensorboard'```

## Evaluation STIG
We provide two evaluation metric, FID and log frequency distance. To evaluate your experiment result, enter the command below. You can choose the evaluation metric in ```[fid / lfd]```. The default option is set to ```fid```.

```python eval.py --eval_root {experiment_name} --eval_mode {metric} --device 0```

## Non-deterministic behavior of the upsampling layer
We use the bi-linear upsampling layer (```nn.Upsample```) in the STIG.

It is well known that ```nn.Upsample``` behaves non-deterministically because the internal layer ```F.interpolate``` of the PyTorch implementation.

# EI339 project - small object dectection based on SSD

&nbsp;
&nbsp;
&nbsp;
&nbsp;


## Training SSD & FFSSD
- First download the fc-reduced [VGG-16](https://arxiv.org/abs/1409.1556) PyTorch base network weights at:              https://s3.amazonaws.com/amdegroot-models/vgg16_reducedfc.pth
- By default, we assume you have downloaded the file in the `ssd.pytorch/weights` dir:

```Shell
mkdir weights
cd weights
wget https://s3.amazonaws.com/amdegroot-models/vgg16_reducedfc.pth
```

- To train SSD and FFSSD using the train script simply specify the parameters listed in `train.py` as a flag or manually change them. And import model in the code need to be changed.

```Shell
python train.py
```

- To tran MDSSD.

```Shell
python train_mdssd.py
```

- Note:
  * For training, an NVIDIA GPU is strongly recommended for speed.
  * For instructions on Visdom usage/installation, see the <a href='#installation'>Installation</a> section.
  * You can pick-up training from a checkpoint by specifying the path as one of the training parameters (again, see `train.py` for options)

## Evaluation
To evaluate a trained network:

```Shell
python eval_for_size_category.py
```

You can specify the parameters listed in the `eval_for_size_category.py` file by flagging them or manually changing them.  


## Performance

#### VOC2007 Test

##### mAP

| | XS | S | M | L | XL |
|:-:|:-:|:-:|:-:|:-:|:-:|
| SSD | 0.42 % | 2.61 % | 2.77 % | 10.69 % | 31.01 % |
| FFSSD | 4.39 % | 2.52 % | 4.82 % | 10.94 % | 32.80 % |
| MDSSD | 4.33 % | 2.91 % | 5.06 % | 10.86 % | 31.67 % |



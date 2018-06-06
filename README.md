# SE-resnet152
This project is the result of the compitition of data analysis and aritficial intelligence held by Baidu and Xi'an Jiaotong University. Our team consists of ZIRUI(Ryan) ZHAO, XINDI(Cindy) WU, YIJUN MAO, SHENGNAN AN, WEIJIAN(Tom) QI and we all from Xi'an Jiaotong University.
This repository contains an implementation of SE-resnet.

You can find the original SE_resnet implementation, using Caffe
-[SE-ResNet* and SE-ResNeXt*](https://github.com/hujie-frank/SENet) thanks to [Alex Parinov](https://github.com/creafz)
-[SENet154](https://github.com/hujie-frank/SENet) thanks to [Alex Parinov](https://github.com/creafz)

## Prerequisites
This requires Ubuntu 16 (Xenial Xeres) or later to get the proper libraries work. It should work for other distributions, but I haven't tested them yet. From a terminal, execute the following:
```
sudo apt-get -y make gcc g++ python perl
```
You also have to install OpenCV 3.3 or higher according to its website [here](https://docs.opencv.org/3.3.1/d7/d9f/tutorial_linux_install.html).
And you also have to install CUDA 8.0 and PyTorch 0.3 and its libraries. Also remember to put `nvcc` (usually in `/usr/local/cuda/bin`) in environment variable `PATH`.

## Build
Use `path.py` to bulid the path of each pictures. And the paths are saved in the `newtrain.txt` and `newtest.txt` which are supposed to be in the folder `./datasets`
Use `preprocess_and_train.py` from a terminal in this folder to run the training codes.
Use `generate_result.py` to generate the result in `result.txt`.
The model of SE_ResNet is in the file `senet.py` and some tools are written in the `tools.py`.
The pictures are in the folder `./datasets/train` and `./datasets/test`

## Usage
You can adjust all the parameters in this project such as the path and datasets to solve the classification problems. Here are the example of the utility of the project in the data compitition held by Baidu and Xi'an Jiaotong University.

### Generate the file path
We use the file `train.txt` and `test.txt` to generate the path of each pictures by using the `path.py`. The folder `dataset` contains of the original dataset and some other pictures which is processed by opencv such as changing the brightness, saturation, and some even filtered by Gaussian Blur in opencv so that we can enlarge the datasets. And these pictures are labeled as `New_xxxxxx.jpg` in the `./datasets/train`. And more specificly, you can see the details of the changes of each picture in the name such as the picture `New_1_5_1.5_0e2442a7d933c895fa281972da1373f082020060.jpg` means the first change of the picture and the parameters of the Gaussian Blur is 5 and 1.5. If the first number of the picture is `2`, it means we improve the brightness and contrast of the picture. If the first number is `3`, it means we reduce the brightness and contrast.
The following command will generates the path of each pictures and saved in the file `./newtrain.txt` and `./newtest`.
```
python path.py 
```
After that, you have to put these two files in the folder `./datasets`.
The codes of processing pictures with opencv is also in the folder named `pre.py`. Attention, the file `pre.py` has to run in the windows environment which requires the cv2 library of python.

### Preprocessed the picture and train the network
The following command will preprocessed all the pictures of train and test and load the pretrained weights of SE_resnet152 in `./models/seresnet/net-20.pth`.
```
python3 preprocess_and_train.py
```
In the file '`preprocess_and_train.py`, we use `torchvision.transforms` for preprocessing the datasets such as randomly rotation, carefully crops the pictures and resize to (224,224). We use different method to preprocess the pictures in different period of training to enlarge the datasets. 
The model of SE-resnet152 is applyed through the file `senet.py`. It contains all the network with S.E. processing bottleneck inserted.

### Generates the result and save
After training the network, we can find the trained network in the folder `models/seresnet/net-epoch_number.pth`, the `epoch_number` means the number of epoch when saving the network. You have to change the number of the network in the file `generate_result.py`. The following command means you can choose the epoch number of the saving network that you want to load.
'''
parser.add_argument('--which_epoch',default='20', type=str, help='0,1,2,3...or last')
'''
After that, run the following command:
'''
python3 generate_result.py
'''
You will get the file `result.py` with the predicted labels.
We accidently find that the number of right result of each labels is 10, se we find the label whose number is not 10 and enlarge the datasets by opencv. This is kind of little trick of this contest.

##Some details of the contest
There are some details of us participating in the contest hold by Baidu and Xi'an Jiaotong University. 

###start
Firstly we start to work on the contest on 28, May. We come up with a convolutional neural network with simply 15 layers which is absolutely not enough for 100 classificaiton problem. After 2 days work, we find that the result of validation is only 55%, so we choose to work on some deeper network.

###change models
Then we choose to work on PyTorch for transferlearning. We choose to work on resnet50 which contains 50 layers of CNN. But we have some problems in preprocessing the image such as randomly flipping the pictures and normalize it with wrong parameters during the training process. Consiquently the accuracy is only 89%.

###deeper models
Firstly I believe the problem is the models is not deep enough, so we change the models to Resnet152 which is much deeper than Resnet50. However, the result is only 90.9%. After carefully read the paper of ResNet we find the problems of preprocessing the pictures and we correct the parameters of normalizatino and delete the randomly filpping. The accuracy of our result is up to 96.1%
Then we analysis the datasets and our result. We accidently find that the picture of 100 brands of test datasets is 1000, and we find that the number of our result of each brand is approximately 10. So we find that the labels whose number is not 10 must have some problems, so we enlarge the datasets of these labels with opencv. 
With enlarged datasets, preprocess the images with `torchvision.transforms.Randomcrop()`, `torchvision.transforms.Randomrotation()` and `torchvision.transforms.Resize()`, our accuracy up to 97.1% with rank 157.

###SE-net models
With the deadline of the contest approaching, we have to got higher accuracy, and we choose to use SE-net which is the champion of ILSVRC 2017. We choose to use SE_ResNet152 which have a much better performance than ResNet152. After 100 epoches of training, our accuracy is up to 98.3% with rank 117.
After that we enlarged the datasets with opencv and got the final accuracy 99.6%.



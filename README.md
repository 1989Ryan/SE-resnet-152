# SE-resnet152
This project is the result of the competition of data analysis and aritficial intelligence held by Baidu and Xi'an Jiaotong University. Our team consists of ZIRUI ZHAO, XINDI WU, YIJUN MAO, SHENGNAN AN, WEIJIAN QI.
This repository contains an implementation of SE-resnet.

You can find the original SE_resnet implementation, using Caffe
-[SE-ResNet* and SE-ResNeXt*](https://github.com/hujie-frank/SENet) thanks to [Alex Parinov](https://github.com/creafz)
-[SENet154](https://github.com/hujie-frank/SENet) thanks to [Alex Parinov](https://github.com/creafz)

## Prerequisites
This requires Ubuntu 16 (Xenial Xeres) or later versions to get the proper libraries work. It should work for other distributions, though not having been tested yet. 

Execute the following in the terminal to install GCC and python developing environment:
```
sudo apt-get -y make gcc g++ python perl
```
You also have to install OpenCV 3.3 or higher according to its website [here](https://docs.opencv.org/3.3.1/d7/d9f/tutorial_linux_install.html).
In addition, CUDA 8.0 and PyTorch 0.3 and its libraries are needed. Remember to put `nvcc` (usually in `/usr/local/cuda/bin`) in environment variable `PATH`.

## Build
Run `path.py` to bulid the path of each pictures, which are saved in the `newtrain.txt` and `newtest.txt` in the folder `.datasets`
Run `preprocess_and_train.py` from the terminal in this folder to conduct the training.
Use `generate_result.py` to generate the result in `result.txt`.
The model of SE_ResNet is in the file `senet.py` and some tools are written in the `tools.py`.
The pictures are in the folder `./datasets/train` and `./datasets/test`

## Usage
All the parameters can be adjusted in this project, including the paths and datasets when solving any classification problems. Here shows an example of using our project to participate in the data compitition held by Baidu and Xi'an Jiaotong University.

### Generate the file path
The files `train.txt` and `test.txt` list complete filenames of the pictures. By using the `path.py`, we generate the paths for all the files referred in the two texts. The folder `dataset` contains the original dataset and pictures processed to enlarge the datasets. by means of changing the brightness or saturation, and some filtered by Gaussian Blur with the help of opencv. These new generated pictures are labeled in the form of `New_xxxxxx.jpg` in the folder `./datasets/train`. More specificly, you can see the details of the processing from the filename. For instance, the picture `New_1_5_1.5_0e2442a7d933c895fa281972da1373f082020060.jpg` means that it is the very first processed picture from its original one and the parameters of the Gaussian Blur is 5 and 1.5. And the first number of the filename indicates that we change the brightness and contrast of the picture, `2`increasing while `3` reducing.
The following command will generates the path of each pictures and save them in the file `./newtrain.txt` and `./newtest`.
```
python path.py 
```
After that, you have to put these two files in the folder `./datasets`.
The codes of processing pictures with opencv is also in the file `pre.py` only run in the windows environment with the cv2 library of python.

### Preprocessed the picture and train the network
The following command will preprocess all the pictures to be trained and tested, as well as load our pretrained weights of SE_resnet152 in `./models/seresnet/net-20.pth`.
```
python3 preprocess_and_train.py
```
In the file '`preprocess_and_train.py`, we use `torchvision.transforms` for preprocessing the datasets such as randomly rotation, carefully crops the pictures and resize them into (224, 224). We use different methods in preprocessing the pictures during different periods of training to enlarge the datasets. 
The model of SE-resnet152 is applied through the file `senet.py`. It contains all the networks with S.E. processing bottleneck inserted.

### Generates the result and save
After training the network, we can find the trained network in the folder `models/seresnet/net-epoch_number.pth`, the `epoch_number` reveals the number of epoch when saving the network. You can change the number of the network in the file `generate_result.py`. The following command enables you to choose the epoch number of the saving network according to what you are willing to load.
'''
parser.add_argument('--which_epoch',default='20', type=str, help='0,1,2,3...or last')
'''
After that, run the following command:
'''
python3 generate_result.py
'''
You will get the file `result.txt` with the predicted labels.
We accidently find that the number of right result of each labels is 10, so we find the label whose number is not 10 and enlarge the datasets by opencv. This is kind of little trick of this contest.

## Some details of the contest
There are some detailed experiences when participating in the contest held by Baidu and Xi'an Jiaotong University. 

### start
We started work on the contest on 28, May. We come up with a convolutional neural network with simply 15 layers. It turns out that 15 layers are absolutely not enough for this 100 classificaiton problem. After 2 days' work, we miserably found that our validation result only reached 55%, so we chose to work on a deeper network.

### change models
Then we chose to work on PyTorch for transferlearning. The resnet50 containing 50 layers of CNN was preferred. But we had some problems in preprocessing the images like randomly flipping the pictures and normalizing them with wrong parameters during the training process. Consequently the accuracy was only 89%.

### deeper models
At first we attributed our unsatisfactory accuracy was to the superficial model, and hence changed the models to Resnet152 which is much deeper than Resnet50. However, the result improved only 1.9 percent, in the end of 90.9%. After carefully reading some papers of ResNet, we found our deficiency in preprocessing the pictures and we corrected the parameters of normalizatino as well as deleted the randomly filpping. Surprisingly, the accuracy reached 96.1%.

Then we analysis the datasets and our result. The order of the provided data was uncovered that the picture of 100 brands in test datasets is 1000, and the number of our result of each brand is approximately 10. So we speculated that the labels whose number is not 10 may have some problems, so we enlarge the datasets of these labels with opencv. 
Thanks to enlarged datasets and more efficient image preprocessing methods like`torchvision.transforms.Randomcrop()`, `torchvision.transforms.Randomrotation()` and `torchvision.transforms.Resize()`, our accuracy up to 97.1% with rank 157.

### SE-net models
With the deadline of the contest approaching, we have to accomplish higher accuracy, and thus choose to use SE-net, the champion of ILSVRC 2017. SE_ResNet152 has a much better performance than ResNet152. 100 epoches of training completed, our accuracy climbs to 98.3% with rank 117.
Having enlarged the datasets with opencv again, we get our final accuracy 99.6%.



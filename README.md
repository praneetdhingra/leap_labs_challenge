
# Leap Laps Challenge  
This repo holds the solution to the coding challenge for Leap Labs. The challenge is to add [adversarial noise](https://christophm.github.io/interpretable-ml-book/adversarial.html) to trick an image classification model into misclassifying the altered image as a specified target class.

**Input**: The user will provide an image and specify a target class.

**Output**: The program should output an image that has been altered with adversarial noise. The altered image should be classified by the model as the target class.

## Solution
Looking from up high, the solution expects an input image, preprocesses it, adds the noise using the Fast Gradient Sign Method, and classifies it to the target class following the pretrained ResNET18 model. The code is broken down into three files:
1. **utils.py**: Supporting functions to the main
2. **model_interface.py**: Function to load and evaluate the classification model
3. **main.py**: Runs the packaged code while expecting arguments like the image and target

Also, there is a **test_utils.py** which contains a couple of unit tests to test the preprocessing and noise generation. The testing coverage can be expanded to account for more functionalities in the code and error handling.

### How to run?
The solution is fairly easy to run but it is important to setup the environment first to ensure that there are no dependency issues. Please follow the following steps for a smooth handling:
1. Create and activate the conda env using the following command(s):
```console
conda create -n challenge_env python=3.10
conda activate challenge_env
```
2. The code should download the ResNET model by itself but if the code breaks due to some error, it is handy to run the following command and download it manually:
```console
curl -O https://download.pytorch.org/models/resnet18-f37072fd.pth
```
3. Its handy to download the ImageNet data labels for the code. Use the following command for it (ensure you are in the working directory):
```console
curl -L https://gist.githubusercontent.com/yrevar/942d3a0ac09ec9e5eb3a/raw/238f720ff059c1f82f368259d1ca4ffa5dd8f9f5/imagenet1000_clsidx_to_labels.txt -o imagenet_classes.txt
```
4. Once everything is downloaded, it is time to install code requirements in the environment:
```console
pip install -r requirements.txt
```
5. Finally, to run the noise generator, you need to call the **main.py** file. This file has a couple of mandatory arguments- **image path**, and the **target class** which need to be passed while calling. Apart from that, there are two optional arguments which can be fed in: **--epsilon** to indicate the strength of noise generator (default = 0.05) and **--output_path** to indicate where to store the noised image (default = working directory). Following command is for a sample panda image and a target class of 100 - black swan (both the images have been uploaded to the repo) running with default optional arguments:
```console
python main.py "./panda.png" 100
```

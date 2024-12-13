<div style="text-align: center; background-color: #8F98E3; font-family: 'Trebuchet MS', Arial, sans-serif; color: white; padding: 10px; font-size: 25px; font-weight: bold; border-radius: 0 0 0 0; box-shadow: 0px 6px 8px rgba(0, 0, 0, 0.2);">
  Pediatric Pneumonia Chest X-ray 🫁
</div>

#### 📖 <font color=red><b>Note that,</b></font> This **README** file is a step-by-step guide to setup. While there are some code snippets here, it is only a shorthand to illustrate what is in the file, so please do not copy the code in this file and put into your file to run, it can cause unwanted errors. </br>
<font color=yellow>Please strictly follow the instructions.</font>

### ⚙️ <font color=Gree><b>0.</b></font> <font color=Gree> Settings </font> </br>
First you need to install and add some necessary libraries for this project.

#### <font color=Purple><b><i> a) Requirements </i></b></font> 
- **seaborn** : ```pip install seaborn```
- **pytorch on CPU** : ```pip3 install torch==1.11.0+cpu torchvision==0.12.0+cpu -f https://download.pytorch.org/whl/torch_stable.html```
- **sklearn** : ```pip3 install scikit-learn```

#### <font color=Purple><b><i> b) Check Torch version </i></b></font> 
Make sure your versions of PyTorch and torchvision are compatible. You can refer to the official PyTorch compatibility table here</br>
| PyTorch Version    | torchvision Version |
| ------------------ | ------------------- |
| 2.0.0              | 0.15.0              |
| 1.13.1             | 0.14.1              |
| 1.12.0             | 0.13.0              |
| 1.11.0             | 0.12.0              |

The below code can be found in the file **"testTorch_ver.py**

```bash
import torch
import torchvision

print("PyTorch version:", torch.__version__)
print("torchvision version:", torchvision.__version__)
```

#### <font color=Purple><b><i> c) Import nescessary libraries </i></b></font> 

The below code can be found in the file <font color=Yellow><i> "libs.py" </i></font>

```bash
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt

.
.
.

# Warnings off : Removes unnecessary warnings for a cleaner interface.
import warnings
warnings.filterwarnings("ignore")

# Fix a seed for PyTorch
torch.manual_seed(42);
```
Overview of the functionality of some libraries used:
- **scikit-learn** Helps us split the data into training and testing sets and evaluate the performance of the classification model.</br>

- **Graphing libraries** use 2 libraries <i>matplotlib</i> and <i>seaborn</i> to draw graphs to visualize the data</br>

- **PyTorch** 
    - **torch** : Build deep learning models, perform matrix and tensor operations on CPU/GPU.
    - **torch.nn** : Provides modules for building neural network layers.
    - **torch.optim** : Provides gradient descent optimization algorithms such as SGD, Adam, which help us update weights in deep learning models.
    - **torch.nn.functional** : Provides activation and loss functions as functions that we can use in the model if we don't want to call the nn layer.

- **torch.manual_seed(42)** : Set seed to ensure reproducible random results. Helps test and compare model runs.


### 🧮 <font color=Gree><b> 1. </b></font> <font color=Gree> Datasets </font> </br>

#### <font color=Purple><b><i> a) Abstract </i></b></font> 
The dataset is organized into two folders, **train** and **test**, and contains subfolders for each image category, **pneumonia** and **normal**.</br>
Overall, there are 5856 Chest X-rays labelled as either pneumonia or normal: 1583 normal (1349 for training, 234 for testing) and 4273 pneumonia (3883 for training, 390 for testing). </br>
Chest X-ray images were selected from <font color=Pink><b><i>Guangzhou Women and Children’s Medical Center, Guangzhou</i></b></font>. All chest X-ray imaging was performed as part of patients’ routine clinical care. For the analysis of chest X-ray images, all chest radiographs were initially screened for quality control by removing all low quality or unreadable scans. 

Link to this dataset:
https://www.kaggle.com/datasets/andrewmvd/pediatric-pneumonia-chest-xray

#### <font color=Purple><b><i> b) Data Evaluation </i></b></font>
However, as mentioned above, the two classes are not balanced. There are more examples for **PNEUMONIA** than **NORMAL**.

In order to properly train a neural network (but in general any classifier), the classes need to be balanced. Let me give some brief explanation for why this is:

- Suppose that a classifier needs to be able to distinguish event A from event B
- If the classifier is trained on some imbalanced data, say 99 events A and 1 event B, then it will always be convenient to predict event A (99% accuracy on the training data)
- This is why, when dealing with (strongly) imbalanced classes, as in this case, intervention is needed.

The code below will analyze the data of **PNEUMONIA** than **NORMAL** in the **"train"** set. You can find it in **"analyze_data"** function in <font color=Yellow><i> analyzeData.py </i></font> file.

```bash
def analyze_data(data_dir):
    ...

    # Count samples
    n_samples_nr_train = len(os.listdir(normal_train_dir))  
    n_samples_pn_train = len(os.listdir(pneumonia_train_dir))  

    # Define result
    class_count = {0: n_samples_nr_train, 1: n_samples_pn_train}
    class_names = {0: 'NORMAL', 1: 'PNEUMONIA'}

    ...

    plt.show()

    return class_count
```
, and the above code will return a comparison chart between the two classes. It'll look like this:
<center>
  <img src="assets/analyze.png" alt="Circular Buffer Animation">
</center>
</br>

The classes are highly unbalanced, so it is necessary to balance them manually. In particular, in this project will be tested balancing via class weighting.

#### <font color=Purple><b><i> c) Solution : Weighted classes </i></b></font>

The approach I propose here is to train a neural network that associates each layer with a certain weight.<br>
Suppose we deal with the case of an event A (99 samples) vs. an event B (1 sample):
* if for an event A the classifier predicts an event B, the weight of the error will be $\hspace{2pt}\frac{1}{99+1}$
* if for an event B the classifier will predict an event A, the weight of the error will be $\hspace{2pt}\frac{99}{99+1}$

In this way for the classifier it will no longer be convenient, as it was previously, to classify everything as an A event.<br>
As already intuited, in this context, the weight for each class is calculated as:

$\qquad class\hspace{2pt}weight = 1 - \frac{number\hspace{2pt}of\hspace{2pt}samples\hspace{2pt}of\hspace{2pt}the\hspace{2pt}class}{total\hspace{2pt}numer\hspace{2pt}of\hspace{2pt}samples}$

The code implementing the above idea can be found in the **"class_weights"** function in the file <font color=Yellow><i> analyzeData.py </i></font>

```bash
def class_weights(class_count):
    ...

    samples_0 = class_count[0]
    samples_1 = class_count[1]
    tot_samples = samples_0 + samples_1

    # Calculate weights
    weight_0 = 1 - samples_0 / tot_samples
    weight_1 = 1 - weight_0  # equivalent to 1 - samples_1 / tot_samples

    ...
    return class_weights_tensor
```
, you will get the following result on the terminal :
```bash
Class weights: [0.7421636085626911, 0.25783639143730885]
```

### 📚 <font color=Gree><b> 2. </b></font> <font color=Gree> Pre-Training </font> </br>

```bash
Classes: ['NORMAL', 'PNEUMONIA']
Total images: 5232
Training set size: 3662
Validation set size: 1570
```


### 📚 <font color=Gree><b> 3. </b></font> <font color=Gree> Training </font> </br>

Put "ChestX-Training.ipynb" into Google Collab, run and save model.

#### <font color=Purple><b> 3.1/ </b></font> <font color=Purple> Model </font> </br>

Model using in this project is DenseNet-161, now let's break down the structure of this model.

**Overall Architecture**

DenseNet-161 is a convolutional neural network (CNN) known for its dense connections between layers. It's characterized by:

*   **Dense Blocks:** The core building blocks of DenseNet, where each layer is connected to every other layer within the block in a feed-forward fashion.
*   **Transition Layers:**  Layers between dense blocks that reduce the spatial dimensions of the feature maps and control the number of channels.
*   **Classification Layer:** The final layer that produces the output probabilities for each class (in this case, 1000 classes, likely for ImageNet).

**Model Breakdown**

This's a summary for the DenseNet-161 model:

```
________________________________________________________________________________________________________________________
Layer (type)                                  Output Shape              Param #     Connected to                     
========================================================================================================================
input_0 (InputLayer)                          [(None, 3, H, W)]         0           []                               
________________________________________________________________________________________________________________________
conv0 (Conv2d)                                (None, 96, H/2, W/2)      14,784      input_0[0][0]                     
________________________________________________________________________________________________________________________
norm0 (BatchNorm2d)                           (None, 96, H/2, W/2)      384         conv0[0][0]                      
________________________________________________________________________________________________________________________
relu0 (ReLU)                                  (None, 96, H/2, W/2)      0           norm0[0][0]                      
________________________________________________________________________________________________________________________
pool0 (MaxPool2d)                             (None, 96, H/4, W/4)      0           relu0[0][0]                       
________________________________________________________________________________________________________________________
denseblock1 (DenseBlock, x6 Layers)          (None, 384, H/4, W/4)     394,752     pool0[0][0]                       
________________________________________________________________________________________________________________________
transition1 (Transition)                      (None, 192, H/8, W/8)     73,920      denseblock1[0][0]                 
________________________________________________________________________________________________________________________
denseblock2 (DenseBlock, x12 Layers)         (None, 768, H/8, W/8)     1,626,624   transition1[0][0]                  
________________________________________________________________________________________________________________________
transition2 (Transition)                      (None, 384, H/16, W/16)   295,296     denseblock2[0][0]                  
________________________________________________________________________________________________________________________
denseblock3 (DenseBlock, x36 Layers)         (None, 2112, H/16, W/16)  10,538,496  transition2[0][0]                  
________________________________________________________________________________________________________________________
transition3 (Transition)                      (None, 1056, H/32, W/32)  2,228,256   denseblock3[0][0]                  
________________________________________________________________________________________________________________________
denseblock4 (DenseBlock, x24 Layers)         (None, 2208, H/32, W/32)  11,358,720  transition3[0][0]                  
________________________________________________________________________________________________________________________
norm5 (BatchNorm2d)                           (None, 2208, H/32, W/32)  8,832       denseblock4[0][0]                  
________________________________________________________________________________________________________________________
classifier (Linear)                           (None, 1000)              2,209,000   norm5[0][0]                        
========================================================================================================================
Total params: 28,748,978
Trainable params: 28,559,986
Non-trainable params: 188,992
```

**Notes:**

*   `H` and `W` are placeholders for the height and width of the input image. They will be actual numbers depending on your input data.



1. **Initial Convolution and Preprocessing (`conv0`, `norm0`, `relu0`, `pool0`)**

    *   `conv0`: A 2D convolutional layer with:
        *   Input channels: 3 (likely representing RGB color channels of an image)
        *   Output channels: 96
        *   Kernel size: 7x7
        *   Stride: 2 (moves the filter 2 pixels at a time)
        *   Padding: 3 (adds padding to maintain spatial dimensions after convolution)
        *   Bias: False (no bias term is added)
    *   `norm0`: Batch normalization. It normalizes the activations of the previous layer, stabilizing and accelerating training.
    *   `relu0`: Rectified Linear Unit (ReLU) activation function. It introduces non-linearity by setting negative values to zero, `f(x) = max(0, x)`.
    *   `pool0`: Max pooling layer. It reduces the spatial dimensions of the feature maps by taking the maximum value within a 3x3 window, with a stride of 2.

2. **Dense Blocks (e.g., `denseblock1`, `denseblock2`, ...)**

    *   **Structure:** Each dense block consists of multiple dense layers (e.g., `denselayer1`, `denselayer2`, ...). The number of layers in each dense block varies in DenseNet-161, following a specific pattern: 6, 12, 36, 24.
    *   **Dense Layers (e.g., `denselayer1`)**
        *   `norm1`, `relu1`, `conv1`: A sequence of batch normalization, ReLU activation, and a 1x1 convolution. The 1x1 convolution expands the number of channels to 192 (bottleneck layer).
        *   `norm2`, `relu2`, `conv2`: Another sequence of batch normalization, ReLU activation, and a 3x3 convolution. The 3x3 convolution extracts features and reduces the number of channels back down to 48 (the growth rate).
        *   **Concatenation (implicit):** The crucial part is that the output of each `denselayer` (48 channels) is concatenated with the input of that `denselayer`. This concatenation is the "dense" connection. For example, `denselayer2` receives input from the output of `denselayer1` and also the original input to `denseblock1`. This is why the input channel size to each layer grows (96, 144, 192, 240, etc.)
    *   **Growth Rate:** In this model, the growth rate is 48. This means each dense layer adds 48 channels to the feature map.

3. **Transition Layers (e.g., `transition1`, `transition2`, ...)**

    *   **Purpose:** These layers connect dense blocks. They reduce the number of feature maps and their spatial dimensions.
    *   `norm`, `relu`, `conv`: Batch normalization, ReLU, and a 1x1 convolution. The 1x1 convolution reduces the number of channels (e.g., from 384 to 192 in `transition1`).
    *   `pool`: Average pooling with a 2x2 kernel and stride of 2. This halves the spatial dimensions (height and width) of the feature maps.

4. **Final Classification (`norm5`, `classifier`)**

    *   `norm5`: Batch normalization before the final classification layer.
    *   `classifier`: A fully connected (linear) layer that maps the 2208 features from the last dense block to 1000 output classes. This layer has a bias term.

**Key Concepts and Advantages of DenseNet**

*   **Dense Connectivity:** The core idea. Connecting all layers directly within a dense block has several benefits:
    *   **Feature Reuse:** Layers can directly access the feature maps from all preceding layers, promoting feature reuse and reducing the number of parameters.
    *   **Vanishing Gradient Mitigation:** Shorter paths to earlier layers improve gradient flow during backpropagation, making training easier.
    *   **Stronger Feature Propagation:** Information can flow more easily through the network.
*   **Bottleneck Layers:** The 1x1 convolutions within dense layers that reduce the number of input channels before the 3x3 convolution help to reduce computational cost.
*   **Parameter Efficiency:** DenseNets often achieve high accuracy with fewer parameters compared to other architectures like ResNets, especially for deeper networks.

**DenseNet-161 Specifics**

*   **161 Layers:** This number refers to the total number of convolutional and fully connected layers (not including batch normalization or pooling).
*   **Layer Pattern:**  6, 12, 36, 24 (number of layers in each of the four dense blocks).
*   **Growth Rate (k):** 48 (as explained earlier).

**In summary,** DenseNet-161 is a powerful and parameter-efficient CNN architecture that leverages dense connectivity to achieve high accuracy in image classification tasks. Its structure allows for effective feature reuse, gradient flow, and information propagation, making it a popular choice in various computer vision applications.






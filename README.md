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

#### <font color=Purple><b> 3.1/ </b></font> <font color=Purple><b><i> Model </i></b></font> </br>

Model using in this project is DenseNet-161, now let's break down the structure of this model.

**Overall Architecture**

<br>
<center>
  <img src="assets/overview.png" alt="DenseNet-161 Overview">
</center>
</br>

DenseNet-161 is a convolutional neural network (CNN) known for its dense connections between layers. It's characterized by:

*   **Initial Convolution:** A 7x7 convolution layer with stride 2, followed by a 3x3 max-pooling layer with stride 2. This reduces the original input size.
*   **Dense Blocks:**  DenseNet-161 has 4 Dense Blocks with the corresponding number of layers:
    *   Dense Layer 1: 6 layers
    *   Dense Layer 2: 12 layers
    *   Dense Layer 3: 36 layers
    *   Dense Layer 4: 24 layers

    Each layer in the Dense Block has the structure: Batch Normalization -> ReLU -> Convolution (1x1) -> Batch Normalization -> ReLU -> Convolution (3x3). And the growth rate (k) of DenseNet-161 is 48.
*   **Transition Layers:**  Between the Dense Blocks, each Transition Layer includes:
    *   Batch Normalization -> ReLU -> Convolution (1x1) (reduce the number of channels)
    *   2x2 Average Pooling with step 2 (reduce dimensionality)

*   **Classification Layer:** Fully Connected Layer with 1000 Outputs (for ImageNet classification problem with 1000 classes).

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

**Notes:** `H` and `W` are placeholders for the height and width of the input image. They will be actual numbers depending on your input data.

, or you can see the detailed parameters of the layers of the denseNet blocks when declaring the model:
```bash
# Base model (transfer learning): DenseNet-161
# Transfer Learning
model = models.densenet161(pretrained=True)
model
```

#### <font color=Purple><b> 3.2/ </b></font> <font color=Purple><b><i> Fine-tuning </i></b></font> </br>

This part involves fine-tuning a pre-trained model (specifically DenseNet, based on **model.classifier.in_features**), replacing the old classifier with a new classifier that is appropriate for this project.






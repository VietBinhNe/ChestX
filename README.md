<div style="text-align: center; background-color: #8F98E3; font-family: 'Trebuchet MS', Arial, sans-serif; color: white; padding: 10px; font-size: 25px; font-weight: bold; border-radius: 0 0 0 0; box-shadow: 0px 6px 8px rgba(0, 0, 0, 0.2);">
  Pediatric Pneumonia Chest X-ray 🫁
</div>

#### 📖 <font color=red><b>Note that,</b></font> This **README** file is a step-by-step guide to setup. While there are some code snippets here, it is only a shorthand to illustrate what is in the file, so please do not copy the code into your file to run it and cause errors. </br>
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

The below code can be found in the file **"libs.py"**

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
Chest X-ray images were selected from  Guangzhou Women and Children’s Medical Center, Guangzhou. All chest X-ray imaging was performed as part of patients’ routine clinical care. For the analysis of chest X-ray images, all chest radiographs were initially screened for quality control by removing all low quality or unreadable scans. 

Link to this dataset:
https://www.kaggle.com/datasets/andrewmvd/pediatric-pneumonia-chest-xray

#### <font color=Purple><b><i> b) Data Evaluation </i></b></font>
However, as mentioned above, the two classes are not balanced. There are more examples for **PNEUMONIA** than **NORMAL**.

In order to properly train a neural network (but in general any classifier), the classes need to be balanced. Let me give some brief explanation for why this is:

- Suppose that a classifier needs to be able to distinguish event A from event B
- If the classifier is trained on some imbalanced data, say 99 events A and 1 event B, then it will always be convenient to predict event A (99% accuracy on the training data)
- This is why, when dealing with (strongly) imbalanced classes, as in this case, intervention is needed.

The code below will analyze the data of **PNEUMONIA** than **NORMAL** in the **"train"** set. You can find full code in <font color=Yellow><i> analyzeData.py </i></font> file.

```bash
from libs import *

data_dir = '' #Fill in ' ' your path
train_dir = os.path.join(data_dir, 'train')  
normal_train_dir = os.path.join(train_dir, 'NORMAL')  
pneumonia_train_dir = os.path.join(train_dir, 'PNEUMONIA')  

.
.
.

fig, ax = plt.subplots(figsize=(5, 5))
ax.bar([class_names[0], class_names[1]], [class_count[0], class_count[1]])
ax.set_title('Class Distribution in Training Set')
ax.set_ylabel('Number of Samples')
ax.set_xlabel('Class')
plt.show()

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





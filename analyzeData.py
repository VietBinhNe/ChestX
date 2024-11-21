from libs import *

data_dir = 'D:\Workspace\ChestX\Data' #Fill in " " your path
train_dir = os.path.join(data_dir, 'train')  
normal_train_dir = os.path.join(train_dir, 'NORMAL')  
pneumonia_train_dir = os.path.join(train_dir, 'PNEUMONIA')  

if not os.path.exists(normal_train_dir) or not os.path.exists(pneumonia_train_dir):
    raise FileNotFoundError("One or more specified data directories do not exist. Please check the paths.")

n_samples_nr_train = len(os.listdir(normal_train_dir))  
n_samples_pn_train = len(os.listdir(pneumonia_train_dir))  

class_names = {0: 'NORMAL', 1: 'PNEUMONIA'}
class_count = {0: n_samples_nr_train, 1: n_samples_pn_train}

print(f'Found {class_count[0]} elements for {class_names[0]}')
print(f'Found {class_count[1]} elements for {class_names[1]}')

fig, ax = plt.subplots(figsize=(5, 5))
ax.bar([class_names[0], class_names[1]], [class_count[0], class_count[1]])
ax.set_title('Class Distribution in Training Set')
ax.set_ylabel('Number of Samples')
ax.set_xlabel('Class')
plt.show()

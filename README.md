# CvT Tensorflow Implementation 
Dear friends: We implemented the Convolutions to Vision Transformers (CvT) into Tensorflow Version > 2.5. 
The goal was to better understand the concept and architecture. Please feel free to use and improve the model.  
CvT original code GitHub: [https://github.com/microsoft/CvT]   
Paper: [CvT: Introducing Convolutions to Vision Transformers](https://arxiv.org/abs/2103.15808)

### Our Implementation Schema 
![](cvtSchema_v3.png "CvT Schema")

### Testing implementation

[comment]: <> (Pretrained on ILSVRC2012 [ImageNet-1k]&#40;https://www.google.com&#41;  )

[comment]: <> (Contains 1.3 million training images and 1000 objects categories.     )

Trained on [CIFAR100](https://www.cs.toronto.edu/~kriz/cifar.html)  
Data set: 60000 images and 100 object categories  
Training set: Contains 50000 images (500 objects per a category)   
Validation set: Contains 10000 images (100 objects per a category) 

### Results 
CIFAR100 was trained from scratch.  
We use augmentation with an image resizing and get state-of-the-art results.
 <table>
  <tr>
    <td>Model</td>
    <td>Resolution</td>
    <td>Param</td>
    <td>Top-1</td>
    <td>Hardware</td>
  </tr>
  <tr>
    <td>CvT-1</td>
    <td>72x72</td>
    <td>3.5M</td>
    <td>59.0</td>
    <td>2x RTX 2080</td>
  </tr>
</table>

Please see option details in config.py
<table>
  <tr>
    <td>Options</td>
    <td>Stage 1</td>
    <td>Stage 2</td>
    <td>Stage 3</td>
    <td>Remark</td>
  </tr>
  <tr>
    <td>Model</td>
    <td></td>
    <td></td>
    <td></td>
    <td></td>
  </tr>
   <tr>
    <td>NUM_STAGES</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    <td>3</td>
  </tr>
  <tr>
    <td>CLS_TOKEN</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    <td>TRUE</td>
  </tr>
  <tr>
    <td>EMBEDDING</td>
    <td></td>
    <td></td>
    <td></td>
    <td></td>
  </tr>
  <tr>
    <td>PATCH_SIZE</td>
    <td>6</td>
    <td>3</td>
    <td>3</td>
    <td></td>
  </tr>
  <tr>
    <td>PATCH_STRIDE</td>
    <td>3</td>
    <td>2</td>
    <td>2</td>
    <td></td>
  </tr>
  <tr>
    <td>DIM_EMBED</td>
    <td>32</td>
    <td>64</td>
    <td>128</td>
    <td></td>
  </tr>
  <tr>
    <td>STAGE</td>
    <td></td>
    <td></td>
    <td></td>
    <td></td>
  </tr>
  <tr>
    <td>DEPTH</td>
    <td>1</td>
    <td>2</td>
    <td>6</td>
    <td>No dropout</td>
  </tr>
  <tr>
    <td>ATTENTION</td>
    <td></td>
    <td></td>
    <td></td>
    <td></td>
  </tr>
  <tr>
    <td>NUM_HEADS</td>
    <td>1</td>
    <td>3</td>
    <td>6</td>
    <td></td>
  </tr>
</table>


### Usage

#### Installation

Before installing the dependencies you should consider using a virtual environment. It can be created by: 

```shell
# activate the environment by running the generated activate
# script in <folder name> for your os. E.g. for windows activate.bat
python3 -m venv <folder name>
```

The necessary packages are listed in requirements.txt.
They can be installed using:

```shell
pip install -r requirements.txt
```

For the installation of the optional CUDA drivers please refer to the [tensorflow](https://www.tensorflow.org/install/gpu) documentation.

#### Configuration 

The Model can be configured with the hyper-parameters in config/config.py.

#### Training

To start the training without changing Datasets, Learning Rate or the Learning Rate Schedule just run main.py:

```shell
python main.py 
``` 

If you want to change these values, open main.py with an editor and change the parameters of the train function at the bottom of the file. 

```python
model, figure = train(cifar_loader,
                      epochs=300,
                      batch_size=512,
                      start_weights="",
                      learning_rate=1e-3,
                      learning_rate_schedule=schedule)
``` 

##### Training Parameters: 

 - cifar_loader  
   > The loader of the Dataset (Consult dataloader/DataLoader.py) for more information.
 
 - epchos
   > The Number of Epochs to train for.

 - batch_size 
   > The Number of Images per batch.
 
 - start_weights 
   > The file name in the weights folder containing pre trained weights to load before starting the training.
 
 - learning_rate
   > The learning rate.
 
 - learning_rate_schedule
   > The learning rate schedule (e. g. a cosine decay)
 
Note that the training can be stopped at any time by focusing on the plot and holding the key 'q'.

Pressing 'h' or 'r' while focusing on the plot will resize it to fit the Data.


#### Testing

To test your Model call the train function found in main.py

```python
figure = test(model, cifar_loader, number_of_images=5000, split="test", seed=None)
```

##### Test Parameters

 - model
   > your trained Model.
 
 - cifar_loader
   > Dataset Loader same as in train.
 
 - number_of_images
   > Determines how many images to use for the test.
 
 - split
   > "test" or "train" the Dataset split to take images from. (usually test : )

 - seed
   > The Random Seed by which to choose images. If the Value is None os.urandom is used instead.

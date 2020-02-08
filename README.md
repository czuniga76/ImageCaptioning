# ImageCaptioning
Image captioning project for Udacity Computer Vision ND using MS COCO Dataset

There are 2 main notebooks, one for training an caption generator model and one for testing. 
2_Training.ipynb
3_Inference.ipynb

A pretrained Resnet50 model is used as a feature generator for images. An embedding layer transforms the features to embeddings for an LSTM
decoder. Captions are provided by the COCO Dataset. The LSTM is trained to predict what words follow given previous word. Initially the image
feature is provided. Currently some captions are reasonably good but others are wrong.

MS COCO dataset must be downloaded separately because of its large size. 

http://cocodataset.org/#home

There are auxiliary files to help with trainig.
data_loader was provided by Udacity and helps load image and caption data in batches.
model.py stores the model I used.

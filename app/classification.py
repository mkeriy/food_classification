import os 
from tqdm.autonotebook import tqdm, trange
# from PIL import Image
# import requests
# import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import torch.nn.functional as F

import pandas as pd
import numpy as np
import torchvision
from torchvision import datasets, models, transforms


# def get_image(url):
#     try:
#         resp = requests.get(url, stream=True).raw
#     except requests.exceptions.RequestException as e:  
#         sys.exit(1)
    
#     try:
#         img = Image.open(resp)
#     except IOError:
#         print("Unable to open image")
#         sys.exit(1)
#     img.save('images/to_class/img2.jpg', 'jpeg')
    

def get_class_names(name_file):
    file = open(name_file, "r")
    data = [line.split("\n") for line in file]
    return data

def predict(model, dataloader_test):
    probs = []
    model.eval()
    
    with torch.no_grad():
        for inputs, y in tqdm(dataloader_test):
            if torch.cuda.is_available():
                inputs = inputs.cuda()
                
            preds = model(inputs).cpu()
            probs.append(preds)
            
    probs = nn.functional.softmax(torch.cat(probs), dim=-1).numpy()
    
    
    return probs


def get_prediction():

    class_names = get_class_names('class_names.txt')

    model_ft = torch.load('food_classification.pkl',  map_location=torch.device('cpu'))
    
    transform = transforms.Compose([
        transforms.Resize((int(244 * 1.05), int(244 * 1.05))),
        transforms.CenterCrop(244),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    # img = transform(img)
    
    dataset_test = datasets.ImageFolder(
        root='uploads/',
        transform=transform
    )
    
    dataloader_test = torch.utils.data.DataLoader(dataset_test, shuffle=False, batch_size=32)

    test_filenames = [fn[0].split('/')[-1] for fn in dataset_test.imgs]

    probs_resnet = predict(model_ft, dataloader_test)
    # print(probs_resnet.shape)
    preds_resnet = np.argmax(probs_resnet, axis=1)
    preds = []
    for i in range(len(preds_resnet)):
        preds.append(class_names[preds_resnet[i]])
    # submission = pd.read_csv('sample_submission.csv')
    submission = pd.DataFrame({'id': test_filenames, 'Expected': preds}).sort_values('id')
    submission.to_csv('./submission.csv', index=False)

    return preds[0][0]



# get_image('http://findopskrift.dk/wp-content/uploads/2013/04/Banan.jpg')
# img = get_image('/home/mkeriy/monty/food_classification/images/to_class/my_apple3.jpg')

# get_prediction()
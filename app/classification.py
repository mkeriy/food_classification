import os 
from tqdm.autonotebook import tqdm, trange
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import torch.nn.functional as F

import pandas as pd
import numpy as np
import torchvision
from torchvision import datasets, models, transforms

    

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
    
    dataset_test = datasets.ImageFolder(
        root='uploads/',
        transform=transform
    )
    
    dataloader_test = torch.utils.data.DataLoader(dataset_test, shuffle=False, batch_size=32)

    test_filenames = [fn[0].split('/')[-1] for fn in dataset_test.imgs]

    probs_resnet = predict(model_ft, dataloader_test)
    preds_resnet = np.argmax(probs_resnet, axis=1)
    preds = []
    for i in range(len(preds_resnet)):
        preds.append(class_names[preds_resnet[i]])
    submission = pd.DataFrame({'id': test_filenames, 'Expected': preds}).sort_values('id')
    submission.to_csv('./submission.csv', index=False)

    return preds[0][0]


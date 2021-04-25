import os
import numpy as np
import torch
from PIL import Image
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
from engine import train_one_epoch, evaluate
import utils
import transforms as T
from chirality_detection.data.chirality_dataset import ChiralityDataset
from PIL import Image, ImageDraw, ImageFont
torch.cuda.set_device(1)


def get_model_instance_segmentation(num_classes):
    # load an instance segmentation model pre-trained pre-trained on COCO
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=False, progress=True,
                                                                 box_nms_thresh=0.001
                                                                 )

    # get number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features

    # replace the pre-trained head with a new one
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    return model

def get_transform(train):
    transforms = []
    transforms.append(T.ToTensor())
    if train:
        transforms.append(T.RandomHorizontalFlip(0.5))
    return T.Compose(transforms)

def main():
    # train on the GPU or on the CPU, if a GPU is not available
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    # our dataset has two classes only - background and sigmoids
    num_classes = 2
    dataset = ChiralityDataset(
        'data/train.json',
        'data/aia_sigmoid_1k_png/',
        True, get_transform(train=True))

    dataset_test = ChiralityDataset(
        'data/train.json',
        'data/aia_sigmoid_1k_png/',
        True, get_transform(train=False))

    # split the dataset in train and test set
    torch.manual_seed(1)
    indices = torch.randperm(len(dataset)).tolist()
    dataset = torch.utils.data.Subset(dataset, indices[:-6178])
    dataset_test = torch.utils.data.Subset(dataset_test, indices[-6178:])
    # define training and validation data loaders
    data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=1, shuffle=False, num_workers=4, collate_fn=utils.collate_fn)

    data_loader_test = torch.utils.data.DataLoader(
        dataset_test, batch_size=1, shuffle=False, num_workers=4, collate_fn=utils.collate_fn)

    # get the model using our helper function
    model = get_model_instance_segmentation(num_classes)

    # move model to the right device
    model.to(device)

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=0.005,
                                momentum=0.9, weight_decay=0.0005)
    # and a learning rate scheduler
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                                   step_size=3,
                                                   gamma=0.1)
    # let's train it for 10 epochsprint metric logger pytorch
    num_epochs = 10
    test = []
    epoch = []
    for epoch in range(num_epochs):
        # train for one epoch, printing every 10 iterations
        temp = train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq=2000)
        test.append(temp)
        # update the learning rate
        lr_scheduler.step()
        # evaluate on the test dataset
        evaluate(model, data_loader_test, device=device)
        print("That's it!")

    torch.save(model, 'data/model.pt')

if __name__ == "__main__":
    main()
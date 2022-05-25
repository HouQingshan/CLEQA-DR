import numpy as np
from sklearn.metrics import confusion_matrix, cohen_kappa_score
from sklearn.metrics import f1_score

import torch

np.seterr(divide='ignore', invalid='ignore')


# miou
def mean_iou(input, target, classes=2):
    """  compute the value of mean iou
	:param input:  2d array, int, prediction
	:param target: 2d array, int, ground truth
	:param classes: int, the number of class
	:return:
		miou: float, the value of miou
	"""
    miou = 0
    input = torch.softmax(input, dim=1)
    _, inputs = input.max(dim=1)
    for i in range(classes):
        intersection = np.logical_and(target == i, inputs == i)
        # print(intersection.any())
        union = np.logical_or(target == i, inputs == i)
        temp = np.sum(intersection) / np.sum(union)
        miou += temp
    return miou / classes


# iou
def iou(input, target, classes=1):
    """  compute the value of iou
	:param input:  2d array, int, prediction
	:param target: 2d array, int, ground truth
	:param classes: int, the number of class
	:return:
		iou: float, the value of iou
	"""
    input = torch.softmax(input, dim=1)
    _, inputs = input.max(dim=1)

    intersection = np.logical_and(target == classes, inputs == classes)
    # print(intersection.any())
    union = np.logical_or(target == classes, inputs == classes)
    iou = np.sum(intersection) / np.sum(union)
    return iou


# acc
def calculate_accuracy(output, target):
    output = torch.softmax(output, dim=1)
    _, predictions = output.max(dim=1)
    Accuracy = torch.true_divide((target == predictions).sum(dim=0), output.size(0)).item()
    return Accuracy


# based matrix's acc
def compute_acc(pred, gt):
    pred = torch.softmax(pred, dim=1)
    _, preds = pred.max(dim=1)
    preds.tolist(), gt.tolist()
    matrix = confusion_matrix(y_true=np.array(gt).flatten(), y_pred=np.array(preds).flatten())
    acc = np.diag(matrix).sum() / matrix.sum()
    return acc

# kappa
def compute_kappa(prediction, target):
    """
	:param prediction: 2d array, int,
			estimated targets as returned by a classifier
	:param target: 2d array, int,
			ground truth
	:return:
		kappa: float
	"""
    prediction = torch.softmax(prediction, dim=1)
    _, predictions = prediction.max(dim=1)
    predictions.tolist(), target.tolist()
    img, target = np.array(predictions.cpu().detach().numpy()).flatten(), np.array(
        target.cpu().detach().numpy()).flatten()
    kappa = cohen_kappa_score(target, img)
    return kappa


# f1-score
def compute_f1(prediction, target):
    """
	:param prediction: 2d array, int,
			estimated targets as returned by a classifier
	:param target: 2d array, int,
			ground truth
	:return:
		f1: float
	"""
    prediction = torch.softmax(prediction, dim=1)
    _, predictions = prediction.max(dim=1)
    predictions.tolist(), target.tolist()
    img, target = np.array(predictions.cpu().detach().numpy()).flatten(), np.array(
        target.cpu().detach().numpy()).flatten()
    f1 = f1_score(y_true=target, y_pred=img, average='micro')
    return f1


# recall
def compute_recall(pred, gt):
    #  返回所有类别的召回率recall
    pred = torch.softmax(pred, dim=1)
    _, preds = pred.max(dim=1)
    preds.tolist(), gt.tolist()
    matrix = confusion_matrix(y_true=np.array(gt.cpu().detach().numpy()).flatten(),
                              y_pred=np.array(pred.cpu().detach().numpy()).flatten())
    recall = np.diag(matrix) / matrix.sum(axis=0)
    return recall

# confusion_matrix
def compute_confusion_matrix(pred, gt):
    pred = torch.softmax(pred, dim=1)
    _,preds = pred.max(dim=1)
    preds.tolist(), gt.tolist()
    y_true = np.array(gt.cpu().detach().numpy()).flatten()
    y_pred = np.array(preds.cpu().detach().numpy()).flatten()
    matrix = confusion_matrix(y_true, y_pred)
    return matrix
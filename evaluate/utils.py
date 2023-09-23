import numpy as np
from PIL import Image
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

def read_image(path):
    """Read an image and return it as a numpy array of 0s and 1s."""
    img = Image.open(path).convert('L')
    arr = np.asarray(img)
    return (arr > 127).astype(int)

def evaluate_single_image(pred_path, true_path):
    """Evaluate single image using accuracy, precision, recall, and f-1 score."""
    pred = read_image(pred_path)
    true = read_image(true_path)
    
    accuracy = accuracy_score(true.ravel(), pred.ravel())
    precision = precision_score(true.ravel(), pred.ravel())
    recall = recall_score(true.ravel(), pred.ravel())
    f1 = f1_score(true.ravel(), pred.ravel())

    return accuracy, precision, recall, f1

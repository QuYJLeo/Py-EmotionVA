import os
import sys
import argparse
import numpy as np
import torch
from torchvision import transforms, datasets
import torch.utils.data as data
from models.DDAM import DDAMNet
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import itertools


def parse_args():
    """
    Parse command line arguments for test/evaluation configuration.

    Returns:
        Namespace object containing test arguments
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--aff_path', type=str, default='/data/affectnet/', help='AffectNet dataset path.')
    parser.add_argument('--batch_size', type=int, default=1, help='Batch size.')
    parser.add_argument('--workers', default=0, type=int, help='Number of data loading workers.')
    parser.add_argument('--num_head', type=int, default=2, help='Number of attention head.')
    parser.add_argument('--num_class', type=int, default=8, help='Number of class.')
    parser.add_argument('--model_path', default='./checkpoints/mp_MFN_epoch15_acc0.5587.pth')
    return parser.parse_args()


def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    Plot and save confusion matrix visualization.

    Args:
        cm: Confusion matrix array
        classes: List of class names
        normalize: Whether to normalize the confusion matrix (default: False)
        title: Title for the plot
        cmap: Colormap for the plot
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    # Create figure and plot
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title, fontsize=16)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    # Format cell text
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j] * 100, fmt) + '%',
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('Actual', fontsize=18)
    plt.xlabel('Predicted', fontsize=18)
    plt.tight_layout()


# Emotion class names for 8-class classification
class8_names = ['Neutral', 'Happy', 'Sad', 'Surprise', 'Fear', 'Disgust', 'Angry', 'Contempt']


def run_test():
    """
    Main test/evaluation function for AffectNet emotion recognition model.
    Loads trained model, evaluates on validation set, and generates confusion matrix.
    """
    args = parse_args()

    # Set device (GPU if available, else CPU)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Initialize model
    model = DDAMNet(num_class=args.num_class, num_head=args.num_head, pretrained=False)

    # Load trained checkpoint
    checkpoint = torch.load(args.model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()

    # Define validation data preprocessing
    data_transforms_val = transforms.Compose([
        transforms.Resize((112, 112)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])])

    # Load validation dataset
    val_dataset = datasets.ImageFolder(f'{args.aff_path}/val', transform=data_transforms_val)

    print('Validation set size:', val_dataset.__len__())

    # Create validation data loader
    val_loader = torch.utils.data.DataLoader(val_dataset,
                                             batch_size=args.batch_size,
                                             num_workers=args.workers,
                                             shuffle=False,
                                             pin_memory=True,
                                             drop_last=True)

    # Initialize metrics tracking
    iter_cnt = 0
    bingo_cnt = 0
    sample_cnt = 0

    # Run evaluation
    for imgs, targets in val_loader:
        imgs = imgs.to(device)
        targets = targets.to(device)

        # Model inference
        out, feat, heads = model(imgs)

        # Get predictions
        _, predicts = torch.max(out, 1)
        correct_num = torch.eq(predicts, targets)
        bingo_cnt += correct_num.sum().cpu()
        sample_cnt += out.size(0)

        # Collect all predictions and targets for confusion matrix
        if iter_cnt == 0:
            all_predicted = predicts
            all_targets = targets
        else:
            all_predicted = torch.cat((all_predicted, predicts), 0)
            all_targets = torch.cat((all_targets, targets), 0)
        iter_cnt += 1

    # Calculate overall accuracy
    acc = bingo_cnt.float() / float(sample_cnt)
    acc = np.around(acc.numpy(), 4)

    print("Validation accuracy:%.4f. " % (acc))

    # Generate and save confusion matrix for 8-class classification
    if args.num_class == 8:
        matrix = confusion_matrix(all_targets.data.cpu().numpy(), all_predicted.cpu().numpy())
        np.set_printoptions(precision=2)
        plt.figure(figsize=(10, 8))
        plot_confusion_matrix(matrix, classes=class8_names, normalize=True,
                              title='AffectNet Confusion Matrix (acc: %0.2f%%)' % (acc * 100))

        # Save confusion matrix plot
        plt.savefig(os.path.join('checkpoints', "affecnet8" + "_acc" + str(acc) + ".png"))
        plt.close()


if __name__ == "__main__":
    run_test()

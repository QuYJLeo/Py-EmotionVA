import os
import sys
from PIL import Image
import json
from tqdm import tqdm
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.utils.data as data
from torchvision import transforms, datasets
from models.DDAM import DDAMNet
import torch.nn.functional as F

# Small epsilon value for numerical stability
eps = sys.float_info.epsilon


def parse_args():
    """
    Parse command line arguments for training configuration.

    Returns:
        Namespace object containing training arguments
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--aff_path', type=str, default='/data/affectnet/', help='AffectNet dataset path.')
    parser.add_argument('--batch_size', type=int, default=10, help='Batch size.')
    parser.add_argument('--lr', type=float, default=0.0001, help='Initial learning rate for adam.')
    parser.add_argument('--workers', default=0, type=int, help='Number of data loading workers.')
    parser.add_argument('--epochs', type=int, default=40, help='Total training epochs.')
    parser.add_argument('--num_head', type=int, default=2, help='Number of attention head.')
    parser.add_argument('--num_class', type=int, default=8, help='Number of class.')
    return parser.parse_args()


class TaskDataset(torch.utils.data.Dataset):
    """
    Custom dataset for AffectNet emotion recognition task.
    Loads images and corresponding emotion labels, valence, and arousal values.

    Args:
        file: Path to JSON annotation file
        root: Root directory containing images
        transform: Image preprocessing transforms
    """
    def __init__(self, file, root, transform):
        # Load dataset from JSON file
        data = json.load(open(file, "r"))

        self.paths = []
        self.targets = []
        self.valences = []
        self.arousals = []
        self.transform = transform
        self.root = root

        # Extract paths and labels from JSON data
        for item in data:
            path = os.path.join(self.root, item['name'])
            self.paths.append(path)
            self.targets.append(item['expression'])
            self.valences.append(item["valence"])
            self.arousals.append(item["arousal"])

        # Convert to numpy arrays for efficient processing
        self.targets = np.array(self.targets)
        self.valences = np.array(self.valences)
        self.arousals = np.array(self.arousals)

    def __len__(self):
        """Return the total number of samples in the dataset."""
        return len(self.paths)

    def __getitem__(self, idx):
        """
        Get a single sample from the dataset.

        Args:
            idx: Index of the sample

        Returns:
            Tuple of (image tensor, labels tuple)
        """
        # Load and preprocess image
        img = Image.open(self.paths[idx]).convert('RGB')
        img = self.transform(img)

        # Extract labels
        emotion_label = self.targets[idx]
        valence = torch.tensor(float(self.valences[idx]), dtype=torch.float32)
        arousal = torch.tensor(float(self.arousals[idx]), dtype=torch.float32)

        return img.data, (emotion_label, valence, arousal)


def ConcordanceCorCoeff(prediction, ground_truth):
    """
    Calculate Concordance Correlation Coefficient (CCC) between predictions and ground truth.
    Measures agreement between two variables on a -1 to 1 scale.

    Args:
        prediction: Predicted values tensor
        ground_truth: Ground truth values tensor

    Returns:
        CCC score (higher is better, 1.0 means perfect agreement)
    """
    mean_gt = torch.mean(ground_truth, 0)
    mean_pred = torch.mean(prediction, 0)
    var_gt = torch.var(ground_truth, 0)
    var_pred = torch.var(prediction, 0)
    v_pred = prediction - mean_pred
    v_gt = ground_truth - mean_gt
    cor = torch.sum(v_pred * v_gt) / (torch.sqrt(torch.sum(v_pred ** 2)) * torch.sqrt(torch.sum(v_gt ** 2)))
    sd_gt = torch.std(ground_truth)
    sd_pred = torch.std(prediction)
    numerator = 2 * cor * sd_gt * sd_pred
    denominator = var_gt + var_pred + (mean_gt - mean_pred) ** 2
    ccc = numerator / denominator
    return ccc


def ConcordanceCorCoeffLoss(prediction, ground_truth):
    """
    Loss function based on Concordance Correlation Coefficient.
    Used for training regression on valence and arousal.

    Args:
        prediction: Predicted values tensor
        ground_truth: Ground truth values tensor

    Returns:
        Loss value (lower is better)
    """
    return (1 - ConcordanceCorCoeff(prediction, ground_truth)) / 2


class AttentionLoss(nn.Module):
    """
    Attention loss to encourage diversity among multiple attention heads.
    Penalizes similarity between different attention head outputs to promote
    each head learning different aspects of the features.

    The loss is computed as the average MSE between all pairs of head outputs,
    weighted by the number of pairs.
    """
    def __init__(self):
        super(AttentionLoss, self).__init__()

    def forward(self, x):
        """
        Calculate attention loss across multiple heads.

        Args:
            x: List of attention head outputs (tensors)

        Returns:
            Attention loss value (scalar tensor)
        """
        num_head = len(x)
        loss = 0
        cnt = 0
        if num_head > 1:
            # Compute pairwise MSE between all head pairs
            for i in range(num_head - 1):
                for j in range(i + 1, num_head):
                    mse = F.mse_loss(x[i], x[j])
                    cnt = cnt + 1
                    loss = loss + mse
            loss = cnt / (loss + eps)
        else:
            loss = 0
        return loss


def run_training():
    """
    Main training loop for AffectNet emotion recognition model.
    Trains DDAMNet on emotion classification and valence-arousal regression.
    """
    args = parse_args()

    # Input size configuration
    input_size = (112, 112)  # For MixedFeatureNet

    # Set device (GPU if available, else CPU)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Configure cuDNN for best performance on GPU
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.enabled = True

    # Initialize model
    model = DDAMNet(num_class=args.num_class, num_head=args.num_head)
    model.to(device)

    # Create training dataset
    train_dataset = TaskDataset(
        file=os.path.join(r'F:\BaiduNetdiskDownload\AffectNetDataset\Manually_Annotated', "mediapipe_train_filter.json"),
        root=r'F:\BaiduNetdiskDownload\AffectNetDataset\Manually_Annotated\Mp_Filter_Annotated_Images',
        transform=transforms.Compose([
            transforms.Resize(input_size),
            transforms.RandomHorizontalFlip(),
            transforms.RandomApply([
                transforms.RandomAffine(20, scale=(0.8, 1), translate=(0.2, 0.2)),
            ], p=0.7),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
            transforms.RandomErasing()]
        ))

    # Calculate class weights for imbalanced dataset
    (unique, counts) = np.unique(train_dataset.targets, return_counts=True)
    cw = 1 / counts
    cw /= cw.min()
    class_weights = {i: cwi for i, cwi in zip(unique, cw)}

    print('Train set size:', train_dataset.__len__())
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=args.batch_size,
                                               num_workers=args.workers,
                                               shuffle=False,
                                               pin_memory=True,
                                               drop_last=True)

    # Create validation dataset
    val_dataset = TaskDataset(
        file=os.path.join(r'F:\BaiduNetdiskDownload\AffectNetDataset\Manually_Annotated', "mediapipe_test_filter.json"),
        root=r'F:\BaiduNetdiskDownload\AffectNetDataset\Manually_Annotated\Mp_Filter_Annotated_Images',
        transform=transforms.Compose([
            transforms.Resize(input_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])]))

    print('Validation set size:', val_dataset.__len__())
    val_loader = torch.utils.data.DataLoader(val_dataset,
                                             batch_size=args.batch_size,
                                             num_workers=args.workers,
                                             shuffle=False,
                                             pin_memory=True,
                                             drop_last=True)

    # Define loss functions
    weights = torch.FloatTensor(list(class_weights.values())).to(device)
    criterion_cls = torch.nn.CrossEntropyLoss(weight=weights).to(device)
    criterion_at = AttentionLoss()
    criterion_valence = ConcordanceCorCoeffLoss
    criterion_arousal = ConcordanceCorCoeffLoss

    # Setup optimizer and learning rate scheduler
    params = list(model.parameters())
    optimizer = torch.optim.Adam(params, args.lr, weight_decay=0)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.6)

    # Training loop
    best_acc = 0
    for epoch in range(1, args.epochs + 1):
        running_loss = 0.0
        correct_sum = 0
        iter_cnt = 0
        train_mse_valence = train_mse_arousal = 0

        model.train()
        for (imgs, labels) in tqdm(train_loader):
            iter_cnt += 1
            optimizer.zero_grad()

            # Move data to device
            imgs = imgs.to(device)
            labels = [l.to(device) for l in labels]
            targets, vLabel, aLabel = labels
            targets = targets.to(device)

            # Forward pass
            out, feat, heads, out2 = model(imgs)

            # Calculate total loss (classification + attention + valence + arousal)
            loss = criterion_cls(out, targets) + 0.1 * criterion_at(heads) + 2.5 * criterion_valence(out2[:, 0],
                                                                                                     vLabel) + 2.5 * criterion_arousal(
                out2[:, 1], aLabel)

            # Backward pass and optimize
            loss.backward()
            optimizer.step()

            # Track metrics
            running_loss += loss
            _, predicts = torch.max(out, 1)
            correct_num = torch.eq(predicts, targets).sum()
            correct_sum += correct_num

            # Track regression metrics
            train_mse_valence += ((out2[:, 0] - vLabel) ** 2).float().sum()
            train_mse_arousal += ((out2[:, 1] - aLabel) ** 2).float().sum()

        # Calculate epoch metrics
        acc = correct_sum.float() / float(train_dataset.__len__())
        running_loss = running_loss / iter_cnt
        mse_valense = train_mse_valence / iter_cnt
        mse_arousal = train_mse_arousal / iter_cnt

        tqdm.write('[Epoch %d] Training accuracy: %.4f. Loss: %.3f. LR %.6f valense(acc) %.4f arousal(acc) %.4f' % (
        epoch, acc, running_loss, optimizer.param_groups[0]['lr'], mse_valense, mse_arousal))

        # Validation loop
        with torch.no_grad():
            running_loss = 0.0
            iter_cnt = 0
            bingo_cnt = 0
            sample_cnt = 0
            val_mse_valence = val_mse_arousal = 0
            model.eval()
            for imgs, labels in tqdm(val_loader):
                imgs = imgs.to(device)
                labels = [l.to(device) for l in labels]
                targets, vLabel, aLabel = labels
                out, feat, heads, out2 = model(imgs)

                # Calculate validation loss
                loss = criterion_cls(out, targets) + 0.1 * criterion_at(heads) + 2.5 * criterion_valence(out2[:, 0],
                                                                                                         vLabel) + 2.5 * criterion_arousal(
                    out2[:, 1], aLabel)

                running_loss += loss
                iter_cnt += 1
                _, predicts = torch.max(out, 1)
                correct_num = torch.eq(predicts, targets)
                bingo_cnt += correct_num.sum().cpu()
                sample_cnt += out.size(0)

                # Track validation regression metrics
                mse_valense = ((out2[:, 0] - vLabel) ** 2).float().sum()
                val_mse_valence += mse_valense
                mse_arousal = ((out2[:, 1] - aLabel) ** 2).float().sum()
                val_mse_arousal += mse_arousal

            # Calculate validation metrics
            running_loss = running_loss / iter_cnt
            mse_valense = val_mse_valence / iter_cnt
            mse_arousal = val_mse_arousal / iter_cnt

            # Update learning rate
            scheduler.step()

            # Calculate accuracy
            acc = bingo_cnt.float() / float(sample_cnt)
            acc = np.around(acc.numpy(), 4)
            best_acc = max(acc, best_acc)
            tqdm.write("[Epoch %d] Validation accuracy:%.4f. Loss:%.3f valense:%.3f arousal:%.3f" % (
            epoch, acc, running_loss, mse_valense, mse_arousal))
            tqdm.write("best_acc:" + str(best_acc))

            # Save model checkpoint if accuracy exceeds threshold
            if acc > 0.5:
                torch.save({'iter': epoch,
                            'model_state_dict': model.state_dict(),
                            'optimizer_state_dict': optimizer.state_dict(), },
                           os.path.join('checkpoints', "affecnet8_epoch" + str(epoch) + "_acc" + str(acc) + ".pth"))
                tqdm.write('Model saved.')


if __name__ == "__main__":
    run_training()

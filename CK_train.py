import os
import argparse
import matplotlib.pyplot as plt
import numpy as np
import math
import itertools
from tqdm import tqdm
from model.GWACO import FERVT
from torchvision import transforms, datasets
from sklearn.metrics import balanced_accuracy_score
from sklearn.metrics import confusion_matrix
import torch
import torch.nn.functional as F

def temperature_scaled_softmax(logits, temperature=1.0):
    return F.softmax(logits / temperature, dim=-1)


class mycs(torch.nn.Module):
    def __init__(self, temperature=1.0, label_smoothing=0.1):
        super(mycs, self).__init__()
        self.temperature = temperature
        self.label_smoothing = label_smoothing

    def forward(self, logits, targets):
        # 应用温度缩放
        scaled_logits = logits / self.temperature
        # 计算交叉熵损失
        n_classes = scaled_logits.size(-1)
        with torch.no_grad():
            # 创建平滑的标签分布
            true_dist = torch.zeros_like(scaled_logits)
            true_dist.fill_(self.label_smoothing / (n_classes - 1))
            true_dist.scatter_(1, targets.data.unsqueeze(1), 1.0 - self.label_smoothing)
        return torch.mean(torch.sum(-true_dist * F.log_softmax(scaled_logits, dim=-1), dim=-1))


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--CK_path', type=str, default='datasets/CK+48', help='CK dataset path.')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size.')
    parser.add_argument('--lr', type=float, default=0.001, help='Initial learning rate for sgd.')
    parser.add_argument('--workers', default=0, type=int, help='Number of data loading workers.')
    parser.add_argument('--epochs', type=int, default=40, help='Total training epochs.')
    return parser.parse_args()


def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title, fontsize=16)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j] * 100, fmt) + '%',
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('Actual', fontsize=18)
    plt.xlabel('Predicted', fontsize=18)
    plt.tight_layout()


#0=angry 1=disgust 2=fear 3=happy 4=neutral 5=sad 6=surprise
class_names = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']

def run_training():
    args = parse_args()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.enabled = True

    model=FERVT(device)

    data_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomApply([
            transforms.RandomRotation(5),
            transforms.RandomCrop(224, padding=8)
        ], p=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
        transforms.RandomErasing(scale=(0.02, 0.25)),
    ])

    train_dataset = datasets.ImageFolder(f'{args.CK_path}/train', transform=data_transforms)

    print('Whole train set size:', train_dataset.__len__())

    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=args.batch_size,
                                               num_workers=args.workers,
                                               shuffle=True,
                                               pin_memory=True)

    data_transforms_val = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])])

    val_dataset = datasets.ImageFolder(f'{args.CK_path}/val', transform=data_transforms_val)

    print('Validation set size:', val_dataset.__len__())

    val_loader = torch.utils.data.DataLoader(val_dataset,
                                             batch_size=args.batch_size,
                                             num_workers=args.workers,
                                             shuffle=False,
                                             pin_memory=True)

    criterion_cls = mycs(temperature=1.1,label_smoothing=0.2)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda x: ((1 + math.cos(x * math.pi / args.epochs)) / 2) * (
                1 - 0.2) + 0.2)

    best_acc = 0
    for epoch in tqdm(range(1, args.epochs + 1)):
        running_loss = 0.0
        correct_sum = 0
        iter_cnt=0
        model.train()
        for (imgs, targets) in train_loader:
            iter_cnt += 1
            optimizer.zero_grad()

            imgs = imgs.to(device)
            targets = targets.to(device)

            output=model(imgs)
            loss=criterion_cls(output,targets)

            loss.backward()
            optimizer.step()

            running_loss += loss

            _, predicts = torch.max(output, 1)
            correct_num = torch.eq(predicts, targets).sum()
            correct_sum += correct_num

        acc = correct_sum.float() / float(train_dataset.__len__())
        running_loss = running_loss / iter_cnt
        tqdm.write('[Epoch %d] Training accuracy: %.4f. Loss: %.3f. LR %.6f' % (epoch, acc, running_loss, optimizer.param_groups[0]['lr']))
        with torch.no_grad():
            running_loss = 0.0
            iter_cnt = 0
            bingo_cnt = 0
            sample_cnt = 0

            ## for calculating balanced accuracy
            y_true = []
            y_pred = []

            model.eval()
            for (imgs, targets) in val_loader:
                imgs = imgs.to(device)
                targets = targets.to(device)

                output = model(imgs)
                loss = criterion_cls(output,targets)

                running_loss += loss

                _, predicts = torch.max(output, 1)
                correct_num = torch.eq(predicts, targets)
                bingo_cnt += correct_num.sum().cpu()
                sample_cnt += output.size(0)

                y_true.append(targets.cpu().numpy())
                y_pred.append(predicts.cpu().numpy())

                if iter_cnt == 0:
                    all_predicted = predicts
                    all_targets = targets
                else:
                    all_predicted = torch.cat((all_predicted, predicts), 0)
                    all_targets = torch.cat((all_targets, targets), 0)
                iter_cnt += 1
            running_loss = running_loss / iter_cnt
            scheduler.step()

            acc = bingo_cnt.float() / float(sample_cnt)
            acc = np.around(acc.numpy(), 4)
            best_acc = max(acc, best_acc)

            y_true = np.concatenate(y_true)
            y_pred = np.concatenate(y_pred)
            balanced_acc = np.around(balanced_accuracy_score(y_true, y_pred), 4)

            tqdm.write(
                "[Epoch %d] Validation accuracy:%.4f. bacc:%.4f. Loss:%.3f" % (epoch, acc, balanced_acc, running_loss))
            tqdm.write("best_acc:" + str(best_acc))

            if acc == 1 and acc == best_acc:
                torch.save({'iter': epoch,
                            'model_state_dict': model.state_dict(),
                            'optimizer_state_dict': optimizer.state_dict(), },
                           os.path.join('checkpoints', "CK+"+".pth"))
                tqdm.write('Model saved.')

                # Compute confusion matrix
                matrix = confusion_matrix(all_targets.data.cpu().numpy(), all_predicted.cpu().numpy())
                np.set_printoptions(precision=2)
                plt.figure(figsize=(10, 8))
                # Plot normalized confusion matrix
                plot_confusion_matrix(matrix, classes=class_names, normalize=True,
                                      title='CK+ Confusion Matrix (acc: %0.2f%%)' % (acc * 100))

                plt.savefig(os.path.join('checkpoints', "CK+_epoch" + str(epoch) + "_acc" + str(acc) + "_bacc" + str(
                    balanced_acc) + ".png"))
                plt.close()

if __name__ == "__main__":
        run_training()



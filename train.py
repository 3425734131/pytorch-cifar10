import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import torch.backends.cudnn as cudnn
import torchvision
import torchvision.transforms as transforms
from torchvision.datasets import CIFAR10
from torch.utils.data import DataLoader
from model import pyramidnet
import argparse
from tensorboardX import SummaryWriter
import pandas as pd
from sklearn.metrics import precision_score, recall_score, f1_score

# 用于保存每一轮的指标
metrics = []


def train(epoch):
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    all_preds = []
    all_targets = []

    for batch_idx, (inputs, targets) in enumerate(train_loader):
        inputs = inputs.to(device)
        targets = targets.to(device)
        outputs = net(inputs)
        loss = criterion(outputs, targets)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        # 保存预测和真实标签用于后续计算精度、召回率
        all_preds.extend(predicted.cpu().numpy())
        all_targets.extend(targets.cpu().numpy())

    acc = 100 * correct / total
    precision = precision_score(all_targets, all_preds, average='macro', zero_division=1)
    recall = recall_score(all_targets, all_preds, average='macro', zero_division=1)
    f1 = f1_score(all_targets, all_preds, average='macro', zero_division=1)

    print(f"Train Epoch: {epoch} | Loss: {train_loss / len(train_loader):.4f} | "
          f"Acc: {acc:.2f}% | Precision: {precision:.4f} | Recall: {recall:.4f} | F1: {f1:.4f}")

    # 将指标保存到 TensorBoard
    writer.add_scalar('log/train error', 100 - acc, epoch)
    writer.add_scalar('log/train precision', precision, epoch)
    writer.add_scalar('log/train recall', recall, epoch)
    writer.add_scalar('log/train f1', f1, epoch)

    return {
        'epoch': epoch,
        'train_loss': train_loss / len(train_loader),
        'train_acc': acc,
        'train_precision': precision,
        'train_recall': recall,
        'train_f1': f1
    }


def test(epoch, best_acc):
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(test_loader):
            inputs = inputs.to(device)
            targets = targets.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            # 保存预测和真实标签用于后续计算精度、召回率
            all_preds.extend(predicted.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())

    acc = 100 * correct / total
    precision = precision_score(all_targets, all_preds, average='macro', zero_division=1)
    recall = recall_score(all_targets, all_preds, average='macro', zero_division=1)
    f1 = f1_score(all_targets, all_preds, average='macro', zero_division=1)

    print(f"Test Epoch: {epoch} | Loss: {test_loss / len(test_loader):.4f} | "
          f"Acc: {acc:.2f}% | Precision: {precision:.4f} | Recall: {recall:.4f} | F1: {f1:.4f}")

    # 将指标保存到 TensorBoard
    writer.add_scalar('log/test error', 100 - acc, epoch)
    writer.add_scalar('log/test precision', precision, epoch)
    writer.add_scalar('log/test recall', recall, epoch)
    writer.add_scalar('log/test f1', f1, epoch)

    # 如果测试精度更好，保存模型
    if acc > best_acc:
        print('==> Saving model..')
        state = {
            'net': net.state_dict(),
            'acc': acc,
            'epoch': epoch,
        }
        if not os.path.isdir('save_model'):
            os.mkdir('save_model')
        torch.save(state, './save_model/ckpt.pth')
        best_acc = acc
    print('best test accuracy is ', best_acc)

    return {
               'epoch': epoch,
               'test_loss': test_loss / len(test_loader),
               'test_acc': acc,
               'test_precision': precision,
               'test_recall': recall,
               'test_f1': f1
           }, best_acc


def save_metrics_to_excel():
    # 将metrics保存为DataFrame并输出到Excel文件
    df = pd.DataFrame(metrics)
    df.to_excel('training_metrics.xlsx', index=False)
    print("Metrics saved to training_metrics.xlsx")


if __name__ == '__main__':
    best_acc = 0

    # 解析命令行参数 python train.py --resume ckpt.pth
    parser = argparse.ArgumentParser(description='cifar10 classification models')
    parser.add_argument('--resume', default=None, help='')
    parser.add_argument('--logdir', type=str, default='logs', help='')
    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    print('==> Preparing data..')
    transforms_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    transforms_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    dataset_train = CIFAR10(root='../data', train=True,
                            download=True, transform=transforms_train)
    dataset_test = CIFAR10(root='../data', train=False,
                           download=True, transform=transforms_test)
    train_loader = DataLoader(dataset_train, batch_size=128,
                              shuffle=True, num_workers=4)
    test_loader = DataLoader(dataset_test, batch_size=100,
                             shuffle=False, num_workers=4)

    # 创建模型并转移到GPU
    net = pyramidnet()
    net = net.to(device)
    num_params = sum(p.numel() for p in net.parameters() if p.requires_grad)
    print('The number of parameters of model is', num_params)

    if args.resume is not None:
        checkpoint = torch.load('./save_model/' + args.resume)
        net.load_state_dict(checkpoint['net'])

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.1,
                          momentum=0.9, weight_decay=1e-4)
    # optimizer = torch.optim.RMSprop(
    #     net.parameters(),
    #     lr=0.001,  # 学习率
    #     alpha=0.99,  # 平滑系数（用于加权平均梯度的平方）
    #     weight_decay=1e-4  # 权重衰减
    # )

    # optimizer = torch.optim.Adam(
    #     net.parameters(),
    #     lr=0.001,  # 学习率
    #     weight_decay=1e-4  # 权重衰减
    # )

    # 使用 Adadelta 优化器
    # optimizer = optim.Adadelta(
    #     net.parameters(),
    #     lr=1.0,
    #     weight_decay=1e-4  # 权重衰减（L2 正则化）
    # )

    decay_epochs = [150, 200]
    step_lr_scheduler = lr_scheduler.MultiStepLR(optimizer, decay_epochs, gamma=0.1)
    writer = SummaryWriter(args.logdir)

    if args.resume is not None:
        best_acc = test(epoch=0, best_acc=0)
        print('best test accuracy is ', best_acc)
    else:
        for epoch in range(200):
            step_lr_scheduler.step()
            train_metrics = train(epoch)
            test_metrics, best_acc = test(epoch, best_acc)
            train_metrics.update(test_metrics)
            metrics.append(train_metrics)

    # 保存指标到Excel文件
    save_metrics_to_excel()

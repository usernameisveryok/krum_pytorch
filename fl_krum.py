from sympy import failing_assumptions
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torch.distributed as dist
import os
import torch.multiprocessing as mp
import torch.nn.functional as F
import logging
import argparse
from tensorboardX import SummaryWriter


# define convnet
class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1)
        self.pool1 = nn.MaxPool2d(kernel_size=3, stride=3)
        self.conv2 = nn.Conv2d(16, 64, kernel_size=4, stride=1, padding=0)
        self.pool2 = nn.MaxPool2d(kernel_size=4, stride=4)
        self.fc1 = nn.Linear(64 * 2 * 2, 384)
        self.fc2 = nn.Linear(384, 192)
        self.fc3 = nn.Linear(192, 10)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.pool1(self.relu(self.conv1(x)))
        x = self.pool2(self.relu(self.conv2(x)))
        x = x.view(-1, 64 * 2 * 2)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x


def pars_args():
    parser = argparse.ArgumentParser(description="Train AWF")
    parser.add_argument("--expname", type=str, required=True)  # run name
    parser.add_argument("-g", type=int, required=True)  # gpu id
    parser.add_argument("-p", type=int, required=True)  # 通信端口
    parser.add_argument("-f", type=int, required=True)  # fail节点个数
    parser.add_argument("-m", type=int, default=1)  # krum选择的个数
    parser.add_argument("-e", type=int, default=200)  # epoch
    parser.add_argument("-b", type=int, default=512)  # batchsize
    parser.add_argument("--momentum", type=int, default=0.9)  # momentum
    parser.add_argument("--lr", type=int, default=1e-1)  # learning rate
    parser.add_argument("--lableflip", action="store_true")
    parser.add_argument("--bitfail", action="store_true")
    parser.add_argument("--krum", action="store_true")
    args = parser.parse_args()

    return args

# 调用用dist.all_reduce，无法实现防御以及bitfail攻击


def all_reduce_gradients(model, world_size):
    for param in model.parameters():
        dist.all_reduce(param.grad.data, op=dist.ReduceOp.SUM)
        param.grad.data /= float(world_size)


def krum_deprecated(model, world_size, args):
    m = args.m
    num_fail = args.f
    faillist = list(range(1, args.f+1))
    assert world_size - m > 2 * num_fail+2 and m > 0
    for param in model.parameters():
        grads = []
        for i in range(world_size):
            grads.append(param.grad.data.clone())
        for i in range(world_size):
            dist.broadcast(grads[i], i)
        # do bitfail
        if args.bitfail:
            grads[faillist[0]] = -grads[faillist[0]]
            for i in range(1, len(faillist)):
                grads[faillist[i]] = grads[faillist[0]]

        s = krum_select_deprecated(grads, num_fail)
        retgrade = grads[s]
        del grads[s]
        for i in range(m-1):
            s = krum_select_deprecated(grads, num_fail)
            retgrade += grads[s]
            del grads[s]
        retgrade = retgrade/m
        param.grad.data.copy_(retgrade)

# 使用广播原语让每个结点得到所有节点的梯度


def all_reduce_gradientsv2(model, world_size, args):
    faillist = list(range(1, args.f+1))
    for param in model.parameters():
        grads = []
        for i in range(world_size):
            grads.append(param.grad.data.clone())
        for i in range(world_size):
            dist.broadcast(grads[i], i)

        if args.bitfail:
            grads[faillist[0]] = -grads[faillist[0]]
            for i in range(1, len(faillist)):
                grads[faillist[i]] = grads[faillist[0]]
        retgrad = grads[0].clone()
        for i in range(1, world_size):
            retgrad += grads[i]
        retgrad /= float(world_size)
        return param.grad.data.copy_(retgrad)

# krum 计算距离+选择 （由于性能原因已废弃）


def krum_select_deprecated(grads, num_faulty_workers):
    n = len(grads)
    b = num_faulty_workers
    assert n - b - 2 >= 0
    scores = []
    for i in range(n):
        dists = []
        for j in range(n):
            if i != j:
                dists.append(torch.norm(grads[i] - grads[j]))
        dists.sort()
        score = sum(dists[:n - b - 2])
        scores.append(score)
    selected_worker = scores.index(min(scores))
    return selected_worker
# krum 计算距离 返回节点梯度之间距离字典


def krum_cal_dis(grads, num_faulty_workers):
    n = len(grads)
    b = num_faulty_workers
    dic = dict()
    assert n - b - 2 >= 0
    for i in range(n):
        for j in range(n):
            if i != j:
                if (i, j)not in dic:
                    dic[(i, j)] = torch.dist(grads[i], grads[j])
                    dic[(j, i)] = dic[(i, j)]
    return dic
# multikrum & krum 当输入m=1的时候自动退化成krum算法


def krum(model, world_size, args):
    m = args.m
    num_fail = args.f
    faillist = list(range(1, args.f+1))
    assert world_size - m > 2 * num_fail+2 and m > 0
    for param in model.parameters():
        grads = []
        for i in range(world_size):
            grads.append(param.grad.data.clone())
        for i in range(world_size):
            dist.broadcast(grads[i], i)
        # do bitfail
        if args.bitfail:
            grads[faillist[0]] = -grads[faillist[0]]
            for i in range(1, len(faillist)):
                grads[faillist[i]] = grads[faillist[0]]
        dic = krum_cal_dis(grads, num_fail)
        worldlist = list(range(world_size))
        retgrad = None
        for i in range(m):
            selected_worker = krum_choose_worker(dic, worldlist, num_fail)
            if retgrad == None:
                retgrad = grads[selected_worker].clone()  # type: ignore
            else:
                retgrad += grads[selected_worker]  # type: ignore
        retgrad = retgrad / m
        param.grad.data.copy_(retgrad)
#


def krum_choose_worker(dic, worldlist, num_fail):
    scores = []
    world_size = len(worldlist)
    selected_worker = None
    for i in worldlist:
        diss = []
        for j in worldlist:
            if i != j:
                diss.append(dic[(i, j)])
        diss.sort()
        scores.append(sum(diss[:world_size - num_fail - 2]))
        # print(scores.index(min(scores)))
        selected_worker = worldlist[scores.index(min(scores))]
    worldlist.remove(selected_worker)
    return selected_worker


# 读取数据集并transofrm
transform = transforms.Compose(
    [transforms.RandomHorizontalFlip(),
     transforms.RandomCrop(32, padding=4),
     transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

# 使用测试集测试模型


def evaluate(model, testloader):
    correct = 0
    total = 0
    with torch.no_grad():
        for data in testloader:
            inputs, labels = data
            inputs, labels = inputs.cuda(), labels.cuda()
            outputs = model(inputs)
            pred = torch.argmax(outputs, 1)
            total += labels.size(0)
            correct += (pred == labels).sum().item()
    return 100 * correct / total


def main_worker(rank, world_size):
    args = pars_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = f'{args.g}'
    logging.basicConfig(
        level=logging.INFO, filename=f"./log/node{rank}.log", format=f'NODE {rank} - %(asctime)s -  %(message)s')
    logger = logging.getLogger(__name__)
    writer = SummaryWriter(f'./runs/{args.expname}/node{rank}')
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = f'{args.p}'
    dist.init_process_group("gloo", rank=rank, world_size=world_size)
    logger.info(f"start")

    model = torchvision.models.resnet18()
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            torch.nn.init.xavier_uniform_(
                m.weight, gain=nn.init.calculate_gain('relu'))
    model = model.cuda()

    # 定义超参数
    learning_rate = args.lr
    batch_size = args.b
    num_epochs = args.e
    momentum = args.momentum

    # 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(),
                          lr=learning_rate, momentum=momentum)
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                            download=True, transform=transform)
    testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                           download=True, transform=transform)
    train_sampler = torch.utils.data.distributed.DistributedSampler(  # type: ignore
        trainset)  # type: ignore
    trainloader = DataLoader(
        trainset, batch_size=batch_size, sampler=train_sampler, num_workers=2)
    testloader = DataLoader(testset, batch_size=batch_size, num_workers=1)
    # 训练模型
    itr = 0
    faillist = list(range(1, args.f+1))
    for epoch in range(num_epochs):
        train_sampler.set_epoch(epoch)
        running_loss = 0.0
        correct = 0
        total = 0
        for i, data in enumerate(trainloader, 0):
            inputs, labels = data
            inputs, labels = inputs.cuda(), labels.cuda()
            # 标签反转
            if args.lableflip and rank in faillist:
                labels = world_size - labels-1
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            if args.krum:
                krum(model, world_size, args)
            else:
                all_reduce_gradientsv2(model, world_size, args)
            optimizer.step()
            running_loss += loss.item()

            # 标签反转回来
            if args.lableflip and rank in faillist:
                labels = world_size - labels-1
            # train acc
            total += labels.size(0)
            pred = torch.argmax(outputs, 1)
            correct += (pred == labels).sum().item()
            itr += 1
            if(itr % 10 == 0):
                writer.add_scalar(f"NODE{rank}/runingloss", running_loss, itr)
        epoch_loss = running_loss / (i + 1)  # type: ignore
        epoch_acc = 100 * correct / total
        logger.info(
            f"Epoch {epoch + 1}, Loss: {epoch_loss}, Accuracy: {epoch_acc}%")
        writer.add_scalar(f"NODE{rank}/acc", epoch_acc, epoch)
        writer.add_scalar(f"NODE{rank}/loss", running_loss, epoch)
        if epoch % 10 == 0:
            tacc = evaluate(model, testloader)
            writer.add_scalar(f"NODE{rank}/testacc", tacc, epoch)
            logger.info(
                f"Epoch {epoch + 1}, TestAccuracy: {tacc}%")
    logger.info("end train")


if __name__ == "__main__":
    world_size = 10
    mp.spawn(main_worker, args=(world_size,),  # type: ignore
             nprocs=world_size, join=True)

import torch
import os
import torch.utils.data as data
from dataset import DataLoader
from mobile_yolo2 import yolo_mobile
from mobilev3_yolov2 import yolo_mobilev3
from region_loss import loss_v2
from train_config import traincfg
#cuda
torch.cuda.set_device(0)


data_train = DataLoader(traincfg.data_list, traincfg)
train_loader = data.DataLoader(data_train, batch_size=traincfg.batch_size, shuffle=True, num_workers=0)
train_len = len(data_train)

obj_class = traincfg.classes
# obj_class = 2
num_anchor = traincfg.nanchors
pre_boxes = traincfg.anchors

net = yolo_mobile(nclass=obj_class, nanchors=num_anchor)
net = net.cuda()
optimizer = torch.optim.SGD(net.parameters(), lr=traincfg.lr, weight_decay=traincfg.weight_decay, momentum=0.9)
criterion = loss_v2(num_classes=obj_class, anchors=pre_boxes, num_anchors=num_anchor)
l_criterion = torch.nn.MSELoss(reduction="sum")
r_criterion = torch.nn.MSELoss(reduction="sum")

def adjust_learning_rate(epoch):
    lr = traincfg.lr
    if epoch > 1000:
        lr = lr / 1000000
    elif epoch > 980:
        lr = lr / 100000
    elif epoch > 900:
        lr = lr / 10000
    elif epoch > 600:
        lr = lr / 1000
    elif epoch > 400:
        lr = lr / 100
    elif epoch > 250:
        lr = lr / 10
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr

max_batch = train_len // traincfg.batch_size

def train():
    for epoch in range(traincfg.epochs):
        print("Epoch {}".format(epoch))
        net.train()
        batch = 0
        for i, (images, target) in enumerate(train_loader):
            batch += 1
            images = images.cuda()

            optimizer.zero_grad()
            out, lt, rt = net(images)
            loss = criterion(out, target)
            loss.backward()
            optimizer.step()

            for params in optimizer.param_groups:
                lr_cur = params['lr']

            if batch % 5 == 0:
                print("Epoch:{}/{} || Epochiter: {}/{} || loss: {:.4f} || LR: {:.8f}"
                      .format(epoch, traincfg.epochs, max_batch, batch, loss.item(), lr_cur))

        adjust_learning_rate(epoch)
        if (epoch % 5 == 0 and epoch > 0):
            torch.save(net.state_dict(), traincfg.model_save + "/" + "yolo_{}.pth".format(epoch))

def train_type():
    for epoch in range(traincfg.epochs):
        print("Epoch {}".format(epoch))
        net.train()
        batch = 0
        for i, (images, target) in enumerate(train_loader):
            batch += 1
            images = images.cuda()

            optimizer.zero_grad()
            out, lt, rt = net(images)
            t_typel = target[:, 5].cuda().unsqueeze(1)
            t_typer = target[:, 6].cuda().unsqueeze(1)
            loss_lt = l_criterion(t_typel, lt)
            loss_rt = r_criterion(t_typer, rt)
            loss_yolo = criterion(out, target)
            loss = loss_yolo + loss_lt + loss_rt
            loss.backward()
            optimizer.step()

            for params in optimizer.param_groups:
                lr_cur = params['lr']

            if batch % 5 == 0:
                print("Epoch:{}/{} || Epochiter: {}/{} || loss: {:.4f} || LR: {:.8f}"
                      .format(epoch, traincfg.epochs, max_batch, batch, loss.item(), lr_cur))

        adjust_learning_rate(epoch)
        if (epoch % 5 == 0 and epoch > 0):
            torch.save(net.state_dict(), traincfg.model_save + "/" + "yolo_{}.pth".format(epoch))

if __name__ == "__main__":
    # train()
    train_type()




























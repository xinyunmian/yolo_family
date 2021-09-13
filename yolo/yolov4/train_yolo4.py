import torch
import os
import torch.utils.data as data
from dataset import DataLoader
from yolov4.yolov4_slim import Yolo4SlimNet
from yolov4.yolov4_config import cfg4 as traincfg
from yolov4.yolov4_loss import Yolov4Loss
#cuda
torch.cuda.set_device(0)

def adjust_learning_rate(epoch, optimizer):
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

def train_slim():
    # data load
    data_train = DataLoader(traincfg.data_list, traincfg)
    train_loader = data.DataLoader(data_train, batch_size=traincfg.batch_size, shuffle=True, num_workers=0)
    train_len = len(data_train)

    obj_class = traincfg.classes
    all_anchors = traincfg.anchors
    anchor_mask = traincfg.anchor_mask

    net = Yolo4SlimNet(config=traincfg)
    net = net.cuda()
    optimizer = torch.optim.SGD(net.parameters(), lr=traincfg.lr, weight_decay=traincfg.weight_decay, momentum=0.9)

    yolo_losses = []
    for i in range(3):
        mask = anchor_mask[i]
        anchors_now = []
        for m in mask:
            anchors_now += all_anchors[m * 2: (m + 1) * 2]
        yolo_losses.append(Yolov4Loss(num_classes=obj_class, anchors=anchors_now, num_anchors=3, anchor_mask=mask))

    max_batch = train_len // traincfg.batch_size

    net.train()
    for epoch in range(traincfg.epochs):
        print("Epoch {}".format(epoch))
        batch = 0
        for i, (images, target) in enumerate(train_loader):
            batch += 1
            images = images.cuda()

            output13, output26, output52= net(images)
            loss52 = yolo_losses[0](output52, target)
            loss26 = yolo_losses[1](output26, target)
            loss13 = yolo_losses[2](output13, target)
            loss = loss52 + loss26 + loss13

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            for params in optimizer.param_groups:
                lr_cur = params['lr']

            if batch % 5 == 0:
                print("Epoch:{}/{} || Epochiter: {}/{} || loss: {:.4f} || LR: {:.8f}"
                      .format(epoch, traincfg.epochs, max_batch, batch, loss.item(), lr_cur))

        adjust_learning_rate(epoch, optimizer)
        if (epoch % 5 == 0 and epoch > 0):
            torch.save(net.state_dict(), traincfg.model_save + "/" + "yolo_{}.pth".format(epoch))

if __name__ == "__main__":
    train_slim()




























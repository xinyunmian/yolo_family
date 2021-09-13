import torch
import os
import torch.utils.data as data
from yolov3.create_label_yolov3 import DataLoader_Yolov3, collate_fn
from yolov3.yolov3_slim import Yolo3LayerSlim
from yolov3.yolov3_config import cfg3 as traincfg
from yolov3.yolov3_loss2 import Yolov3LossNew
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

def train_slimNew():
    # data load
    data_train = DataLoader_Yolov3(traincfg.data_list, traincfg)
    train_loader = data.DataLoader(data_train, batch_size=traincfg.batch_size, shuffle=True, num_workers=0, collate_fn=collate_fn)
    train_len = len(data_train)

    obj_class = traincfg.classes
    all_anchors = traincfg.anchors
    anchor_mask = traincfg.anchor_mask
    strides = traincfg.strides

    net = Yolo3LayerSlim(config=traincfg)
    net = net.cuda()

    # create optimizer
    backbone_params = list(map(id, net.backbone.parameters()))
    logits_params = filter(lambda p: id(p) not in backbone_params, net.parameters())
    params = [
        {"params": logits_params, "lr": traincfg.lr},
        {"params": net.backbone.parameters(), "lr": traincfg.lr},
    ]
    optimizer = torch.optim.SGD(params, lr=traincfg.lr, weight_decay=traincfg.weight_decay, momentum=0.9)

    yolo_losses = []
    for i in range(3):
        stride = strides[i]
        mask = anchor_mask[i]
        anchors_now = [(all_anchors[m * 2], all_anchors[m * 2 + 1]) for m in mask]
        yolo_losses.append(Yolov3LossNew(num_classes=obj_class, anchors=anchors_now, num_anchors=3, stride=stride))

    max_batch = train_len // traincfg.batch_size

    net.train()
    for epoch in range(traincfg.epochs):
        print("Epoch {}".format(epoch))
        batch = 0
        for i, (images, label) in enumerate(train_loader):
            batch += 1
            images = images.cuda()
            label = label.cuda()

            output13, output26, output52= net(images)
            loss52 = yolo_losses[0](output52, label)
            loss26 = yolo_losses[1](output26, label)
            loss13 = yolo_losses[2](output13, label)
            loss = loss52 + loss26 + loss13

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            for params in optimizer.param_groups:
                lr_cur = params['lr']

            if batch % 5 == 0:
                print("Epoch:{}/{} || Epochiter: {}/{} || loss: {:.4f} || loss8: {:.4f} || loss16: {:.4f} || loss32: {:.4f} || LR: {:.8f}"
                      .format(epoch, traincfg.epochs, max_batch, batch, loss.item(), loss52.item(), loss26.item(), loss13.item(), lr_cur))

        adjust_learning_rate(epoch, optimizer)
        if (epoch % 5 == 0 and epoch > 0):
            torch.save(net.state_dict(), traincfg.model_save + "/" + "yolo_{}.pth".format(epoch))

if __name__ == "__main__":
    train_slimNew()




























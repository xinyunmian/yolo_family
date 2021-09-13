import torch
import os
import torch.utils.data as data
from create_data import DataLoader_Yolo
from darknet2pytorch import darknetCfg_to_pytorchModel
from train_config import traincfg as datacfg
#cuda
torch.cuda.set_device(0)

data_train = DataLoader_Yolo(datacfg.data_list, datacfg)
train_loader = data.DataLoader(data_train,batch_size=datacfg.batch_size,shuffle=True,num_workers=0)
train_len = len(data_train)

model = darknetCfg_to_pytorchModel(datacfg.cfgfile)
model.cuda()

optimizer = torch.optim.SGD(model.parameters(), lr=datacfg.lr, weight_decay=datacfg.weight_decay, momentum=0.9)

def adjust_learning_rate(epoch):
    lr = datacfg.lr
    if epoch > 800000000000000000:
        lr = lr / 1000000
    elif epoch > 65000000000000:
        lr = lr / 100000
    elif epoch > 600000000000:
        lr = lr / 10000
    elif epoch > 2500:
        lr = lr / 1000
    elif epoch > 1000:
        lr = lr / 100
    elif epoch > 500:
        lr = lr / 10
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr

max_batch = train_len // datacfg.batch_size

def train_yolo():
    for epoch in range(datacfg.epochs):
        print("Epoch {}".format(epoch))
        model.train()
        batch = 0
        for i, (images, target) in enumerate(train_loader):
            batch += 1
            images = images.cuda()

            optimizer.zero_grad()
            loss = model(images, target)
            loss.backward()
            optimizer.step()

            for params in optimizer.param_groups:
                lr_cur = params['lr']

            if batch % 2 == 0:
                print("Epoch:{}/{} || Epochiter: {}/{} || loss: {:.4f} || LR: {:.8f}"
                      .format(epoch, datacfg.epochs, max_batch, batch, loss.item(), lr_cur))

        adjust_learning_rate(epoch)
        if (epoch % 100 == 0 and epoch > 0):
            model.seen = (epoch + 1) * train_len
            model.save_weights(datacfg.model_save + "/" + "yolo_{}.weights".format(epoch))
            # torch.save(model.state_dict(), datacfg.model_save + "/" + "yolo_{}.pth".format(epoch))

if __name__ == "__main__":
    import numpy as np
    train_yolo()















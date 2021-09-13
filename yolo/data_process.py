import numpy as np
import os
import cv2
import random
import shutil
import xml.etree.ElementTree as ET

def create_age_gender_label(imgdir, txtsave):
    age_gender = open(txtsave, mode = "w+")
    for root, dirs, files in os.walk(imgdir):
        for file in files:
            if file.endswith("jpg"):
                root = root.replace('\\', '/')
                imgpath = root + "/" + file
                dir, imgname = os.path.split(imgpath)
                splitname = file.split("_")#获取年龄，性别
                tage = splitname[0]#年龄
                tgender = splitname[1]#性别  0:男  1:女
                splitdir = dir.split("/")#获取文件夹名称
                subdir_name = splitdir[-1]

                savep = subdir_name + "/" + imgname
                savedata = savep + " " + tage + " " + tgender#utkface_align1/a.jpg 100 1
                age_gender.write(savedata)
                age_gender.write("\n")
    age_gender.close()

def create_age_gender_label2(imgdir, txtsave):
    age_gender = open(txtsave, mode = "w+")
    for root, dirs, files in os.walk(imgdir):
        for file in files:
            if file.endswith("jpg"):
                root = root.replace('\\', '/')
                splitroot = root.split("/")
                imgpath = root + "/" + file
                dir, imgname = os.path.split(imgpath)
                splitname = file.split("A")#获取年龄，性别
                splitname2 = splitname[1].split(".")  # 获取年龄
                tage = splitname2[0].strip('r')#年龄
                tgender = splitroot[-1]#性别  0:男  1:女
                splitdir = dir.split("/")#获取文件夹名称
                subdir_name = splitdir[-1]

                savep = "all_faces2/" + imgname
                savedata = savep + " " + tage + " " + tgender#utkface_align1/a.jpg 100 1
                age_gender.write(savedata)
                age_gender.write("\n")
    age_gender.close()

def create_classfication_label(imgdir, txtsave):
    label_classfication = open(txtsave, mode="w+")
    for root, dirs, files in os.walk(imgdir):
        for file in files:
            if file.endswith("jpg"):
                root = root.replace('\\', '/')
                splitroot = root.split("/")
                dirname = splitroot[-1]
                imgpath = dirname + "/" + file
                savedata = imgpath + " " + dirname
                label_classfication.write(savedata)
                label_classfication.write("\n")
    label_classfication.close()

def img_augment(imgdir, savedir):
    for root, dirs, files in os.walk(imgdir):
        for file in files:
            if file.endswith("jpg") or file.endswith("png") or file.endswith("bmp"):
                root = root.replace('\\', '/')
                splitname = file.split(".")
                name = splitname[0]
                imgpath = root + "/" + file
                img = cv2.imread(imgpath, cv2.IMREAD_COLOR)
                imgmirror = cv2.flip(img, 1)
                savepath1 = savedir + "/" + name + ".jpg"
                savepath2 = savedir + "/m" + name + ".jpg"
                cv2.imwrite(savepath1, img)
                cv2.imwrite(savepath2, imgmirror)

def shuffle_txt(srctxt, shuffletxt):
    FileNamelist = []
    files = open(srctxt, 'r+')
    for line in files:
        line = line.strip('\n')  # 删除每一行的\n
        FileNamelist.append(line)
    print('len ( FileNamelist ) = ', len(FileNamelist))
    files.close()
    random.shuffle(FileNamelist)

    file_handle = open(shuffletxt, mode='w+')
    for idx in range(len(FileNamelist)):
        str = FileNamelist[idx]
        file_handle.write(str)
        file_handle.write("\n")
    file_handle.close()

#文件夹1有若干图片，文件夹2有很多图片，移除和文件夹1相同的文件
def remove_filesjpg(imgdir, dirremove, save):
    for root, dirs, files in os.walk(imgdir):
        for file in files:
            if file.endswith("jpg"):
                filelist = list(file)
                filelist.pop(-5)
                filep = "".join(filelist)

                remove_name = dirremove + "/" + filep
                save_name = save + "/" + filep
                img = cv2.imread(remove_name, cv2.IMREAD_COLOR)
                if img is not None:
                    shutil.copy(remove_name, save_name)

def select_txt_by_img(imgpath, txtdir, save):
    f1 = open(imgpath, 'r+')
    lines = f1.readlines()
    for i in range(len(lines)):
        line = lines[i].strip("\n")
        img_name = line.split("/")[-1]
        txtname = img_name.replace("jpg", "txt")
        savetxt = save + "/" + txtname

        txtpath = txtdir + "/" + txtname
        shutil.move(txtpath, savetxt)


#txt中有若干图片路径，将图片移动到另一个文件夹
def remove_filestxt(txtpath, dir, save):
    f1 = open(txtpath, 'r+')
    lines = f1.readlines()
    for i in range(len(lines)):
        line = lines[i].strip("\n")
        img_name = line.split("/")[-1]
        imgpath = dir + "/" + img_name
        txtpath = imgpath.replace("jpg", "txt")
        saveimg = save + "/" + img_name
        savetxt = saveimg.replace("jpg", "txt")

        img = cv2.imread(imgpath, cv2.IMREAD_COLOR)
        if img is not None:
            shutil.move(imgpath, saveimg)
            shutil.move(txtpath, savetxt)

def idl_to_txt(idl_path, txtdir):
    if not os.path.exists(txtdir):
        os.mkdir(txtdir)
    f1 = open(idl_path, 'r+')
    lines = f1.readlines()
    for i in range(len(lines)):
        line = lines[i]
        line = line.replace(":", ";")
        img_dir = line.split(";")[0]
        img_boxs = line.split(";")[1]
        img_dir = img_dir.replace('"', "")
        img_name = img_dir.split("/")[1]
        txt_name = img_name.split(".")[0]
        img_extension = img_name.split(".")[1]
        img_boxs = img_boxs.replace(",", "")
        img_boxs = img_boxs.replace("(", "")
        img_boxs = img_boxs.split(")")

        # imgpath = "D:/data/imgs/head/brainwash/" + img_dir
        # imgsave = txtdir + "/" + txt_name + ".jpg"
        # imgmat = cv2.imread(imgpath, cv2.IMREAD_COLOR)
        # cv2.imwrite(imgsave, imgmat)

        if (img_extension == 'jpg'):
            f = open(txtdir + "/" + txt_name + ".txt", 'a')
            imgpath = "D:/data/imgs/head/brainwash/" + img_dir
            imgsave = txtdir + "/" + txt_name + ".jpg"
            imgmat = cv2.imread(imgpath, cv2.IMREAD_COLOR)
            cv2.imwrite(imgsave, imgmat)
            for n in range(len(img_boxs) - 1):
                box = img_boxs[n]
                box = box.split(" ")
                f.write(' '.join(['0', str((float(box[1]) + float(box[3])) / (2 * 640)),
                                      str((float(box[2]) + float(box[4])) / (2 * 480)),
                                      str((float(box[3]) - float(box[1])) / 640),
                                      str((float(box[4]) - float(box[2])) / 480)]) + '\n')
            f.close()
        f1.close()

def convert(size, box):
    dw = 1./size[0]
    dh = 1./size[1]
    x = (box[0] + box[1])/2.0
    y = (box[2] + box[3])/2.0
    w = box[1] - box[0]
    h = box[3] - box[2]
    x = x*dw
    w = w*dw
    y = y*dh
    h = h*dh
    return (x,y,w,h)

def xml_to_txt(xmldir, txtsavedir):
    classes = ["person", "hat"]
    for root, dirs, files in os.walk(xmldir):
        for file in files:
            if file.endswith("xml"):
                xmlpath = xmldir + "/" + file
                xml_name = file.split(".")[0]
                in_file = open(xmlpath, encoding="utf-8", mode='r+')
                txt_name = xml_name + ".txt"
                txtpath = txtsavedir + "/" + txt_name
                out_file = open(txtpath, 'w+')
                tree = ET.parse(in_file)
                root = tree.getroot()
                size = root.find('size')
                w = int(size.find('width').text)
                h = int(size.find('height').text)

                for obj in root.iter('object'):
                    cls = obj.find('name').text
                    if cls not in classes:
                        continue
                    cls_id = classes.index(cls)
                    xmlbox = obj.find('bndbox')
                    b = (
                    float(xmlbox.find('xmin').text), float(xmlbox.find('xmax').text), float(xmlbox.find('ymin').text),
                    float(xmlbox.find('ymax').text))
                    bb = convert((w, h), b)
                    out_file.write(str(cls_id) + " " + " ".join([str(a) for a in bb]) + '\n')
            out_file.close()
            in_file.close()

def get_img_list(imgdir, listpath, endname):
    list_file = open(listpath, 'w+')
    for root, dirs, files in os.walk(imgdir):
        for file in files:
            if file.endswith(endname):
                root = root.replace('\\', '/')
                imgpath = root + "/" + file
                list_file.write(imgpath + "\n")
    list_file.close()

def get_img_norepeat(imgdir, endname):
    list_file = []
    for root, dirs, files in os.walk(imgdir):
        for file in files:
            if file.endswith(endname):
                if file not in list_file:
                    list_file.append(file)
    return list_file

def get_img_bydir(imgdir1, imgdir2, savedir, endname):
    for root, dirs, files in os.walk(imgdir1):
        for file in files:
            if file.endswith(endname):
                imgpath = imgdir2 + "/" + file
                savepath = savedir + "/" + file
                imgmat = cv2.imread(imgpath, cv2.IMREAD_COLOR)
                cv2.imwrite(savepath, imgmat)


def hattxt_headtxt(txtdir, savedir):
    for root, dirs, files in os.walk(txtdir):
        for file in files:
            if file.endswith("txt"):
                root = root.replace('\\', '/')
                txtpath = root + "/" + file
                dir, name = os.path.split(txtpath)
                in_file = open(txtpath, encoding="utf-8", mode='r+')

                imgpath = txtpath.replace('txt', 'jpg')
                img_mat = cv2.imread(imgpath, cv2.IMREAD_COLOR)
                im_h, im_w, _ = img_mat.shape

                outpath = savedir + "/" + name
                out_file = open(outpath, 'w+')
                lines = in_file.readlines()
                for i in range(len(lines)):
                    line = lines[i]
                    label = line.split(" ")
                    objc = int(label[0])
                    cx = float(label[1])
                    cy = float(label[2])
                    cw = float(label[3])
                    ch = float(label[4])
                    if objc == 1:
                        hatw = cw * im_w
                        hath = ch * im_h
                        cxs = cx
                        cys = (cy * im_h + 0.2 * hath) / im_h
                        cws = hatw * 1.22 / im_w
                        chs = hath * 1.21 / im_h
                        out_file.write("1" + " " + str(cxs) + " " + str(cys) + " " + str(cws) + " " + str(chs) + '\n')
                in_file.close()
                out_file.close()

def ears_errorlab_process(pathlist, errortxt):
    out_file = open(errortxt, 'w+')
    f = open(pathlist, 'r', encoding='utf-8')
    for imgp in f.readlines():
        num0 = 0
        num1 = 0
        num2 = 0
        num3 = 0
        num4 = 0
        num5 = 0
        imgp = imgp.rstrip()
        labelp = imgp.replace('jpg', 'txt')
        labs = open(labelp, 'r')
        labline = labs.readlines()
        if len(labline) == 0:
            continue
        else:
            for lab in labline:
                lab = lab.rstrip().split(" ")
                class_id = int(lab[0])
                if class_id==0:
                    num0 += 1
                if class_id==1:
                    num1+= 1
                if class_id==2:
                    num2 += 1
                if class_id==3:
                    num3 += 1
                if class_id==4:
                    num4 += 1
                if class_id==5:
                    num5 += 1
        if num0>1 or num1>1 or num2>1 or num3>1 or num4>1 or num5>1:
            print(labelp)
            savedata = labelp + "\n"
            out_file.write(savedata)

def show_ears_lable(pathlist):
    f = open(pathlist, 'r', encoding='utf-8')
    cv2.namedWindow('result2', cv2.WINDOW_NORMAL)
    for imgp in f.readlines():
        imgp = imgp.rstrip()
        imgmat = cv2.imread(imgp)
        hei, wid, _ = imgmat.shape
        labelp = imgp.replace('jpg', 'txt')
        labs = open(labelp, 'r')
        labline = labs.readlines()
        if len(labline) == 0:
            continue
        else:
            for lab in labline:
                lab = lab.rstrip().split(" ")
                class_id = str(int(lab[0]))
                cx = float(lab[1]) * wid
                cy = float(lab[2]) * hei
                iw = float(lab[3]) * wid
                ih = float(lab[4]) * hei
                xmin = int(cx - 0.5 * iw)
                ymin = int(cy - 0.5 * ih)
                xmax = int(cx + 0.5 * iw)
                ymax = int(cy + 0.5 * ih)
                imgmat = cv2.rectangle(imgmat, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
                imgmat = cv2.putText(imgmat, class_id, (int(cx), ymin), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 4)
        cv2.imshow('result2', imgmat)
        cv2.waitKey(0)

if __name__ == "__main__":
    imgpath = "D:/data/imgs/facePicture/glasses/train"
    txtpath = "D:/data/imgs/facePicture/glasses/train.txt"
    shuffletxt = "D:/data/imgs/facePicture/glasses/shuffle_train.txt"
    # create_age_gender_label(imgpath, txtpath)
    # create_age_gender_label2(imgpath, txtpath)
    # create_classfication_label(imgpath, txtpath)
    # shuffle_txt(txtpath, shuffletxt)

    txtpath2 = "D:/data/imgs/facePicture/ears/ears.txt"
    shufflepath = "D:/data/imgs/facePicture/ears/earss.txt"
    # shuffle_txt(txtpath2, shufflepath)

    dir1 = "D:/data/imgs/rename/watermask_ID/train3/mark_ID2_resize"
    dir2 = "D:/data/imgs/rename/watermask_ID/create_label/mask_ID2_resize"
    dir3 = "D:/data/imgs/rename/watermask_ID/create_label/mask_wide"
    # remove_files(dir1, dir2, dir3)
    # remove_filesjpg(dir1, dir2, dir3)
    # img_augment(dir1, dir2)
    # get_img_bydir(dir1, dir2, dir3, "jpg")

    path1 = "C:/Users/xym/Desktop/rename/widerface.txt"
    dir4 = "D:/data/imgs/widerface/train/widerface_origintxt"
    dir5 = "D:/data/imgs/widerface/train/widerface_select"
    # remove_filestxt(path1, dir4, dir5)
    # select_txt_by_img(path1, dir4, dir5)

    idlp = "D:/data/imgs/head/brainwash/brainwash_train.idl"
    txtd = "D:/data/imgs/head/head1"
    # idl_to_txt(idlp, txtd)

    xmld = "D:/BaiduNetdiskDownload/VOC2028/Annotations"
    txtsd = "D:/BaiduNetdiskDownload/VOC2028/txt"
    # xml_to_txt(xmld, txtsd)

    imgd = "D:/data/imgs/facePicture/ears/bb"
    txtlist = "D:/data/imgs/facePicture/ears/ears.txt"
    # get_img_list(imgd, txtlist, endname="jpg")
    # imglist = get_img_norepeat(imgd, endname="png")

    hattxt = "D:/data/imgs/head/hatbelt" #D:/data/imgs/head/hatbelt
    savetxt = "D:/data/imgs/head/hatbelt_txt"
    # hattxt_headtxt(hattxt, savetxt)

    earlist = "D:/data/imgs/facePicture/ears/earss.txt"
    earerrortxt = "D:/data/imgs/facePicture/ears/error.txt"
    # ears_errorlab_process(earlist, earerrortxt)
    show_ears_lable(earlist)




















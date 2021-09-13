import os
import cv2
import sys
import math
import time
import torch
import random
import numpy as np
from yolo_last_layers import convert2cpu, convert2cpu_long, bbox_iou

# boxes: [x, y, w, h, det_conf, cls_max_conf, cls_max_id]
def nms(boxes, nms_thresh):
	if len(boxes) == 0:
		return boxes
	det_confs = torch.zeros(len(boxes))
	for i in range(len(boxes)):
		det_confs[i] = boxes[i][4]
	# descending
	_, sortIds = torch.sort(det_confs, descending=True)
	out_boxes = []
	for i in range(len(boxes)):
		box_i = boxes[sortIds[i]]
		if box_i[4] > 0:
			out_boxes.append(box_i)
			for j in range(i+1, len(boxes)):
				box_j = boxes[sortIds[j]]
				if bbox_iou(box_i, box_j, x1y1x2y2=False) > nms_thresh:
					box_j[4] = 0
	return out_boxes

def plot_boxes_cv2(img, boxes):
	try:
		width = img.shape[1]
		height = img.shape[0]
	except:
		print('[Error]: The type of image in <plot_boxes_cv2> unsupported...')
		sys.exit(-1)
	for i in range(len(boxes)):
		box = boxes[i]
		x1 = (box[0] - box[2]/2.0) * width
		y1 = (box[1] - box[3]/2.0) * height
		x2 = (box[0] + box[2]/2.0) * width
		y2 = (box[1] + box[3]/2.0) * height
		cls_id = str(int(box[6]))
		img = cv2.rectangle(img, (x1.data, y1.data), (x2.data, y2.data), (0, 255, 255), 2)
		img = cv2.putText(img, cls_id, (x1.data, y1.data), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 4)
	return img

def get_boxes_yolo2(output, anchors, num_anchors, num_classes, conf_thresh, use_sigmoid, stride):
	anchors = [anchor / stride for anchor in anchors]
	anchor_step = len(anchors) // num_anchors
	if output.dim() == 3:
		output = output.unsqueeze(0)
	batch_size = output.size(0)
	assert output.size(1) == (5 + num_classes) * num_anchors
	h = output.size(2)
	w = output.size(3)
	all_boxes = []
	output = output.view(batch_size*num_anchors, 5+num_classes, h*w).transpose(0, 1).contiguous().view(5+num_classes, batch_size*num_anchors*h*w)
	grid_x = torch.linspace(0, w-1, w).repeat(h, 1).repeat(batch_size*num_anchors, 1, 1).view(batch_size*num_anchors*h*w).type_as(output)
	grid_y = torch.linspace(0, h-1, h).repeat(w, 1).t().repeat(batch_size*num_anchors, 1, 1).view(batch_size*num_anchors*h*w).type_as(output)
	xs = torch.sigmoid(output[0]) + grid_x
	ys = torch.sigmoid(output[1]) + grid_y
	anchor_w = torch.Tensor(anchors).view(num_anchors, anchor_step).index_select(1, torch.LongTensor([0]))
	anchor_h = torch.Tensor(anchors).view(num_anchors, anchor_step).index_select(1, torch.LongTensor([1]))
	anchor_w = anchor_w.repeat(batch_size, 1).repeat(1, 1, h*w).view(batch_size*num_anchors*h*w).type_as(output)
	anchor_h = anchor_h.repeat(batch_size, 1).repeat(1, 1, h*w).view(batch_size*num_anchors*h*w).type_as(output)
	ws = torch.exp(output[2]) * anchor_w
	hs = torch.exp(output[3]) * anchor_h
	det_confs = torch.sigmoid(output[4])
	if use_sigmoid:
		cls_confs = torch.nn.Sigmoid()(torch.autograd.Variable(output[5: 5+num_classes].transpose(0, 1))).data
	else:
		cls_confs = torch.nn.Softmax(dim=1)(torch.autograd.Variable(output[5: 5+num_classes].transpose(0, 1))).data
	cls_max_confs, cls_max_ids = torch.max(cls_confs, 1)
	cls_max_confs = cls_max_confs.view(-1)
	cls_max_ids = cls_max_ids.view(-1)
	sz_hw = h * w
	sz_hwa = sz_hw * num_anchors
	det_confs = convert2cpu(det_confs)
	cls_max_confs = convert2cpu(cls_max_confs)
	cls_max_ids = convert2cpu_long(cls_max_ids)
	xs = convert2cpu(xs)
	ys = convert2cpu(ys)
	ws = convert2cpu(ws)
	hs = convert2cpu(hs)
	for b in range(batch_size):
		boxes = []
		for cy in range(h):
			for cx in range(w):
				for i in range(num_anchors):
					ind = b*sz_hwa + i*sz_hw + cy*w + cx
					det_conf = det_confs[ind]
					if det_conf > conf_thresh:
						bcx = xs[ind]
						bcy = ys[ind]
						bw = ws[ind]
						bh = hs[ind]
						cls_max_conf = cls_max_confs[ind]
						cls_max_id = cls_max_ids[ind]
						box = [bcx/w, bcy/h, bw/w, bh/h, det_conf, cls_max_conf, cls_max_id]
						boxes.append(box)
		all_boxes.append(boxes)
	return all_boxes

def plot_boxes_cv2_type(img, boxes):
	try:
		width = img.shape[1]
		height = img.shape[0]
	except:
		print('[Error]: The type of image in <plot_boxes_cv2> unsupported...')
		sys.exit(-1)
	for i in range(len(boxes)):
		box = boxes[i]
		x1 = (box[0] - box[2]/2.0) * width
		y1 = (box[1] - box[3]/2.0) * height
		x2 = (box[0] + box[2]/2.0) * width
		y2 = (box[1] + box[3]/2.0) * height
		cls_id = str(int(box[6]))
		cx = int(0.1 * width)
		if int(box[6]) == 0:
			cy = int(0.25 * height)
		if int(box[6]) == 1:
			cy = int(0.45 * height)
		typelr = np.around(box[7].data.cpu().numpy(), 3)
		img = cv2.rectangle(img, (x1.data, y1.data), (x2.data, y2.data), (0, 255, 255), 2)
		img = cv2.putText(img, cls_id, (x1.data, y1.data), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 4)
		img = cv2.putText(img, str(typelr), (cx, cy), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
	return img

def get_boxes_yolo2_type(output, anchors, num_anchors, num_classes, conf_thresh, use_sigmoid, stride):
	anchors = [anchor / stride for anchor in anchors]
	anchor_step = len(anchors) // num_anchors
	if output.dim() == 3:
		output = output.unsqueeze(0)
	batch_size = output.size(0)
	assert output.size(1) == (6 + num_classes) * num_anchors
	h = output.size(2)
	w = output.size(3)
	all_boxes = []
	output = output.view(batch_size*num_anchors, 6+num_classes, h*w).transpose(0, 1).contiguous().view(6+num_classes, batch_size*num_anchors*h*w)
	grid_x = torch.linspace(0, w-1, w).repeat(h, 1).repeat(batch_size*num_anchors, 1, 1).view(batch_size*num_anchors*h*w).type_as(output)
	grid_y = torch.linspace(0, h-1, h).repeat(w, 1).t().repeat(batch_size*num_anchors, 1, 1).view(batch_size*num_anchors*h*w).type_as(output)
	xs = torch.sigmoid(output[0]) + grid_x
	ys = torch.sigmoid(output[1]) + grid_y
	anchor_w = torch.Tensor(anchors).view(num_anchors, anchor_step).index_select(1, torch.LongTensor([0]))
	anchor_h = torch.Tensor(anchors).view(num_anchors, anchor_step).index_select(1, torch.LongTensor([1]))
	anchor_w = anchor_w.repeat(batch_size, 1).repeat(1, 1, h*w).view(batch_size*num_anchors*h*w).type_as(output)
	anchor_h = anchor_h.repeat(batch_size, 1).repeat(1, 1, h*w).view(batch_size*num_anchors*h*w).type_as(output)
	ws = torch.exp(output[2]) * anchor_w
	hs = torch.exp(output[3]) * anchor_h
	if use_sigmoid:
		cls_confs = torch.nn.Sigmoid()(torch.autograd.Variable(output[5: 5+num_classes].transpose(0, 1))).data
	else:
		cls_confs = torch.nn.Softmax(dim=1)(torch.autograd.Variable(output[5: 5+num_classes].transpose(0, 1))).data
	cls_max_confs, cls_max_ids = torch.max(cls_confs, 1)
	cls_max_confs = cls_max_confs.view(-1)
	cls_max_ids = cls_max_ids.view(-1)
	sz_hw = h * w
	sz_hwa = sz_hw * num_anchors
	det_confs = torch.sigmoid(output[4])
	det_confs = convert2cpu(det_confs)

	det_conf_lr = torch.sigmoid(output[5+num_classes])
	det_conf_lr = convert2cpu(det_conf_lr)

	cls_max_confs = convert2cpu(cls_max_confs)
	cls_max_ids = convert2cpu_long(cls_max_ids)
	xs = convert2cpu(xs)
	ys = convert2cpu(ys)
	ws = convert2cpu(ws)
	hs = convert2cpu(hs)
	for b in range(batch_size):
		boxes = []
		for cy in range(h):
			for cx in range(w):
				for i in range(num_anchors):
					ind = b*sz_hwa + i*sz_hw + cy*w + cx
					det_conf = det_confs[ind]
					if det_conf > conf_thresh:
						bcx = xs[ind]
						bcy = ys[ind]
						bw = ws[ind]
						bh = hs[ind]
						cls_max_conf = cls_max_confs[ind]
						cls_max_id = cls_max_ids[ind]

						lrt = det_conf_lr[ind]


						box = [bcx/w, bcy/h, bw/w, bh/h, det_conf, cls_max_conf, cls_max_id, lrt]
						boxes.append(box)
		all_boxes.append(boxes)
	return all_boxes

def get_boxes_yolo3(output, anchors, anchor_masks, allnum_anchors, num_classes, conf_thresh, use_sigmoid, strides):
	anchor_step = len(anchors) // allnum_anchors
	all_boxes = []
	for i in range(len(output)):
		anchor_mask = anchor_masks[i]
		stride = strides[i]
		op = output[i].data
		anchors_now = []

		for m in anchor_mask:
			anchors_now += anchors[m * anchor_step: (m+1) * anchor_step]

		num_anchors = len(anchors_now) // anchor_step
		boxes = get_boxes_yolo2(op, conf_thresh=conf_thresh, num_classes=num_classes, anchors=anchors_now,
								num_anchors=num_anchors, use_sigmoid=use_sigmoid, stride=stride)
		all_boxes.append(boxes[0])
	return all_boxes











from pathlib import Path
import logging
import time
import math
import glob
import re
from multiprocessing import Pool
import argparse
import os
import sys

import numpy as np
import torch
import torchvision
import cv2
import onnxruntime

FILE = Path(__file__).resolve()
ROOT = FILE.parents[1]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative


from utils.patch_classify.general import xywh2xyxy, clip_boxes, increment_path, print_args, yaml_save
from models.common import DetectMultiBackend
from utils.torch_utils import select_device, smart_inference_mode

logging.basicConfig(format="%(message)s", level=logging.INFO)
logger = logging.getLogger(__name__)
logger.setLevel(level=logging.INFO)


class CrackDetector:
    # detection classes
    names = ['TransverseCrack', 'LongitudinalCrack', 'AlligatorCrack', 'StripRepair', 'Marking']
    # plot colors
    color = [(0, 0, 255), (255, 0, 255), (3, 97, 255), (255, 0, 0), (255, 56, 56)]
    # match box and grid by class, grid_0 match box_0,1,2; grid_1 match box_3; grid_2 match box_4
    class_grid2box = {0: (0, 1, 2), 1: (3,), 2: (4,)}

    def __init__(self, model,  # onnx or pytorch model
                 device,  # device: cpu or cuda or cuda:0 or cuda:1 ....
                 max_bs=48,  # max batch_size
                 infer_size=640,  # inference size
                 # preprocess
                 num_workers=0, pad=1,  # padding for better performance(pad to multiple of 32)
                 # box params
                 conf_threshold_box=0.1, iou_threshold=0.5, classes_box=(0, 1, 2, 3), max_det=100,
                 ras_threshold=0.5, crack_agnostic=True, max_w=4096,
                 # grid params
                 conf_threshold_grid=0.7, match=True,
                 ):
        device = select_device(device)
        model = DetectMultiBackend(model, device=device)
        stride = model.stride
        self.device = device
        self.model = model
        self.max_bs = max_bs
        new_size = math.ceil(infer_size / stride) * stride
        if new_size != infer_size:
            logger.warning(f'infer-size {infer_size} must be multiple of max stride {stride}, updating to {new_size}')
        self.infer_size = new_size
        self.stride = stride
        assert 0 <= conf_threshold_box <= 1, f'Invalid Confidence threshold {conf_threshold_box}, ' \
                                             f'valid values are between 0.0 and 1.0'
        assert 0 <= conf_threshold_grid <= 1, f'Invalid Confidence threshold {conf_threshold_grid},' \
                                              f' valid values are between 0.0 and 1.0'
        assert 0 <= iou_threshold <= 1, f'Invalid IoU {iou_threshold}, valid values are between 0.0 and 1.0'
        assert 0 <= ras_threshold <= 1, f'Invalid IoU {ras_threshold}, valid values are between 0.0 and 1.0'
        self.conf_threshold_box = conf_threshold_box
        self.iou_threshold = iou_threshold
        self.conf_threshold_grid = conf_threshold_grid
        self.max_det = max_det
        self.match = match
        self.classes_box = classes_box
        self.ras_threshold = ras_threshold
        self.crack_agnostic = crack_agnostic
        self.crack_classes = []
        for index, name in enumerate(self.names):
            if 'Crack' in name or 'crack' in name:
                self.crack_classes.append(index)
        self.max_w = max_w
        self.num_workers = num_workers
        self.pad = int(pad*self.stride)

    @smart_inference_mode()
    def run(self, paths, save, view_img=False, save_img=False, save_txt=True):
        # directory
        save_dir = Path(save)
        if view_img or save_img or save_txt:
            for i in ['box_image', 'box_txt', 'grid_image', 'grid_txt']:
                (save_dir / i).mkdir(exist_ok=True, parents=True)
            if isinstance(paths, (str, Path)):  # path-->[path]
                paths = [paths]

        # inferences
        start = time.time()
        paths, images, raw_images, ratio = self.preprocess(paths)
        batch_size = len(paths)
        box_pred, grid_pred = self.model(images)[:2]
        box_pred, grid_pred, paths = self.postprocess(box_pred, grid_pred, paths, ratio)
        logger.info(f'batch_size is {batch_size},total inference time is {time.time() - start}s')
        assert len(box_pred) == len(grid_pred) == len(paths) == len(raw_images)

        # save result
        self.export('box', save_dir, box_pred, paths, raw_images, view_img, save_img, save_txt)
        self.export('grid', save_dir, grid_pred, paths, raw_images, view_img, save_img, save_txt)
        return box_pred, grid_pred, paths

    def preprocess(self, paths):
        paths = [str(Path(path).absolute()) for path in paths]
        images, raw_images, ratio = [], [], []
        if self.num_workers > 0:
            pool = Pool(self.num_workers)
            for path in paths:
                raw_images.append(pool.apply_async(func=read_img, args=(path,)))
            pool.close()
            pool.join()
            raw_images = [p.get() for p in raw_images]
        else:
            for path in paths:
                raw_images.append(read_img(path))
        for path, img0 in zip(paths, raw_images):
            h, w, _ = img0.shape
            new_img = resize(img0, self.infer_size, self.stride)
            new_h, new_w, _ = new_img.shape
            new_img = cv2.copyMakeBorder(new_img, self.pad, self.pad, self.pad, self.pad,
                                         cv2.BORDER_CONSTANT, value=(114, 114, 114))  # add border
            new_img = np.ascontiguousarray(new_img.transpose((2, 0, 1))[::-1])
            new_img = torch.from_numpy(new_img).to(self.model.device)
            new_img = new_img.half() if self.model.fp16 else new_img.float()  # uint8 to fp16/32
            new_img /= 255  # 0 - 255 to 0.0 - 1.0
            images.append(new_img)
            ratio.append((w / new_w, h / new_h))
        assert 0 < len(paths) <= self.max_bs, 'There are too many(no) images to process'
        self.new_size = (new_h, new_w)
        return paths, torch.stack(images), raw_images, ratio

    def postprocess(self, box_pred, grid_pred, paths, ratio):
        box_pred = self.ras(self.nms(box_pred))
        grid_pred = self.grid2box(grid_pred)
        for d in box_pred:
            d[:, :4] -= self.pad
            clip_boxes(d, self.new_size)
        for g in grid_pred:
            g[:, :4] -= self.pad
        assert len(box_pred) == len(grid_pred)
        # match gird to box
        for i, (d, g, r) in enumerate(zip(box_pred, grid_pred, ratio)):
            d[:, :4:2] *= r[0]
            d[:, 1:4:2] *= r[1]
            g[:, :4:2] *= r[0]
            g[:, 1:4:2] *= r[1]
            g_class = g[:, 5]
            # change crack classes
            new_g = []
            for gc, v in self.class_grid2box.items():
                for bc in v:
                    g_tmp = g[g_class == gc].clone()
                    g_tmp[:, 5:] = bc
                    new_g.append(g_tmp)
            g = torch.cat(new_g)
            if g.numel() > 0 and self.match:
                d_class = d[:, 5:]
                d_boxes = d[:, :4] + d_class * self.max_w
                g_boxes = g[:, :4] + g[:, 5:] * self.max_w
                iou = box_ioa(g_boxes, d_boxes)
                index = iou.max(1)[0] > 0 if d.numel() > 0 else torch.zeros(g.shape[0], device=self.device).bool()
                g = g[index]
            box_pred[i] = d
            grid_pred[i] = g

        return box_pred, grid_pred, paths

    def grid2box(self, prediction):  # [bs,gy,gx,3]-->[bs,nc,6]
        output = [torch.zeros((0, 6), device=prediction.device)] * prediction.shape[0]
        for xi, x in enumerate(prediction):
            # keep only the qualified pred
            conf, j = x.max(0)
            top, left = (conf > self.conf_threshold_grid).nonzero(as_tuple=False).T
            # generate boxes
            x = torch.stack([left, top, left + 1, top + 1, conf[top, left], j[top, left]], dim=1)
            x[:, :4] *= self.stride
            output[xi] = x
        return output

    def nms(self, prediction):  # [bs,nc,11]-->[bs,new_nc,6]
        xc = prediction[..., 4] > 0.001  # candidates
        # Settings
        max_nms = 30000  # maximum number of boxes into torchvision.ops.nms()
        time_limit = 10.0  # seconds to quit after

        t = time.time()
        output = [torch.zeros((0, 6), device=prediction.device)] * prediction.shape[0]
        for xi, x in enumerate(prediction):  # image index, image inference
            x = x[xc[xi]]  # confidence
            if not x.shape[0]:
                continue
            x[:, 5:] *= x[:, 4:5]  # conf = obj_conf * cls_conf
            box = xywh2xyxy(x[:, :4])
            conf, j = x[:, 5:].max(1, keepdim=True)
            x = torch.cat((box, conf, j.float()), 1)[conf.view(-1) > self.conf_threshold_box]
            if self.classes_box is not None:  # filter by class
                x = x[(x[:, 5:6] == torch.tensor(self.classes_box, device=x.device)).any(1)]
            n = x.shape[0]  # number of boxes
            if not n:  # no boxes
                continue
            elif n > max_nms:  # excess boxes
                x = x[x[:, 4].argsort(descending=True)[:max_nms]]  # sort by confidence
            classes = x[:, 5:6].clone()
            if self.crack_agnostic:
                classes[(classes == torch.tensor(self.crack_classes, device=x.device)).any(1)] = self.crack_classes[0]
            bias = classes * self.infer_size  # nms on each class separately
            boxes, scores = x[:, :4] + bias, x[:, 4]  # boxes (offset by class), scores
            i = torchvision.ops.nms(boxes, scores, self.iou_threshold)  # NMS
            if i.shape[0] > self.max_det:  # limit nums of boxes
                i = i[:self.max_det]
            output[xi] = x[i]
            if (time.time() - t) > time_limit:
                logger.warning(f'NMS time limit {time_limit}s exceeded')
                break  # time limit exceeded
        return output

    def ras(self, prediction):
        output = [torch.zeros((0, 6), device=prediction[0].device)] * len(prediction)
        for xi, x in enumerate(prediction):
            n = x.shape[0]
            _, order = torch.sort(x[:, 4])
            x = x[order]
            # if group the three classes of crack together
            classes = x[:, 5:6].clone()
            if self.crack_agnostic:
                classes[(classes == torch.tensor(self.crack_classes, device=x.device)).any(1)] = self.crack_classes[0]
            bias = classes * self.infer_size  # nms on each class separately
            boxes = x[:, :4] + bias
            ioa = box_ioa(boxes, boxes)
            mask = torch.arange(n).repeat(n, 1) <= torch.arange(n).view(-1, 1)  # mask low conf boxes
            ioa[mask] = 0
            x = x[ioa.sum(1) < self.ras_threshold]
            output[xi] = x
        return output

    def export(self, stage, save_dir, pred, paths, raw_images, view_img, save_img, save_txt):
        for p, dp, img in zip(paths, pred, raw_images):
            p = Path(p)
            img_path = save_dir / f'{stage}_image' / p.name
            txt_path = save_dir / f'{stage}_txt' / (p.stem + '.txt')
            img0 = img.copy()
            for *xyxy, conf, cls in dp:
                c = int(cls)
                label = f'{conf:.2f}' if stage == 'grid' else f'{self.names[c]} {conf:.2f}'
                if save_txt:  # Write to file
                    line = (self.names[c], *xyxy, conf)
                    with open(txt_path, 'a') as f:
                        f.write(('%s ' + '%g ' * (len(line) - 1)).rstrip() % line + '\n')
                if save_img or view_img:
                    img0 = plot_one_box(xyxy, img0, label=label, color=self.color[c])
            if save_img:
                cv2.imwrite(str(img_path), img0)
            if view_img:
                cv2.imshow(str(p), img0)
                cv2.waitKey(1)


def inter(box1, box2):
    return (torch.min(box1[:, None, 2:], box2[:, 2:]) - torch.max(box1[:, None, :2], box2[:, :2])).clamp(0).prod(2)


def box_ioa(box1, box2):
    area = (box1[:, 2] - box1[:, 0]) * (box1[:, 3] - box1[:, 1])
    # inter(N,M) = (rb(N,M,2) - lt(N,M,2)).clamp(0).prod(2)
    return inter(box1, box2) / area.view(-1, 1)  # iou = inter / (area1 + area2 - inter)


def plot_one_box(box, im, color=(128, 128, 128), txt_color=(255, 255, 255), label=None, line_width=3):
    # Plots one xyxy box on image im with label
    assert im.data.contiguous, 'Image not contiguous. Apply np.ascontiguousarray(im) to plot_on_box() input image.'
    lw = line_width or max(int(min(im.size) / 200), 2)  # line width
    c1, c2 = (int(box[0]), int(box[1])), (int(box[2]), int(box[3]))
    cv2.rectangle(im, c1, c2, color, thickness=lw, lineType=cv2.LINE_AA)
    if label:
        tf = max(lw - 1, 1)  # font thickness
        txt_width, txt_height = cv2.getTextSize(label, 0, fontScale=lw / 3, thickness=tf)[0]
        c2 = c1[0] + txt_width, c1[1] - txt_height - 3
        cv2.rectangle(im, c1, c2, color, -1, cv2.LINE_AA)  # filled
        cv2.putText(im, label, (c1[0], c1[1] - 2), 0, lw / 3, txt_color, thickness=tf, lineType=cv2.LINE_AA)
    return im


def resize(img, new_size, stride=32):
    h, w, _ = img.shape
    if h != 640 or w % 32 != 0:
        ratio = h / new_size
        w = round(w / ratio / stride) * stride
        img = cv2.resize(img, (w, new_size), interpolation=cv2.INTER_LINEAR)
    return img


def read_img(path):
    img = cv2.imread(path)
    # img=cv2.imdecode(np.fromfile(path,dtype=np.uint8),-1)# Chinese in Path
    assert img is not None, 'Image Not Found ' + path
    return img


def read_paths(path):
    if isinstance(path, str) and Path(path).suffix == ".txt":  # *.txt file with img/vid/dir on each line
        path = Path(path).read_text().rsplit()
    files = []
    for p in sorted(path) if isinstance(path, (list, tuple)) else [path]:
        p = str(Path(p).resolve())
        if '*' in p:
            files.extend(sorted(glob.glob(p, recursive=True)))  # glob
        elif os.path.isdir(p):
            files.extend(sorted(glob.glob(os.path.join(p, '*.*'))))  # dir
        elif os.path.isfile(p):
            files.append(p)  # files
        else:
            raise FileNotFoundError(f'{p} does not exist')
    return files


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default=ROOT / 'runs_pc/train/exp42/weights/best.pt',
                        help='model path or triton URL')
    parser.add_argument('--data', type=str, default='/home/qiyu/datasets/pavement_paper/pavement_box_grid/detect.txt',
                        help='(optional) dataset.yaml path')
    parser.add_argument('--imgsz', '--img', '--img-size', type=int, default=640, help='inference image height')
    parser.add_argument('--workers', type=int, default=0, help='number of workers')
    parser.add_argument('--bs', type=int, default=100, help='batch size')
    parser.add_argument('--pad', type=int, default=1, help='padding=pad*stride')
    parser.add_argument('--conf-thres-box', type=float, default=0.174, help='confidence threshold')
    parser.add_argument('--conf-thres-grid', type=float, default=0.576, help='confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.5, help='NMS IoU threshold')
    parser.add_argument('--ras-thres', type=float, default=0.5, help='RAS threshold')
    parser.add_argument('--max-det', type=int, default=1000, help='maximum detections per image')
    parser.add_argument('--device', default='cuda:1', help='cuda:0, cuda:1 or cpu')
    parser.add_argument('--classes', nargs='+', type=int, default=[0, 1, 2, 3],
                        help='filter by class: --classes 0, or --classes 0 1 2 3')
    parser.add_argument('--view-img', action='store_true', help='show results')
    parser.add_argument('--no-save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--no-save-img', action='store_true', help='save results to *.jpg')
    parser.add_argument('--no-match', action='store_true', help='no match grid and box')
    parser.add_argument('--no-crack-agnostic', action='store_true', help='no class-agnostic NMS&RAS for crack')
    parser.add_argument('--project', default=ROOT /'runs_pc/detect', help='save results to project/name')
    parser.add_argument('--name', default='exp', help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    opt = parser.parse_args()
    print_args(vars(opt))
    return opt


def main(opt):
    save_dir = increment_path(Path(opt.project) / opt.name, exist_ok=opt.exist_ok, mkdir=True)  # increment run
    yaml_save(save_dir / 'opt.yaml', vars(opt))
    my_model = CrackDetector(model=opt.weights,
                             device=opt.device,
                             max_bs=opt.bs,
                             infer_size=opt.imgsz,
                             num_workers=opt.workers,
                             pad=opt.pad,
                             conf_threshold_box=opt.conf_thres_box,
                             iou_threshold=opt.iou_thres,
                             classes_box=opt.classes,
                             max_det=opt.max_det,
                             ras_threshold=opt.ras_thres,
                             crack_agnostic=not opt.no_crack_agnostic,
                             max_w=4096,
                             conf_threshold_grid=opt.conf_thres_grid,
                             match=not opt.no_match)
    paths = read_paths(opt.data)
    for i in range(0, len(paths), opt.bs):
        paths_batch = paths[i:i + opt.bs]
        box_pred, grid_pred, _ = my_model.run(paths_batch, save=save_dir, save_img=not opt.no_save_img,
                                                  save_txt=not opt.no_save_txt, view_img=opt.view_img)


if __name__ == '__main__':
    opt = parse_opt()
    main(opt)

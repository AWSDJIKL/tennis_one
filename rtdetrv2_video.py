"""Copyright(c) 2023 lyuwenyu. All Rights Reserved.
"""

import torch
import torch.nn as nn
import torchvision.transforms as T
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import os
import sys
# sys.path.append(str(os.getcwd()))
from rtdetrv2.src.core import YAMLConfig
from tqdm import tqdm
import torchreid
from sklearn.cluster import DBSCAN
from rtdetrv2.references.deploy.yolo11_print_pose import *
import vis

class_names = [
    'swing',  # 1
    'other',  # 2
]


def load_osnet_model():
    model = torchreid.models.build_model(name='osnet_x1_0', num_classes=1, pretrained=True, use_gpu=True)
    torchreid.utils.load_pretrained_weights(model, './rtdetrv2/osnet_x1_0_imagenet.pth')  # 下载预训练权重
    model.eval()
    return model


def preprocess_image(image, input_size=(256, 128)):
    transform = T.Compose([
        T.Resize(input_size),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    # image = Image.open(image_path).convert('RGB')
    return transform(image).unsqueeze(0)  # 添加批次维度


def extract_features(model, image_list):
    features = []
    for img in tqdm(image_list):
        input_tensor = preprocess_image(img)
        with torch.no_grad():
            feature = model(input_tensor)  # 提取特征向量
        features.append(feature.cpu().numpy().flatten())  # 转为一维数组
    return np.array(features)


def cluster_images(features, threshold=0.4):
    # 使用 DBSCAN 进行聚类
    clustering = DBSCAN(eps=threshold, min_samples=1, metric='cosine').fit(features)
    labels = clustering.labels_
    return labels


def draw(images, labels, boxes, scores, video_writer, thrh=0.6, draw_boxes=True):
    font = ImageFont.truetype(font='./rtdetrv2/arial.ttf', size=30)  # 获得字体
    draw = ImageDraw.Draw(images)
    scr = scores
    lab = labels[scr > thrh]
    box = boxes[scr > thrh]
    scrs = scores[scr > thrh]
    if draw_boxes:
        for j, b in enumerate(box):
            draw.rectangle(list(b), outline='red', )
            draw.text((b[0], b[1]), text=f"{class_names[lab[j].item()]} {round(scrs[j].item(), 2)}", fill='blue',
                      font=font)
    opencv_image = cv2.cvtColor(np.array(images), cv2.COLOR_RGB2BGR)
    video_writer.write(opencv_image)


def extract_box(frame: Image, labels, boxes, scores, max_size, thrh=0.6):
    scr = scores
    box = boxes[scr > thrh]

    # 1.只提取swing类的框
    box = boxes[labels == 0]
    scr = scores[labels == 0]
    # 2.对box按分数排序，只要第一个
    box = box[torch.argmax(scr.squeeze())]
    # print("box: ", box)
    # print("--" * 20)
    # 3.截取box区域，写进视频
    # 扩展边框，为了把整个人都截取到
    # args.box_padding_rate = 0.4
    box_height = box[2] - box[0]
    box_width = box[3] - box[1]
    x1 = max(float(box[0] - box_width * args.box_padding_rate), 0)
    y1 = max(float(box[1] - box_height * args.box_padding_rate), 0)
    x2 = min(float(box[2] + box_width * args.box_padding_rate), frame.size[0])
    y2 = min(float(box[3] + box_height * args.box_padding_rate), frame.size[1])

    frame = frame.crop((x1, y1, x2, y2))
    iw, ih = frame.size
    if iw > max_size[0] or ih > max_size[1]:
        max_size = (iw, ih)
    return frame, max_size
    black_bg = Image.new('RGB', video_size, color='black')
    bw, bh = black_bg.size
    iw, ih = frame.size
    if iw > max_size[0] or ih > max_size[1]:
        max_size = (iw, ih)
    paste_position = ((bw - iw) // 2, (bh - ih) // 2)
    # 粘贴裁剪后的图片到黑色背景上
    black_bg.paste(frame, paste_position)

    frame = cv2.cvtColor(np.array(black_bg), cv2.COLOR_RGB2BGR)
    return frame, max_size


def paste_frame(frame_list: list[Image], max_size):
    f_list = []
    for f in frame_list:
        black_bg = Image.new('RGB', max_size, color='black')
        bw, bh = black_bg.size
        iw, ih = f.size
        paste_position = ((bw - iw) // 2, (bh - ih) // 2)
        # 粘贴裁剪后的图片到黑色背景上
        black_bg.paste(f, paste_position)
        f_list.append(cv2.cvtColor(np.array(black_bg), cv2.COLOR_RGB2BGR))
    return f_list


def main(args, ):
    """main
    """
    cfg = YAMLConfig(args.config, resume=args.resume)

    if args.resume:
        checkpoint = torch.load(args.resume, map_location='cpu')
        if 'ema' in checkpoint:
            state = checkpoint['ema']['module']
        else:
            state = checkpoint['model']
    else:
        raise AttributeError('Only support resume to load model.state_dict by now.')

    # NOTE load train mode state -> convert to deploy mode
    cfg.model.load_state_dict(state)

    class Model(nn.Module):

        def __init__(self, ) -> None:
            super().__init__()
            self.model = cfg.model.deploy()
            self.postprocessor = cfg.postprocessor.deploy()

        def forward(self, images, orig_target_sizes):
            outputs = self.model(images)
            outputs = self.postprocessor(outputs, orig_target_sizes)
            return outputs

    model = Model().to(args.device)

    # im_pil = Image.open(args.im_file).convert('RGB')
    print(os.getcwd())
    print("------------------------------------------")
    if not os.path.exists(args.video_file):
        print(f"[错误] 视频文件 {args.video_file} 不存在")
        sys.exit(1)
    cap = cv2.VideoCapture(args.video_file)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    # 处理无法获取总帧数的情况
    if total_frames <= 0:
        print("[警告] 无法获取总帧数，进度条将显示为未知模式")
        pbar = tqdm(desc="处理进度", unit="帧", dynamic_ncols=True)
    else:
        pbar = tqdm(total=total_frames, desc="处理进度", unit="帧", dynamic_ncols=True)

    ret, frame = cap.read()
    if not ret:
        pass
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    output_dir = "./video/output"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    print(f"视频宽度: {w}, 视频高度: {h}")
    video_name = os.path.basename(args.video_file)
    video_writer = cv2.VideoWriter(os.path.join(output_dir, video_name), cv2.VideoWriter_fourcc(*'mp4v'), 30, (w, h))
    crop_image_list = []
    max_size = (0, 0)

    # crop_video_writer = cv2.VideoWriter(os.path.join("video/output", "crop_" + video_name), cv2.VideoWriter_fourcc(*'mp4v'), 30, (w, h))
    while True:
        orig_size = torch.tensor([w, h])[None].to(args.device)
        transforms = T.Compose([
            T.Resize((640, 640)),
            T.ToTensor(),
        ])
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = Image.fromarray(frame)

        im_data = transforms(frame)[None].to(args.device)
        output = model(im_data, orig_size)
        labels, boxes, scores = output
        # print("labels: ", labels)
        # print("boxes: ", boxes)
        # 标注边框和动作分类
        draw(frame, labels, boxes, scores, video_writer, draw_boxes=False)

        f, max_size = extract_box(frame, labels, boxes, scores, max_size)
        crop_image_list.append(f)
        ret, frame = cap.read()
        if not ret:
            break
        pbar.update(1)
    cap.release()
    video_writer.release()
    # 计算这些图片的特征距离，进行分组，然后将图片最多的组合并成一个视频
    osnet_model = load_osnet_model()
    features = extract_features(osnet_model, crop_image_list)
    labels = cluster_images(features, threshold=0.2)
    groups = {}
    for img_path, label in zip(crop_image_list, labels):
        if label not in groups:
            groups[label] = []
        groups[label].append(img_path)
    # print("groups: ", groups.keys())
    main_id = 0
    max_count = 0
    for group_id, imgs in groups.items():
        if len(imgs) > max_count:
            max_count = len(imgs)
            main_id = group_id
    crop_image_list = groups[main_id]

    crop_video_writer = cv2.VideoWriter(os.path.join(output_dir, "crop_" + video_name), cv2.VideoWriter_fourcc(*'mp4v'),
                                        30, max_size)
    crop_image_list = paste_frame(crop_image_list, max_size)
    for f in tqdm(crop_image_list):
        crop_video_writer.write(f)
    crop_video_writer.release()
    vis.main(os.path.join(output_dir, "crop_" + video_name), args)

    # video_path = os.path.join(output_dir, "crop_" + video_name)
    # frames, keypoints, w, h = get_yolo11_keypoints(video_path)
    # output_video = cv2.VideoWriter(os.path.join(output_dir, "yolo11_crop" + video_name),
    #                                cv2.VideoWriter_fourcc(*'mp4v'), 30, (w + INFOWEIGHT, h))
    # for f, k in tzip(frames, keypoints):
    #     f = show2Dpose(k, f)
    #     output_video.write(f)
    # output_video.release()


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument('-c', '--config', type=str,
                        default="./rtdetrv2/configs/rtdetrv2/rtdetrv2_r18vd_sp3_120e_coco.yml")
    parser.add_argument('-r', '--resume', type=str, default="./rtdetrv2/best.pth")
    parser.add_argument('-bpr', '--box_padding_rate', type=float, default=0.2)
    parser.add_argument('-vf', '--video_file', type=str, default='./video/input/zed_test.mp4')
    parser.add_argument('-d', '--device', type=str, default='cuda:0')
    parser.add_argument('--gpu', type=str, default='0', help='input video')
    parser.add_argument('--fix_z', action='store_true', help='fix z axis')
    parser.add_argument('--p', action='store_true', help='create 3D')
    args = parser.parse_args()
    main(args)

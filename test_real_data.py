from __future__ import print_function, division

import argparse
import logging
import numpy as np
import cv2
import os
from pathlib import Path
from tqdm import tqdm

from lib.human_loader import StereoHumanDataset
from lib.network import RtStereoHumanModel
from config.stereo_human_config import ConfigStereoHuman as config
from lib.utils import get_novel_calib
from lib.GaussianRender import pts2render

import torch
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

import matplotlib.pyplot as plt
import torch.nn.functional as F
from torch.cuda.amp import autocast as autocast


class StereoHumanRender:
    def __init__(self, cfg_file, phase):
        self.cfg = cfg_file
        self.bs = self.cfg.batch_size

        self.model = RtStereoHumanModel(self.cfg, with_gs_render=True)
        self.dataset = StereoHumanDataset(self.cfg.dataset, phase=phase)
        self.model.cuda()
        if self.cfg.restore_ckpt:
            self.load_ckpt(self.cfg.restore_ckpt)
        self.model.eval()

    def infer_seqence(self, view_select, ratio=0.5):
        cv2.cuda.setDevice(0)
        nvof = cv2.cuda.NvidiaOpticalFlow_2_0.create(
            imageSize=(1024, 1024),   # (width, height)
            perfPreset=cv2.cuda.NvidiaOpticalFlow_2_0_NV_OF_PERF_LEVEL_SLOW,
            enableTemporalHints=False,
            enableExternalHints=False,
            enableCostBuffer=False,
            gpuId=0
        )
        total_frames = len(os.listdir(os.path.join(self.cfg.dataset.test_data_root, 'img')))
        interp_frame_count = 2

        previous_frame_depth_l = None
        previous_frame_image_l = None
        previous_frame_depth_r = None
        previous_frame_image_r = None
        
        for idx in tqdm(range(total_frames)):
            print(f"processing idx {idx}")
            item = self.dataset.get_test_item(idx, source_id=view_select)
            data = self.fetch_data(item)
            data = get_novel_calib(data, self.cfg.dataset, ratio=ratio, intr_key='intr_ori', extr_key='extr_ori')
            with torch.no_grad():
                bs = data['lmain']['img'].shape[0]
                image = torch.cat([data['lmain']['img'], data['rmain']['img']], dim=0)

                with autocast(enabled=self.cfg.raft.mixed_precision):
                    img_feat = self.model.img_encoder(image)

                if idx % interp_frame_count == 0: # this is the first frame, store the depth and image results
                    flow_up = self.model.raft_stereo(img_feat[2], iters=self.model.val_iters, test_mode=True)
                    data['lmain']['flow_pred'] = flow_up[0]
                    data['rmain']['flow_pred'] = flow_up[1]
                    self.model.flow2gsparms(image, img_feat, data, bs)
                    previous_frame_image_l = data['lmain']['grayscale'].contiguous()
                    previous_frame_depth_l = data['lmain']['depth']
                    previous_frame_image_r = data['rmain']['grayscale'].contiguous()
                    previous_frame_depth_r = data['rmain']['depth']
                else: # interpolation frames
                    # (1024, 1024, 2)
                    flow_l = self.find_opt_flow(nvof, data['lmain']['grayscale'][0], previous_frame_image_l[0])
                    flow_r = self.find_opt_flow(nvof, data['rmain']['grayscale'][0], previous_frame_image_r[0])

                    flow_l= flow_l.to(torch.float32) / 32.0
                    flow_r= flow_r.to(torch.float32) / 32.0

                    y_grid_l, x_grid_l = torch.meshgrid(
                        torch.arange(1024, device=flow_l.device, dtype=torch.float32).cuda(),
                        torch.arange(1024, device=flow_l.device, dtype=torch.float32).cuda(),
                        indexing='ij'
                    )

                    x_grid_l = x_grid_l + flow_l[:, :, 0]
                    y_grid_l = y_grid_l + flow_l[:, :, 1]

                    x_normalized_l = 2.0 * x_grid_l / (1024 - 1) - 1.0
                    y_normalized_l = 2.0 * y_grid_l / (1024 - 1) - 1.0

                    grid_l = torch.stack([x_normalized_l, y_normalized_l], dim=-1).unsqueeze(0).type(torch.float32).cuda()
                    
                    warped_l = F.grid_sample(
                        previous_frame_depth_l,
                        grid_l, 
                        mode='bilinear', 
                        padding_mode='zeros',
                    ).cuda()

                    y_grid_r, x_grid_r = torch.meshgrid(
                        torch.arange(1024, device=previous_frame_image_r.device, dtype=float).cuda(),
                        torch.arange(1024, device=previous_frame_image_r.device, dtype=float).cuda(),
                        indexing='ij'
                    )

                    x_grid_r = x_grid_r + flow_r[:, :, 0]
                    y_grid_r = y_grid_r + flow_r[:, :, 1]
                    x_normalized_r = 2.0 * x_grid_r / (1024 - 1) - 1.0
                    y_normalized_r = 2.0 * y_grid_r / (1024 - 1) - 1.0

                    grid_r = torch.stack([x_normalized_r, y_normalized_r], dim=-1).unsqueeze(0).type(torch.float32).cuda()
                    
                    warped_r = F.grid_sample(
                        previous_frame_depth_r,
                        grid_r, 
                        mode='bilinear', 
                        padding_mode='zeros',
                        align_corners=True
                    )

                    frame_data = self.model.flow2gsparms(
                        image, 
                        img_feat, 
                        data, bs, 
                        override_depth = {
                            'lmain': warped_l,
                            'rmain': warped_r, 
                        }
                    )
                    # data, _, _ = self.model(data, is_train=False)

                data = pts2render(data, bg_color=self.cfg.dataset.bg_color)

            render_novel = self.tensor2np(data['novel_view']['img_pred'])
            cv2.imwrite(self.cfg.test_out_path + '/%s_novel.jpg' % (data['name']), render_novel)

    def tensor2np(self, img_tensor):
        img_np = img_tensor.permute(0, 2, 3, 1)[0].detach().cpu().numpy()
        img_np = img_np * 255
        img_np = img_np[:, :, ::-1].astype(np.uint8)
        return img_np

    def fetch_data(self, data):
        for view in ['lmain', 'rmain']:
            for item in data[view].keys():
                data[view][item] = data[view][item].cuda().unsqueeze(0)
            # grayscale_mat = cv2.cuda_GpuMat()
            # grayscale_mat.upload(data[view]['grayscale'])
            # data[view]['grayscale'] = grayscale_mat
        return data

    def load_ckpt(self, load_path):
        assert os.path.exists(load_path)
        logging.info(f"Loading checkpoint from {load_path} ...")
        ckpt = torch.load(load_path, map_location='cuda')
        self.model.load_state_dict(ckpt['network'], strict=True)
        logging.info(f"Parameter loading done")

    def find_opt_flow(self, nvof, img1, img2):
        gpumat1 = cv2.cuda_GpuMat()
        gpumat1.upload(img1.cpu().numpy())
        gpumat2 = cv2.cuda_GpuMat()
        gpumat2.upload(img2.cpu().numpy())

        flow, cost = nvof.calc(gpumat1, gpumat2, None)
        return self.gpumat_to_tensor(flow)

    def gpumat_to_tensor(self, gpu_mat: cv2.cuda_GpuMat) -> torch.Tensor:
        h, w = gpu_mat.size()[0], gpu_mat.size()[0]
        c = gpu_mat.channels()

        class GpuMatWrapper:
            __cuda_array_interface__ = {
                "version": 3,
                "shape": (h, w, c),
                "typestr": "<i2",  # uint8, change to "<f4" for float32
                "data": (gpu_mat.cudaPtr(), False),
                "strides": (gpu_mat.step, gpu_mat.elemSize(), gpu_mat.elemSize1()),
            }

        return torch.as_tensor(GpuMatWrapper(), device="cuda").clone()


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s')
    parser = argparse.ArgumentParser()
    parser.add_argument('--test_data_root', type=str, required=True)
    parser.add_argument('--ckpt_path', type=str, required=True)
    parser.add_argument('--src_view', type=int, nargs='+', required=True)
    parser.add_argument('--ratio', type=float, default=0.5)
    arg = parser.parse_args()

    cfg = config()
    cfg_for_train = os.path.join('./config', 'stage2.yaml')
    cfg.load(cfg_for_train)
    cfg = cfg.get_cfg()

    cfg.defrost()
    cfg.batch_size = 1
    cfg.dataset.test_data_root = arg.test_data_root
    cfg.dataset.use_processed_data = False
    cfg.restore_ckpt = arg.ckpt_path
    cfg.test_out_path = './test_out'
    Path(cfg.test_out_path).mkdir(exist_ok=True, parents=True)
    cfg.freeze()

    render = StereoHumanRender(cfg, phase='test')
    render.infer_seqence(view_select=arg.src_view, ratio=arg.ratio)

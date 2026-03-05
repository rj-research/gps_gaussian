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
            perfPreset=cv2.cuda.NvidiaOpticalFlow_2_0_NV_OF_PERF_LEVEL_FAST,
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
                if idx % interp_frame_count == 0: # this is the first frame, store the depth and image results
                    print("store on frame 1?")
                    data, _, _ = self.model(data, is_train=False)
                    previous_frame_image_l = data['lmain']['img'].contiguous()
                    previous_frame_depth_l = data['lmain']['depth']
                    previous_frame_image_r = data['rmain']['img'].contiguous()
                    previous_frame_depth_r = data['rmain']['depth']
                else: # interpolation frames
                    print("interpolate")
                    flow_l = self.find_opt_flow(nvof, previous_frame_image_l, data['lmain']['img'])
                    flow_r = self.find_opt_flow(nvof, previous_frame_image_r, data['rmain']['img'])
                    data, _, _ = self.model(data, is_train=False)

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
        return data

    def load_ckpt(self, load_path):
        assert os.path.exists(load_path)
        logging.info(f"Loading checkpoint from {load_path} ...")
        ckpt = torch.load(load_path, map_location='cuda')
        self.model.load_state_dict(ckpt['network'], strict=True)
        logging.info(f"Parameter loading done")

    def find_opt_flow(self, nvof, img1, img2):
        gpumat1 = self.tensor_to_gpumat(img1)
        gpumat2 = self.tensor_to_gpumat(img2)
        flow, cost = nvof.calc(gpumat1, gpumat2, None)

        return self.gpumat_to_tensor(flow)

    def tensor_to_gpumat(self, t: torch.Tensor) -> cv2.cuda_GpuMat:
        assert t.is_cuda, "Tensor must be on GPU"
        assert t.is_contiguous(), "Tensor must be contiguous"
        assert t.dtype == torch.uint8, "Expects uint8 for grayscale"
    
        rows, cols = t.shape
        # step = t.stride(0) * t.element_size()   # bytes per row
        return cv2.cuda_GpuMat(rows, cols, cv2.CV_8UC1, t.data_ptr())
    
    def gpumat_to_tensor(self, gpu_mat: cv2.cuda_GpuMat) -> torch.Tensor:
        """Convert cv2.cuda_GpuMat to PyTorch tensor (zero-copy via DLPack)."""
        return torch.from_dlpack(gpu_mat.toDLpack())


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

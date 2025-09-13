import argparse
import os
from omegaconf import OmegaConf
import numpy as np
import cv2
import torch
import glob
import pickle
import sys
from tqdm import tqdm
import copy
import json
from transformers import WhisperModel

from musetalk.utils.face_parsing import FaceParsing
from musetalk.utils.utils import datagen
from musetalk.utils.preprocessing import get_landmark_and_bbox, read_imgs
from musetalk.utils.blending import get_image_prepare_material, get_image_blending
from musetalk.utils.utils import load_all_model
from musetalk.utils.audio_processor import AudioProcessor

import shutil
import threading
import queue
import time
import subprocess


def fast_check_ffmpeg():
    """快速检查ffmpeg是否已安装"""
    try:
        subprocess.run(["ffmpeg", "-version"], capture_output=True, check=True)
        return True
    except:
        return False


def osmakedirs(path_list):
    """创建多个目录
    Args:
        path_list: 目录路径列表
    """
    for path in path_list:
        os.makedirs(path) if not os.path.exists(path) else None


@torch.no_grad()  # 禁用梯度计算装饰器
class Avatar:
    """虚拟人物类，用于处理基于单张图片的视频和音频生成"""
    def __init__(self, avatar_id, image_path, bbox_shift, batch_size, preparation):
        """初始化Avatar类
        Args:
            avatar_id: 虚拟人物ID
            image_path: 图片文件路径
            bbox_shift: 边界框偏移量
            batch_size: 批处理大小
            preparation: 是否进行预处理
        """
        self.avatar_id = avatar_id
        self.image_path = image_path
        self.bbox_shift = bbox_shift
        # 根据版本设置不同的基础路径
        if args.version == "v15":
            self.base_path = f"./results/{args.version}/avatars/{avatar_id}"
        else:  # v1
            self.base_path = f"./results/avatars/{avatar_id}"
            
        # 设置各种路径
        self.avatar_path = self.base_path
        self.full_imgs_path = f"{self.avatar_path}/full_imgs"  # 完整图片路径
        self.coords_path = f"{self.avatar_path}/coords.pkl"  # 坐标文件路径
        self.latents_out_path = f"{self.avatar_path}/latents.pt"  # 潜在向量输出路径
        self.video_out_path = f"{self.avatar_path}/vid_output/"  # 视频输出路径
        self.mask_out_path = f"{self.avatar_path}/mask"  # 遮罩输出路径
        self.mask_coords_path = f"{self.avatar_path}/mask_coords.pkl"  # 遮罩坐标路径
        self.avatar_info_path = f"{self.avatar_path}/avator_info.json"  # 虚拟人物信息路径
        
        # 保存虚拟人物信息
        self.avatar_info = {
            "avatar_id": avatar_id,
            "image_path": image_path,  # 更新为image_path以反映实际用途
            "bbox_shift": bbox_shift,
            "version": args.version
        }
        self.preparation = preparation
        self.batch_size = batch_size
        self.idx = 0
        self.init()  # 初始化

    def init(self):
        """初始化虚拟人物，包括创建必要的目录和加载或准备数据"""
        if self.preparation:  # 如果需要预处理
            if os.path.exists(self.avatar_path):  # 检查虚拟人物目录是否存在
                # response = input(f"{self.avatar_id} exists, Do you want to re-create it ? (y/n)")
                response = "y"  # TODO: for testing
                if response.lower() == "y":  # 如果用户选择重新创建
                    shutil.rmtree(self.avatar_path)  # 删除现有目录
                    print("*********************************")
                    print(f"  creating avator: {self.avatar_id}")
                    print("*********************************")
                    osmakedirs([self.avatar_path, self.full_imgs_path, self.video_out_path, self.mask_out_path])  # 创建必要的目录
                    self.prepare_material()  # 准备材料
                else:  # 如果用户选择不重新创建
                    # 加载已存在的数据
                    # 加载单帧基础数据
                    self.input_latent = torch.load(self.latents_out_path)
                    with open(self.coords_path, 'rb') as f:
                        self.base_bbox = pickle.load(f)
                    # 读取基础帧
                    input_img_list = glob.glob(os.path.join(self.full_imgs_path, '*.[jpJP][pnPN]*[gG]'))
                    input_img_list = sorted(input_img_list, key=lambda x: int(os.path.splitext(os.path.basename(x))[0]))
                    self.base_frame = read_imgs([input_img_list[0]])[0]
                    with open(self.mask_coords_path, 'rb') as f:
                        self.base_mask_crop_box = pickle.load(f)
                    input_mask_list = glob.glob(os.path.join(self.mask_out_path, '*.[jpJP][pnPN]*[gG]'))
                    input_mask_list = sorted(input_mask_list, key=lambda x: int(os.path.splitext(os.path.basename(x))[0]))
                    self.base_mask = read_imgs([input_mask_list[0]])[0]
                    # 为 datagen 提供一个单元素列表即可循环使用
                    self.input_latent_list_cycle = [self.input_latent]
            else:  # 如果虚拟人物目录不存在
                print("*********************************")
                print(f"  creating avator: {self.avatar_id}")
                print("*********************************")
                osmakedirs([self.avatar_path, self.full_imgs_path, self.video_out_path, self.mask_out_path])  # 创建必要的目录
                self.prepare_material()  # 准备材料
        else:  # 如果不需要预处理
            if not os.path.exists(self.avatar_path):  # 检查虚拟人物目录是否存在
                print(f"{self.avatar_id} does not exist, you should set preparation to True")
                sys.exit()

            with open(self.avatar_info_path, "r") as f:
                avatar_info = json.load(f)  # 加载虚拟人物信息

            if avatar_info['bbox_shift'] != self.avatar_info['bbox_shift']:  # 检查边界框偏移是否改变
                response = input(f" 【bbox_shift】 is changed, you need to re-create it ! (c/continue)")
                if response.lower() == "c":  # 如果用户选择继续
                    shutil.rmtree(self.avatar_path)  # 删除现有目录
                    print("*********************************")
                    print(f"  creating avator: {self.avatar_id}")
                    print("*********************************")
                    osmakedirs([self.avatar_path, self.full_imgs_path, self.video_out_path, self.mask_out_path])  # 创建必要的目录
                    self.prepare_material()  # 准备材料
                else:
                    sys.exit()
            else:  # 如果边界框偏移未改变
                # 加载已存在的单帧基础数据
                self.input_latent = torch.load(self.latents_out_path)
                with open(self.coords_path, 'rb') as f:
                    self.base_bbox = pickle.load(f)
                input_img_list = glob.glob(os.path.join(self.full_imgs_path, '*.[jpJP][pnPN]*[gG]'))
                input_img_list = sorted(input_img_list, key=lambda x: int(os.path.splitext(os.path.basename(x))[0]))
                self.base_frame = read_imgs([input_img_list[0]])[0]
                with open(self.mask_coords_path, 'rb') as f:
                    self.base_mask_crop_box = pickle.load(f)
                input_mask_list = glob.glob(os.path.join(self.mask_out_path, '*.[jpJP][pnPN]*[gG]'))
                input_mask_list = sorted(input_mask_list, key=lambda x: int(os.path.splitext(os.path.basename(x))[0]))
                self.base_mask = read_imgs([input_mask_list[0]])[0]
                self.input_latent_list_cycle = [self.input_latent]

    def prepare_material(self):
        """准备虚拟人物所需的材料，仅使用单张图片在内存中处理"""
        print("preparing data materials ... ...")
        with open(self.avatar_info_path, "w") as f:
            json.dump(self.avatar_info, f)  # 保存虚拟人物信息

        if not os.path.isfile(self.image_path):  # 如果输入不是文件
            print(f"Error: Input must be an image file, not a directory: {self.image_path}")
            sys.exit(1)
            
        ext = os.path.splitext(self.image_path)[1].lower()
        # 支持的图片后缀
        image_exts = {".png", ".jpg", ".jpeg"}
        
        if ext not in image_exts:
            print(f"Error: Unsupported file extension: {ext}. Only image files (.png, .jpg, .jpeg) are supported.")
            sys.exit(1)
            
        # 复制图片作为首帧（仅为缓存与复用，处理过程驻留内存）
        print(f"copy image {self.image_path} as base frame")
        base_dst = f"{self.full_imgs_path}/00000000{ext}"
        shutil.copyfile(self.image_path, base_dst)
        input_img_list = [base_dst]

        print("extracting landmarks...")
        coord_list, frame_list = get_landmark_and_bbox(input_img_list, self.bbox_shift)
        # 仅使用第一帧（也是唯一的一帧）
        bbox = coord_list[0]
        frame = frame_list[0]
        coord_placeholder = (0.0, 0.0, 0.0, 0.0)
        if bbox == coord_placeholder:
            print("Error: Face bbox not found in the input image.")
            sys.exit(1)
        x1, y1, x2, y2 = bbox
        if args.version == "v15":
            y2 = y2 + args.extra_margin
            y2 = min(y2, frame.shape[0])
            bbox = [x1, y1, x2, y2]
        crop_frame = frame[y1:y2, x1:x2]
        resized_crop_frame = cv2.resize(crop_frame, (256, 256), interpolation=cv2.INTER_LANCZOS4)
        latents = vae.get_latents_for_unet(resized_crop_frame)

        # 生成一次遮罩和裁剪框
        mode = args.parsing_mode if args.version == "v15" else "raw"
        mask, crop_box = get_image_prepare_material(frame, [x1, y1, x2, y2], fp=fp, mode=mode)

        # 将结果驻留在内存中供后续推理使用
        self.base_frame = frame
        self.base_bbox = bbox
        self.base_mask = mask
        self.base_mask_crop_box = crop_box
        self.input_latent = latents
        # 为 datagen 提供一个单元素列表即可循环使用
        self.input_latent_list_cycle = [self.input_latent]

        # 将关键数据保存到磁盘，便于下次复用（非必须）
        with open(self.coords_path, 'wb') as f:
            pickle.dump(self.base_bbox, f)
        with open(self.mask_coords_path, 'wb') as f:
            pickle.dump(self.base_mask_crop_box, f)
        # 保存基础遮罩与基础帧（可视化/调试用）
        cv2.imwrite(f"{self.mask_out_path}/00000000.png", self.base_mask)
        cv2.imwrite(f"{self.full_imgs_path}/00000000.png", self.base_frame)
        torch.save(self.input_latent, self.latents_out_path)

    def process_frames(self, res_frame_queue, video_len, skip_save_images):
        """
        处理视频帧的主要功能
        
        该函数的主要功能：
        1. 从结果帧队列中获取生成的视频帧
        2. 将生成的帧与原始帧进行混合处理
        3. 根据边界框调整帧的大小和位置
        4. 应用遮罩进行图像融合
        5. 可选择性地保存处理后的帧到磁盘
        6. 管理帧索引，确保按顺序处理所有帧
        
        参数:
            res_frame_queue: 包含生成结果帧的队列
            video_len: 视频总长度（帧数）
            skip_save_images: 是否跳过保存图片到磁盘
        
        处理流程:
            - 循环处理直到所有帧都处理完成
            - 从队列中获取生成的结果帧
            - 获取对应的原始帧和边界框信息
            - 调整结果帧大小以匹配边界框
            - 使用遮罩进行图像混合
            - 可选择保存混合后的帧
            - 更新帧索引继续处理下一帧
        """
        
        print(video_len)
        while True:
            if self.idx >= video_len - 1:  # 检查是否处理完所有帧
                break
            try:
                start = time.time()
                res_frame = res_frame_queue.get(block=True, timeout=1)  # 从队列获取结果帧
            except queue.Empty:  # 如果队列为空，继续等待
                continue

            # 使用单张基础图片与其对应的 bbox 与 mask
            bbox = self.base_bbox
            ori_frame = copy.deepcopy(self.base_frame)
            x1, y1, x2, y2 = bbox
            try:
                res_frame = cv2.resize(res_frame.astype(np.uint8), (x2 - x1, y2 - y1))  # 调整结果帧大小
            except:
                continue
            mask = self.base_mask
            mask_crop_box = self.base_mask_crop_box
            combine_frame = get_image_blending(ori_frame, res_frame, bbox, mask, mask_crop_box)  # 混合图像

            if skip_save_images is False:  # 如果需要保存图片
                cv2.imwrite(f"{self.avatar_path}/tmp/{str(self.idx).zfill(8)}.png", combine_frame)  # 保存混合后的帧
            self.idx = self.idx + 1

    @torch.no_grad()
    def inference(self, audio_path, out_vid_name, fps, skip_save_images):
        """执行推理过程，生成视频
        Args:
            audio_path: 音频文件路径
            out_vid_name: 输出视频名称
            fps: 帧率
            skip_save_images: 是否跳过保存图片
        """
        os.makedirs(self.avatar_path + '/tmp', exist_ok=True)  # 创建临时目录
        print("start inference")
        ############################################## extract audio feature ##############################################
        start_time = time.time()
        # 提取音频特征
        whisper_input_features, librosa_length = audio_processor.get_audio_feature(audio_path, weight_dtype=weight_dtype)  # 获取音频特征
        whisper_chunks = audio_processor.get_whisper_chunk(  # 获取Whisper模型的分块
            whisper_input_features,
            device,
            weight_dtype,
            whisper,
            librosa_length,
            fps=fps,
            audio_padding_length_left=args.audio_padding_length_left,
            audio_padding_length_right=args.audio_padding_length_right,
        )
        print(f"processing audio:{audio_path} costs {(time.time() - start_time) * 1000}ms")
        ############################################## inference batch by batch ##############################################
        video_num = len(whisper_chunks)  # 获取视频帧数
        res_frame_queue = queue.Queue()  # 创建结果帧队列
        self.idx = 0
        # 创建并启动处理线程
        process_thread = threading.Thread(target=self.process_frames, args=(res_frame_queue, video_num, skip_save_images))
        process_thread.start()

        gen = datagen(whisper_chunks,  # 生成数据批次
                     self.input_latent_list_cycle,
                     self.batch_size)
        start_time = time.time()
        res_frame_list = []

        # 批量处理数据。*** 主要的视频帧就在这里完成 ***
        for i, (whisper_batch, latent_batch) in enumerate(tqdm(gen, total=int(np.ceil(float(video_num) / self.batch_size)))):
            """
            pe是位置编码器（Positional Encoder）的缩写
            - 这个编码器用于为音频特征添加位置信息
            - 位置编码可以帮助模型理解音频特征在时间序列中的位置关系
            - 这对于生成与音频同步的嘴型动作非常重要
            """
            audio_feature_batch = pe(whisper_batch.to(device))  # 处理音频特征
            latent_batch = latent_batch.to(device=device, dtype=unet.model.dtype)  # 转换潜在向量

            pred_latents = unet.model(latent_batch,  # 使用UNet模型生成预测
                                    timesteps,
                                    encoder_hidden_states=audio_feature_batch).sample
            pred_latents = pred_latents.to(device=device, dtype=vae.vae.dtype)  # 转换预测结果
            recon = vae.decode_latents(pred_latents)  # 解码潜在向量
            for res_frame in recon:
                res_frame_queue.put(res_frame)  # 将结果放入队列
        # 等待处理线程完成
        process_thread.join()

        if args.skip_save_images is True:  # 如果跳过保存图片
            print('Total process time of {} frames without saving images = {}s'.format(
                video_num,
                time.time() - start_time))
        else:  # 如果保存图片
            print('Total process time of {} frames including saving images = {}s'.format(
                video_num,
                time.time() - start_time))

        if out_vid_name is not None and args.skip_save_images is False:  # 如果需要生成视频
            # 将图片序列转换为视频
            cmd_img2video = f"ffmpeg -y -v warning -r {fps} -f image2 -i {self.avatar_path}/tmp/%08d.png -vcodec libx264 -vf format=yuv420p -crf 18 {self.avatar_path}/temp.mp4"
            print(cmd_img2video)
            os.system(cmd_img2video)

            output_vid = os.path.join(self.video_out_path, out_vid_name + ".mp4")  # 设置输出视频路径
            # 将音频和视频合并
            cmd_combine_audio = f"ffmpeg -y -v warning -i {audio_path} -i {self.avatar_path}/temp.mp4 {output_vid}"
            print(cmd_combine_audio)
            os.system(cmd_combine_audio)

            os.remove(f"{self.avatar_path}/temp.mp4")  # 删除临时视频文件
            shutil.rmtree(f"{self.avatar_path}/tmp")  # 删除临时目录
            print(f"result is save to {output_vid}")
        print("\n")

    @torch.no_grad()
    def inference_stream(self, audio_path, out_vid_name, fps, skip_save_images):
        """执行推理过程，生成视频
        Args:
            audio_path: 音频文件路径
            out_vid_name: 输出视频名称
            fps: 帧率
            skip_save_images: 是否跳过保存图片
        """
        os.makedirs(self.avatar_path + '/tmp', exist_ok=True)  # 创建临时目录
        print("start inference")
        ############################################## extract audio feature ##############################################
        start_time = time.time()
        # 提取音频特征
        # 从文件创建音频流
        audio_stream_chunks = []
        for chunk in audio_processor.create_audio_stream_from_file(audio_path, chunk_size=16000):  # 1秒的块
            audio_stream_chunks.append(chunk)
        # 将流数据合并为完整音频数据
        audio_stream_data = np.concatenate(audio_stream_chunks)
        stream_features, stream_length = audio_processor.get_audio_stream_feature(audio_stream_data, weight_dtype=weight_dtype) # 获取音频特征

        whisper_chunks = audio_processor.get_whisper_chunk(  # 获取Whisper模型的分块
            stream_features,
            device,
            weight_dtype,
            whisper,
            stream_length,
            fps=fps,
            audio_padding_length_left=args.audio_padding_length_left,
            audio_padding_length_right=args.audio_padding_length_right,
        )
        print(f"processing audio:{audio_path} costs {(time.time() - start_time) * 1000}ms")
        ############################################## inference batch by batch ##############################################
        video_num = len(whisper_chunks)  # 获取视频帧数
        res_frame_queue = queue.Queue()  # 创建结果帧队列
        self.idx = 0
        # 创建并启动处理线程
        process_thread = threading.Thread(target=self.process_frames, args=(res_frame_queue, video_num, skip_save_images))
        process_thread.start()

        gen = datagen(whisper_chunks,  # 生成数据批次
                     self.input_latent_list_cycle,
                     self.batch_size)
        start_time = time.time()
        res_frame_list = []

        # 批量处理数据。*** 主要的视频帧就在这里完成 ***
        for i, (whisper_batch, latent_batch) in enumerate(tqdm(gen, total=int(np.ceil(float(video_num) / self.batch_size)))):
            """
            pe是位置编码器（Positional Encoder）的缩写
            - 这个编码器用于为音频特征添加位置信息
            - 位置编码可以帮助模型理解音频特征在时间序列中的位置关系
            - 这对于生成与音频同步的嘴型动作非常重要
            """
            audio_feature_batch = pe(whisper_batch.to(device))  # 处理音频特征
            latent_batch = latent_batch.to(device=device, dtype=unet.model.dtype)  # 转换潜在向量

            pred_latents = unet.model(latent_batch,  # 使用UNet模型生成预测
                                    timesteps,
                                    encoder_hidden_states=audio_feature_batch).sample
            pred_latents = pred_latents.to(device=device, dtype=vae.vae.dtype)  # 转换预测结果
            recon = vae.decode_latents(pred_latents)  # 解码潜在向量
            for res_frame in recon:
                res_frame_queue.put(res_frame)  # 将结果放入队列
        # 等待处理线程完成
        process_thread.join()

        if args.skip_save_images is True:  # 如果跳过保存图片
            print('Total process time of {} frames without saving images = {}s'.format(
                video_num,
                time.time() - start_time))
        else:  # 如果保存图片
            print('Total process time of {} frames including saving images = {}s'.format(
                video_num,
                time.time() - start_time))

        if out_vid_name is not None and args.skip_save_images is False:  # 如果需要生成视频
            # 将图片序列转换为视频
            cmd_img2video = f"ffmpeg -y -v warning -r {fps} -f image2 -i {self.avatar_path}/tmp/%08d.png -vcodec libx264 -vf format=yuv420p -crf 18 {self.avatar_path}/temp.mp4"
            print(cmd_img2video)
            os.system(cmd_img2video)

            output_vid = os.path.join(self.video_out_path, out_vid_name + ".mp4")  # 设置输出视频路径
            # 将音频和视频合并
            cmd_combine_audio = f"ffmpeg -y -v warning -i {audio_path} -i {self.avatar_path}/temp.mp4 {output_vid}"
            print(cmd_combine_audio)
            os.system(cmd_combine_audio)

            os.remove(f"{self.avatar_path}/temp.mp4")  # 删除临时视频文件
            shutil.rmtree(f"{self.avatar_path}/tmp")  # 删除临时目录
            print(f"result is save to {output_vid}")
        print("\n")


if __name__ == "__main__":
    '''
    这个脚本用于从单张图片生成虚拟人物，并提前进行必要的预处理，如人脸检测和人脸解析。
    仅支持图片文件输入（.png, .jpg, .jpeg），不支持视频文件或目录输入。
    在推理过程中，只涉及UNet和VAE解码器，这使得MuseTalk能够实时运行。
    '''

    parser = argparse.ArgumentParser()  # 创建参数解析器
    parser.add_argument("--version", type=str, default="v15", choices=["v1", "v15"], help="Version of MuseTalk: v1 or v15")  # 版本参数
    parser.add_argument("--ffmpeg_path", type=str, default="./ffmpeg-4.4-amd64-static/", help="Path to ffmpeg executable")  # ffmpeg路径
    parser.add_argument("--gpu_id", type=int, default=0, help="GPU ID to use")  # GPU ID
    parser.add_argument("--vae_type", type=str, default="sd-vae", help="Type of VAE model")  # VAE模型类型
    parser.add_argument("--unet_config", type=str, default="./models/musetalk/musetalk.json", help="Path to UNet configuration file")  # UNet配置文件路径
    parser.add_argument("--unet_model_path", type=str, default="./models/musetalk/pytorch_model.bin", help="Path to UNet model weights")  # UNet模型权重路径
    parser.add_argument("--whisper_dir", type=str, default="./models/whisper", help="Directory containing Whisper model")  # Whisper模型目录
    parser.add_argument("--inference_config", type=str, default="configs/inference/realtime.yaml")  # 推理配置文件
    parser.add_argument("--bbox_shift", type=int, default=0, help="Bounding box shift value")  # 边界框偏移值
    parser.add_argument("--result_dir", default='./results', help="Directory for output results")  # 结果输出目录
    parser.add_argument("--extra_margin", type=int, default=10, help="Extra margin for face cropping")  # 人脸裁剪额外边距
    parser.add_argument("--fps", type=int, default=25, help="Video frames per second")  # 视频帧率
    parser.add_argument("--audio_padding_length_left", type=int, default=2, help="Left padding length for audio")  # 音频左填充长度
    parser.add_argument("--audio_padding_length_right", type=int, default=2, help="Right padding length for audio")  # 音频右填充长度
    parser.add_argument("--batch_size", type=int, default=20, help="Batch size for inference")  # 推理批处理大小
    parser.add_argument("--output_vid_name", type=str, default=None, help="Name of output video file")  # 输出视频名称
    parser.add_argument("--use_saved_coord", action="store_true", help='Use saved coordinates to save time')  # 使用保存的坐标
    parser.add_argument("--saved_coord", action="store_true", help='Save coordinates for future use')  # 保存坐标
    parser.add_argument("--parsing_mode", default='jaw', help="Face blending parsing mode")  # 人脸混合解析模式
    parser.add_argument("--left_cheek_width", type=int, default=90, help="Width of left cheek region")  # 左脸颊区域宽度
    parser.add_argument("--right_cheek_width", type=int, default=90, help="Width of right cheek region")  # 右脸颊区域宽度
    parser.add_argument("--skip_save_images",  # 跳过保存图片
                       action="store_true",
                       help="Whether skip saving images for better generation speed calculation",
                       )

    args = parser.parse_args()  # 解析命令行参数

    # 配置ffmpeg路径
    if not fast_check_ffmpeg():  # 检查ffmpeg是否可用
        print("Adding ffmpeg to PATH")
        # 根据操作系统选择路径分隔符
        path_separator = ';' if sys.platform == 'win32' else ':'
        os.environ["PATH"] = f"{args.ffmpeg_path}{path_separator}{os.environ['PATH']}"
        if not fast_check_ffmpeg():  # 再次检查ffmpeg是否可用
            print("Warning: Unable to find ffmpeg, please ensure ffmpeg is properly installed")

    # 设置计算设备
    device = torch.device(f"cuda:{args.gpu_id}" if torch.cuda.is_available() else "cpu")

    # 加载模型权重
    vae, unet, pe = load_all_model(  # 加载所有模型
        unet_model_path=args.unet_model_path,
        vae_type=args.vae_type,
        unet_config=args.unet_config,
        device=device
    )
    timesteps = torch.tensor([0], device=device)  # 设置时间步

    # 将模型转换为半精度并移至指定设备
    pe = pe.half().to(device)
    vae.vae = vae.vae.half().to(device)
    unet.model = unet.model.half().to(device)

    # 初始化音频处理器和Whisper模型
    audio_processor = AudioProcessor(feature_extractor_path=args.whisper_dir)  # 创建音频处理器
    weight_dtype = unet.model.dtype  # 获取权重数据类型
    whisper = WhisperModel.from_pretrained(args.whisper_dir)  # 加载Whisper模型
    whisper = whisper.to(device=device, dtype=weight_dtype).eval()  # 将模型移至设备并设置为评估模式
    whisper.requires_grad_(False)  # 禁用梯度计算

    # 根据版本初始化人脸解析器
    if args.version == "v15":
        fp = FaceParsing(  # 创建v15版本的人脸解析器
            left_cheek_width=args.left_cheek_width,
            right_cheek_width=args.right_cheek_width
        )
    else:  # v1
        fp = FaceParsing()  # 创建v1版本的人脸解析器

    inference_config = OmegaConf.load(args.inference_config)  # 加载推理配置
    print(inference_config)

    # 处理每个虚拟人物
    for avatar_id in inference_config:
        data_preparation = inference_config[avatar_id]["preparation"]  # 获取预处理标志
        image_path = inference_config[avatar_id]["pic_path"]  # 获取图片文件路径
        if args.version == "v15":
            bbox_shift = 0
        else:
            bbox_shift = inference_config[avatar_id]["bbox_shift"]  # 获取边界框偏移
        avatar = Avatar(  # 创建虚拟人物实例
            avatar_id=avatar_id,
            image_path=image_path,
            bbox_shift=bbox_shift,
            batch_size=args.batch_size,
            preparation=data_preparation)

        audio_clips = inference_config[avatar_id]["audio_clips"]  # 获取音频片段
        for audio_num, audio_path in audio_clips.items():  # 处理每个音频片段
            print("Inferring using:", audio_path)
            avatar.inference_stream(audio_path,  # 执行推理
                           audio_num,
                           args.fps,
                           args.skip_save_images)


"""
command:
    python -m scripts.realtime_infer_pic --inference_config configs/inference/pic.yaml --result_dir results/realtime --unet_model_path models/musetalkV15/unet.pth --unet_config models/musetalkV15/musetalk.json --version v15 --fps 25 --ffmpeg_path ffmpeg-master-latest-win64-gpl-shared/bin
"""
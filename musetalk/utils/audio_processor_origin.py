import math
import os

import librosa
import numpy as np
import torch
from einops import rearrange
from transformers import AutoFeatureExtractor


class AudioProcessor:
    def __init__(self, feature_extractor_path="openai/whisper-tiny/"):
        self.feature_extractor = AutoFeatureExtractor.from_pretrained(feature_extractor_path)

    def get_audio_feature(self, wav_path, start_index=0, weight_dtype=None):
        """提取音频特征
        Args:
            wav_path: 音频文件路径
            start_index: 起始索引,默认为0
            weight_dtype: 权重数据类型,默认为None
            
        Returns:
            features: 音频特征列表
            len(librosa_output): 音频长度(采样点数)
        """
        # 检查音频文件是否存在
        if not os.path.exists(wav_path):
            return None
            
        # 加载音频文件,采样率为16000Hz
        librosa_output, sampling_rate = librosa.load(wav_path, sr=16000)
        assert sampling_rate == 16000
        
        # 将音频分割成30秒的片段
        segment_length = 30 * sampling_rate
        segments = [librosa_output[i:i + segment_length] for i in range(0, len(librosa_output), segment_length)]

        # 提取每个片段的特征
        features = []
        for segment in segments:
            # 使用特征提取器提取音频特征
            """
            # 输出是一个字典，包含：
                {
                    'input_features': tensor(...),  # 形状为 (batch_size, 50, 384)
                    'attention_mask': tensor(...)   # 注意力掩码
                }
            """
            audio_feature = self.feature_extractor(
                segment,
                return_tensors="pt",
                sampling_rate=sampling_rate
            ).input_features

            # 如果指定了权重数据类型,则进行转换
            if weight_dtype is not None:
                audio_feature = audio_feature.to(dtype=weight_dtype)
            features.append(audio_feature)

        return features, len(librosa_output)

             
    def get_audio_stream_feature(self, audio_chunk, weight_dtype=None):
        """提取音频流特征
        
        Args:
            audio_chunk: 音频数据块(16000字节)
                每次处理大小：
                    # 第一块数据（16000字节）
                    chunk1 = b'\x07\x01\xb6\xfc\x17\x0a...'  # 约0.36秒的音频
            weight_dtype: 权重数据类型,默认为None
            
        Returns:
            features: 音频特征
            chunk_length: 音频块长度(采样点数)
        """
        # 将字节数据转换为numpy数组
        audio_data = np.frombuffer(audio_chunk, dtype=np.int16)
        
        # 确保采样率为16000Hz
        sampling_rate = 16000
        
        # 使用特征提取器提取音频特征
        """
        # 输出是一个字典，包含：
            {
                'input_features': tensor(...),  # 形状为 (batch_size, 50, 384)
                'attention_mask': tensor(...)   # 注意力掩码
            }
        """
        audio_feature = self.feature_extractor(
            audio_data,
            return_tensors="pt",
            sampling_rate=sampling_rate
        ).input_features
        
        # 如果指定了权重数据类型,则进行转换
        if weight_dtype is not None:
            audio_feature = audio_feature.to(dtype=weight_dtype)
            
        return audio_feature, len(audio_data)


    def get_whisper_chunk(
        self,
        whisper_input_features,
        device,
        weight_dtype,
        whisper,
        librosa_length,
        fps=25,
        audio_padding_length_left=2,
        audio_padding_length_right=2,
    ):
        """
        将音频特征转换为Whisper模型所需的格式，并生成每一帧对应的音频特征序列

        Args:
            whisper_input_features: 从音频提取的Whisper特征列表
            device: 计算设备(CPU/GPU)
            weight_dtype: 权重数据类型
            whisper: Whisper模型实例
            librosa_length: 音频长度(采样点数)
            fps: 视频帧率,默认25fps
            audio_padding_length_left: 音频特征左侧填充长度,默认2
            audio_padding_length_right: 音频特征右侧填充长度,默认2

        Returns:
            audio_prompts: 每一帧对应的音频特征序列,形状为[num_frames, seq_len, hidden_dim]
        """
        # 计算每帧的音频特征长度
        audio_feature_length_per_frame = 2 * (audio_padding_length_left + audio_padding_length_right + 1)
        whisper_feature = []
        # Process multiple 30s mel input features
        # 处理多个30秒的梅尔频谱输入特征
        for input_feature in whisper_input_features:
            # 将特征移动到指定设备并转换数据类型
            input_feature = input_feature.to(device).to(weight_dtype)
            # 使用Whisper编码器提取音频特征
            audio_feats = whisper.encoder(input_feature, output_hidden_states=True).hidden_states
            # 将隐藏状态堆叠在一起
            audio_feats = torch.stack(audio_feats, dim=2)
            whisper_feature.append(audio_feats)

        # 连接所有特征
        whisper_feature = torch.cat(whisper_feature, dim=1)
        # Trim the last segment to remove padding
        # 裁剪最后一个片段以移除填充
        sr = 16000  # 音频采样率
        audio_fps = 50  # 音频帧率
        fps = int(fps)  # 视频帧率
        whisper_idx_multiplier = audio_fps / fps  # 音频到视频帧率的比率
        num_frames = math.floor((librosa_length / sr) * fps)  # 总视频帧数
        actual_length = math.floor((librosa_length / sr) * audio_fps)  # 实际音频长度
        whisper_feature = whisper_feature[:,:actual_length,...]  # 将特征裁剪到实际长度

        # 计算填充量并添加填充
        padding_nums = math.ceil(whisper_idx_multiplier)
        # Add padding at start and end
        # 在开始和结束处添加填充
        whisper_feature = torch.cat([
            torch.zeros_like(whisper_feature[:, :padding_nums * audio_padding_length_left]),  # 左侧填充
            whisper_feature,  # 原始特征
            # Add extra padding to prevent out of bounds
            # 添加额外填充以防止越界
            torch.zeros_like(whisper_feature[:, :padding_nums * 3 * audio_padding_length_right])  # 右侧填充
        ], 1)

        # 为每一帧生成音频特征
        audio_prompts = []
        for frame_index in range(num_frames):
            try:
                # 计算当前帧的音频索引
                audio_index = math.floor(frame_index * whisper_idx_multiplier)
                # 提取当前帧的音频片段
                audio_clip = whisper_feature[:, audio_index: audio_index + audio_feature_length_per_frame]
                # 确保音频片段长度正确
                assert audio_clip.shape[1] == audio_feature_length_per_frame
                audio_prompts.append(audio_clip)
            except Exception as e:
                # 错误处理：打印详细的调试信息
                print(f"Error occurred: {e}")
                print(f"whisper_feature.shape: {whisper_feature.shape}")
                print(f"audio_clip.shape: {audio_clip.shape}")
                print(f"num frames: {num_frames}, fps: {fps}, whisper_idx_multiplier: {whisper_idx_multiplier}")
                print(f"frame_index: {frame_index}, audio_index: {audio_index}-{audio_index + audio_feature_length_per_frame}")
                exit()

        # 连接所有音频提示并重新排列维度
        audio_prompts = torch.cat(audio_prompts, dim=0)  # 形状：[T, 10, 5, 384]
        audio_prompts = rearrange(audio_prompts, 'b c h w -> b (c h) w')  # 重新排列维度
        return audio_prompts

if __name__ == "__main__":
    audio_processor = AudioProcessor()
    wav_path = "./2.wav"
    audio_feature, librosa_feature_length = audio_processor.get_audio_feature(wav_path)
    print("Audio Feature shape:", audio_feature.shape)
    print("librosa_feature_length:", librosa_feature_length)


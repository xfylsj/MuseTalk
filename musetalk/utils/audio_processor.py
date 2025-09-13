import math
import os

import librosa
import numpy as np
import torch
from einops import rearrange
from transformers import AutoFeatureExtractor


class AudioProcessor:
    def __init__(self, feature_extractor_path="openai/whisper-tiny"):
        self.feature_extractor = AutoFeatureExtractor.from_pretrained(feature_extractor_path)

    def get_audio_feature(self, wav_path, start_index=0, weight_dtype=None):
        if not os.path.exists(wav_path):
            return None
        librosa_output, sampling_rate = librosa.load(wav_path, sr=16000)
        assert sampling_rate == 16000
        # Split audio into 30s segments
        segment_length = 30 * sampling_rate
        segments = [librosa_output[i:i + segment_length] for i in range(0, len(librosa_output), segment_length)]

        features = []
        for segment in segments:
            audio_feature = self.feature_extractor(
                segment,
                return_tensors="pt",
                sampling_rate=sampling_rate
            ).input_features
            if weight_dtype is not None:
                audio_feature = audio_feature.to(dtype=weight_dtype)
            features.append(audio_feature)

        return features, len(librosa_output)

    def get_audio_stream_feature(self, audio_stream_data, start_index=0, weight_dtype=None):
        """
        处理音频流数据并提取特征
        
        Args:
            audio_stream_data: 音频流数据 (numpy array 或 list)
            start_index: 开始索引
            weight_dtype: 权重数据类型
            
        Returns:
            features: 提取的音频特征列表
            total_length: 音频流总长度
        """
        # 检查音频流数据是否有效
        if audio_stream_data is None or len(audio_stream_data) == 0:
            return None, 0
            
        # 将音频流数据转换为numpy数组
        if isinstance(audio_stream_data, list):
            audio_stream_data = np.array(audio_stream_data)
        
        # 确保采样率为16000Hz
        sampling_rate = 16000
        
        # 将音频流分割成30秒的片段进行处理
        segment_length = 30 * sampling_rate
        segments = []
        
        # 按30秒片段分割音频流
        for i in range(0, len(audio_stream_data), segment_length):
            segment = audio_stream_data[i:i + segment_length]
            # 如果最后一个片段不足30秒，用零填充
            if len(segment) < segment_length:
                segment = np.pad(segment, (0, segment_length - len(segment)), 'constant')
            segments.append(segment)

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

        return features, len(audio_stream_data)

    def create_audio_stream_from_file(self, wav_path, chunk_size=1024):
        """
        从音频文件创建音频流数据
        
        Args:
            wav_path: 音频文件路径
            chunk_size: 每次读取的块大小
            
        Yields:
            audio_chunk: 音频数据块
        """
        if not os.path.exists(wav_path):
            raise FileNotFoundError(f"Audio file not found: {wav_path}")
            
        # 加载音频文件
        librosa_output, sampling_rate = librosa.load(wav_path, sr=16000)
        
        # 按块大小生成音频流
        for i in range(0, len(librosa_output), chunk_size):
            chunk = librosa_output[i:i + chunk_size]
            yield chunk

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
        audio_feature_length_per_frame = 2 * (audio_padding_length_left + audio_padding_length_right + 1)
        whisper_feature = []
        # Process multiple 30s mel input features
        for input_feature in whisper_input_features:
            input_feature = input_feature.to(device).to(weight_dtype)
            audio_feats = whisper.encoder(input_feature, output_hidden_states=True).hidden_states
            audio_feats = torch.stack(audio_feats, dim=2)
            whisper_feature.append(audio_feats)

        whisper_feature = torch.cat(whisper_feature, dim=1)
        # Trim the last segment to remove padding
        sr = 16000
        audio_fps = 50
        fps = int(fps)
        whisper_idx_multiplier = audio_fps / fps
        num_frames = math.floor((librosa_length / sr) * fps)
        actual_length = math.floor((librosa_length / sr) * audio_fps)
        whisper_feature = whisper_feature[:,:actual_length,...]

        # Calculate padding amount
        padding_nums = math.ceil(whisper_idx_multiplier)
        # Add padding at start and end
        whisper_feature = torch.cat([
            torch.zeros_like(whisper_feature[:, :padding_nums * audio_padding_length_left]),
            whisper_feature,
            # Add extra padding to prevent out of bounds
            torch.zeros_like(whisper_feature[:, :padding_nums * 3 * audio_padding_length_right])
        ], 1)

        audio_prompts = []
        for frame_index in range(num_frames):
            try:
                audio_index = math.floor(frame_index * whisper_idx_multiplier)
                audio_clip = whisper_feature[:, audio_index: audio_index + audio_feature_length_per_frame]
                assert audio_clip.shape[1] == audio_feature_length_per_frame
                audio_prompts.append(audio_clip)
            except Exception as e:
                print(f"Error occurred: {e}")
                print(f"whisper_feature.shape: {whisper_feature.shape}")
                print(f"audio_clip.shape: {audio_clip.shape}")
                print(f"num frames: {num_frames}, fps: {fps}, whisper_idx_multiplier: {whisper_idx_multiplier}")
                print(f"frame_index: {frame_index}, audio_index: {audio_index}-{audio_index + audio_feature_length_per_frame}")
                exit()

        audio_prompts = torch.cat(audio_prompts, dim=0)  # T, 10, 5, 384
        audio_prompts = rearrange(audio_prompts, 'b c h w -> b (c h) w')
        return audio_prompts

if __name__ == "__main__":
    audio_processor = AudioProcessor("./models/whisper")
    wav_path = "./data/audio/dd.wav"
    
    # --- 测试传统音频文件处理
    audio_feature, librosa_feature_length = audio_processor.get_audio_feature(wav_path)
    print("Audio Feature shape:", audio_feature[0].shape if audio_feature else "None")
    print("librosa_feature_length:", librosa_feature_length)
    
    # --- 测试音频流处理
    print("\n--- Testing Audio Stream Processing ---")
    try:
        # 方法1: 从文件创建音频流
        audio_stream_chunks = []
        for chunk in audio_processor.create_audio_stream_from_file(wav_path, chunk_size=16000):  # 1秒的块
            audio_stream_chunks.append(chunk)
        
        # 将流数据合并为完整音频数据
        audio_stream_data = np.concatenate(audio_stream_chunks)
        
        # 使用音频流数据提取特征
        stream_features, stream_length = audio_processor.get_audio_stream_feature(audio_stream_data)
        print("Stream Features shape:", stream_features[0].shape if stream_features else "None")
        print("Stream length:", stream_length)
        
    except FileNotFoundError:
        print("Test audio file not found, skipping stream test")

"""
command: (run in musetalk directory)
    python -m musetalk.utils.audio_processor
"""
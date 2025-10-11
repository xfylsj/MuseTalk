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
        # segment_length = 30 * sampling_rate
        segment_length = 1 * sampling_rate
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
        segment_length = 1 * sampling_rate
        segments = []
        
        # 按30秒片段分割音频流
        for i in range(0, len(audio_stream_data), segment_length):
            segment = audio_stream_data[i:i + segment_length]
            # 如果最后一个片段不足30秒，用零填充
            if len(segment) < segment_length:
                segment = np.pad(segment, (0, segment_length - len(segment)), 'constant')
            segments.append(segment)

        print(f"---segments length: {len(segments)}")

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

            print(f"--- audio_feature.shape: {audio_feature.shape}")

            """
                audio_feature.shape = torch.Size([1, 80, 3000])
                - 形状含义: [batch, n_mels, n_frames] = [1, 80, 3000]
                -- 1: 单个样本的 batch
                -- 80: 梅尔滤波器组数量（Whisper 固定 80 维）
                -- 3000: 时间帧数。Whisper 提取器以 16 kHz 采样、hop_length=160（每帧 10 ms，100 fps）生成 30 s 的对数梅尔谱，所以 30 s × 100 fps = 3000 帧

                - 短音频也会被右侧零填充到 3000 帧；长音频会被分段处理，每段各自得到一个 [1, 80, 3000]。
                - 进入 whisper.encoder 后，时间维会因下采样从 3000 变为 1500（约 50 fps），随后你的代码把各层隐藏态堆叠，变成类似 [1, 1500, num_layers, hidden_dim]，再按时间维拼接多个片段。
            """

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
            # 使用Whisper编码器提取音频特征（在这里会长度会填充到30秒）
            audio_feats = whisper.encoder(input_feature, output_hidden_states=True).hidden_states
            # 将隐藏状态堆叠在一起
            audio_feats = torch.stack(audio_feats, dim=2)

            # 计算当前段的有效长度（3秒对应的帧数）
            # Whisper encoder下采样后：3秒 * 50fps = 150帧
            segment_duration_frames = 50 * 1  # 3秒对应的帧数 TODO: 时间长度可能会改
            # 裁剪到实际3秒长度，避免30秒填充的影响
            audio_feats = audio_feats[:, :segment_duration_frames, :, :]

            whisper_feature.append(audio_feats)
            print(f"---audio_feats.shape: {audio_feats.shape}")

            """
            - batch=1: 单段/单样本处理。
            - 时间步=1500: Whisper 编码器把梅尔谱 [1, 80, 3000] 经过下采样（卷积 stride=2）变成约 50 fps，所以 30s × 50 = 1500。
            - 层数=5: 取了 5 个 Transformer 编码层的隐藏状态并在“层”维度堆叠，比如从若干指定的层索引收集到的特征；这个 5 就是所选层的数量。
            - 隐藏维度=384: Whisper 模型的通道维（比如 tiny 模型的 d_model=384）。不同规模模型该维度会不同。
            
            因此 audio_feats 的语义是：
            - 形状 [batch, time, num_selected_layers, hidden_dim] = [1, 1500, 5, 384]
            - 后续会在时间维（dim=1）把多个片段首尾相接，并按滑动窗口从时间轴裁剪出每个视频帧需要的音频提示特征。
            """

        # 连接所有特征
        whisper_feature = torch.cat(whisper_feature, dim=1)
        # Trim the last segment to remove padding
        # 裁剪最后一个片段以移除填充
        sr = 16000  # 音频采样率
        audio_fps = 50  # 音频帧率
        fps = int(fps)  # 25, 视频帧率
        whisper_idx_multiplier = audio_fps / fps  # 2, 音频到视频帧率的比率
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

        print(">>>>>>>>>>>>>>>>>>>>")
        print(f"whisper_feature.shape: {whisper_feature.shape}")
        print(f"num_frames: {num_frames}, fps: {fps}, whisper_idx_multiplier: {whisper_idx_multiplier}")
        print(f"actual_length: {actual_length}")
        print(f"audio_padding_length_left: {audio_padding_length_left}, audio_padding_length_right: {audio_padding_length_right}")
        print(f"padding_nums: {padding_nums}")
        print(f"audio_feature_length_per_frame: {audio_feature_length_per_frame}")
        print(f"audio_prompts.shape: {audio_prompts.shape}")
        print(">>>>>>>>>>>>>>>>>>>>")

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
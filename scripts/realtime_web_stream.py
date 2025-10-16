

"""
prompt:
创建一个html客户端页面，主要功能如下：
1. 一个上传音频文件按钮，一个播放按钮和一个视频播放控件
2. 音频文件上传之后，服务端使用MuseTalk进行流式推理，生成对应的视频帧
3. 当页面上的播放按钮被点击以后，后端实时推送合成好的视频帧和音频帧到前端
4. 前端实时播放合成的视频和音频
"""

import os
from flask import Flask, render_template_string, request, Response, send_file
from werkzeug.utils import secure_filename
from flask_socketio import SocketIO, emit
import subprocess
import threading
import queue
import tempfile
import shutil
import time
import wave
import numpy as np
import cv2
import base64
import io
import argparse
from omegaconf import OmegaConf
import torch
import glob
import pickle
import sys
from tqdm import tqdm
import json
import gc
from transformers import WhisperModel

# 导入MuseTalk相关模块
from musetalk.utils.face_parsing import FaceParsing
from musetalk.utils.utils import datagen
from musetalk.utils.preprocessing import get_landmark_and_bbox, read_imgs
from musetalk.utils.blending import get_image_prepare_material, get_image_blending
from musetalk.utils.utils import load_all_model
from musetalk.utils.audio_processor import AudioProcessor

app = Flask(__name__)
socketio = SocketIO(app, cors_allowed_origins="*")

# 全局变量
temp_dir = tempfile.mkdtemp()
current_avatar = None  # 当前Avatar实例
is_inferring = False  # 是否正在推理
inference_thread = None  # 推理线程

# CUDA内存管理
def clear_cuda_cache():
    """清理CUDA缓存"""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        gc.collect()

def get_gpu_memory_info():
    """获取GPU内存信息"""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**3  # GB
        reserved = torch.cuda.memory_reserved() / 1024**3     # GB
        return f"GPU Memory - Allocated: {allocated:.2f}GB, Reserved: {reserved:.2f}GB"
    return "CUDA not available"

# MuseTalk模型相关
device = None
vae = None
unet = None
pe = None
whisper = None
fp = None
audio_processor = None
weight_dtype = None
timesteps = None

# HTML 客户端页面
HTML = """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <title>MuseTalk实时语音驱动</title>
        <script src="https://cdnjs.cloudflare.com/ajax/libs/socket.io/4.0.1/socket.io.js"></script>
    </head>
    <body>
        <h2>MuseTalk实时语音驱动 Demo</h2>
        <div>
            <button id="playBtn">开始推理</button>
            <button id="stopBtn" disabled>停止</button>
        </div>
        <br><br>
        <div>
            <canvas id="videoCanvas" width="512" height="512"></canvas>
            <audio id="audioPlayer" controls></audio>
        </div>
        <p id="status">准备就绪：使用默认音频 ./data/audio/dd.wav</p>
        <div id="debugInfo" style="display:none; background:#f0f0f0; padding:10px; margin:10px 0;"></div>
        <script>
            let isPlaying = false;
            let playBtn = document.getElementById("playBtn");
            let stopBtn = document.getElementById("stopBtn");
            let statusText = document.getElementById("status");
            let videoCanvas = document.getElementById("videoCanvas");
            let videoCtx = videoCanvas.getContext('2d', { alpha: false, desynchronized: true });
            let audioPlayer = document.getElementById("audioPlayer");
            let audioCtx = null;
            let audioQueue = [];
            let nextStartTime = 0;
            function ensureAudioContext() {
                if (!audioCtx) {
                    try {
                        audioCtx = new (window.AudioContext || window.webkitAudioContext)({ sampleRate: 16000 });
                    } catch (e) {
                        audioCtx = new (window.AudioContext || window.webkitAudioContext)();
                    }
                }
                if (audioCtx.state === 'suspended') {
                    audioCtx.resume();
                }
            }
            
            // 初始化Socket.IO连接
            const socket = io();
            
            // 监听服务器事件（接收二进制JPEG帧）
            socket.on('video_frame', async function(payload) {
                if (!isPlaying || !payload) return;
                try {
                    let arrayBuffer = null;
                    if (payload instanceof ArrayBuffer) {
                        arrayBuffer = payload;
                    } else if (payload instanceof Blob) {
                        arrayBuffer = await payload.arrayBuffer();
                    } else if (payload.data) {
                        if (payload.data instanceof ArrayBuffer) {
                            arrayBuffer = payload.data;
                        } else if (payload.data instanceof Blob) {
                            arrayBuffer = await payload.data.arrayBuffer();
                        }
                    }
                    if (!arrayBuffer) return;
                    const blob = new Blob([arrayBuffer], { type: 'image/jpeg' });
                    const bitmap = await createImageBitmap(blob);
                    // 固定尺寸绘制，避免频繁resize
                    videoCtx.drawImage(bitmap, 0, 0, videoCanvas.width, videoCanvas.height);
                } catch (e) {
                    // 忽略单帧错误，保证连续性
                }
            });
            
            socket.on('audio_chunk', async function(payload) {
                if (!isPlaying) return;
                try {
                    let arrayBuffer = null;
                    if (payload instanceof ArrayBuffer) {
                        arrayBuffer = payload;
                    } else if (payload instanceof Blob) {
                        arrayBuffer = await payload.arrayBuffer();
                    } else if (payload && payload.data) {
                        if (payload.data instanceof ArrayBuffer) arrayBuffer = payload.data;
                        else if (payload.data instanceof Blob) arrayBuffer = await payload.data.arrayBuffer();
                    }
                    if (!arrayBuffer) return;
                    ensureAudioContext();
                    const wavBlob = new Blob([arrayBuffer], { type: 'audio/wav' });
                    const wavBuf = await wavBlob.arrayBuffer();
                    audioCtx.decodeAudioData(wavBuf.slice(0), (decoded) => {
                        const src = audioCtx.createBufferSource();
                        src.buffer = decoded;
                        src.connect(audioCtx.destination);
                        const startAt = Math.max(audioCtx.currentTime + 0.05, nextStartTime || audioCtx.currentTime + 0.05);
                        try { src.start(startAt); } catch (_) { src.start(); }
                        nextStartTime = startAt + decoded.duration;
                    }, (err) => {
                        // ignore decode errors for individual chunks
                    });
                } catch (e) {
                    // ignore
                }
            });
            
            socket.on('inference_start', function(data) {
                statusText.textContent = "开始推理...";
                playBtn.disabled = true;
                stopBtn.disabled = false;
                isPlaying = true;
            });
            
            socket.on('inference_complete', function(data) {
                statusText.textContent = "推理完成！";
                playBtn.disabled = false;
                stopBtn.disabled = true;
                isPlaying = false;
            });
            
            socket.on('inference_error', function(data) {
                statusText.textContent = "推理错误: " + data.error;
                playBtn.disabled = false;
                stopBtn.disabled = true;
                isPlaying = false;
            });

            playBtn.onclick = function() {
                // 发送开始推理请求
                ensureAudioContext();
                socket.emit('start_inference');
            };
            
            stopBtn.onclick = function() {
                // 发送停止推理请求
                socket.emit('stop_inference');
                statusText.textContent = "正在停止...";
                if (audioCtx && audioCtx.state !== 'closed') {
                    try { audioCtx.close(); } catch (e) {}
                }
                audioCtx = null;
                nextStartTime = 0;
            };
        </script>
    </body>
    </html>
    """

# MuseTalk Avatar类（简化版）
@torch.no_grad()
class WebAvatar:
    def __init__(self, avatar_id, image_path, bbox_shift, batch_size):
        self.avatar_id = avatar_id
        self.image_path = image_path
        self.bbox_shift = bbox_shift
        self.batch_size = batch_size
        
        # 内存存储，不使用磁盘路径
        self.base_bbox = None
        self.base_mask = None
        self.base_mask_crop_box = None
        self.input_latent = None
        self.input_latent_list_cycle = None
        self.base_frame = None
        
        self.avatar_info = {
            "avatar_id": avatar_id,
            "image_path": image_path,
            "bbox_shift": bbox_shift,
            "version": "v15"
        }
        self.idx = 0
        self.init()

    def init(self):
        """初始化Avatar - 完全基于内存操作"""
        # 直接从图片路径加载并处理，不保存到磁盘
        self.process_image_from_path()

    def process_image_from_path(self):
        """直接从图片路径处理，所有操作在内存中完成"""
        print("Processing image in memory...")
        
        if not os.path.isfile(self.image_path):
            print(f"Error: Input must be an image file: {self.image_path}")
            return False
            
        ext = os.path.splitext(self.image_path)[1].lower()
        image_exts = {".png", ".jpg", ".jpeg"}
        
        if ext not in image_exts:
            print(f"Error: Unsupported file extension: {ext}")
            return False

        # 直接读取图片到内存，不复制到磁盘
        input_img_list = [self.image_path]
        print("extracting landmarks...")
        coord_list, frame_list = get_landmark_and_bbox(input_img_list, self.bbox_shift)
        bbox = coord_list[0]
        frame = frame_list[0]
        coord_placeholder = (0.0, 0.0, 0.0, 0.0)
        if bbox == coord_placeholder:
            print("Error: Face bbox not found in the input image.")
            return False
        x1, y1, x2, y2 = bbox
        y2 = y2 + 10  # extra_margin
        y2 = min(y2, frame.shape[0])
        bbox = [x1, y1, x2, y2]
        crop_frame = frame[y1:y2, x1:x2]
        resized_crop_frame = cv2.resize(crop_frame, (256, 256), interpolation=cv2.INTER_LANCZOS4)
        latents = vae.get_latents_for_unet(resized_crop_frame)

        # 生成遮罩和裁剪框
        mask, crop_box = get_image_prepare_material(frame, [x1, y1, x2, y2], fp=fp, mode="jaw")

        # 保存到内存
        self.base_frame = frame
        self.base_bbox = bbox
        self.base_mask = mask
        self.base_mask_crop_box = crop_box
        self.input_latent = latents
        self.input_latent_list_cycle = [self.input_latent]
        
        print("Image processed successfully in memory")
        return True

    def inference_stream_web_origin(self, audio_path, fps=25):
        """Web版本的流式推理"""
        print("start web inference")
        self.idx = 0
        
        try:
            # 从文件按1秒块读取并处理
            for chunk in audio_processor.create_audio_stream_from_file(audio_path, chunk_size=16000):
                chunk_start = time.time()
                
                # 清理内存
                # clear_cuda_cache()
                
                stream_features, stream_length = audio_processor.get_audio_stream_feature(chunk, weight_dtype=weight_dtype)
                whisper_chunks = audio_processor.get_whisper_chunk(
                    stream_features,
                    device,
                    weight_dtype,
                    whisper,
                    stream_length,
                    fps=fps,
                    audio_padding_length_left=2,
                    audio_padding_length_right=2,
                )

                gen = datagen(whisper_chunks, self.input_latent_list_cycle, self.batch_size)
                for _, (whisper_batch, latent_batch) in enumerate(gen):
                    pic_infer_start = time.time()

                    # 编码音频特征并推理
                    audio_feature_batch = pe(whisper_batch.to(device))
                    latent_batch = latent_batch.to(device=device, dtype=unet.model.dtype)
                    
                    # 使用torch.no_grad()节省内存
                    with torch.no_grad():
                        pred_latents = unet.model(latent_batch, timesteps, encoder_hidden_states=audio_feature_batch).sample
                        pred_latents = pred_latents.to(device=device, dtype=vae.vae.dtype)
                        recon = vae.decode_latents(pred_latents)

                    print(f"pic_infer_time: {(time.time() - pic_infer_start) * 1000:.1f}ms")
                    
                    # 简单的帧率限制：仅保留每秒fps帧
                    send_start = time.time()
                    frame_interval = max(1, int(len(recon) / max(1, fps)))
                    for i, res_frame in enumerate(recon):
                        if i % frame_interval != 0:
                            continue
                        # 处理帧并发送到前端
                        self.process_and_send_frame(res_frame)
                        
                        # 发送对应的音频块
                        print("send audio chunk to frontend:")
                        # 将 float32 PCM [-1,1] 转为 16-bit PCM 并封装 WAV 头
                        try:
                            pcm16 = (np.clip(chunk, -1.0, 1.0) * 32767.0).astype(np.int16)
                            wav_buffer = io.BytesIO()
                            with wave.open(wav_buffer, 'wb') as wf:
                                wf.setnchannels(1)
                                wf.setsampwidth(2)  # 16-bit
                                wf.setframerate(16000)
                                wf.writeframes(pcm16.tobytes())
                            audio_chunk_bytes = wav_buffer.getvalue()
                            socketio.emit('audio_chunk', {'audio': audio_chunk_bytes})
                        except Exception as _e:
                            print(f"audio chunk encode error: {_e}")
                    print(f"send current chunk done. send_time: {(time.time() - send_start) * 1000:.1f}ms ")
         

                print(f"processed 1s audio chunk in {(time.time() - chunk_start) * 1000:.1f}ms")
                print(f"Memory status: {get_gpu_memory_info()}")
                
                # 检查是否需要停止
                if not is_inferring:
                    break
                    
        except Exception as e:
            print(f"Inference error: {e}")
            socketio.emit('inference_error', {'error': str(e)})
            return
            
        socketio.emit('inference_complete', {'message': 'Inference completed'})

    def inference_stream_web(self, audio_path, fps=25):
        """Web版本的流式推理（队列化，去除多层for嵌套）"""
        print("start web inference")
        self.idx = 0

        # 队列：音频块 -> whisper块 -> 模型重建帧
        audio_q: queue.Queue = queue.Queue(maxsize=8)
        whisper_q: queue.Queue = queue.Queue(maxsize=16)
        frames_q: queue.Queue = queue.Queue(maxsize=16)

        stop_flag = threading.Event()

        def producer_audio():
            try:
                for chunk in audio_processor.create_audio_stream_from_file(audio_path, chunk_size=4000):
                    if stop_flag.is_set() or not is_inferring:
                        break
                    audio_q.put(chunk)
            finally:
                # 结束标记
                audio_q.put(None)

        def stage_whisper():
            while True:
                if stop_flag.is_set() or not is_inferring:
                    break
                chunk = audio_q.get()
                if chunk is None:
                    # 传递结束标记
                    whisper_q.put((None, None))
                    break
                stream_features, stream_length = audio_processor.get_audio_stream_feature(chunk, weight_dtype=weight_dtype)
                whisper_chunks = audio_processor.get_whisper_chunk(
                    stream_features,
                    device,
                    weight_dtype,
                    whisper,
                    stream_length,
                    fps=fps,
                    audio_padding_length_left=2,
                    audio_padding_length_right=2,
                )
                # 增量产出：在本阶段就按 batch 切分并入队，降低下游等待时间
                gen_local = datagen(whisper_chunks, self.input_latent_list_cycle, self.batch_size)
                for _, (whisper_batch, _) in enumerate(gen_local):
                    if stop_flag.is_set() or not is_inferring:
                        break
                    whisper_q.put((whisper_batch, chunk))

        def stage_infer_and_decode():
            while True:
                if stop_flag.is_set() or not is_inferring:
                    break
                whisper_item, raw_chunk = whisper_q.get()
                if whisper_item is None:
                    # 结束标记
                    frames_q.put((None, None))
                    break

                # 直接消费单个 whisper 批并推理（latent 批来自循环单帧 latent 列表）
                # 这里每个批的 latent_batch 与 whisper_batch 数量一致，由 datagen 的逻辑保证
                # 重新生成一次对应大小的 latent 批
                gen_latent = datagen(whisper_item, self.input_latent_list_cycle, self.batch_size)
                for _, (whisper_batch, latent_batch) in enumerate(gen_latent):
                    if stop_flag.is_set() or not is_inferring:
                        break
                    pic_infer_start = time.time()
                    audio_feature_batch = pe(whisper_batch.to(device))
                    latent_batch = latent_batch.to(device=device, dtype=unet.model.dtype)
                    with torch.no_grad():
                        pred_latents = unet.model(latent_batch, timesteps, encoder_hidden_states=audio_feature_batch).sample
                        pred_latents = pred_latents.to(device=device, dtype=vae.vae.dtype)
                        recon = vae.decode_latents(pred_latents)
                    frames_q.put((recon, raw_chunk))
                    print(f"pic_infer_time: {(time.time() - pic_infer_start) * 1000:.1f}ms")

        def consumer_emit():
            try:
                while True:
                    if stop_flag.is_set() or not is_inferring:
                        break
                    recon, raw_chunk = frames_q.get()
                    if recon is None:
                        break
                    # 帧率限制
                    frame_interval = max(1, int(len(recon) / max(1, fps)))
                    send_start = time.time()
                    for i, res_frame in enumerate(recon):
                        if i % frame_interval != 0:
                            continue
                        self.process_and_send_frame(res_frame)
                    # 发送对应的音频块（与该批次视频对应）
                    if raw_chunk is not None:
                        try:
                            pcm16 = (np.clip(raw_chunk, -1.0, 1.0) * 32767.0).astype(np.int16)
                            wav_buffer = io.BytesIO()
                            with wave.open(wav_buffer, 'wb') as wf:
                                wf.setnchannels(1)
                                wf.setsampwidth(2)
                                wf.setframerate(16000)
                                wf.writeframes(pcm16.tobytes())
                            audio_chunk_bytes = wav_buffer.getvalue()
                            # 直接以二进制发送，避免JSON封装影响
                            socketio.emit('audio_chunk', audio_chunk_bytes)
                        except Exception as _e:
                            print(f"audio chunk encode error: {_e}")
                    print(f"send current batch done. send_time: {(time.time() - send_start) * 1000:.1f}ms ")
            finally:
                pass

        # 启动各阶段线程
        t_prod = threading.Thread(target=producer_audio, daemon=True)
        t_wsp = threading.Thread(target=stage_whisper, daemon=True)
        t_infer = threading.Thread(target=stage_infer_and_decode, daemon=True)
        t_emit = threading.Thread(target=consumer_emit, daemon=True)

        try:
            t_prod.start(); t_wsp.start(); t_infer.start(); t_emit.start()
            # 等待管线完成或被外部停止
            while any(t.is_alive() for t in (t_prod, t_wsp, t_infer)) and is_inferring:
                time.sleep(0.01)
            # 通知消费端结束
            frames_q.put((None, None))
        except Exception as e:
            print(f"Inference error: {e}")
            socketio.emit('inference_error', {'error': str(e)})
            return
        finally:
            stop_flag.set()

        socketio.emit('inference_complete', {'message': 'Inference completed'})


    def process_and_send_frame(self, res_frame):
        """处理帧并发送到前端"""
        try:
            bbox = self.base_bbox
            ori_frame = self.base_frame  # no deepcopy needed; blending does not mutate input
            x1, y1, x2, y2 = bbox
            res_frame = cv2.resize(res_frame.astype(np.uint8), (x2 - x1, y2 - y1))
            mask = self.base_mask
            mask_crop_box = self.base_mask_crop_box
            combine_frame = get_image_blending(ori_frame, res_frame, bbox, mask, mask_crop_box)
            
            # 将帧编码为JPEG并以二进制发送
            encode_params = [int(cv2.IMWRITE_JPEG_QUALITY), 80]
            ok, buffer = cv2.imencode('.jpg', combine_frame, encode_params)
            if ok:
                # 直接发送字节，SocketIO会按合适的传输方式发送
                socketio.emit('video_frame', buffer.tobytes())
            
            self.idx += 1
        except Exception as e:
            print(f"Frame processing error: {e}")


@app.route('/')
def index():
    return render_template_string(HTML)

# WebSocket事件处理
@socketio.on('start_inference')
def handle_start_inference():
    """处理开始推理请求"""
    global is_inferring, inference_thread
    
    if is_inferring:
        emit('inference_error', {'error': 'Already inferring'})
        return
    
    if current_avatar is None:
        emit('inference_error', {'error': 'No avatar available'})
        return
    
    # 使用固定音频路径
    audio_path = os.path.abspath(os.path.join('.', 'data', 'audio', 'dd.wav'))
    if not os.path.exists(audio_path):
        emit('inference_error', {'error': f'Default audio not found: {audio_path}'})
        return
    is_inferring = True
    
    # 启动推理线程
    inference_thread = threading.Thread(target=run_inference, args=(audio_path,))
    inference_thread.start()
    
    emit('inference_start', {'message': 'Inference started'})

@socketio.on('stop_inference')
def handle_stop_inference():
    """处理停止推理请求"""
    global is_inferring
    
    is_inferring = False
    emit('inference_complete', {'message': 'Inference stopped'})

def run_inference(audio_path):
    """运行推理的线程函数"""
    try:
        print(f"Starting inference. {get_gpu_memory_info()}")
        current_avatar.inference_stream_web(audio_path)
    except Exception as e:
        print(f"Inference thread error: {e}")
        socketio.emit('inference_error', {'error': str(e)})
    finally:
        global is_inferring
        is_inferring = False
        # 清理内存
        clear_cuda_cache()
        print(f"Inference finished. {get_gpu_memory_info()}")

@app.route('/upload', methods=['POST'])
def upload_file():
    """处理音频文件上传"""
    if 'file' not in request.files:
        return {"status": "error", "error": "No file provided"}
    
    f = request.files['file']
    if f.filename == '':
        return {"status": "error", "error": "No file selected"}
    
    # 检查文件类型
    if not f.filename.lower().endswith(('.mp3', '.wav', '.mpeg')):
        return {"status": "error", "error": "Unsupported file type. Please upload MP3 or WAV files."}
    
    # 将文件保存到临时目录，供推理线程读取
    filename = secure_filename(f.filename)
    # 清理旧的音频文件，避免混淆
    for old in os.listdir(temp_dir):
        if old.lower().endswith((".mp3", ".wav", ".mpeg")):
            try:
                os.remove(os.path.join(temp_dir, old))
            except Exception:
                pass
    save_path = os.path.join(temp_dir, filename)
    f.save(save_path)
    file_size = os.path.getsize(save_path) if os.path.exists(save_path) else 0
    print(f"Audio file saved: {filename} -> {save_path} ({file_size} bytes)")
    return {"status": "ok", "message": "File uploaded successfully", "filename": filename, "size": file_size}



def initialize_musetalk():
    """初始化MuseTalk模型"""
    global device, vae, unet, pe, whisper, fp, audio_processor, weight_dtype, timesteps, current_avatar
    
    # 设置参数
    args = argparse.Namespace()
    args.version = "v15"
    args.gpu_id = 0
    args.vae_type = "sd-vae"
    args.unet_config = "./models/musetalkV15/musetalk.json"
    args.unet_model_path = "./models/musetalkV15/unet.pth"
    args.whisper_dir = "./models/whisper"
    args.bbox_shift = 0
    args.extra_margin = 10
    args.fps = 25
    args.audio_padding_length_left = 2
    args.audio_padding_length_right = 2
    args.batch_size = 5  # 减少batch size以节省内存
    args.parsing_mode = "jaw"
    args.left_cheek_width = 90
    args.right_cheek_width = 90
    
    # 设置计算设备
    device = torch.device(f"cuda:{args.gpu_id}" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # 设置CUDA内存管理
    if torch.cuda.is_available():
        torch.cuda.set_per_process_memory_fraction(0.8)  # 限制使用80%的GPU内存
        os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
        print("CUDA memory management configured")
    
    # 清理内存
    clear_cuda_cache()
    
    # 加载模型权重
    vae, unet, pe = load_all_model(
        unet_model_path=args.unet_model_path,
        vae_type=args.vae_type,
        unet_config=args.unet_config,
        device=device
    )
    timesteps = torch.tensor([0], device=device)
    
    # 将模型转换为半精度并移至指定设备
    pe = pe.half().to(device)
    vae.vae = vae.vae.half().to(device)
    unet.model = unet.model.half().to(device)
    
    # 启用梯度检查点以节省内存
    if hasattr(unet.model, 'enable_gradient_checkpointing'):
        unet.model.enable_gradient_checkpointing()
    
    # 清理内存
    clear_cuda_cache()
    print(f"Models loaded. {get_gpu_memory_info()}")
    
    # 初始化音频处理器和Whisper模型
    audio_processor = AudioProcessor(feature_extractor_path=args.whisper_dir)
    weight_dtype = unet.model.dtype
    whisper = WhisperModel.from_pretrained(args.whisper_dir)
    whisper = whisper.to(device=device, dtype=weight_dtype).eval()
    whisper.requires_grad_(False)
    
    # 初始化人脸解析器
    fp = FaceParsing(
        left_cheek_width=args.left_cheek_width,
        right_cheek_width=args.right_cheek_width
    )
    
    # 创建默认Avatar（使用data/pic/dd.png）
    default_image_path = "./data/pic/dd.png"
    if os.path.exists(default_image_path):
        current_avatar = WebAvatar(
            avatar_id="default",
            image_path=default_image_path,
            bbox_shift=args.bbox_shift,
            batch_size=args.batch_size
        )
        print("Default avatar initialized successfully")
    else:
        print(f"Warning: Default image not found at {default_image_path}")

if __name__ == '__main__':
    print("启动MuseTalk实时推理服务器...")
    print("访问: http://localhost:8080")
    
    # 检查ffmpeg是否可用
    try:
        result = subprocess.run(['ffmpeg', '-version'], capture_output=True, text=True)
        if result.returncode == 0:
            print("✓ FFmpeg 可用")
        else:
            print("✗ FFmpeg 不可用")
    except:
        print("✗ FFmpeg 未安装")
        print("请安装FFmpeg: brew install ffmpeg")
    
    # 初始化MuseTalk模型
    try:
        initialize_musetalk()
        print("✓ MuseTalk模型初始化成功")
        print(f"Final memory status: {get_gpu_memory_info()}")
    except Exception as e:
        print(f"✗ MuseTalk模型初始化失败: {e}")
        print("请确保模型文件存在且路径正确")
        print("如果仍然出现内存不足，请尝试:")
        print("1. 关闭其他占用GPU的程序")
        print("2. 重启Python进程")
        print("3. 使用更小的batch_size")
    
    # 启动服务器（关闭debug与reloader以避免多进程占用GPU）
    socketio.run(app, debug=False, use_reloader=False, port=8080, host='0.0.0.0')


"""
cmd:
    python -m scripts.realtime_web_stream
"""


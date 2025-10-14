


"""
prompt:
创建一个html客户端页面，主要功能如下：
1. 一个上传文件按钮，一个播放按钮和一个视频播放控件
2. 视频文件上传之后，服务端需要将视mp4频文件拆分成视频帧和音频帧，分别放入视频queue和音频queue中。
3. 当页面上的播放按钮被点击以后，页面上的视频控件需要从视频queue和音频queue读取视频帧和音频帧，然后合并播放出来。
"""

import os
from flask import Flask, render_template_string, request, Response, send_file
import subprocess
import threading
import queue
import tempfile
import shutil
import time
import wave
import numpy as np

app = Flask(__name__)

# 全局队列和路径
video_frame_queue = queue.Queue()
audio_frame_queue = queue.Queue()
video_frames_list = []  # 存储所有视频帧，用于重复播放
temp_dir = tempfile.mkdtemp()
mp4_file_path = None  # 预生成的MP4文件路径
audio_file_path = None  # 预生成的音频文件路径
sync_start_time = None  # 同步开始时间

# HTML 客户端页面
HTML = """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <title>视频帧音频帧实时播放</title>
    </head>
    <body>
        <h2>MP4视频分帧实时播放 Demo</h2>
        <input type="file" id="fileInput" accept="video/mp4">
        <button id="uploadBtn">上传视频</button>
        <button id="playBtn" disabled>播放</button>
        <br><br>
        <video id="videoPlayer" width="400" controls></video>
        <audio id="audioPlayer" controls style="display: none;"></audio>
        <p id="status"></p>
        <button onclick="checkDebug()">检查状态</button>
        <button onclick="createTestFrames()">创建测试帧</button>
        <div id="debugInfo" style="display:none; background:#f0f0f0; padding:10px; margin:10px 0;"></div>
        <script>
            let uploaded = false;
            let playBtn = document.getElementById("playBtn");
            let uploadBtn = document.getElementById("uploadBtn");
            let statusText = document.getElementById("status");
            let videoPlayer = document.getElementById("videoPlayer");
            let audioPlayer = document.getElementById("audioPlayer");
            let fileInput = document.getElementById("fileInput");

            uploadBtn.onclick = function() {
                let file = fileInput.files[0];
                if (!file) {
                    alert("请选择一个mp4视频文件");
                    return;
                }
                let formData = new FormData();
                formData.append("file", file);

                statusText.textContent = "上传中...";
                playBtn.disabled = true;
                fetch("/upload", {
                    method: "POST",
                    body: formData
                }).then(resp => resp.json())
                .then(data => {
                    if (data.status === "ok") {
                        uploaded = true;
                        statusText.textContent = "上传并切分完成！";
                        playBtn.disabled = false;
                    } else {
                        statusText.textContent = "上传失败";
                    }
                }).catch(e=>{
                    statusText.textContent = "服务器出错: " + e;
                });
            };

            playBtn.onclick = function() {
                if (!uploaded) {
                    alert("请先上传一个视频文件！"); return;
                }
                
                statusText.textContent = "正在播放...";
                
                // 设置视频和音频源
                videoPlayer.src = "/video_stream";
                audioPlayer.src = "/audio_stream";
                audioPlayer.style.display="none";
                
                // 添加错误处理
                videoPlayer.onerror = function(e) {
                    console.error("视频播放错误:", e);
                    statusText.textContent = "视频播放失败: " + e.message;
                };
                
                audioPlayer.onerror = function(e) {
                    console.error("音频播放错误:", e);
                    statusText.textContent = "音频播放失败: " + e.message;
                };
                
                // 同步播放逻辑
                let syncAttempts = 0;
                const maxSyncAttempts = 3;
                
                function attemptSyncPlay() {
                    syncAttempts++;
                    
                    // 等待两个媒体都准备好
                    Promise.all([
                        new Promise((resolve) => {
                            if (videoPlayer.readyState >= 2) {
                                resolve();
                            } else {
                                videoPlayer.addEventListener('canplay', resolve, { once: true });
                            }
                        }),
                        new Promise((resolve) => {
                            if (audioPlayer.readyState >= 2) {
                                resolve();
                            } else {
                                audioPlayer.addEventListener('canplay', resolve, { once: true });
                            }
                        })
                    ]).then(() => {
                        // 同时开始播放
                        const playPromises = [
                            videoPlayer.play(),
                            audioPlayer.play()
                        ];
                        
                        Promise.allSettled(playPromises).then((results) => {
                            const videoResult = results[0];
                            const audioResult = results[1];
                            
                            if (videoResult.status === 'fulfilled' && audioResult.status === 'fulfilled') {
                                statusText.textContent = "播放中...";
                                console.log("视频和音频同步播放成功");
                            } else {
                                console.error("播放失败:", videoResult, audioResult);
                                if (syncAttempts < maxSyncAttempts) {
                                    console.log(`同步尝试 ${syncAttempts}/${maxSyncAttempts}，重试...`);
                                    setTimeout(attemptSyncPlay, 100);
                                } else {
                                    statusText.textContent = "播放失败，请重试";
                                }
                            }
                        });
                    }).catch((error) => {
                        console.error("同步播放错误:", error);
                        if (syncAttempts < maxSyncAttempts) {
                            console.log(`同步尝试 ${syncAttempts}/${maxSyncAttempts}，重试...`);
                            setTimeout(attemptSyncPlay, 100);
                        } else {
                            statusText.textContent = "播放失败，请重试";
                        }
                    });
                }
                
                // 开始同步播放
                attemptSyncPlay();
            };
            
            function checkDebug() {
                fetch('/debug')
                    .then(response => response.json())
                    .then(data => {
                        const debugInfo = document.getElementById('debugInfo');
                        debugInfo.style.display = 'block';
                        debugInfo.innerHTML = '<h3>调试信息:</h3><pre>' + JSON.stringify(data, null, 2) + '</pre>';
                    })
                    .catch(e => {
                        const debugInfo = document.getElementById('debugInfo');
                        debugInfo.style.display = 'block';
                        debugInfo.innerHTML = '<h3>调试信息获取失败:</h3><pre>' + e + '</pre>';
                    });
            }
            
            function createTestFrames() {
                fetch('/test_frames')
                    .then(response => response.json())
                    .then(data => {
                        statusText.textContent = `创建了 ${data.frames_count} 个测试帧`;
                        playBtn.disabled = false;
                        uploaded = true;
                    })
                    .catch(e => {
                        statusText.textContent = "创建测试帧失败: " + e.message;
                    });
            }
        </script>
    </body>
    </html>
    """

@app.route('/')
def index():
    return render_template_string(HTML)

def split_video_audio(file_path):
    """
    使用FFmpeg提取视频帧和音频帧
    将视频帧和音频帧放到各自queue中。
    """
    global video_frames_list
    
    # 清空队列和列表
    while not video_frame_queue.empty():
        video_frame_queue.get()
    while not audio_frame_queue.empty():
        audio_frame_queue.get()
    video_frames_list.clear()

    # 视频帧: 使用FFmpeg提取JPEG帧
    frame_dir = os.path.join(temp_dir, 'frames')
    os.makedirs(frame_dir, exist_ok=True)
    
    frame_pattern = os.path.join(frame_dir, "frame_%04d.jpg")
    cmd = [
        'ffmpeg', '-i', file_path,
        '-vf', 'fps=25',  # 25fps
        '-q:v', '2',      # 高质量
        frame_pattern,
        '-y'              # 覆盖现有文件
    ]
    
    print(f"执行视频帧提取命令: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"FFmpeg视频帧提取错误: {result.stderr}")
        return
    
    # 读取所有帧文件
    frame_files = sorted([f for f in os.listdir(frame_dir) if f.startswith('frame_') and f.endswith('.jpg')])
    frame_count = 0
    
    for frame_file in frame_files:
        frame_path = os.path.join(frame_dir, frame_file)
        with open(frame_path, 'rb') as f:
            frame_data = f.read()
            video_frame_queue.put(frame_data)
            video_frames_list.append(frame_data)
            frame_count += 1
    
    print(f"提取了 {frame_count} 帧视频")

    # 音频帧: 用ffmpeg导出为wav（16k/单声/16位）
    wav_path = os.path.join(temp_dir, 'audio.wav')
    cmd = [
        'ffmpeg', '-i', file_path,
        '-vn',           # 不要视频
        '-acodec', 'pcm_s16le',  # PCM 16位
        '-ar', '16000',  # 16kHz采样率
        '-ac', '1',      # 单声道
        wav_path,
        '-y'
    ]
    
    print(f"执行音频提取命令: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"FFmpeg音频提取错误: {result.stderr}")
        return
    
    # 将整个wav文件读入内存
    with open(wav_path, "rb") as f:
        wav_bytes = f.read()
        audio_frame_queue.put(wav_bytes)
        print(f"音频文件大小: {len(wav_bytes)} 字节")
    
    # 预生成MP4文件
    global mp4_file_path, audio_file_path
    mp4_file_path = os.path.join(temp_dir, 'output.mp4')
    audio_file_path = wav_path
    
    # 使用ffmpeg将JPEG帧转换为MP4
    frame_pattern = os.path.join(temp_dir, 'frames', 'frame_%04d.jpg')
    cmd = [
        'ffmpeg',
        '-framerate', '25',  # 25fps
        '-i', frame_pattern,
        '-c:v', 'libx264',  # H.264编码
        '-preset', 'fast',  # 快速编码
        '-crf', '23',  # 质量设置
        '-pix_fmt', 'yuv420p',  # 兼容性好的像素格式
        '-movflags', '+faststart',  # 优化流媒体播放
        mp4_file_path,
        '-y'
    ]
    
    print(f"预生成MP4文件: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"FFmpeg MP4预生成错误: {result.stderr}")
        mp4_file_path = None
    else:
        mp4_size = os.path.getsize(mp4_file_path)
        print(f"MP4文件预生成成功，大小: {mp4_size} 字节")

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return {"status": "no file"}
    f = request.files['file']
    save_path = os.path.join(temp_dir, f.filename)
    f.save(save_path)
    # 分离帧
    split_video_audio(save_path)
    return {"status": "ok"}

def generate_video_stream():
    """使用预生成的MP4文件进行流式传输"""
    global mp4_file_path
    
    if not mp4_file_path or not os.path.exists(mp4_file_path):
        print("没有预生成的MP4文件可播放")
        return
    
    print(f"开始播放预生成的MP4文件: {mp4_file_path}")
    
    # 读取MP4文件并流式传输
    try:
        with open(mp4_file_path, 'rb') as f:
            while True:
                chunk = f.read(8192)  # 8KB chunks
                if not chunk:
                    f.seek(0)  # 循环播放
                    continue
                yield chunk
    except Exception as e:
        print(f"MP4流传输错误: {e}")

@app.route('/video_stream')
def video_stream():
    """视频流端点"""
    if not video_frames_list:
        return Response("No video frames available", status=404)
    
    response = Response(generate_video_stream(),
                       mimetype='video/mp4')
    response.headers['Cache-Control'] = 'no-cache'
    response.headers['Connection'] = 'close'
    return response

@app.route('/audio_stream')
def audio_stream():
    # 直接返回完整wav, 并设定浏览器下载/播放
    if audio_frame_queue.empty():
        print("音频队列为空，返回静音音频")
        # 返回一个短的静音WAV文件
        silence_wav = b'RIFF$\x00\x00\x00WAVEfmt \x10\x00\x00\x00\x01\x00\x01\x00\x80>\x00\x00\x00}\x00\x00\x02\x00\x10\x00data\x00\x00\x00\x00'
        return Response(silence_wav, mimetype="audio/wav")
    
    wav_bytes = audio_frame_queue.get()
    print(f"返回音频流，大小: {len(wav_bytes)} 字节")
    return Response(wav_bytes, mimetype="audio/wav")

@app.route('/test_frames')
def test_frames():
    """创建测试帧"""
    create_test_frames()
    return {"status": "ok", "frames_count": len(video_frames_list)}

@app.route('/debug')
def debug():
    """调试端点，显示当前状态"""
    import subprocess
    
    # 检查ffmpeg是否可用
    ffmpeg_available = False
    try:
        result = subprocess.run(['ffmpeg', '-version'], capture_output=True, text=True)
        ffmpeg_available = result.returncode == 0
    except:
        pass
    
    return {
        "video_frames_count": len(video_frames_list),
        "video_queue_size": video_frame_queue.qsize(),
        "audio_queue_size": audio_frame_queue.qsize(),
        "temp_dir": temp_dir,
        "ffmpeg_available": ffmpeg_available,
        "first_frame_size": len(video_frames_list[0]) if video_frames_list else 0,
        "frame_files_exist": os.path.exists(os.path.join(temp_dir, 'frames')) if temp_dir else False
    }

def create_test_frames():
    """创建测试用的彩色帧和音频"""
    global video_frames_list
    
    # 清空队列
    while not video_frame_queue.empty():
        video_frame_queue.get()
    while not audio_frame_queue.empty():
        audio_frame_queue.get()
    
    # 创建一些测试帧
    video_frames_list = []
    colors = [
        (255, 0, 0),    # 红色
        (0, 255, 0),    # 绿色
        (0, 0, 255),    # 蓝色
        (255, 255, 0),  # 黄色
        (255, 0, 255),  # 紫色
        (0, 255, 255),  # 青色
    ]
    
    # 确保frames目录存在
    frames_dir = os.path.join(temp_dir, 'frames')
    os.makedirs(frames_dir, exist_ok=True)
    
    for i, color in enumerate(colors):
        # 创建一个简单的JPEG帧文件
        frame_path = os.path.join(frames_dir, f'frame_{i+1:04d}.jpg')
        cmd = [
            'ffmpeg', '-f', 'lavfi', '-i', f'color=c={color[0]:02x}{color[1]:02x}{color[2]:02x}:size=640x480:duration=0.1',
            '-frames:v', '1',
            '-f', 'image2',
            '-vcodec', 'mjpeg',
            '-q:v', '2',
            frame_path,
            '-y'
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode == 0:
            # 读取文件内容到内存
            with open(frame_path, 'rb') as f:
                frame_data = f.read()
                video_frames_list.append(frame_data)
                video_frame_queue.put(frame_data)
                print(f"创建测试帧 {i+1}: {len(frame_data)} 字节")
        else:
            print(f"创建测试帧 {i+1} 失败: {result.stderr}")
    
    # 创建测试音频 - 生成一个简单的音调
    audio_cmd = [
        'ffmpeg', '-f', 'lavfi', '-i', 'sine=frequency=440:duration=2',
        '-acodec', 'pcm_s16le',
        '-ar', '16000',
        '-ac', '1',
        '-f', 'wav',
        'pipe:1'
    ]
    
    print("创建测试音频...")
    audio_result = subprocess.run(audio_cmd, capture_output=True)
    if audio_result.returncode == 0:
        audio_frame_queue.put(audio_result.stdout)
        print(f"创建测试音频: {len(audio_result.stdout)} 字节")
    else:
        print(f"音频创建失败: {audio_result.stderr}")
    
    print(f"创建了 {len(video_frames_list)} 个测试帧")
    
    # 预生成MP4文件
    global mp4_file_path, audio_file_path
    mp4_file_path = os.path.join(temp_dir, 'output.mp4')
    audio_file_path = os.path.join(temp_dir, 'test_audio.wav')
    
    # 保存测试音频到文件
    if audio_result.returncode == 0:
        with open(audio_file_path, 'wb') as f:
            f.write(audio_result.stdout)
    
    # 使用ffmpeg将JPEG帧转换为MP4
    frame_pattern = os.path.join(temp_dir, 'frames', 'frame_%04d.jpg')
    cmd = [
        'ffmpeg',
        '-framerate', '25',  # 25fps
        '-i', frame_pattern,
        '-c:v', 'libx264',  # H.264编码
        '-preset', 'fast',  # 快速编码
        '-crf', '23',  # 质量设置
        '-pix_fmt', 'yuv420p',  # 兼容性好的像素格式
        '-movflags', '+faststart',  # 优化流媒体播放
        mp4_file_path,
        '-y'
    ]
    
    print(f"预生成测试MP4文件: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"FFmpeg测试MP4预生成错误: {result.stderr}")
        mp4_file_path = None
    else:
        mp4_size = os.path.getsize(mp4_file_path)
        print(f"测试MP4文件预生成成功，大小: {mp4_size} 字节")

if __name__ == '__main__':
    print("启动实时播放服务器...")
    print("访问: http://localhost:8080")
    
    # 检查ffmpeg是否可用
    try:
        result = subprocess.run(['ffmpeg', '-version'], capture_output=True, text=True)
        if result.returncode == 0:
            print("✓ FFmpeg 可用")
            # 创建测试帧
            create_test_frames()
        else:
            print("✗ FFmpeg 不可用")
    except:
        print("✗ FFmpeg 未安装")
        print("请安装FFmpeg: brew install ffmpeg")
    
    app.run(debug=True, port=8080, host='0.0.0.0')

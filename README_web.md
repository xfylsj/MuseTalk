# MuseTalk实时Web推理服务器

## 功能说明

这个Web服务器实现了MuseTalk的实时语音驱动功能：

1. **音频上传**: 用户可以通过Web界面上传MP3或WAV音频文件
2. **实时推理**: 后端使用MuseTalk模型对音频进行流式推理
3. **实时播放**: 通过WebSocket实时推送合成的视频帧和音频帧到前端
4. **实时显示**: 前端实时播放合成的视频和音频

## 安装依赖

```bash
pip install -r requirements_web.txt
```

## 使用方法

1. **启动服务器**:
   ```bash
   python scripts/realtime_web_stream.py
   ```

2. **访问Web界面**:
   打开浏览器访问 `http://localhost:8080`

3. **使用步骤**:
   - 上传音频文件（MP3或WAV格式）
   - 点击"开始推理"按钮
   - 观看实时生成的视频和音频

## 技术实现

### 后端架构
- **Flask + SocketIO**: Web服务器和WebSocket通信
- **MuseTalk模型**: 语音驱动的视频生成
- **流式处理**: 按1秒音频块进行推理
- **实时推送**: 通过WebSocket推送合成帧

### 前端架构
- **HTML5**: 基础界面
- **Socket.IO**: WebSocket客户端
- **Canvas API**: 视频帧渲染
- **Web Audio API**: 音频播放

### 核心流程
1. 音频文件上传到服务器
2. MuseTalk模型初始化Avatar（使用默认图片）
3. 音频按1秒块进行流式处理
4. 每个音频块生成对应的视频帧
5. 视频帧和音频块通过WebSocket实时推送到前端
6. 前端实时播放合成的视频和音频

## 配置说明

默认配置：
- 使用 `./data/pic/dd.png` 作为Avatar图片
- 模型路径：`./models/musetalk/`
- Whisper模型：`./models/whisper/`
- 输出路径：`./results/web_avatars/`

## 注意事项

1. 确保MuseTalk模型文件已正确下载
2. 确保默认Avatar图片存在
3. 需要GPU支持以获得最佳性能
4. 音频文件建议不超过几分钟长度


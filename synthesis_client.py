from Processor import stt_processor, tts_generator, speech2cmd, sport_client
from unitree_sdk2py.idl.unitree_go.msg.dds_ import LowState_
from unitree_sdk2py.core.channel import ChannelSubscriber
from collections import deque
import sounddevice as sd
import numpy as np
import threading
import requests
import logging
import socket
import wave
import time
import json
import os
import re


"""
version 1.0.2
增加了一个简单的输入分流
对于所有输入，首先进行控制指令的提取，同时将该内容与提示词一起传给指定的大模型
在传给大模型的同时，在提示词中要求大模型对输入内容进行判断，如果需要进行回答，则额外输出flag=1，否则flag=0
如果提取到了指令，就执行并忽略大模型的返回值；
如果没有提取到指令，就等待大模型的回答，如果flag=1，就提取回答并生成语音，反之则忽略。
"""

logging.getLogger("funasr.utils.cli_utils").disabled = True

# 配置参数
device_id = len(sd.query_devices()) - 1  # use default input, change in settings
SampleRate = int(sd.query_devices(device_id, 'input')['default_samplerate'])
BlockSize = 1024
Low_Threshold = 15  # 监测响应音量
High_Threshold = 20  # 开始录音音量
PreRecord = 1  # 预录音时长/s
SilenceCut = 1  # 结束录音检测时长
Gain_Factor = 2  # 增益系数
Save_Path = "recordings"
os.makedirs(Save_Path, exist_ok=True)
os.makedirs('voices', exist_ok=True)

# 远程控制及服务器交互参数
equip_id = "1001"
SERVER_IP = "47.111.140.142"
SERVER_PORT = 9000
running = True
state_freq = 10
lock = threading.Lock()

# 大模型通信参数
llm_url_root = "****"
llm_chat_route = "/ChatMessages"


class AudioRecorder:
    def __init__(self):
        self.recording = False
        self.last_loud_time = 0
        self.last_activity_time = 0
        self.should_stop = False

        # 预录音缓冲区和数据缓存
        buffer_size = int(SampleRate * PreRecord / BlockSize)
        self.buffer = deque(maxlen=buffer_size)
        self.data_list = []

        # 监听控制标志和音频流对象
        self.is_listening = False
        self.stream = None

        self._lock = threading.Lock()

    def audio_callback(self, indata, frames, time_info, status):
        if not self.is_listening:  # 仅在监听状态下处理音频
            return
        # 持续更新预录音区，超出部分会从头开始自动删除
        current_time = time.time()
        self.buffer.append(indata.copy())

        # 计算音量
        rms = np.sqrt(np.mean(indata.astype(np.float32) ** 2)) * Gain_Factor
        volume_percent = (rms / 32767) * 100

        with self._lock:
            if self.should_stop:
                return

        # 录音逻辑
        if self.recording:
            if volume_percent >= High_Threshold:
                self.last_loud_time = current_time
            elif current_time - self.last_loud_time >= SilenceCut:
                # 保留阈值激活后两秒的内容，防止信息丢失
                self.stop_recording()
            self.data_list.append(indata.copy())
        else:
            if volume_percent >= High_Threshold:
                self.start_recording()
                print("开始录音")
                self.last_loud_time = current_time
        # 发送提示逻辑
        if volume_percent >= Low_Threshold and current_time - self.last_activity_time > SilenceCut/2:
            self.last_activity_time = current_time

    def start_recording(self):
        with self._lock:
            self.recording = True
            self.data_list = list(self.buffer)

    def stop_recording(self):
        with self._lock:
            if not self.recording:
                return

            self.should_stop = True
            self.recording = False

            # 裁剪末尾静音
            total_frames = len(self.data_list) * BlockSize
            remove_frames = SilenceCut * SampleRate
            keep_frames = total_frames - remove_frames

            if keep_frames <= 0:
                # print("录音过短，已丢弃")
                self.data_list = []
                self.should_stop = False
                return

            full_blocks = keep_frames // BlockSize
            remainder = keep_frames % BlockSize

            truncated = self.data_list[:full_blocks]
            if remainder > 0 and full_blocks < len(self.data_list):
                last_block = self.data_list[full_blocks][:remainder]
                truncated.append(last_block)

            filename = os.path.join(Save_Path, f"recording_{int(self.last_loud_time)}.wav")
            with wave.open(filename, "wb") as f:
                f.setnchannels(1)
                f.setsampwidth(2)
                f.setframerate(SampleRate)
                for chunk in truncated:
                    f.writeframes(chunk.tobytes())
                # print(f"文件保存：{filename}")
            
            def async_process():
                execution = 0
                try:
                    result = stt_processor.process(filename)
                    execution = speech2cmd(result, execution)
                    
                    if not execution:
                        tts_path = os.path.join("voices", f"{str(time.time())}.wav")
                        
                        # 使用LLMClient处理请求
                        response = llm_client.query(result)  # 直接获取完整响应
                        
                        # 生成语音
                        tts_generator.generate(response, tts_path)
                        tts_generator.play_audio(tts_path)
                except Exception as e:
                    print(f"{type(e).__name__} - {e}")

            threading.Thread(target=async_process).start()

            self.data_list = []
            self.should_stop = False

    # 控制音频流的方法
    def start_listening(self):
        with self._lock:
            if self.is_listening:
                return
            self.is_listening = True
            self.stream = sd.InputStream(
                samplerate=SampleRate,
                blocksize=BlockSize,
                device=device_id,
                channels=1,
                dtype="int16",
                callback=self.audio_callback
            )
            self.stream.start()

    def stop_listening(self):
        with self._lock:
            if not self.is_listening:
                return
            self.is_listening = False
            if self.stream:
                self.stream.stop()
                self.stream.close()
                self.stream = None


class Go2Monitor:
    def __init__(self, id):
        self.low_state = None

        # 创建状态订阅器
        self.sub = ChannelSubscriber("rt/lowstate", LowState_)
        self.sub.Init(self._state_handler, 10)
        
        self.id = id
        
        # 等待初始数据
        time.sleep(0.5)

    def _state_handler(self, msg: LowState_):
        """内部状态处理回调"""
        self.low_state = msg

    def get_battery_info(self):
        """获取电池信息
        Returns:
            dict: 包含电池信息的字典，结构如下:
            {
                'voltage': 总电压(V),
                'current': 总电流(A),
                'soc': 剩余电量百分比(0-100),
                'cycle_count': 充电循环次数,
                'mainboard':主板温度,
                'temperatures': {
                    'bat1': 电池内部温度1(°C),
                    'bat2': 电池内部温度2(°C),
                    'mcu_res': MCU电阻温度(°C),
                    'mcu_mos': MCU MOS温度(°C)
                }
            }
        """
        if not self.low_state:
            return None

        bms = self.low_state.bms_state
        return {
            'voltage': self.low_state.power_v,
            'current': self.low_state.power_a,
            'soc': bms.soc,
            'mainboard': self.low_state.temperature_ntc1,
            'temperatures': {
                'bat1': bms.bq_ntc[0],
                'bat2': bms.bq_ntc[1],
                'mcu_res': bms.mcu_ntc[0],
                'mcu_mos': bms.mcu_ntc[1]
            }
        }


class TCPClient:
    def __init__(self, sport_client):
        self.sock = None
        self.connected = False
        self.sport_client = sport_client
        self.reconnect_interval = 5
        self.buffer = b""  # 添加接收缓冲区
        self.known_commands = ["forward", "backward", "left", "right", "stop", "sitdown", "standup"]  # 已知指令列表
    
    def connect(self):
        while True:
            try:
                self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                self.sock.settimeout(10)
                self.sock.connect((SERVER_IP, SERVER_PORT))
                
                # 设置 TCP_NODELAY 禁用 Nagle 算法
                self.sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
                
                self.connected = True
                self.sock.settimeout(None)
                print("✅ 已连接云服务器")
                return True
            except socket.timeout:
                print("⌛ 连接超时，重试中...")
            except Exception as e:
                print(f"❌ 连接失败: {str(e)}")
            print(f"⏳ {self.reconnect_interval}秒后尝试重新连接...")
            time.sleep(self.reconnect_interval)
    
    def receive_commands(self):
        while True:
            if not self.connected:
                self.connect()
                continue
            
            try:
                data = self.sock.recv(256)
                if not data:
                    raise ConnectionError("连接中断")
                
                # 添加到缓冲区
                self.buffer += data
                
                # 处理缓冲区中的所有完整消息
                while self.buffer:
                    # 方案1: 尝试按消息边界分割
                    if b'\n' in self.buffer:
                        line, _, self.buffer = self.buffer.partition(b'\n')
                        message = line.decode('utf-8').strip()
                        self.process_message(message)
                    # 方案2: 尝试识别已知指令
                    else:
                        found = False
                        for cmd in self.known_commands:
                            cmd_bytes = cmd.encode('utf-8')
                            if self.buffer.startswith(cmd_bytes):
                                # 提取并处理指令
                                message = cmd_bytes.decode('utf-8')
                                self.buffer = self.buffer[len(cmd_bytes):]
                                self.process_message(message)
                                found = True
                                break
                        
                        # 如果没有找到完整指令，等待更多数据
                        if not found:
                            break
            except Exception as e:
                print(f"⚠️ 接收错误: {str(e)}")
                self.connected = False
    
    def process_message(self, message):
        """处理单条消息"""
        print(f"📥 收到指令: {message}")
        self.execute_command(message)

    def execute_command(self, cmd):
        # 使用更精确的指令匹配
        cmd = cmd.strip().lower()

        if cmd == "forward":
            self.sport_client.Move(0.6, 0, 0)
            print("执行: 前进")
        elif cmd == "backward":
            self.sport_client.Move(-0.6, 0, 0)
            print("执行: 后退")
        elif cmd == "left":
            self.sport_client.Move(0, 0, 1.0)
            print("执行: 左转")
        elif cmd == "right":
            self.sport_client.Move(0, 0, -1.0)
            print("执行: 右转")
        elif cmd == "sitdown":
            self.sport_client.StandDown()
            print("执行: 坐下")
        elif cmd == "standup":
            self.sport_client.StandUp()
            time.sleep(0.5)
            self.sport_client.BalanceStand()
            print("执行: 站立")
        else:
            print(f"⚠️ 未知指令: {cmd}")

    def send_state(self, data_type, data_id, content):
        with lock:
            if not self.connected:
                print("⚠️ 未连接服务器，无法发送数据")
                return False

            try:
                payload = json.dumps({
                    "type": data_type,
                    "id": data_id,
                    "content": content
                }) + '\n'  # 添加换行符作为分隔符
                self.sock.sendall(payload.encode('utf-8'))
                return True
            except Exception as e:
                print(f"❌ 发送失败: {str(e)}")
                self.connected = False
                return False

    def close(self):
        global running
        running = False
        if self.sock:
            self.sock.close()


class LLMClient:
    def __init__(self, base_url=llm_url_root, timeout=10): # timeout暂时存疑
        self.base_url = base_url
        self.timeout = timeout
        self.session = requests.Session()
    
    def clean_response(self, text):
        """清理特殊符号并提取有效内容"""
        if not text:
            return ""
        
        # 移除Markdown符号
        text = re.sub(r'[#*_`~]', '', text)
        # 移除HTML标签
        text = re.sub(r'<.*?>', '', text)
        # 合并多余空格和换行
        text = re.sub(r'\s+', ' ', text)
        # 处理特殊转义序列
        text = text.replace('\\n', '\n').replace('\\t', '\t')
        # 移除JSON元数据标记
        text = re.sub(r'\{.*?\}', '', text)
        return text.strip()
    
    def stream_query(self, query, chat_url=llm_chat_route):
        """处理流式响应，提取answer字段"""
        url = self.base_url + chat_url
        headers = {"Content-Type": "application/json"}
        payload = {
            "query": query,
            "ResponseMode": "streaming",
            "UserId": monitor.id,
            "ProjectId": "string",
            "limit": "string"
        }
        
        full_response = ""
        answer_complete = False
        metadata_detected = False
        
        try:
            # 使用流式接收
            with self.session.post(
                    url,
                    json=payload,
                    headers=headers,
                    stream=True,
                    timeout=self.timeout
            ) as response:
                
                if response.status_code != 200:
                    print(f"LLM请求失败: HTTP {response.status_code}")
                    return "服务暂时不可用，请稍后再试"
                
                # 逐行处理流式响应
                for line in response.iter_lines():
                    if not line:
                        continue
                    
                    try:
                        # 解码JSON
                        chunk = json.loads(line.decode('utf-8'))
                        
                        # 检查是否包含answer字段
                        if "answer" in chunk and not metadata_detected:
                            # 提取并清理内容
                            cleaned = self.clean_response(chunk["answer"])
                            full_response += cleaned
                            
                            # 检查answer是否完整
                            if cleaned.endswith(('.', '?', '!', '。', '？', '！')):
                                answer_complete = True
                        
                        # 检查metadata标记
                        if "metadata" in chunk:
                            metadata_detected = True
                        
                        # 如果检测到元数据或answer已完成，停止处理
                        if metadata_detected or answer_complete:
                            break
                    
                    except json.JSONDecodeError:
                        # 尝试提取可能的文本内容
                        line_str = line.decode('utf-8', errors='ignore')
                        if '"answer":' in line_str:
                            # 尝试手动提取answer内容
                            match = re.search(r'"answer":\s*"([^"]+)"', line_str)
                            if match:
                                cleaned = self.clean_response(match.group(1))
                                full_response += cleaned
                    
                    except Exception as e:
                        print(f"处理响应行时出错: {str(e)}")
        
        except requests.exceptions.Timeout:
            print("LLM请求超时")
            if not full_response:
                return "请求超时，请稍后再试"
        
        except Exception as e:
            print(f"未知错误: {str(e)}")
            if not full_response:
                return "处理请求时出错"
        
        return full_response
    
    def query(self, query, chat_url=llm_chat_route):
        """发送查询并获取响应"""
        url = self.base_url + chat_url
        headers = {"Content-Type": "application/json"}
        payload = {
            "query": query,
            "ResponseMode": "streaming",
            "UserId": monitor.id,
            "ProjectId": "string",
            "limit": "string"
        }
        
        try:
            # 发送普通POST请求
            response = self.session.post(
                url,
                json=payload,
                headers=headers,
                timeout=self.timeout
            )
            
            # 检查响应状态
            if response.status_code != 200:
                print(f"LLM请求失败: HTTP {response.status_code}")
                return "服务暂时不可用，请稍后再试"
            
            # 解析JSON响应
            response_data = response.json()
            
            # 提取answer内容
            if "data" in response_data and isinstance(response_data["data"], dict):
                data = response_data["data"]
                if "answer" in data:
                    answer = data["answer"]
                    # 清理内容
                    cleaned_answer = self.clean_response(answer)
                    return cleaned_answer
            
            # 如果找不到预期结构，尝试直接提取answer
            if "answer" in response_data:
                answer = response_data["answer"]
                return self.clean_response(answer)
            
            # 如果都没有，返回错误信息
            print(f"响应中缺少answer字段: {response_data}")
            return "响应中缺少有效内容"
        
        except requests.exceptions.Timeout:
            print("LLM请求超时")
            return "请求超时，请稍后再试"
        
        except Exception as e:
            print(f"未知错误: {str(e)}")
            return "处理请求时出错"

# 初始化网络控制
tcp_client = TCPClient(sport_client)

# 初始化语音控制
recorder = AudioRecorder()
recorder.start_listening()
tcp_thread = threading.Thread(target=tcp_client.receive_commands, daemon=True)
tcp_thread.start()

# 初始化大模型请求处理
llm_client = LLMClient()

# 初始化状态监控
monitor = Go2Monitor(equip_id)

try:
    while True:
        battery_info = monitor.get_battery_info()
        tcp_client.send_state("state", monitor.id,
                              {
                                  "energy_remain": battery_info['soc'],
                                  "mainboard_tempera": battery_info['mainboard'],
                              })
        time.sleep(state_freq)
except KeyboardInterrupt:
    recorder.stop_listening()
    tcp_client.close()
    # print("\n程序已终止")
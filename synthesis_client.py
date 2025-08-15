from unitree_sdk2py.core.channel import ChannelSubscriber, ChannelFactoryInitialize
from unitree_sdk2py.go2.sport.sport_client import SportClient
from unitree_sdk2py.idl.unitree_go.msg.dds_ import LowState_
from Processor import processor, tts
from pypinyin import lazy_pinyin
from collections import deque
import sounddevice as sd
import numpy as np
import threading
import logging
import socket
import cn2an
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
Save_Path = "recordings"
Gain_Factor = 2  # 增益系数
os.makedirs(Save_Path, exist_ok=True)

# 远程控制参数
SERVER_IP = "47.111.140.142"
SERVER_PORT = 9000
tcp_socket = None
running = True
lock = threading.Lock()

# 初始化运动控制及低层状态监控
ChannelFactoryInitialize(0, "eth0")
sport_client = SportClient()
sport_client.SetTimeout(10.0)
sport_client.Init()
state_freq = 10
print("运动控制初始化完成！")

#个性化参数
wake_up = ["go to", "gou2", "go 2", "go two", "gou to", "goto"]  # 唤醒词
is_wake = 0  # 是否被唤醒，默认不启用
command_mode = "Interactive"  # 控制模式：规划模式可以识别一连串指令；交互模式则只支持单句话
validity = 6  # 被唤醒之后的几句话将可被识别
is_sleeping = True  # 被唤醒一定时间后无指令将沉睡
speed_list = [0.3, 0.5, 0.7]
default_speed = 1
default_distance = 1
default_angle = 45


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
                    result = processor.process(filename)
                    execution = speech2cmd(result, execution)
                    if not execution:
                        response = tcp_client.send2llm(result) # 目标大模型是流式传输，因此有多个回传
                        tts(response)
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
    def __init__(self, name):
        """初始化机器狗监控模块
        Args:
            ether_name: 网络接口名称 (可选)
        """
        self.low_state = None

        # 创建状态订阅器
        self.sub = ChannelSubscriber("rt/lowstate", LowState_)
        self.sub.Init(self._state_handler, 10)

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


    def get_combined_info(self):
        """获取电池和温度的综合信息

        Returns:
            dict: 包含所有监控信息的字典
        """
        return {
            'battery': self.get_battery_info(),
            'temperature': self.get_temperature_info()
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
            
    def send2llm(self, query):
        return "test"

    def close(self):
        global running
        running = False
        if self.sock:
            self.sock.close()


def speech2cmd(result, exec_flag):
    global validity, default_distance, default_speed, default_angle, is_wake
    is_wake = 0
    def process_command(content, flag):
        global command_mode
        if "设置模式" in content:
            if "计划模式" in content:
                command_mode = "Planning"
                print("mode set to Planing")
                flag = 1
            elif "交互模式" in content:
                command_mode = "Interactive"
                flag = 1
        else:
            if command_mode == "Interactive":
                flag = interact_execute(content, flag)
            if command_mode == "Planning":
                flag = planning_execute(content, flag)
        return flag
    for word in wake_up:
        if ''.join(lazy_pinyin(word)) in ''.join(lazy_pinyin(result)):
            is_wake = 1
            print("detect wake-up word")
            validity = 6
            exec_flag = process_command(result, exec_flag)
            break
    if not is_wake and validity:
        print(f"validity remain {validity} times")
        validity -= 1
        exec_flag = process_command(result, exec_flag)
    return exec_flag

def interact_execute(command, get_cmd):
    global execution
    print(f"interactive mode: {command}")
    global default_distance, default_speed
    move_pattern = re.compile(r'([前后])走?(\d+|[零一二三四五六七八九十百]+)?米?')
    turn_pattern = re.compile(r'([左右])转?(\d+|[零一二三四五六七八九十百]+)?度?')
    setting_pattern = re.compile(r'(上调|增大|减小|下调)默认(速度|距离)')
    
    # 尝试匹配移动指令（走）
    if match := move_pattern.search(command):
        print("forward or backward detected")
        speed = speed_list[default_speed]
        direction = speed if match.group(1) == "前" else -speed
        # 提取距离或使用默认值
        step = int(cn2an.cn2an(match.group(2), "smart")/speed) if match.group(2) else int(default_distance/speed)
        print(f"direction:{direction}, step:{step}")
        for i in range(step):
            sport_client.Move(direction, 0, 0)
            time.sleep(1)
        get_cmd = 1
        

    # 尝试匹配旋转指令（转）
    elif match := turn_pattern.search(command):
        print("left or right detected")
        angle_list = [0, 45, 90, 135, 180]
        direction = 1 if match.group(1) == "左" else -1
        # 提取角度或使用默认值
        angle = cn2an.cn2an(match.group(2), "smart") if match.group(2) else default_angle
        for i in range(len(angle_list)):
            if angle_list[i] < angle <= angle_list[i + 1]:
                angle = angle / (i + 1)
                distance = 3 * (i + 1)
                break
        print(f"abgle:{angle}, step:{distance}")
        for i in range(distance):
            sport_client.Move(0, 0, direction * angle / 180 * 3.14)
            time.sleep(1)
        get_cmd = 1

    elif "站起" in command:
        sport_client.StandUp()
        time.sleep(0.5)
        sport_client.BalanceStand()
        get_cmd = 1
    elif "坐下" in command:
        sport_client.StandDown()
        get_cmd = 1
    elif match := setting_pattern.search(command):
        action, target = match.group(1), match.group(2)
        if target == "速度" and 0 <= default_speed <= 2:
            default_speed += 1 if action in ["上调", "增大"] else -1

            sport_client.SwitchGait(1)
            time.sleep(1)
            sport_client.SwitchGait(0)
            get_cmd = 1
        if target == "距离" and 1 <= default_distance <= 5:
            default_distance += 1 if action in ["上调", "增大"] else -1

            sport_client.SwitchGait(1)
            time.sleep(1)
            sport_client.SwitchGait(0)
            get_cmd = 1
    else:
        get_cmd = 0
        print("interactive mode: no matched results")
    return get_cmd

def planning_execute(text, get_cmd):
    global default_distance, default_speed, execution
    move_pattern = re.compile(r'([前后])走?(\d+|[零一二三四五六七八九十百]+)?米?')
    turn_pattern = re.compile(r'([左右])转?(\d+|[零一二三四五六七八九十百]+)?度?')

    command_list = split_command(text)
    print(f"planning mode: {command_list}")
    for command in command_list:
        if match := move_pattern.search(command):
            print("forward or backward detected")
            speed = speed_list[default_speed]
            direction = speed if match.group(1) == "前" else -speed
            # 提取距离或使用默认值
            if match.group(2):
                print(f"步为{int(cn2an.cn2an(match.group(2), 'smart'))/speed}")
            step = int(cn2an.cn2an(match.group(2), "smart")/speed) if match.group(2) else int(default_distance/speed)
            print(f"direction:{direction}, step:{step}")
            for i in range(step):
                sport_client.Move(direction, 0, 0)
                time.sleep(1)
            get_cmd = 1

        # 尝试匹配旋转指令（转）
        elif match := turn_pattern.search(command):
            print("left or right detected")
            angle_list = [0, 45, 90, 135, 180]
            direction = 1 if match.group(1) == "左" else -1
            # 提取角度或使用默认值
            angle = cn2an.cn2an(match.group(2), "smart") if match.group(2) else default_angle
            for i in range(len(angle_list)):
                if angle_list[i] < angle <= angle_list[i + 1]:
                    angle = angle / (i + 1)
                    distance = 3 * (i + 1)
                    break
            print(f"abgle:{angle}, step:{distance}")
            for i in range(distance):
                sport_client.Move(0, 0, direction * angle / 180 * 3.14)
                time.sleep(1)
            get_cmd = 1
    return get_cmd

def split_command(text):
    result = []
    current = []

    for char in text:
        if char in '，。':  # 遇到逗号或句号
            if current:  # 确保当前片段非空
                result.append(''.join(current))
                current = []
        else:
            current.append(char)

    # 处理最后一个片段（如果存在）
    if current:
        result.append(''.join(current))

    return result


# 初始化网络控制
tcp_client = TCPClient(sport_client)

# 初始化语音控制
recorder = AudioRecorder()
recorder.start_listening()
tcp_thread = threading.Thread(target=tcp_client.receive_commands, daemon=True)
tcp_thread.start()

# 初始化状态监控
monitor = Go2Monitor("eth0")

try:
    while True:
        battery_info = monitor.get_battery_info()
        tcp_client.send_state("state", 1001,
                             {
                                 "energy_remain": battery_info['soc'],
                                 "mainboard_tempera": battery_info['mainboard'],
                             })
        time.sleep(state_freq)
except KeyboardInterrupt:
    recorder.stop_listening()
    tcp_client.close()
    # print("\n程序已终止")

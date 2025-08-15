from funasr.utils.postprocess_utils import rich_transcription_postprocess
from unitree_sdk2py.core.channel import ChannelFactoryInitialize
from unitree_sdk2py.go2.sport.sport_client import SportClient
from pypinyin import lazy_pinyin
from funasr import AutoModel
import cn2an
import time
import re

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


class AudioProcessor:
    def __init__(self, model_dir="/home/unitree/unitree_sdk2_python/example/go2/Speech_control/SenseVoiceSmall", vad_model="/home/unitree/unitree_sdk2_python/example/go2/Speech_control/fsmn_vad"):
        # 初始化耗时资源
        print("正在初始化处理资源...")
        self.model = AutoModel(
            model=model_dir,
            vad_model=vad_model,  # 将长语音切割成短句
            vad_kwargs={"max_single_segment_time": 30000},
            device="cuda:0",
            disable_download=True,
            disable_update=True,
            disable_log=True,
            disable_pbar=True,
            log_level='ERROR'
        )
        print("资源初始化完成！")

    def process(self, file_path):
        # 使用已初始化的资源处理文件
        result = self.model.generate(
            input=file_path,
            cache={},
            language="zn",  # "zn", "en", "yue", "ja", "ko", "nospeech"
            use_itn=True,
            batch_size_s=60,
            merge_vad=True,
            merge_length_s=15,
            disable_log=True
        )
        result = rich_transcription_postprocess(result[0]["text"])
        return result


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
        step = int(cn2an.cn2an(match.group(2), "smart") / speed) if match.group(2) else int(default_distance / speed)
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
                print(f"步为{int(cn2an.cn2an(match.group(2), 'smart')) / speed}")
            step = int(cn2an.cn2an(match.group(2), "smart") / speed) if match.group(2) else int(
                default_distance / speed)
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

def tts(text):
    pass
# 创建全局单例实例
processor = AudioProcessor()

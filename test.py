from kokoro import KPipeline, KModel
import soundfile as sf
from playsound import playsound
import requests
import torch
import json
import re
import os

tts_model = 'hexgrad/Kokoro-82M-v1.1-zh'
tts_model_path = 'ckpts/kokoro-v1.1/kokoro-v1_1-zh.pth'
tts_config_path = 'ckpts/kokoro-v1.1/config.json'
tts_timbre_path = "ckpts/kokoro-v1.1/voices/zm_014.pt"


class LLMClient:
    def __init__(self, base_url="*****", timeout=10):  # timeout暂时存疑
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
    
    def stream_query(self, query, chat_url=":8000/test"):
        """处理流式响应，提取answer字段"""
        url = self.base_url + chat_url
        headers = {"Content-Type": "application/json"}
        payload = {
            "query": query,
            "ResponseMode": "streaming",
            "UserId": 1,
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
    
    def query(self, query, chat_url=":8000/test"):
        """发送查询并获取响应"""
        url = self.base_url + chat_url
        headers = {"Content-Type": "application/json"}
        payload = {
            "query": query,
            "ResponseMode": "streaming",
            "UserId": 1,
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
        

class Kokoro:
    def __init__(self, repo_id, model_path, config_path, timbre):
        print("正在初始化语音生成模型...")
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model = KModel(model=model_path, config=config_path, repo_id=repo_id).to(device).eval()
        self.zh_pipeline = KPipeline(lang_code='z', repo_id=repo_id, model=self.model)
        self.timbre_tensor = torch.load(timbre, weights_only=True)
        print("语音模型初始化完成！")
    
    def generate(self, text, output_path):
        # 生成语音
        self.generator = self.zh_pipeline(text, voice=self.timbre_tensor, speed=1.1)
        result = next(self.generator)
        wav = result.audio
        # 保存文件
        sf.write(output_path, wav, 24000)
    
    def play_audio(self, audio_path):
        abs_path = os.path.abspath(audio_path)
        # 使用异步播放避免阻塞问题
        playsound(abs_path, block=True)
        '''abs_path = os.path.abspath(audio_path)
        # 将路径转换为Windows原生格式
        if os.name == 'nt':
            from ctypes import windll, create_unicode_buffer
            buf = create_unicode_buffer(512)
            windll.kernel32.GetShortPathNameW(abs_path, buf, 512)
            abs_path = buf.value
        
        playsound(abs_path, block=True)'''
        
        
llm_client = LLMClient()
response = llm_client.query("你好")  # 直接获取完整响应

tts_generator = Kokoro(tts_model, tts_model_path, tts_config_path, tts_timbre_path)
tts_path = "output.wav"
# 生成语音
tts_generator.generate(response, tts_path)
tts_generator.play_audio(tts_path)
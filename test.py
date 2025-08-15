import torch
import time
from kokoro import KPipeline, KModel
import soundfile as sf
from playsound import playsound
import os


# 创建存储目录
os.makedirs('voice', exist_ok=True)

# 加载模型和管道（全局只加载一次）
repo_id = 'hexgrad/Kokoro-82M-v1.1-zh'
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model_path = 'ckpts/kokoro-v1.1/kokoro-v1_1-zh.pth'
config_path = 'ckpts/kokoro-v1.1/config.json'
model = KModel(model=model_path, config=config_path, repo_id=repo_id).to(device).eval()
zh_pipeline = KPipeline(lang_code='z', repo_id=repo_id, model=model)

# 获取所有音色文件
voice_dir = 'ckpts/kokoro-v1.1/voices'
voice_files = [
    os.path.join(voice_dir, f)
    for f in os.listdir(voice_dir)
    if f.endswith('.pt')
]

if not voice_files:
    print('未找到音色文件!')
    exit()

print(f'找到 {len(voice_files)} 个音色文件')

# 修改为指定文本
text = "这是一个测试"

# 批量生成所有音色的语音
start_time = time.time()
generated_files = []  # 用于记录生成的文件路径

for voice_file in voice_files:
    try:
        voice_name = os.path.splitext(os.path.basename(voice_file))[0]
        output_path = f'voice/{voice_name}.wav'
        
        # 加载音色
        voice_tensor = torch.load(voice_file, weights_only=True)
        
        # 生成语音
        generator = zh_pipeline(text, voice=voice_tensor, speed=1.1)
        result = next(generator)
        wav = result.audio
        
        # 保存文件
        sf.write(output_path, wav, 24000)
        print(f'已生成: {output_path}')
        generated_files.append('./'+ output_path)
    
    except Exception as e:
        print(f'处理 {voice_file} 时出错: {str(e)}')

# 生成阶段结束
print(f'生成完成，总耗时: {time.time() - start_time:.2f}秒')

# 播放所有生成的语音文件
print("开始播放所有生成的语音文件...")
for file_path in generated_files:
    print(f"播放: {file_path}")
    
    try:
        # 确保使用绝对路径
        abs_path = os.path.abspath(file_path)
        # 使用异步播放避免阻塞问题
        playsound(abs_path, block=True)
    except Exception as e:
        print(f"播放 {file_path} 时出错: {str(e)}")

print("全部播放完毕。")

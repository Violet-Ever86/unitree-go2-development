import torch
import time
from kokoro import KPipeline, KModel
import soundfile as sf
from playsound import playsound
import time


start = time.time()
print(start)
voice_zf = "zf_001"
voice_zf_tensor = torch.load(f'ckpts/kokoro-v1.1/voices/{voice_zf}.pt', weights_only=True)
voice_af = 'af_maple'
voice_af_tensor = torch.load(f'ckpts/kokoro-v1.1/voices/{voice_af}.pt', weights_only=True)
repo_id = 'hexgrad/Kokoro-82M-v1.1-zh'
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model_path = 'ckpts/kokoro-v1.1/kokoro-v1_1-zh.pth'
config_path = 'ckpts/kokoro-v1.1/config.json'
model = KModel(model=model_path, config=config_path, repo_id=repo_id).to(device).eval()
zh_pipeline = KPipeline(lang_code='z', repo_id=repo_id, model=model)
sentence = '如果您愿意帮助进一步完成这一使命，请考虑为此贡献许可的音频数据。'
start_time = time.time()
generator = zh_pipeline(sentence, voice=voice_zf_tensor, speed=1.1)
result = next(generator)
wav = result.audio
speech_len = len(wav) / 24000
print('yield speech len {}, rtf {}'.format(speech_len, (time.time() - start_time) / speech_len))
sf.write('output.wav', wav, 24000)

print(f'转换耗时{time.time() - start_time}')

playsound('output.wav')
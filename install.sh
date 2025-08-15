# pip3 install kokoro
# pip3 install ordered-set
# pip3 install pypinyin_dict
# pip3 install soundfile
# pip3 install jieba

export HF_ENDPOINT=https://hf-mirror.com # 引入镜像地址
huggingface-cli download --resume-download hexgrad/Kokoro-82M-v1.1-zh --local-dir ./ckpts/kokoro-v1.1
from funasr.utils.postprocess_utils import rich_transcription_postprocess
from transformers import pipeline
from funasr import AutoModel
import logging


classifier = pipeline("text-classification", model="bert-tiny-finetuned-sst2")


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


def judgementor(text):
    result = classifier(text)
    if result[0]["label"] == "QUESTION":  # 需自行标注数据训练
        print("疑问句")
    
    logging.getLogger("funasr.utils.cli_utils").disabled = True
# 创建全局单例实例
processor = AudioProcessor()

import sounddevice as sd
import numpy as np

Gain_Factor = 2


def list_devices():
    """列出所有音频设备"""
    print("\n可用音频设备：")
    devices = sd.query_devices()
    print(len(devices))
    for i, dev in enumerate(devices):
        print(f"{i}. {dev['name']} (输入通道: {dev['max_input_channels']})")


def audio_test(device_id=None):
    """音频输入测试程序"""

    def callback(indata, frames, time_info, status):
        # 计算音量（分贝和百分比）
        rms = np.sqrt(np.mean(indata.astype(np.float32) ** 2)) * Gain_Factor
        db = 20 * np.log10(rms + 1e-6)  # 防止log(0)
        percent = min(max(int(100 * rms), 0), 100)

        # 显示实时音量
        print(f"\r输入电平: {percent:3d}% | {db:5.1f} dB", end="", flush=True)

        print("\n开始音频输入测试...")
        print("请对着麦克风说话（按Ctrl+C停止）")

    try:
        with sd.InputStream(
                device=device_id,
                channels=1,
                samplerate=int(sd.query_devices(device_id, 'input')['default_samplerate']),
                # samplerate=1600,
                blocksize=1024,
                callback=callback
        ):
            while True:
                sd.sleep(1000)
    except KeyboardInterrupt:
        print("\n测试已停止")


if __name__ == "__main__":
    list_devices()
    device_id = int(input("\n请输入要测试的设备ID: "))
    audio_test(device_id)
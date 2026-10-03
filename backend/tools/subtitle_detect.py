
from backend.tools.paddle_model_config import PaddleModelConfig
from backend.tools.hardware_accelerator import HardwareAccelerator
import numpy as np

try:
    from paddleocr import TextDetection
except ImportError:
    TextDetection = None


class SubtitleDetect:
    """
    文本框检测类，用于检测视频帧中是否存在文本框
    """

    def __init__(self):
        hardware_accelerator = HardwareAccelerator.instance()
        model_config = PaddleModelConfig(hardware_accelerator)
        self.text_detector = self._build_detector(model_config, hardware_accelerator)

    @staticmethod
    def _paddle_det_kwargs(model_config):
        """把管线风格的参数名换成 TextDetection 认的那两个。"""
        kwargs = {'enable_mkldnn': False}
        for key, value in model_config.paddle_engine_kwargs().items():
            if key == 'text_detection_model_dir':
                kwargs['model_dir'] = value
            elif key == 'text_detection_model_name':
                kwargs['model_name'] = value
        return kwargs

    @staticmethod
    def _build_detector(model_config, hardware_accelerator):
        """
        构建检测器, 优先用 ONNX Runtime 引擎。

        ONNX Runtime 是非 CUDA 显卡(如 DirectML)用上 GPU 的唯一官方途径。它初始化失败
        就退回 paddle 引擎。
        """
        onnx_kwargs = model_config.onnx_engine_kwargs()
        if onnx_kwargs:
            source = model_config.onnx_model_source('text_detection')
            kind, value = source
            try:
                print(f"使用 ONNX Runtime 推理引擎: {hardware_accelerator.accelerator_name}")
                return TextDetection(
                    engine='onnxruntime',
                    engine_config=onnx_kwargs['engine_config'],
                    device='cpu',
                    **{f'model_{kind}': value},
                )
            except Exception as e:
                print(f"ONNX Runtime 推理初始化失败，回退到默认推理引擎: {e}")
        return TextDetection(**SubtitleDetect._paddle_det_kwargs(model_config))

    def detect_subtitle(self, img):
        """
        检测图像中的文本框
        :param img: 输入图像
        :return: (dt_boxes, elapse) dt_boxes为numpy数组，elapse为耗时
        """
        results = list(self.text_detector.predict(img))
        if not results:
            return np.array([]), 0
        res = results[0]
        dt_polys = res.get('dt_polys', np.array([]))
        return dt_polys, 0

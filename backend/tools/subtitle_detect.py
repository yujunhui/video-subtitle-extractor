
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
        # 使用 TextDetection 公开 API（PaddleOCR 3.x）
        default_kwargs = {'model_dir': model_config.DET_MODEL_PATH}
        if model_config.DET_MODEL_NAME:
            default_kwargs['model_name'] = model_config.DET_MODEL_NAME
        # 见 ocr.py:绕开 paddlepaddle 3.3 + oneDNN 执行 PIR 格式模型的缺陷
        default_kwargs['enable_mkldnn'] = False

        # 优先使用 ONNX Runtime 引擎（DirectML 等非 CUDA 后端唯一的 GPU 途径）
        engine_config = hardware_accelerator.onnx_engine_config()
        if engine_config:
            source = model_config.onnx_model_source('text_detection')
            if source is not None:
                kind, value = source
                onnx_kwargs = dict(
                    default_kwargs,
                    engine='onnxruntime',
                    engine_config=engine_config,
                    device='cpu',
                )
                # 用 ONNX 来源覆盖默认的目录/模型名
                onnx_kwargs.pop('model_dir', None)
                onnx_kwargs.pop('model_name', None)
                onnx_kwargs[f'model_{kind}'] = value
                try:
                    print(f"使用 ONNX Runtime 推理引擎: {hardware_accelerator.accelerator_name}")
                    self.text_detector = TextDetection(**onnx_kwargs)
                    return
                except Exception as e:
                    print(f"ONNX Runtime 推理初始化失败，回退到默认推理引擎: {e}")

        self.text_detector = TextDetection(**default_kwargs)

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

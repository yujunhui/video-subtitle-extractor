from backend.config import tr

try:
    import paddle
except ImportError:  # ONNX Runtime 推理路径不依赖 paddlepaddle
    paddle = None


# ONNX Runtime 官方 provider 名称 -> 展示名称
PROVIDER_DISPLAY_NAMES = {
    "DmlExecutionProvider": "DirectML",
    "CUDAExecutionProvider": "CUDA",
    "ROCMExecutionProvider": "ROCm",
    "MIGraphXExecutionProvider": "MIGraphX",
    "VitisAIExecutionProvider": "VitisAI",
    "OpenVINOExecutionProvider": "OpenVINO",
    "MetalExecutionProvider": "Metal",
    "CoreMLExecutionProvider": "CoreML",
}

# 本项目认可的 ONNX Runtime 加速后端
SUPPORTED_ONNX_PROVIDERS = list(PROVIDER_DISPLAY_NAMES.keys())


class HardwareAccelerator:

    # 类变量，用于存储单例实例
    _instance = None

    # DirectML 所使用的显卡序号(0 = 默认适配器)
    DIRECTML_DEVICE_ID = 0

    @classmethod
    def instance(cls):
        """获取单例实例"""
        if cls._instance is None:
            cls._instance = HardwareAccelerator()
            cls._instance.initialize()
        return cls._instance

    def __init__(self):
        self.__cuda = False
        self.__onnx_providers = []
        self.__enabled = True

    def initialize(self):
        self.check_paddle()
        self.check_onnx()

    def check_paddle(self):
        if paddle is None:
            return
        # 如果paddlepaddle编译了gpu的版本
        if paddle.is_compiled_with_cuda():
        # 查看是否有可用的gpu
            if len(paddle.static.cuda_places()) > 0:
                # 如果有GPU则使用GPU
                self.__cuda = True

    def check_onnx(self):
        if self.__cuda:
            return
        try:
            import onnxruntime as ort
            available_providers = ort.get_available_providers()
            for provider in available_providers:
                if provider in [
                    "CPUExecutionProvider"
                ]:
                    continue
                if provider not in SUPPORTED_ONNX_PROVIDERS:
                    print(tr['Main']['OnnxExectionProviderNotSupportedSkipped'].format(provider))
                    continue
                print(tr['Main']['OnnxExecutionProviderDetected'].format(provider))
                self.__onnx_providers.append(provider)
        except ModuleNotFoundError as e:
            print(tr['Main']['OnnxRuntimeNotInstall'])

    def has_accelerator(self):
        if not self.__enabled:
            return False
        return self.__cuda or len(self.__onnx_providers) > 0

    @property
    def accelerator_name(self):
        if not self.__enabled:
            return "CPU"
        if self.__cuda:
            return "GPU"
        elif len(self.__onnx_providers) > 0:
            return ", ".join(
                PROVIDER_DISPLAY_NAMES.get(p, p) for p in self.__onnx_providers
            )
        else:
            return "CPU"

    @property
    def onnx_providers(self):
        if not self.__enabled:
            return []
        return self.__onnx_providers

    def has_cuda(self):
        if not self.__enabled:
            return False
        return self.__cuda

    def has_directml(self):
        """是否可用 DirectML(Windows 上面向 AMD/Intel 等非 NVIDIA 显卡的 GPU 加速)"""
        if not self.__enabled:
            return False
        return "DmlExecutionProvider" in self.__onnx_providers

    @property
    def use_onnx_engine(self):
        """
        是否应改用 PaddleOCR 的 ONNX Runtime 推理引擎。

        PaddleOCR 3.5+ 支持 engine='onnxruntime' + engine_config，
        这是目前唯一能让 AMD/Intel 显卡用上 GPU 的官方途径。
        CUDA 可用时仍走 paddle 自身的 GPU 路径，无需切换。
        """
        if not self.__enabled or self.__cuda:
            return False
        return len(self.__onnx_providers) > 0

    def onnx_engine_config(self):
        """
        构造 PaddleOCR `engine_config`(engine='onnxruntime')。

        注意 DirectML 的限制:不支持 memory pattern,也不支持并行执行,
        因此必须 enable_mem_pattern=False 且 execution_mode='sequential',
        否则 ORT 会直接报错。
        """
        if not self.use_onnx_engine:
            return None
        providers = list(self.__onnx_providers)
        if "CPUExecutionProvider" not in providers:
            # 兜底:未覆盖的算子回落到 CPU
            providers.append("CPUExecutionProvider")
        provider_options = [
            {"device_id": self.DIRECTML_DEVICE_ID} if p == "DmlExecutionProvider" else {}
            for p in providers
        ]
        config = {
            "providers": providers,
            "provider_options": provider_options,
            # 仅在报错级别输出,避免 ORT 图优化的中文告警刷屏
            "log_severity_level": 3,
        }
        if "DmlExecutionProvider" in providers:
            config["enable_mem_pattern"] = False
            config["execution_mode"] = "sequential"
        return config

    def set_enabled(self, enable):
        self.__enabled = enable

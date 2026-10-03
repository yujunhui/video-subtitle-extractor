import os
from backend.config import BASE_DIR, config
from backend.tools.constant import (
    V5_EQUIVALENT_MODEL, resolve_det_model, resolve_rec_model, apply_v5_override,
)

# PaddleX 官方提供 ONNX 格式的模型清单(下载名为 `<model_name>_onnx`)。
# 直接复用 PaddleX 自己的清单, 免得硬编码一份以后跟着版本漂移。
try:
    from paddlex.inference.utils.official_models import (
        ONNX_SUPPORTED_MODELS as _ONNX_SUPPORTED_MODELS,
    )
except Exception:
    _ONNX_SUPPORTED_MODELS = frozenset()

# 本地预转换的 ONNX 文件名(PaddleX 约定)
ONNX_MODEL_FILENAME = 'inference.onnx'

# 排障开关: 设为 0 时按档位把 V6 型号降级成 V5 等价型号, 优先级高于界面选择。
USE_V6_MODELS = os.environ.get('VSE_USE_V6_MODELS', '1').lower() not in ('0', 'false', 'no')


class PaddleModelConfig:
    def __init__(self, hardware_accelerator):
        self.hardware_accelerator = hardware_accelerator
        self.REC_CHAR_TYPE = config.language.value

        self.MODEL_BASE = os.path.join(BASE_DIR, 'models')
        # 仓库自带的模型是 V5
        self.MODEL_VERSION = 'V5'
        # V5 识别模型要求的输入尺寸
        self.REC_IMAGE_SHAPE = '3,48,320'
        self.REC_MODEL_PATH = None
        self.DET_MODEL_PATH = None
        self.DET_MODEL_NAME = None
        self.REC_MODEL_NAME = None
        # 模型被替换的原因, 启动日志会打出来
        self.fallback_notes = []

        self._resolve_selected_models()

    def _local_model_dirs(self):
        """
        扫描 MODEL_BASE 下的模型目录, 建立 官方模型名 -> 本地目录 的映射。

        模型名从各目录的 inference.yml 里读 Global.model_name。按 sorted() 顺序先到
        先得, 所以万一两个目录声明了同一个模型名, V5 那个会静默胜出。
        """
        mapping = {}
        if not os.path.isdir(self.MODEL_BASE):
            return mapping
        for version in sorted(os.listdir(self.MODEL_BASE)):
            version_dir = os.path.join(self.MODEL_BASE, version)
            if not os.path.isdir(version_dir):
                continue
            for dirname in sorted(os.listdir(version_dir)):
                model_dir = os.path.join(version_dir, dirname)
                if not os.path.isdir(model_dir):
                    continue
                model_name = self._read_model_name_from_yaml(model_dir)
                if model_name and model_name not in mapping:
                    mapping[model_name] = model_dir
        return mapping

    def _downgrade_to_local(self, det, rec, local_dirs):
        """
        把没有本地目录的型号换成同档位的 V5 型号。

        paddle 路径只能加载仓库自带的东西, 而 V6 只有 ONNX 格式。
        """
        out = []
        for model in (det, rec):
            if model in local_dirs:
                out.append(model)
                continue
            equivalent = V5_EQUIVALENT_MODEL.get(model, model)
            if equivalent != model:
                self.fallback_notes.append(
                    f'{model} 无法用 ONNX 加载，改用仓库自带的 {equivalent}')
            out.append(equivalent)
        return out[0], out[1]

    def _onnx_reachable(self, det, rec, local_dirs):
        """
        两个模型是否都能真正交给 ONNX Runtime 引擎。

        比 HardwareAccelerator.use_onnx_engine 严格: 那个只回答"有没有 ONNX provider",
        而这里还要求每个模型要么本地有 inference.onnx, 要么官方发了 ONNX 版。只要有一个
        不满足, 整条流水线就得留在 paddle 上。

        必须和 onnx_engine_kwargs() 保持一致: 那个方法返回 None 的情形, 正是这里拒绝的
        情形。两者不一致是出过 bug 的, 当时模型既没有本地目录, paddle 兜底也没东西可加载,
        于是跑去联网下载。

        返回 True 表示 onnx_engine_kwargs() 会产出可用的配置。
        """
        accelerator = self.hardware_accelerator
        if not (accelerator and accelerator.use_onnx_engine):
            return False
        for model in (det, rec):
            if self._has_local_onnx(local_dirs.get(model)):
                continue
            if self._is_onnx_supported(model):
                continue
            return False
        return True

    def _resolve_selected_models(self):
        """
        把界面上的选择换算成这次实际要加载的模型。

        有三件事会顶掉用户的选择:
          1. 识别模型读不了所选字幕语言
          2. VSE_USE_V6_MODELS=0, 把 V6 降级成 V5 等价型号
          3. ONNX 引擎用不了, 只剩仓库自带的本地模型
        """
        det, det_note = resolve_det_model(config.detModel.value)
        rec, rec_note = resolve_rec_model(config.recModel.value, self.REC_CHAR_TYPE)
        if det_note:
            self.fallback_notes.append(det_note)
        if rec_note:
            self.fallback_notes.append(rec_note)

        if not USE_V6_MODELS:
            det, rec = apply_v5_override(det, rec)
            self.fallback_notes.append(
                'VSE_USE_V6_MODELS=0，V6 型号已按档位换成 V5 等价型号')

        local_dirs = self._local_model_dirs()
        if not self._onnx_reachable(det, rec, local_dirs):
            det, rec = self._downgrade_to_local(det, rec, local_dirs)

        self.DET_MODEL_NAME = det
        self.REC_MODEL_NAME = rec
        self.DET_MODEL_PATH = local_dirs.get(det)
        self.REC_MODEL_PATH = local_dirs.get(rec)

    @staticmethod
    def _read_model_name_from_yaml(model_dir):
        """从 inference.yml 中读取 Global.model_name"""
        yaml_path = os.path.join(model_dir, 'inference.yml')
        if not os.path.exists(yaml_path):
            return None
        try:
            with open(yaml_path, 'r', encoding='utf-8') as f:
                in_global = False
                for line in f:
                    stripped = line.strip()
                    if stripped == 'Global:':
                        in_global = True
                        continue
                    if in_global:
                        if stripped and not stripped.startswith('#') and ':' in stripped:
                            if stripped.startswith('model_name:'):
                                return stripped.split(':', 1)[1].strip().strip('"').strip("'")
                        # 遇到下一个顶级 section 则退出
                        if stripped and not stripped.startswith('model_name') and not stripped.startswith(' ') and stripped.endswith(':'):
                            break
        except Exception:
            pass
        return None

    # ------------------------------------------------------------------
    # ONNX Runtime 推理引擎支持 (PaddleOCR >= 3.5)
    # ------------------------------------------------------------------

    @staticmethod
    def _has_local_onnx(model_dir):
        """该模型目录下是否已有预转换好的 ONNX 模型(离线场景)。"""
        if not model_dir:
            return False
        return os.path.exists(os.path.join(model_dir, ONNX_MODEL_FILENAME))

    @staticmethod
    def _is_onnx_supported(model_name):
        """该模型是否有官方 ONNX 版可供 PaddleX 自动下载。"""
        return bool(model_name) and model_name in _ONNX_SUPPORTED_MODELS

    def onnx_model_source(self, which):
        """
        该模型在 ONNX Runtime 引擎下该从哪里加载。

        本地已有 inference.onnx 就优先用目录, 否则给模型名让 PaddleX 下载转换。

        参数 which 只接受 'text_detection' / 'text_recognition', 传别的会抛 ValueError。
        """
        if which == 'text_detection':
            model_name, model_dir = self.DET_MODEL_NAME, self.DET_MODEL_PATH
        elif which == 'text_recognition':
            model_name, model_dir = self.REC_MODEL_NAME, self.REC_MODEL_PATH
        else:
            raise ValueError(f'unknown model kind: {which}')

        if self._has_local_onnx(model_dir):
            return 'dir', model_dir
        if self._is_onnx_supported(model_name):
            return 'name', model_name
        return None

    def display_model_names(self):
        """
        实际会被加载的 检测/识别 模型标识, 用于启动日志与排障。

        只有两个模型都走通了 ONNX 才报 ONNX 名字。以前只看 onnx_engine_config(),
        只要某个模型没有 ONNX 版, 这里就打成 "None", 排障时很误导。
        """
        det_source = self.onnx_model_source('text_detection')
        rec_source = self.onnx_model_source('text_recognition')
        if det_source and rec_source:
            return det_source[1], rec_source[1]
        det = os.path.basename(self.DET_MODEL_PATH) if self.DET_MODEL_PATH \
            else (self.DET_MODEL_NAME or 'None')
        rec = os.path.basename(self.REC_MODEL_PATH) if self.REC_MODEL_PATH \
            else (self.REC_MODEL_NAME or 'None')
        return det, rec

    def paddle_engine_kwargs(self):
        """
        默认(paddle)推理引擎所需的模型参数。

        本地有目录就给目录, 不给模型名: 两个都传是多余的, 而且传了和目录不一致的名字
        PaddleX 会直接报错。
        """
        kwargs = {}
        for prefix, model_name, model_dir in (
            ('text_detection', self.DET_MODEL_NAME, self.DET_MODEL_PATH),
            ('text_recognition', self.REC_MODEL_NAME, self.REC_MODEL_PATH),
        ):
            if model_dir:
                kwargs[f'{prefix}_model_dir'] = model_dir
            elif model_name:
                kwargs[f'{prefix}_model_name'] = model_name
        return kwargs

    def onnx_engine_kwargs(self):
        """
        在 DirectML 等 ONNX Runtime 后端可用时, 返回 PaddleOCR 管线所需的
        engine / engine_config / 模型来源参数; 不可用时返回 None。

        返回 None 是"留在 paddle"的信号, 调用方要把它当回退而不是错误。
        """
        accelerator = self.hardware_accelerator
        engine_config = accelerator.onnx_engine_config() if accelerator else None
        if not engine_config:
            return None

        kwargs = {'engine': 'onnxruntime', 'engine_config': engine_config}
        for prefix in ('text_detection', 'text_recognition'):
            source = self.onnx_model_source(prefix)
            if source is None:
                # 有模型无法以 ONNX 方式加载,整体回退
                return None
            kind, value = source
            kwargs[f'{prefix}_model_{kind}'] = value
        return kwargs

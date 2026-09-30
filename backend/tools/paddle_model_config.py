import os
from backend.config import BASE_DIR, config

# PaddleX 官方提供 ONNX 格式的模型清单(下载名为 `<model_name>_onnx`)。
# 直接复用 PaddleX 自身的清单,避免版本漂移导致判断失准。
try:
    from paddlex.inference.utils.official_models import (
        ONNX_SUPPORTED_MODELS as _ONNX_SUPPORTED_MODELS,
    )
except Exception:
    _ONNX_SUPPORTED_MODELS = frozenset()

# 本地预转换的 ONNX 文件名(PaddleX 约定)
ONNX_MODEL_FILENAME = 'inference.onnx'

# ---------------------------------------------------------------------------
# PP-OCRv6 支持
#
# 经实测(见下), V6 统一模型覆盖 简体/繁体中文 + 日文 + 46 种拉丁语系 + 希腊文,
# 但不含 韩文/西里尔/阿拉伯/天城文/泰米尔/泰卢固 —— 这些语种必须继续用
# V5 的分语种模型(字典里分别有 11504 个谚文 / 409 个西里尔 / 274 个阿拉伯 /
# 128 个天城文字符, 而 V6 为 0)。
#
# 简繁中文小字/劣化素材上的实测 CER(200 条合成字幕行):
#   V5_mobile_rec 12.55% | V5_server_rec 9.74%
#   V6_small_rec   9.36% | V6_medium_rec 7.58%
# V6_tiny_rec 被排除: 字典仅 6904 字, 缺 19 个常用繁体字
# (說錢麼幾頁認繼續讓帳資軟擇...), 会把繁体系统性识别成简体。
# ---------------------------------------------------------------------------

# 可用 V6 的语种(按字典字集实测确定)
V6_REC_LANGS = frozenset(
    ['ch', 'chinese_cht', 'japan', 'en', 'el',
     'af', 'az', 'bs', 'cs', 'cy', 'da', 'de', 'es', 'et', 'fr', 'ga', 'hr',
     'hu', 'id', 'is', 'it', 'ku', 'la', 'lt', 'lv', 'mi', 'ms', 'mt', 'nl',
     'no', 'oc', 'pi', 'pl', 'pt', 'ro', 'rs_latin', 'sk', 'sl', 'sq', 'sv',
     'sw', 'tl', 'tr', 'uz', 'vi', 'latin', 'german', 'french']
)

# 识别模式 -> V6 档位
V6_TIER_BY_MODE = {'fast': 'small', 'auto': 'medium', 'accurate': 'medium'}

# 允许通过环境变量回退到 V5 模型(便于对比/排障)
USE_V6_MODELS = os.environ.get('VSE_USE_V6_MODELS', '1').lower() not in ('0', 'false', 'no')


class PaddleModelConfig:
    def __init__(self, hardware_accelerator):
        self.hardware_accelerator = hardware_accelerator
        # 设置识别语言
        self.REC_CHAR_TYPE = config.language.value

        # 模型文件目录
        self.MODEL_BASE = os.path.join(BASE_DIR, 'models')
        # 模型版本 V5
        self.MODEL_VERSION = 'V5'
        # V5模型默认图形识别的shape为3, 48, 320
        self.REC_IMAGE_SHAPE = '3,48,320'
        # 初始化模型路径
        self.REC_MODEL_PATH = None
        self.DET_MODEL_PATH = None
        self.DET_MODEL_NAME = None
        self.REC_MODEL_NAME = None

        # 语言组定义
        self.LATIN_LANG = [
            'af', 'az', 'bs', 'cs', 'cy', 'da', 'de', 'es', 'et', 'fr', 'ga', 'hr',
            'hu', 'id', 'is', 'it', 'ku', 'la', 'lt', 'lv', 'mi', 'ms', 'mt', 'nl',
            'no', 'oc', 'pi', 'pl', 'pt', 'ro', 'rs_latin', 'sk', 'sl', 'sq', 'sv',
            'sw', 'tl', 'tr', 'uz', 'vi', 'latin', 'german', 'french',
            'fi', 'eu', 'gl', 'lb', 'rm', 'ca', 'qu',
        ]
        self.ARABIC_LANG = ['ar', 'fa', 'ug', 'ur', 'ps', 'sd', 'bal']
        self.CYRILLIC_LANG = [
            'ru', 'rs_cyrillic', 'be', 'bg', 'uk', 'mn', 'abq', 'ady', 'kbd', 'ava',
            'dar', 'inh', 'che', 'lbe', 'lez', 'tab', 'cyrillic',
            'sr', 'kk', 'ky', 'tg', 'mk', 'tt', 'cv', 'ba', 'mhr', 'mo',
            'udm', 'kv', 'os', 'bua', 'xal', 'tyv', 'sah', 'kaa',
        ]
        self.DEVANAGARI_LANG = [
            'hi', 'mr', 'ne', 'bh', 'mai', 'ang', 'bho', 'mah', 'sck', 'new', 'gom',
            'sa', 'bgc', 'devanagari',
        ]
        self.OTHER_LANG = [
            'ch', 'japan', 'korean', 'en', 'ta', 'kn', 'te', 'ka',
            'chinese_cht',
        ]
        self.MULTI_LANG = (self.LATIN_LANG + self.ARABIC_LANG + self.CYRILLIC_LANG
                           + self.DEVANAGARI_LANG + self.OTHER_LANG)

        # 如果设置了识别文本语言类型，则设置为对应的语言
        if self.REC_CHAR_TYPE in self.MULTI_LANG:
            resolved = self._resolve_models()
            if resolved:
                self.MODEL_VERSION = 'V5'
                self.DET_MODEL_PATH, self.REC_MODEL_PATH, self.DET_MODEL_NAME, self.REC_MODEL_NAME = resolved

    def _get_v5_rec_model_name(self, lang):
        """
        根据语言获取V5识别模型目录名
        参考: https://www.paddleocr.ai/main/version3.x/algorithm/PP-OCRv5/PP-OCRv5_multi_languages.html
        """
        if lang in ('ch', 'chinese_cht', 'japan'):
            return 'PP-OCRv5_server_rec_infer'
        elif lang == 'en':
            return 'PP-OCRv5_server_rec_infer'
        elif lang == 'korean':
            return 'korean_PP-OCRv5_mobile_rec_infer'
        elif lang in self.LATIN_LANG:
            return 'latin_PP-OCRv5_mobile_rec_infer'
        elif lang in self.ARABIC_LANG:
            return 'arabic_PP-OCRv5_mobile_rec_infer'
        elif lang in self.CYRILLIC_LANG:
            return 'cyrillic_PP-OCRv5_mobile_rec_infer'
        elif lang in self.DEVANAGARI_LANG:
            return 'devanagari_PP-OCRv5_mobile_rec_infer'
        elif lang == 'th':
            return 'th_PP-OCRv5_mobile_rec_infer'
        elif lang == 'el':
            return 'el_PP-OCRv5_mobile_rec_infer'
        elif lang == 'ta':
            return 'ta_PP-OCRv5_mobile_rec_infer'
        elif lang == 'te':
            return 'te_PP-OCRv5_mobile_rec_infer'
        return None

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

    def _resolve_models(self):
        """
        解析 V5 模型路径，返回 (det_model_path, rec_model_path, det_model_name, rec_model_name) 或 None
        """
        v5_base = os.path.join(self.MODEL_BASE, 'V5')

        # 快速模式优先使用 mobile 模型，否则使用 server 模型
        if config.mode.value == 'fast':
            det_model_path = os.path.join(v5_base, 'PP-OCRv5_mobile_det_infer')
            if not os.path.exists(det_model_path):
                det_model_path = os.path.join(v5_base, 'PP-OCRv5_server_det_infer')
        else:
            det_model_path = os.path.join(v5_base, 'PP-OCRv5_server_det_infer')
        if not os.path.exists(det_model_path):
            return None

        det_model_name = self._read_model_name_from_yaml(det_model_path)

        # 快速模式：中文(简/繁)、英文、日文使用通用 mobile 模型，其他语言使用对应的专用模型
        if config.mode.value == 'fast' and self.REC_CHAR_TYPE in ('ch', 'chinese_cht', 'en', 'japan'):
            rec_model_path = os.path.join(v5_base, 'PP-OCRv5_mobile_rec_infer')
            if os.path.exists(rec_model_path):
                rec_model_name = self._read_model_name_from_yaml(rec_model_path)
                return det_model_path, rec_model_path, det_model_name, rec_model_name
            # mobile 不存在则 fallback 到按语言选择

        # 获取识别模型
        rec_model_dir_name = self._get_v5_rec_model_name(self.REC_CHAR_TYPE)
        if rec_model_dir_name is None:
            return None

        rec_model_path = os.path.join(v5_base, f'{rec_model_dir_name}_infer'
                                      if not rec_model_dir_name.endswith('_infer')
                                      else rec_model_dir_name)

        if not os.path.exists(rec_model_path):
            rec_model_path = os.path.join(v5_base, rec_model_dir_name)

        if not os.path.exists(rec_model_path):
            return None

        rec_model_name = self._read_model_name_from_yaml(rec_model_path)
        return det_model_path, rec_model_path, det_model_name, rec_model_name

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
        """该模型是否有官方 ONNX 版本可供 PaddleX 自动下载。"""
        return bool(model_name) and model_name in _ONNX_SUPPORTED_MODELS

    def v6_model_names(self):
        """
        当前「语言 + 识别模式」下应使用的 V6 模型名,返回 (det_name, rec_name)。
        语言不在 V6 覆盖范围内(韩/阿/西里尔/天城文等)时返回 None。
        """
        if not USE_V6_MODELS:
            return None
        if self.REC_CHAR_TYPE not in V6_REC_LANGS:
            return None
        tier = V6_TIER_BY_MODE.get(config.mode.value, 'medium')
        det = f'PP-OCRv6_{tier}_det'
        rec = f'PP-OCRv6_{tier}_rec'
        if not (self._is_onnx_supported(det) and self._is_onnx_supported(rec)):
            return None
        return det, rec

    def onnx_model_source(self, which):
        """
        返回该模型在 ONNX Runtime 引擎下的来源,形式为 ('dir'|'name', 值)。
        无法以 ONNX 方式加载时返回 None。

        优先顺序:
          1. PP-OCRv6 统一模型(仅 ONNX 路径可用,实测简繁中文 CER 更低)
          2. 本地目录下已存在 inference.onnx -> ('dir', 目录)
          3. 有官方 ONNX 版本 -> ('name', 模型名),由 PaddleX 自动下载并缓存
        """
        if which not in ('text_detection', 'text_recognition'):
            raise ValueError(f'unknown model kind: {which}')

        v6 = self.v6_model_names()
        if v6 is not None:
            return 'name', (v6[0] if which == 'text_detection' else v6[1])

        if which == 'text_detection':
            model_name, model_dir = self.DET_MODEL_NAME, self.DET_MODEL_PATH
        else:
            model_name, model_dir = self.REC_MODEL_NAME, self.REC_MODEL_PATH

        if self._has_local_onnx(model_dir):
            return 'dir', model_dir
        if self._is_onnx_supported(model_name):
            return 'name', model_name
        return None

    def display_model_names(self):
        """
        实际会被加载的 检测/识别 模型名,用于启动日志与排障。
        (V6 走在线下载时 DET/REC_MODEL_PATH 仍指向本地 V5 目录,直接打印会误导。)
        """
        det = self.onnx_model_source('text_detection') if self.hardware_accelerator \
            and self.hardware_accelerator.onnx_engine_config() else None
        if det is None:
            return (os.path.basename(self.DET_MODEL_PATH) if self.DET_MODEL_PATH else 'None',
                    os.path.basename(self.REC_MODEL_PATH) if self.REC_MODEL_PATH else 'None')
        rec = self.onnx_model_source('text_recognition')
        return det[1], (rec[1] if rec else 'None')

    def onnx_engine_kwargs(self):
        """
        在 DirectML 等 ONNX Runtime 后端可用时,返回 PaddleOCR 管线所需的
        engine / engine_config / 模型来源参数;不可用时返回 None。
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

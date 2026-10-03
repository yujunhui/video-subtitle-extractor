from enum import Enum


# 默认字幕出现的大致区域
class SubtitleArea(Enum):
    # 字幕区域出现在下半部分
    LOWER_PART = 0
    # 字幕区域出现在上半部分
    UPPER_PART = 1
    # 不知道字幕区域可能出现的位置
    UNKNOWN = 2
    # 明确知道字幕区域出现的位置
    # CUSTOM = 3


class BackgroundColor(Enum):
    # 字幕背景
    WHITE = 0
    DARK = 1
    UNKNOWN = 2


class VideoSubFinderDecoder(Enum):
    OPENCV = "OpenCV"
    FFMPEG = "FFmpeg"


# ---------------------------------------------------------------------------
# OCR 模型清单与语种覆盖
#
# 这里刻意不放 PP-OCRv6_tiny: 它的识别字典只有 6904 字, 缺了 19 个常用繁体字
# (說 錢 麼 幾 頁 認 繼 續 讓 帳 資 軟 擇 等), 会把繁体成段地认成简体。实测繁体
# 字符错误率 23.16%, 是所有候选里最差的。
# ---------------------------------------------------------------------------

DET_MODEL_CHOICES = (
    "PP-OCRv6_medium_det",
    "PP-OCRv6_small_det",
    "PP-OCRv5_server_det",
    "PP-OCRv5_mobile_det",
)

REC_MODEL_CHOICES = (
    "PP-OCRv6_medium_rec",
    "PP-OCRv6_small_rec",
    "PP-OCRv5_server_rec",
    "PP-OCRv5_mobile_rec",
)

DEFAULT_DET_MODEL = "PP-OCRv6_small_det"
DEFAULT_REC_MODEL = "PP-OCRv6_small_rec"

# 抽帧策略。以前它是识别模式顺带管的事, 现在独立成一个选项, 由用户自己选。
#
# 默认给 VideoSubFinder: 检测抽帧要把整片每一帧都过一遍检测器, 实测比 VSF 慢 3.6~6.3 倍
# (2.4 分钟 1080p30 的视频: VSF 约 36s, 检测抽帧 130s 起, 检测模型越重越慢), 长视频上
# 差距会拉到几百倍。旧版本的默认模式同样走 VSF, 所以默认值保持不变。
#
# vsf 放第一项, 这样旧配置里已经不存在的 auto 被校验器纠正时会落到它身上。
FRAME_EXTRACTION_DETECT = "detect"
FRAME_EXTRACTION_VSF = "vsf"
FRAME_EXTRACTION_CHOICES = (
    FRAME_EXTRACTION_VSF,
    FRAME_EXTRACTION_DETECT,
)
DEFAULT_FRAME_EXTRACTION = FRAME_EXTRACTION_VSF

# 语言分组, 用来把字幕语言映射到对应的 V5 分语种识别模型。
LATIN_LANG = [
    'af', 'az', 'bs', 'cs', 'cy', 'da', 'de', 'es', 'et', 'fr', 'ga', 'hr',
    'hu', 'id', 'is', 'it', 'ku', 'la', 'lt', 'lv', 'mi', 'ms', 'mt', 'nl',
    'no', 'oc', 'pi', 'pl', 'pt', 'ro', 'rs_latin', 'sk', 'sl', 'sq', 'sv',
    'sw', 'tl', 'tr', 'uz', 'vi', 'latin', 'german', 'french',
    'fi', 'eu', 'gl', 'lb', 'rm', 'ca', 'qu',
]
ARABIC_LANG = ['ar', 'fa', 'ug', 'ur', 'ps', 'sd', 'bal']
CYRILLIC_LANG = [
    'ru', 'rs_cyrillic', 'be', 'bg', 'uk', 'mn', 'abq', 'ady', 'kbd', 'ava',
    'dar', 'inh', 'che', 'lbe', 'lez', 'tab', 'cyrillic',
    'sr', 'kk', 'ky', 'tg', 'mk', 'tt', 'cv', 'ba', 'mhr', 'mo',
    'udm', 'kv', 'os', 'bua', 'xal', 'tyv', 'sah', 'kaa',
]
DEVANAGARI_LANG = [
    'hi', 'mr', 'ne', 'bh', 'mai', 'ang', 'bho', 'mah', 'sck', 'new', 'gom',
    'sa', 'bgc', 'devanagari',
]
OTHER_LANG = ['ch', 'japan', 'korean', 'en', 'ta', 'kn', 'te', 'ka', 'chinese_cht']
MULTI_LANG = LATIN_LANG + ARABIC_LANG + CYRILLIC_LANG + DEVANAGARI_LANG + OTHER_LANG

# 各识别模型能读哪些语种, 依据仓库自带字典实际覆盖的文字系统统计:
#   V6 small/medium: 汉字 15906, 平假名 87 + 片假名 152, 拉丁 613, 希腊 64
#   V5 mobile/server: 汉字 15906, 平假名 87 + 片假名 152, 拉丁 294, 希腊 64
#   韩文 / 西里尔 / 阿拉伯 / 天城文 / 泰米尔 / 泰卢固 在两者的字典里都是 0
V6_REC_LANGS = frozenset(['ch', 'chinese_cht', 'japan', 'en', 'el'] + LATIN_LANG)
V5_GENERIC_REC_LANGS = frozenset(['ch', 'chinese_cht', 'japan', 'en'])

# 同档位的 V5 等价型号, VSE_USE_V6_MODELS=0 时用。
V5_EQUIVALENT_MODEL = {
    "PP-OCRv6_small_det": "PP-OCRv5_mobile_det",
    "PP-OCRv6_medium_det": "PP-OCRv5_server_det",
    "PP-OCRv6_small_rec": "PP-OCRv5_mobile_rec",
    "PP-OCRv6_medium_rec": "PP-OCRv5_server_rec",
}


# 旧 Mode 取值 -> (检测模型, 识别模型, 抽帧策略)。
# 用来把移除模式选项之前写下的 config.json 迁移过来, 保证升级后实际跑的东西不变。
#   fast     -> 轻量模型 + VideoSubFinder 抽帧
#   auto     -> 大模型   + VideoSubFinder 抽帧
#   accurate -> 大模型   + 检测抽帧
MODE_TO_MODEL_CHOICES = {
    "fast": ("PP-OCRv6_small_det", "PP-OCRv6_small_rec", FRAME_EXTRACTION_VSF),
    "auto": ("PP-OCRv6_medium_det", "PP-OCRv6_medium_rec", FRAME_EXTRACTION_VSF),
    "accurate": ("PP-OCRv6_medium_det", "PP-OCRv6_medium_rec", FRAME_EXTRACTION_DETECT),
}


def v5_lang_rec_model(lang):
    """
    该语种对应的 V5 分语种识别模型。

    通用 V5 字典已经覆盖的语种不需要专用模型, 所以 'ch' / 'en' / 'japan' 返回 None。
    会命中的例子: 'korean' / 'latin' / 'ru'。
    """
    if lang == 'korean':
        return 'korean_PP-OCRv5_mobile_rec'
    if lang in LATIN_LANG:
        return 'latin_PP-OCRv5_mobile_rec'
    if lang in ARABIC_LANG:
        return 'arabic_PP-OCRv5_mobile_rec'
    if lang in CYRILLIC_LANG:
        return 'cyrillic_PP-OCRv5_mobile_rec'
    if lang in DEVANAGARI_LANG:
        return 'devanagari_PP-OCRv5_mobile_rec'
    if lang == 'th':
        return 'th_PP-OCRv5_mobile_rec'
    if lang == 'el':
        return 'el_PP-OCRv5_mobile_rec'
    if lang == 'ta':
        return 'ta_PP-OCRv5_mobile_rec'
    if lang == 'te':
        return 'te_PP-OCRv5_mobile_rec'
    return None


def resolve_rec_model(selected, lang):
    """
    校验所选识别模型能否读当前字幕语言, 读不了就换掉。

    字典里没有对应文字的模型会被换成 V5 分语种模型: V6 读不了韩文 / 西里尔 /
    阿拉伯 / 天城文, 通用 V5 模型同样读不了。另外, 不在 REC_MODEL_CHOICES 里的
    名字也会回退到默认模型。

    返回 (实际使用的模型名, 回退原因或 None)。
    """
    if selected not in REC_MODEL_CHOICES:
        return DEFAULT_REC_MODEL, \
            f"识别模型 {selected} 不在候选清单里，改用 {DEFAULT_REC_MODEL}"

    if selected.startswith("PP-OCRv6_"):
        covered = lang in V6_REC_LANGS
    else:
        covered = lang in V5_GENERIC_REC_LANGS
    if covered:
        return selected, None

    fallback = v5_lang_rec_model(lang)
    if fallback is None and lang in V5_GENERIC_REC_LANGS:
        fallback = "PP-OCRv5_server_rec"
    if fallback is None:
        return selected, f"{selected} 读不了 {lang} 语言，且没有可用的替代模型"
    return fallback, f"{selected} 读不了 {lang} 语言，改用 {fallback}"


def resolve_det_model(selected):
    """
    校验检测模型名。

    检测与文字系统无关, 所以 DET_MODEL_CHOICES 里的名字都接受; 只有旧配置文件里
    留下的过时名字需要挡掉。
    """
    if selected not in DET_MODEL_CHOICES:
        return DEFAULT_DET_MODEL, \
            f"检测模型 {selected} 不在候选清单里，改用 {DEFAULT_DET_MODEL}"
    return selected, None


def apply_v5_override(det_model, rec_model):
    """
    把 V6 型号映射到同档位的 V5 型号。

    只给 VSE_USE_V6_MODELS=0 这个排障开关用。没有 V5 对应型号的原样返回。
    """
    return (V5_EQUIVALENT_MODEL.get(det_model, det_model),
            V5_EQUIVALENT_MODEL.get(rec_model, rec_model))


def should_use_detection_extraction(strategy):
    """
    是否用检测抽帧。

    检测抽帧要逐帧扫完整个视频, 只适合短片。长视频上 VideoSubFinder 快几百倍, 所以这项
    默认关着, 由用户明确选择。
    """
    return strategy == FRAME_EXTRACTION_DETECT


BGR_COLOR_GREEN = (0, 0xff, 0)
BGR_COLOR_BLUE = (0xff, 0, 0)
BGR_COLOR_RED = (0, 0, 0xff)
BGR_COLOR_WHITE = (0xff, 0xff, 0xff)

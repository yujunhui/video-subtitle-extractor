"""预下载 backend 运行所需的 PP-OCRv5 ONNX 模型。

复刻 PaddleX 的缓存布局(<CACHE_DIR>/official_models/<model_name>_onnx/)，
下载完成后应用即可离线使用，无需再联网。

支持三种下载源:
  huggingface  官方的 PaddlePaddle/<name>_onnx 仓库(可配合代理)
  modelscope    国内可直连
  aistudio      国内可直连

代理通过环境变量生效(requests / huggingface_hub 均原生支持):
  HTTP_PROXY / HTTPS_PROXY / ALL_PROXY

用法示例
--------
# 走代理 + HuggingFace
set HTTP_PROXY=http://127.0.0.1:7890
set HTTPS_PROXY=http://127.0.0.1:7890
python scripts/download_onnx_models.py --source huggingface

# 走 hf-mirror(不需要代理)
python scripts/download_onnx_models.py --source huggingface --endpoint https://hf-mirror.com

# 国内直连 ModelScope
python scripts/download_onnx_models.py --source modelscope

# 只下 fast 模式用得到的
python scripts/download_onnx_models.py --source modelscope --set fast

# 先看看要下什么，不实际下载
python scripts/download_onnx_models.py --dry-run
"""
import argparse
import os
import sys
import time

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# PaddleX 的 hoster 别名(必须完全一致, 见 paddlex/utils/flags.py + official_models.py)
SOURCE_ALIASES = ("huggingface", "modelscope", "aistudio", "bos")

DEFAULT_CACHE_DIR = os.path.join(os.path.expanduser("~"), ".paddlex")
DEFAULT_ENDPOINT = "https://huggingface.co"


def _read_model_name_from_yml(yml_path):
    """读取 inference.yml 里的 Global.model_name（与 backend 的实现一致）。"""
    try:
        with open(yml_path, "r", encoding="utf-8") as f:
            in_global = False
            for line in f:
                stripped = line.strip()
                if stripped == "Global:":
                    in_global = True
                    continue
                if in_global:
                    if stripped.startswith("model_name:"):
                        return stripped.split(":", 1)[1].strip().strip('"').strip("'")
                    if stripped and not stripped.startswith(" ") and stripped.endswith(":"):
                        break
    except OSError:
        pass
    return None


def _local_model_names():
    """
    扫描 backend/models 下所有模型目录的 inference.yml，取出 model_name。

    这条路径只依赖标准库，不需要 qfluentwidgets/PySide6，
    因此在任何装了 paddlex 的环境里都能用。
    """
    import glob

    models_root = os.path.join(BASE_DIR, "backend", "models")
    names = set()
    for yml in glob.glob(os.path.join(models_root, "*", "*", "inference.yml")):
        name = _read_model_name_from_yml(yml)
        if name:
            names.add(name)
    return names


def _onnx_supported_models():
    """PaddleX 认可的可下载 ONNX 模型清单。"""
    try:
        from paddlex.inference.utils.official_models import ONNX_SUPPORTED_MODELS

        return set(ONNX_SUPPORTED_MODELS)
    except Exception:
        return set()


def _resolve_models_from_local(mode_filter):
    """
    退化方案：以「本地随仓库分发的模型」为基准，取其中有官方 ONNX 版的。

    是精确清单的超集（会多带几个项目当前语言列表选不到的模型），
    但完全不需要 GUI 依赖。
    """
    supported = _onnx_supported_models()
    if not supported:
        raise SystemExit(
            "无法导入 paddlex 以获取 ONNX 模型清单。\n"
            "请在项目的运行环境里执行，例如:\n"
            "    conda activate vse-lion\n"
            "    python scripts/download_onnx_models.py --source modelscope"
        )

    local = _local_model_names()
    if not local:
        raise SystemExit(
            f"未能从 {os.path.join(BASE_DIR, 'backend', 'models')} 解析出任何模型名。"
        )

    names = local & supported
    # 近似区分快慢模式：fast 不含 server，accurate 不需要 mobile det / 通用 mobile rec
    if mode_filter == "fast":
        names = {n for n in names if "server" not in n}
    elif mode_filter == "accurate":
        names = {n for n in names if not (n == "PP-OCRv5_mobile_rec" or "mobile_det" in n)}
    return {n for n in names if "det" in n}, {n for n in names if "det" not in n}


def resolve_models(mode_filter, langs=None):
    """
    推导出「语言 x 模式」组合下会用到的模型名，返回 (det_names, rec_names)。

    优先用项目自身的 PaddleModelConfig（唯一真源）；若当前环境缺少 GUI 依赖
    （qfluentwidgets/PySide6），则退化为扫描本地模型目录。
    """
    sys.path.insert(0, BASE_DIR)
    os.environ.setdefault("PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK", "True")

    try:
        from backend.config import config
        from backend.tools.hardware_accelerator import HardwareAccelerator
        from backend.tools.paddle_model_config import PaddleModelConfig
    except ImportError as e:
        print(f"注意: 无法导入项目模块({type(e).__name__}: {e})")
        print("      改用本地模型目录清单(超集)。如需精确清单，请用应用所在环境运行：")
        print("      conda activate vse-lion\n")
        return _resolve_models_from_local(mode_filter)

    accelerator = HardwareAccelerator()      # 仅用于构造，不做后端探测
    accelerator.set_enabled(False)

    available_langs = list(config.language.validator.options)
    use_langs = langs or available_langs
    invalid = [l for l in use_langs if l not in available_langs]
    if invalid:
        raise SystemExit(f"未知语言: {invalid}\n可选: {available_langs}")

    modes = [mode_filter] if mode_filter in ("fast", "accurate") else ["fast", "accurate"]

    det, rec = set(), set()
    original = (config.language.value, config.mode.value)
    config_file = os.path.join(BASE_DIR, "config", "config.json")
    config_existed = os.path.exists(config_file)
    try:
        for lang in use_langs:
            for mode in modes:
                config.set(config.language, lang)
                config.set(config.mode, mode)
                mc = PaddleModelConfig(accelerator)
                if mc.DET_MODEL_NAME:
                    det.add(mc.DET_MODEL_NAME)
                if mc.REC_MODEL_NAME:
                    rec.add(mc.REC_MODEL_NAME)
    finally:
        config.set(config.language, original[0])
        config.set(config.mode, original[1])
        if not config_existed and os.path.exists(config_file):
            os.remove(config_file)

    return det, rec


def check_sources(timeout=10):
    """
    连通性自检：验证当前(可能带代理的)网络能否访问各下载源，
    并实际探测一个 ONNX 模型仓库是否存在、可达。
    """
    import requests

    endpoint = os.environ.get("PADDLE_PDX_HUGGING_FACE_ENDPOINT", DEFAULT_ENDPOINT)
    # (名称, 站点根, 用于验证模型仓库的 URL 模板或 None 表示该源不做模型包探测)
    targets = [
        ("huggingface (官方)", endpoint,
         f"{endpoint.rstrip('/')}/api/models/PaddlePaddle/{{name}}_onnx"),
        ("hf-mirror (镜像)", "https://hf-mirror.com",
         "https://hf-mirror.com/api/models/PaddlePaddle/{name}_onnx"),
        ("modelscope", "https://modelscope.cn",
         "https://modelscope.cn/api/v1/models/PaddlePaddle/{name}_onnx"),
        # AIStudio 的仓库 API 路径由 aistudio_sdk 动态拼接，这里不做模型包探测，
        # 以免给出误导性的 404。实际可用性以 --source aistudio 的下载结果为准。
        ("aistudio", "https://aistudio.baidu.com", None),
    ]
    probe_model = "PP-OCRv5_mobile_det"

    print("网络连通性自检")
    print(f"  HTTP_PROXY  = {os.environ.get('HTTP_PROXY') or os.environ.get('http_proxy') or '(未设置)'}")
    print(f"  HTTPS_PROXY = {os.environ.get('HTTPS_PROXY') or os.environ.get('https_proxy') or '(未设置)'}")
    proxy = os.environ.get("HTTPS_PROXY") or os.environ.get("https_proxy") \
        or os.environ.get("HTTP_PROXY") or os.environ.get("http_proxy")
    print(f"  requests 实际使用: {proxy or '(直连)'}")
    print()

    reachable = []
    for label, root, model_url_tpl in targets:
        sw = time.time()
        try:
            r = requests.head(root, timeout=timeout, allow_redirects=True)
            site = f"站点 {r.status_code}"
        except Exception as e:
            site = f"站点 不可达 ({type(e).__name__})"
        if model_url_tpl is None:
            model = "模型包 未探测"
        else:
            try:
                r = requests.get(model_url_tpl.format(name=probe_model), timeout=timeout)
                if r.status_code == 200:
                    model = "模型仓库 200 OK"
                    reachable.append((label, model_url_tpl.format(name=probe_model)))
                else:
                    model = f"模型仓库 HTTP {r.status_code}"
            except Exception as e:
                model = f"模型仓库 不可达 ({type(e).__name__})"
        ms = (time.time() - sw) * 1000
        print(f"  {label:22} {site:28} {model:30} ({ms:.0f}ms)")

    print()
    if reachable:
        print("已验证可用的下载源:")
        for label, url in reachable:
            print(f"  * {label}   (例如 {url})")
        print("\n建议:")
        best = reachable[0][0]
        if "huggingface" in best:
            print("  python scripts/download_onnx_models.py --source huggingface")
        elif "mirror" in best:
            print("  python scripts/download_onnx_models.py --source huggingface --endpoint https://hf-mirror.com")
        else:
            print("  python scripts/download_onnx_models.py --source modelscope")
    else:
        print("没有任何源可达。请检查代理是否生效（可试 curl -x <proxy> https://huggingface.co）。")
        sys.exit(1)


def download_one(source, model_name, cache_root, endpoint, dry_run=False):
    """下载单个模型的 ONNX 包到 PaddleX 缓存目录。

    返回 (状态描述, 体积MB, 是否为本次新下载)
    """
    repo_id = f"PaddlePaddle/{model_name}_onnx"
    local_dir = os.path.join(cache_root, "official_models", f"{model_name}_onnx")

    onnx_file = os.path.join(local_dir, "inference.onnx")
    if os.path.exists(onnx_file):
        size = sum(
            os.path.getsize(os.path.join(r, f))
            for r, _, fs in os.walk(local_dir)
            for f in fs
        )
        return "已存在, 跳过", size / 1e6, False

    if dry_run:
        return "待下载", 0.0, False

    if source == "huggingface":
        import huggingface_hub as hf_hub

        hf_hub.snapshot_download(repo_id=repo_id, local_dir=local_dir, endpoint=endpoint)
    elif source == "modelscope":
        import modelscope

        modelscope.snapshot_download(repo_id=repo_id, local_dir=local_dir)
    elif source == "aistudio":
        from aistudio_sdk.snapshot_download import snapshot_download

        snapshot_download(repo_id=repo_id, local_dir=local_dir)
    else:
        raise SystemExit(
            f"source={source!r} 不支持 ONNX 包下载。\n"
            f"bos 只提供 paddle 格式的 *_infer.tar，没有 *_onnx 包。"
        )

    size = sum(
        os.path.getsize(os.path.join(r, f))
        for r, _, fs in os.walk(local_dir)
        for f in fs
    )
    return "完成", size / 1e6, True


def main():
    parser = argparse.ArgumentParser(
        description="预下载 PP-OCRv5 ONNX 模型到 PaddleX 缓存目录",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--source",
        default=os.environ.get("PADDLE_PDX_MODEL_SOURCE", "huggingface").lower(),
        choices=SOURCE_ALIASES,
        help="下载源(默认取 PADDLE_PDX_MODEL_SOURCE，否则 huggingface)",
    )
    parser.add_argument(
        "--endpoint",
        default=os.environ.get("PADDLE_PDX_HUGGING_FACE_ENDPOINT", DEFAULT_ENDPOINT),
        help=f"huggingface 端点，可设为 https://hf-mirror.com(默认 {DEFAULT_ENDPOINT})",
    )
    parser.add_argument(
        "--cache-dir",
        default=os.environ.get("PADDLE_PDX_CACHE_HOME", DEFAULT_CACHE_DIR),
        help=f"PaddleX 缓存根目录(默认 {DEFAULT_CACHE_DIR})",
    )
    parser.add_argument(
        "--set",
        dest="mode_set",
        default="all",
        choices=["all", "fast", "accurate"],
        help="只下载某个识别模式会用到的模型(默认 all)",
    )
    parser.add_argument("--langs", default=None, help="逗号分隔的语言子集，默认全部")
    parser.add_argument(
        "--models",
        default=None,
        help="直接指定要下载的模型名(逗号分隔)，跳过模型清单解析。"
             "例: --models PP-OCRv6_medium_det,PP-OCRv6_medium_rec",
    )
    parser.add_argument("--dry-run", action="store_true", help="只列出计划，不下载")
    parser.add_argument(
        "--check",
        action="store_true",
        help="只做网络连通性自检(验证代理/镜像能否打通各下载源)，不解析也不下载模型",
    )
    args = parser.parse_args()

    if args.check:
        check_sources()
        return

    langs = [s.strip() for s in args.langs.split(",")] if args.langs else None
    if args.models:
        models = [s.strip() for s in args.models.split(",") if s.strip()]
        det, rec = set(), set()
    else:
        det, rec = resolve_models(args.mode_set, langs)
        models = sorted(det | rec)

    print(f"下载源     : {args.source}" + (f"  (endpoint={args.endpoint})" if args.source == "huggingface" else ""))
    print(f"缓存目录   : {args.cache_dir}")
    print(f"模式       : {args.mode_set}    语言数: {len(langs) if langs else '全部'}")
    print(f"需要模型   : {len(models)} 个\n")

    for var, label in (("HTTP_PROXY", "HTTP_PROXY"), ("HTTPS_PROXY", "HTTPS_PROXY"), ("ALL_PROXY", "ALL_PROXY")):
        val = os.environ.get(var) or os.environ.get(var.lower())
        print(f"  {label:12} = {val or '(未设置)'}")
    print()

    total = 0.0
    skipped = 0
    pending = []
    failures = []
    started = time.time()
    for i, name in enumerate(models, 1):
        try:
            status, size, is_new = download_one(
                args.source, name, args.cache_dir, args.endpoint, args.dry_run
            )
            if is_new:
                total += size
            elif status.startswith("已存在"):
                skipped += 1
            else:
                pending.append(name)
            print(f"  [{i}/{len(models)}] {name:34} {status}" + (f"  ({size:.1f} MB)" if size else ""))
        except Exception as e:
            failures.append((name, e))
            print(f"  [{i}/{len(models)}] {name:34} 失败: {type(e).__name__}: {str(e)[:160]}")

    print(f"\n耗时 {time.time()-started:.1f}s | 已存在 {skipped} 个 | 本次新增 {total:.1f} MB")
    if pending:
        print(f"待下载 {len(pending)} 个(本次为 --dry-run，未实际下载)")

    if failures:
        print(f"\n{len(failures)} 个模型下载失败:")
        for name, e in failures:
            print(f"  - {name}: {str(e)[:200]}")
        print(
            "\n排查建议:\n"
            "  * 走代理: set HTTP_PROXY=http://127.0.0.1:7890 & set HTTPS_PROXY=http://127.0.0.1:7890\n"
            "  * 换镜像: --endpoint https://hf-mirror.com\n"
            "  * 换源  : --source modelscope   (国内可直连)\n"
            "  * Xet 传输异常时可试: set HF_HUB_DISABLE_XET=1\n"
            "  * 先自检连通性: python scripts/download_onnx_models.py --dry-run"
        )
        sys.exit(1)

    if args.dry_run:
        print("\n(dry-run 结束，去掉 --dry-run 即开始实际下载)")
    else:
        print("全部就绪，应用现在可以离线运行。")


if __name__ == "__main__":
    main()

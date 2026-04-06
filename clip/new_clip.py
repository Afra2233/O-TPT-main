import hashlib
import os
import urllib
import warnings
from typing import Any, Union, List
from pkg_resources import packaging

import torch
from PIL import Image
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize
from tqdm import tqdm

from .model import build_model
from .simple_tokenizer import SimpleTokenizer as _Tokenizer

try:
    from torchvision.transforms import InterpolationMode
    BICUBIC = InterpolationMode.BICUBIC
except ImportError:
    BICUBIC = Image.BICUBIC


if packaging.version.parse(torch.__version__) < packaging.version.parse("1.7.1"):
    warnings.warn("PyTorch version 1.7.1 or higher is recommended")


__all__ = ["available_models", "load", "tokenize", "load_custom_checkpoint"]
_tokenizer = _Tokenizer()

_MODELS = {
    "RN50": "https://openaipublic.azureedge.net/clip/models/afeb0e10f9e5a86da6080e35cf09123aca3b358a0c3e3b6c78a7b63bc04b6762/RN50.pt",
    "RN101": "https://openaipublic.azureedge.net/clip/models/8fa8567bab74a42d41c5915025a8e4538c3bdbe8804a470a72f30b0d94fab599/RN101.pt",
    "RN50x4": "https://openaipublic.azureedge.net/clip/models/7e526bd135e493cef0776de27d5f42653e6b4c8bf9e0f653bb11773263205fdd/RN50x4.pt",
    "RN50x16": "https://openaipublic.azureedge.net/clip/models/52378b407f34354e150460fe41077663dd5b39c54cd0bfd2b27167a4a06ec9aa/RN50x16.pt",
    "RN50x64": "https://openaipublic.azureedge.net/clip/models/be1cfb55d75a9666199fb2206c106743da0f6468c9d327f3e0d0a543a9919d9c/RN50x64.pt",
    "ViT-B/32": "https://openaipublic.azureedge.net/clip/models/40d365715913c9da98579312b702a82c18be219cc2a73407c4526f58eba950af/ViT-B-32.pt",
    "ViT-B/16": "https://openaipublic.azureedge.net/clip/models/5806e77cd80f8b59890b7e101eabd078d9fb84e6937f9e85e4ecb61988df416f/ViT-B-16.pt",
    "ViT-L/14": "https://openaipublic.azureedge.net/clip/models/b8cca3fd41ae0c99ba7e8951adf17d267cdb84cd88be6f7c2e0eca1737a03836/ViT-L-14.pt",
}


def _download(url: str, root: str):
    os.makedirs(root, exist_ok=True)
    filename = os.path.basename(url)

    expected_sha256 = url.split("/")[-2]
    download_target = os.path.join(root, filename)

    if os.path.exists(download_target) and not os.path.isfile(download_target):
        raise RuntimeError(f"{download_target} exists and is not a regular file")

    if os.path.isfile(download_target):
        if hashlib.sha256(open(download_target, "rb").read()).hexdigest() == expected_sha256:
            return download_target
        else:
            warnings.warn(
                f"{download_target} exists, but the SHA256 checksum does not match; re-downloading the file"
            )

    with urllib.request.urlopen(url) as source, open(download_target, "wb") as output:
        with tqdm(
            total=int(source.info().get("Content-Length")),
            ncols=80,
            unit='iB',
            unit_scale=True,
            unit_divisor=1024
        ) as loop:
            while True:
                buffer = source.read(8192)
                if not buffer:
                    break

                output.write(buffer)
                loop.update(len(buffer))

    if hashlib.sha256(open(download_target, "rb").read()).hexdigest() != expected_sha256:
        raise RuntimeError("Model has been downloaded but the SHA256 checksum does not not match")

    return download_target


def _convert_image_to_rgb(image):
    return image.convert("RGB")


def _transform(n_px):
    return Compose([
        Resize(n_px, interpolation=BICUBIC),
        CenterCrop(n_px),
        _convert_image_to_rgb,
        ToTensor(),
        Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
    ])


def available_models() -> List[str]:
    return list(_MODELS.keys())


# def _unwrap_checkpoint(ckpt):
#     if isinstance(ckpt, dict):
#         for k in ["state_dict", "model", "model_state_dict", "clip_state_dict"]:
#             if k in ckpt and isinstance(ckpt[k], dict):
#                 return ckpt[k]
#     return ckpt


# def _clean_state_dict_keys(state_dict):
#     new_state = {}
#     for k, v in state_dict.items():
#         nk = k
#         for prefix in ["module.", "model.", "clip."]:
#             if nk.startswith(prefix):
#                 nk = nk[len(prefix):]
#         new_state[nk] = v
#     return new_state


# def load_custom_checkpoint(model, ckpt_path, verbose=True):
#     ckpt = torch.load(ckpt_path, map_location="cpu")
#     state_dict = _unwrap_checkpoint(ckpt)
#     state_dict = _clean_state_dict_keys(state_dict)

#     model_state = model.state_dict()
#     matched = {}
#     skipped = []

#     for k, v in state_dict.items():
#         if k in model_state and model_state[k].shape == v.shape:
#             matched[k] = v
#         else:
#             skipped.append(k)

#     msg = model.load_state_dict(matched, strict=False)

#     if verbose:
#         print(f"[load_custom_checkpoint] loaded from: {ckpt_path}")
#         print(f"[load_custom_checkpoint] matched keys: {len(matched)}")
#         print(f"[load_custom_checkpoint] missing keys: {len(msg.missing_keys)}")
#         print(f"[load_custom_checkpoint] unexpected keys: {len(msg.unexpected_keys)}")
#         print(f"[load_custom_checkpoint] skipped keys due to mismatch/not found: {len(skipped)}")

#         if len(skipped) > 0:
#             print("[load_custom_checkpoint] first 20 skipped keys:")
#             for k in skipped[:20]:
#                 print("  ", k)

#     return model
def _unwrap_checkpoint(ckpt):
    if isinstance(ckpt, dict):
        for k in ["state_dict", "model", "model_state_dict", "clip_state_dict"]:
            if k in ckpt and isinstance(ckpt[k], dict):
                return ckpt[k]
    return ckpt


def _clean_state_dict_keys(state_dict):
    new_state = {}
    for k, v in state_dict.items():
        nk = k
        for prefix in ["module.", "model.", "clip."]:
            if nk.startswith(prefix):
                nk = nk[len(prefix):]
        new_state[nk] = v
    return new_state


def load_custom_checkpoint(model, ckpt_path, verbose=True):
    ckpt = torch.load(ckpt_path, map_location="cpu")
    state_dict = _unwrap_checkpoint(ckpt)
    state_dict = _clean_state_dict_keys(state_dict)

    model_state = model.state_dict()
    matched = {}
    skipped = []

    for k, v in state_dict.items():
        candidate_keys = [k]

        # case 1: checkpoint stores only visual backbone keys like
        # conv1.weight -> map to visual.conv1.weight
        if not k.startswith("visual.") and ("visual." + k) in model_state:
            candidate_keys.append("visual." + k)

        # choose first candidate that matches both key and shape
        loaded = False
        for ck in candidate_keys:
            if ck in model_state and model_state[ck].shape == v.shape:
                matched[ck] = v
                loaded = True
                break

        if not loaded:
            skipped.append(k)

    msg = model.load_state_dict(matched, strict=False)

    if verbose:
        print(f"[load_custom_checkpoint] loaded from: {ckpt_path}")
        print(f"[load_custom_checkpoint] matched keys: {len(matched)}")
        print(f"[load_custom_checkpoint] missing keys: {len(msg.missing_keys)}")
        print(f"[load_custom_checkpoint] unexpected keys: {len(msg.unexpected_keys)}")
        print(f"[load_custom_checkpoint] skipped keys due to mismatch/not found: {len(skipped)}")

        matched_visual = [k for k in matched.keys() if k.startswith("visual.")]
        skipped_visual = [k for k in skipped if (not k.startswith("visual.")) or k.startswith("visual.")]
        print(f"[load_custom_checkpoint] matched visual keys: {len(matched_visual)}")

        if len(matched_visual) > 0:
            print("[load_custom_checkpoint] first 20 matched visual keys:")
            for k in matched_visual[:20]:
                print("  ", k)

        if len(skipped) > 0:
            print("[load_custom_checkpoint] first 20 skipped keys:")
            for k in skipped[:20]:
                print("  ", k)

    return model

def load(
    name: str,
    device: Union[str, torch.device] = "cuda" if torch.cuda.is_available() else "cpu",
    jit: bool = False,
    download_root: str = None
):
    """
    Load a CLIP model

    Parameters
    ----------
    name : str
        A model name listed by `clip.available_models()`, or the path to a model checkpoint containing the state_dict

    device : Union[str, torch.device]
        The device to put the loaded model

    jit : bool
        Whether to load the optimized JIT model (default: False)

    download_root: str
        Path to download the model files; by default, "~/.cache/clip"

    Returns
    -------
    model : torch.nn.Module
    preprocess : Callable[[PIL.Image], torch.Tensor]
    """
    if name in _MODELS:
        model_path = _download(_MODELS[name], download_root or os.path.expanduser("~/.cache/clip"))
    elif os.path.isfile(name):
        model_path = name
    else:
        raise RuntimeError(f"Model {name} not found; available models = {available_models()}")

    try:
        model = torch.jit.load(model_path, map_location=device if jit else "cpu").eval()
        state_dict = None
    except RuntimeError:
        if jit:
            warnings.warn(f"File {model_path} is not a JIT archive. Loading as a state dict instead")
            jit = False
        state_dict = torch.load(model_path, map_location="cpu")

    embed_dim = model.state_dict()["text_projection"].shape[1]

    if not jit:
        model = build_model(state_dict or model.state_dict()).to(device)
        if str(device) == "cpu":
            model.float()
        return model, embed_dim, _transform(model.visual.input_resolution)

    device_holder = torch.jit.trace(lambda: torch.ones([]).to(torch.device(device)), example_inputs=[])
    device_node = [n for n in device_holder.graph.findAllNodes("prim::Constant") if "Device" in repr(n)][-1]

    def patch_device(module):
        try:
            graphs = [module.graph] if hasattr(module, "graph") else []
        except RuntimeError:
            graphs = []

        if hasattr(module, "forward1"):
            graphs.append(module.forward1.graph)

        for graph in graphs:
            for node in graph.findAllNodes("prim::Constant"):
                if "value" in node.attributeNames() and str(node["value"]).startswith("cuda"):
                    node.copyAttributes(device_node)

    model.apply(patch_device)
    patch_device(model.encode_image)
    patch_device(model.encode_text)

    if str(device) == "cpu":
        float_holder = torch.jit.trace(lambda: torch.ones([]).float(), example_inputs=[])
        float_input = list(float_holder.graph.findNode("aten::to").inputs())[1]
        float_node = float_input.node()

        def patch_float(module):
            try:
                graphs = [module.graph] if hasattr(module, "graph") else []
            except RuntimeError:
                graphs = []

            if hasattr(module, "forward1"):
                graphs.append(module.forward1.graph)

            for graph in graphs:
                for node in graph.findAllNodes("aten::to"):
                    inputs = list(node.inputs())
                    for i in [1, 2]:
                        if inputs[i].node()["value"] == 5:
                            inputs[i].node().copyAttributes(float_node)

        model.apply(patch_float)
        patch_float(model.encode_image)
        patch_float(model.encode_text)

        model.float()

    return model, embed_dim, _transform(model.input_resolution.item())


def tokenize(texts: Union[str, List[str]], context_length: int = 77, truncate: bool = False) -> torch.LongTensor:
    """
    Returns the tokenized representation of given input string(s)
    """
    if isinstance(texts, str):
        texts = [texts]

    sot_token = _tokenizer.encoder["<|startoftext|>"]
    eot_token = _tokenizer.encoder["<|endoftext|>"]
    all_tokens = [[sot_token] + _tokenizer.encode(text) + [eot_token] for text in texts]

    result = torch.zeros(len(all_tokens), context_length, dtype=torch.long)

    for i, tokens in enumerate(all_tokens):
        if len(tokens) > context_length:
            if truncate:
                tokens = tokens[:context_length]
                tokens[-1] = eot_token
            else:
                raise RuntimeError(f"Input {texts[i]} is too long for context length {context_length}")
        result[i, :len(tokens)] = torch.tensor(tokens)

    return result
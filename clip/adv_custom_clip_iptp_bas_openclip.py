import math
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    import open_clip
except ImportError:
    open_clip = None

from clip import load as openai_clip_load
from clip import tokenize as openai_clip_tokenize
from .simple_tokenizer import SimpleTokenizer as _Tokenizer
from data.imagnet_prompts import imagenet_classes
from data.covid_prompts import covid_classes
from data.fewshot_datasets import fewshot_datasets
from data.cls_to_names import *
import ipdb

_tokenizer = _Tokenizer()

DOWNLOAD_ROOT = '~/.cache/clip'


# def _build_backbone_and_tokenizer(
#     clip_impl='openai',
#     arch='ViT-B/16',
#     pretrained=None,
#     device='cuda',
#     download_root=DOWNLOAD_ROOT
# ):
def _build_backbone_and_tokenizer(
    clip_impl='openai',
    arch='ViT-B/16',
    pretrained=None,
    checkpoint_path=None,
    device='cuda',
    download_root='~/.cache/clip'
):
    """
    Returns:
        clip_model: underlying CLIP/OpenCLIP model
        tokenizer_fn: callable that takes str or list[str] and returns token ids
    """
    if clip_impl == 'openai':
        clip_model, _, _ = openai_clip_load(
            arch, device=device, download_root=download_root
        )
        tokenizer_fn = openai_clip_tokenize
        return clip_model, tokenizer_fn

    # elif clip_impl == 'open_clip':
    #     if open_clip is None:
    #         raise ImportError("open_clip is not installed, but clip_impl='open_clip' was requested.")

    #     if pretrained is None:
    #         raise ValueError("For clip_impl='open_clip', please provide pretrained model name or hf-hub path.")

    #     # HF hub path
    #     if pretrained.startswith('hf-hub:'):
    #         clip_model, _, _ = open_clip.create_model_and_transforms(
    #             pretrained,
    #             device=device
    #         )
    #         tokenizer_fn = open_clip.get_tokenizer(pretrained)
    #     else:
    #         clip_model, _, _ = open_clip.create_model_and_transforms(
    #             arch,
    #             pretrained=pretrained,
    #             device=device
    #         )
    #         tokenizer_fn = open_clip.get_tokenizer(arch)

    #     return clip_model, tokenizer_fn
    elif clip_impl == 'open_clip':
        if open_clip is None:
            raise ImportError("open_clip is not installed, but clip_impl='open_clip' was requested.")

        # local checkpoint path has highest priority
        if checkpoint_path is not None:
            clip_model = open_clip.create_model(
                arch,
                pretrained=None,
                device=device
            )
            tokenizer_fn = open_clip.get_tokenizer(arch)

            ckpt = torch.load(checkpoint_path, map_location='cpu')

            # common checkpoint layouts
            if isinstance(ckpt, dict):
                if 'state_dict' in ckpt:
                    state_dict = ckpt['state_dict']
                elif 'model' in ckpt:
                    state_dict = ckpt['model']
                else:
                    state_dict = ckpt
            else:
                state_dict = ckpt

            # strip common prefixes
            new_state_dict = {}
            for k, v in state_dict.items():
                nk = k
                if nk.startswith('module.'):
                    nk = nk[len('module.'):]
                if nk.startswith('model.'):
                    nk = nk[len('model.'):]
                new_state_dict[nk] = v

            missing, unexpected = clip_model.load_state_dict(new_state_dict, strict=False)
            print("Loaded open_clip checkpoint from:", checkpoint_path)
            print("Missing keys:", len(missing))
            print("Unexpected keys:", len(unexpected))

            clip_model = clip_model.to(device)
            return clip_model, tokenizer_fn

        if pretrained is None:
            raise ValueError("For clip_impl='open_clip', provide either pretrained or checkpoint_path.")

        if pretrained.startswith('hf-hub:'):
            clip_model, _, _ = open_clip.create_model_and_transforms(
                pretrained,
                device=device
            )
            tokenizer_fn = open_clip.get_tokenizer(pretrained)
        else:
            clip_model, _, _ = open_clip.create_model_and_transforms(
                arch,
                pretrained=pretrained,
                device=device
            )
            tokenizer_fn = open_clip.get_tokenizer(arch)

        return clip_model, tokenizer_fn

    # else:
    #     raise ValueError(f"Unknown clip_impl: {clip_impl}")


def _get_visual_module(clip_model):
    if hasattr(clip_model, 'visual'):
        return clip_model.visual
    raise AttributeError("Could not find visual tower on the provided CLIP model.")


def _get_token_embedding_module(clip_model):
    if hasattr(clip_model, 'token_embedding'):
        return clip_model.token_embedding
    if hasattr(clip_model, 'text') and hasattr(clip_model.text, 'token_embedding'):
        return clip_model.text.token_embedding
    raise AttributeError("Could not find token_embedding on the provided CLIP model.")


def _get_logit_scale_tensor(clip_model):
    if hasattr(clip_model, 'logit_scale'):
        return clip_model.logit_scale
    raise AttributeError("Could not find logit_scale on the provided CLIP model.")


class ClipImageEncoder(nn.Module):
    def __init__(
        self,
        device,
        arch="ViT-L/14",
        image_resolution=224,
        n_class=1000,
        clip_impl='openai',
        pretrained=None,
        download_root=DOWNLOAD_ROOT
    ):
        super(ClipImageEncoder, self).__init__()
        clip_model, _ = _build_backbone_and_tokenizer(
            clip_impl=clip_impl,
            arch=arch,
            pretrained=pretrained,
            device=device,
            download_root=download_root
        )
        self.encoder = _get_visual_module(clip_model)

        if hasattr(clip_model, 'transformer'):
            del clip_model.transformer
        torch.cuda.empty_cache()

        if hasattr(clip_model, 'text_projection'):
            embed_dim = clip_model.text_projection.shape[1]
        elif hasattr(clip_model, 'text') and hasattr(clip_model.text, 'text_projection'):
            embed_dim = clip_model.text.text_projection.shape[1]
        else:
            raise AttributeError("Unable to infer embed_dim from CLIP model.")

        self.cls_head = nn.Linear(embed_dim, n_class)

    @property
    def dtype(self):
        return self.encoder.conv1.weight.dtype

    def forward(self, image):
        x = self.encoder(image.type(self.dtype))
        output = self.cls_head(x)
        return output


class TextEncoder(nn.Module):
    def __init__(self, clip_model):
        super().__init__()

        # OpenAI CLIP style
        # if hasattr(clip_model, 'transformer'):
        #     self.transformer = clip_model.transformer
        #     self.positional_embedding = clip_model.positional_embedding
        #     self.ln_final = clip_model.ln_final
        #     self.text_projection = clip_model.text_projection
        #     self.dtype = clip_model.dtype
        #     self.mode = 'openai'
        #     return
        if hasattr(clip_model, 'transformer'):
            self.transformer = clip_model.transformer
            self.positional_embedding = clip_model.positional_embedding
            self.ln_final = clip_model.ln_final
            self.text_projection = clip_model.text_projection

            if hasattr(clip_model, 'dtype'):
                self.dtype = clip_model.dtype
            elif hasattr(clip_model, 'token_embedding'):
                self.dtype = clip_model.token_embedding.weight.dtype
            elif hasattr(clip_model, 'visual') and hasattr(clip_model.visual, 'conv1'):
                self.dtype = clip_model.visual.conv1.weight.dtype
            else:
                self.dtype = torch.float32

            self.mode = 'openai'
            return
        # OpenCLIP common style: text transformer stored under clip_model.transformer as well
        # if hasattr(clip_model, 'text') and hasattr(clip_model.text, 'transformer'):
        #     self.transformer = clip_model.text.transformer
        #     self.positional_embedding = clip_model.text.positional_embedding
        #     self.ln_final = clip_model.text.ln_final
        #     self.text_projection = clip_model.text.text_projection
        #     self.dtype = clip_model.visual.conv1.weight.dtype
        #     self.mode = 'openclip_text_submodule'
        #     return
        if hasattr(clip_model, 'text') and hasattr(clip_model.text, 'transformer'):
            self.transformer = clip_model.text.transformer
            self.positional_embedding = clip_model.text.positional_embedding
            self.ln_final = clip_model.text.ln_final
            self.text_projection = clip_model.text.text_projection

            if hasattr(clip_model.text, 'token_embedding'):
                self.dtype = clip_model.text.token_embedding.weight.dtype
            elif hasattr(clip_model, 'token_embedding'):
                self.dtype = clip_model.token_embedding.weight.dtype
            elif hasattr(clip_model, 'visual') and hasattr(clip_model.visual, 'conv1'):
                self.dtype = clip_model.visual.conv1.weight.dtype
            else:
                self.dtype = torch.float32

            self.mode = 'openclip_text_submodule'
            return

        raise AttributeError("Unsupported CLIP text tower structure for TextEncoder.")

    def forward(self, prompts, tokenized_prompts):
        x = prompts + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(self.dtype)

        x = x[torch.arange(x.shape[0], device=x.device), tokenized_prompts.argmax(dim=-1)] @ self.text_projection
        return x


class PromptLearner(nn.Module):
    def __init__(
        self,
        clip_model,
        tokenizer_fn,
        classnames,
        batch_size=None,
        n_ctx=16,
        ctx_init=None,
        ctx_position='end',
        learned_cls=False
    ):
        super().__init__()
        n_cls = len(classnames)
        self.learned_cls = learned_cls
        self.tokenizer_fn = tokenizer_fn

        if hasattr(clip_model, 'dtype'):
            dtype = clip_model.dtype
        else:
            dtype = clip_model.visual.conv1.weight.dtype

        self.dtype = dtype
        self.device = clip_model.visual.conv1.weight.device

        if hasattr(clip_model, 'ln_final'):
            ctx_dim = clip_model.ln_final.weight.shape[0]
        elif hasattr(clip_model, 'text') and hasattr(clip_model.text, 'ln_final'):
            ctx_dim = clip_model.text.ln_final.weight.shape[0]
        else:
            raise AttributeError("Could not infer ctx_dim from clip model.")

        self.ctx_dim = ctx_dim
        self.batch_size = batch_size

        token_embedding = _get_token_embedding_module(clip_model)

        if ctx_init:
            print("Initializing the contect with given words: [{}]".format(ctx_init))
            ctx_init = ctx_init.replace("_", " ")

            if '[CLS]' in ctx_init:
                ctx_list = ctx_init.split(" ")
                split_idx = ctx_list.index("[CLS]")
                ctx_init = ctx_init.replace("[CLS] ", "")
                ctx_position = "middle"
            else:
                split_idx = None

            self.split_idx = split_idx
            n_ctx = len(ctx_init.split(" "))

            prompt = tokenizer_fn(ctx_init).to(self.device)
            with torch.no_grad():
                embedding = token_embedding(prompt).type(dtype)

            ctx_vectors = embedding[0, 1:1 + n_ctx, :]
            prompt_prefix = ctx_init
        else:
            print("Random initialization: initializing a generic context")
            ctx_vectors = torch.empty(n_ctx, ctx_dim, dtype=dtype, device=self.device)
            nn.init.normal_(ctx_vectors, std=0.02)
            prompt_prefix = " ".join(["X"] * n_ctx)

        self.prompt_prefix = prompt_prefix

        print(f'Initial context: "{prompt_prefix}"')
        print(f"Number of context words (tokens): {n_ctx}")

        if self.batch_size is not None:
            ctx_vectors = ctx_vectors.repeat(batch_size, 1, 1)

        self.ctx_init_state = ctx_vectors.detach().clone()
        self.ctx = nn.Parameter(ctx_vectors)

        if not self.learned_cls:
            classnames = [name.replace("_", " ") for name in classnames]
            name_lens = [len(_tokenizer.encode(name)) for name in classnames]
            prompts = [prompt_prefix + " " + name + "." for name in classnames]
        else:
            print("Random initialization: initializing a learnable class token")
            cls_vectors = torch.empty(n_cls, 1, ctx_dim, dtype=dtype, device=self.device)
            nn.init.normal_(cls_vectors, std=0.02)
            cls_token = "X"
            name_lens = [1 for _ in classnames]
            prompts = [prompt_prefix + " " + cls_token + "." for _ in classnames]

            self.cls_init_state = cls_vectors.detach().clone()
            self.cls = nn.Parameter(cls_vectors)

        tokenized_prompts = tokenizer_fn(prompts).to(self.device)

        with torch.no_grad():
            embedding = token_embedding(tokenized_prompts).type(dtype)

        self.register_buffer("token_prefix", embedding[:, :1, :])  # SOS
        if self.learned_cls:
            self.register_buffer("token_suffix", embedding[:, 1 + n_ctx + 1:, :])
        else:
            self.register_buffer("token_suffix", embedding[:, 1 + n_ctx:, :])

        self.ctx_init = ctx_init
        self.tokenized_prompts = tokenized_prompts
        self.name_lens = name_lens
        self.class_token_position = ctx_position
        self.n_cls = n_cls
        self.n_ctx = n_ctx
        self.classnames = classnames
        self.clip_impl = None
        self.pretrained = None
        self.arch = None

    def reset(self):
        ctx_vectors = self.ctx_init_state
        self.ctx.data.copy_(ctx_vectors)
        if self.learned_cls:
            cls_vectors = self.cls_init_state
            self.cls.data.copy_(cls_vectors)

    def reset_classnames(self, classnames, arch, clip_impl='openai', pretrained=None, download_root=DOWNLOAD_ROOT):
        self.n_cls = len(classnames)

        if not self.learned_cls:
            classnames = [name.replace("_", " ") for name in classnames]
            name_lens = [len(_tokenizer.encode(name)) for name in classnames]
            prompts = [self.prompt_prefix + " " + name + "." for name in classnames]
        else:
            cls_vectors = torch.empty(self.n_cls, 1, self.ctx_dim, dtype=self.dtype, device=self.device)
            nn.init.normal_(cls_vectors, std=0.02)
            cls_token = "X"
            name_lens = [1 for _ in classnames]
            prompts = [self.prompt_prefix + " " + cls_token + "." for _ in classnames]
            self.cls_init_state = cls_vectors.detach().clone()

        tokenized_prompts = self.tokenizer_fn(prompts).to(self.device)

        clip_model, _ = _build_backbone_and_tokenizer(
            clip_impl=clip_impl,
            arch=arch,
            pretrained=pretrained,
            device=self.device,
            download_root=download_root
        )
        token_embedding = _get_token_embedding_module(clip_model)

        with torch.no_grad():
            embedding = token_embedding(tokenized_prompts).type(self.dtype)

        self.token_prefix = embedding[:, :1, :]
        self.token_suffix = embedding[:, 1 + self.n_ctx:, :]
        self.name_lens = name_lens
        self.tokenized_prompts = tokenized_prompts
        self.classnames = classnames

    def forward(self, init=None):
        if init is not None:
            ctx = init
        else:
            ctx = self.ctx

        if ctx.dim() == 2:
            ctx = ctx.unsqueeze(0).expand(self.n_cls, -1, -1)
        elif not ctx.size()[0] == self.n_cls:
            ctx = ctx.unsqueeze(1).expand(-1, self.n_cls, -1, -1)

        prefix = self.token_prefix
        suffix = self.token_suffix

        if self.batch_size is not None:
            prefix = prefix.repeat(self.batch_size, 1, 1, 1)
            suffix = suffix.repeat(self.batch_size, 1, 1, 1)

        if self.learned_cls:
            assert self.class_token_position == "end"

        if self.class_token_position == "end":
            if self.learned_cls:
                cls = self.cls
                prompts = torch.cat([prefix, ctx, cls, suffix], dim=-2)
            else:
                prompts = torch.cat([prefix, ctx, suffix], dim=-2)

        elif self.class_token_position == "middle":
            if self.split_idx is not None:
                half_n_ctx = self.split_idx
            else:
                half_n_ctx = self.n_ctx // 2

            prompts = []
            for i in range(self.n_cls):
                name_len = self.name_lens[i]
                prefix_i = prefix[i:i + 1, :, :]
                class_i = suffix[i:i + 1, :name_len, :]
                suffix_i = suffix[i:i + 1, name_len:, :]
                ctx_i_half1 = ctx[i:i + 1, :half_n_ctx, :]
                ctx_i_half2 = ctx[i:i + 1, half_n_ctx:, :]
                prompt = torch.cat(
                    [prefix_i, ctx_i_half1, class_i, ctx_i_half2, suffix_i],
                    dim=1,
                )
                prompts.append(prompt)
            prompts = torch.cat(prompts, dim=0)

        elif self.class_token_position == "front":
            prompts = []
            for i in range(self.n_cls):
                name_len = self.name_lens[i]
                prefix_i = prefix[i:i + 1, :, :]
                class_i = suffix[i:i + 1, :name_len, :]
                suffix_i = suffix[i:i + 1, name_len:, :]
                ctx_i = ctx[i:i + 1, :, :]
                prompt = torch.cat([prefix_i, class_i, ctx_i, suffix_i], dim=1)
                prompts.append(prompt)
            prompts = torch.cat(prompts, dim=0)

        else:
            raise ValueError

        return prompts


class ClipTestTimeTuning(nn.Module):
    def __init__(
        self,
        device,
        classnames,
        batch_size,
        criterion='cosine',
        arch="ViT-L/14",
        n_ctx=16,
        ctx_init=None,
        ctx_position='end',
        learned_cls=False,
        clip_impl='openai',
        pretrained=None,
        download_root=DOWNLOAD_ROOT,
        checkpoint_path=None
    ):
        super(ClipTestTimeTuning, self).__init__()

        # clip_model, tokenizer_fn = _build_backbone_and_tokenizer(
        #     clip_impl=clip_impl,
        #     arch=arch,
        #     pretrained=pretrained,
        #     device=device,
        #     download_root=download_root
        # )
        clip_model, tokenizer_fn = _build_backbone_and_tokenizer(
        clip_impl=clip_impl,
        arch=arch,
        pretrained=pretrained,
        checkpoint_path=checkpoint_path,
        device=device,
        download_root=download_root
)

        self.clip_impl = clip_impl
        self.pretrained = pretrained
        self.arch = arch
        self.download_root = download_root

        self.image_encoder = _get_visual_module(clip_model)
        self.text_encoder = TextEncoder(clip_model)
        self.logit_scale = _get_logit_scale_tensor(clip_model).data

        self.prompt_learner = PromptLearner(
            clip_model,
            tokenizer_fn,
            classnames,
            batch_size,
            n_ctx,
            ctx_init,
            ctx_position,
            learned_cls
        )

        self.criterion = criterion
        self.l2_norm_cal = False
        self.textfeatures_ = None

    @property
    def dtype(self):
        return self.image_encoder.conv1.weight.dtype

    def reset(self):
        self.prompt_learner.reset()

    def reset_classnames(self, classnames, arch):
        self.prompt_learner.reset_classnames(
            classnames,
            arch,
            clip_impl=self.clip_impl,
            pretrained=self.pretrained,
            download_root=self.download_root
        )

    def get_text_features(self):
        text_features = []
        prompts = self.prompt_learner()
        tokenized_prompts = self.prompt_learner.tokenized_prompts
        t_features = self.text_encoder(prompts, tokenized_prompts)
        text_features.append(t_features / t_features.norm(dim=-1, keepdim=True))
        text_features = torch.stack(text_features, dim=0)
        return torch.mean(text_features, dim=0)

    def inference(self, image, cons, args, enable_image_grad=False):
        if enable_image_grad:
            image_features = self.image_encoder(image.type(self.dtype))
        else:
            with torch.no_grad():
                image_features = self.image_encoder(image.type(self.dtype))

        text_features = self.get_text_features()
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)

        if self.l2_norm_cal:
            self.textfeatures_ = text_features

        logit_scale = self.logit_scale.exp()
        logits = logit_scale * image_features @ text_features.t()
        return logits

    def forward(self, input, cons, args, enable_image_grad=False):
        if isinstance(input, Tuple):
            view_0, view_1, view_2 = input
            return self.contrast_prompt_tuning(view_0, view_1, view_2)
        elif len(input.size()) == 2:
            return self.directional_prompt_tuning(input)
        else:
            return self.inference(input, cons, args, enable_image_grad=enable_image_grad)


def get_coop(
    clip_arch,
    test_set,
    device,
    n_ctx,
    ctx_init,
    cons,
    learned_cls=False,
    clip_impl='openai',
    pretrained=None,
    download_root=DOWNLOAD_ROOT,
    checkpoint_path=None
):
    if test_set in fewshot_datasets:
        classnames = eval("{}_classes".format(test_set.lower()))
    elif test_set == 'bongard':
        if learned_cls:
            classnames = ['X', 'X']
        else:
            classnames = ['True', 'False']
    elif test_set == 'C':
        classnames = covid_classes
    else:
        classnames = imagenet_classes
        model = ClipTestTimeTuning(
            device,
            classnames,
            None,
            arch=clip_arch,
            n_ctx=n_ctx,
            ctx_init=ctx_init,
            learned_cls=learned_cls,
            clip_impl=clip_impl,
            pretrained=pretrained,
            checkpoint_path=checkpoint_path,
            download_root=download_root
        )

    # model = ClipTestTimeTuning(
    #     device,
    #     classnames,
    #     None,
    #     arch=clip_arch,
    #     n_ctx=n_ctx,
    #     ctx_init=ctx_init,
    #     learned_cls=learned_cls,
    #     clip_impl=clip_impl,
    #     pretrained=pretrained,
    #     download_root=download_root
    # )

    return model
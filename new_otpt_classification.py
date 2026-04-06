import argparse
import time
from copy import deepcopy
from datetime import datetime
from PIL import Image
import numpy as np
import csv
import os
import math
import pickle

import torch
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torch.nn.functional as F
import torchvision.models as models

import matplotlib.pyplot as plt
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from scipy.spatial.distance import pdist, squareform
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.metrics.pairwise import cosine_similarity
import seaborn as sns
import matplotlib.cm as cm
from scipy.stats import wasserstein_distance

try:
    from torchvision.transforms import InterpolationMode
    BICUBIC = InterpolationMode.BICUBIC
except ImportError:
    BICUBIC = Image.BICUBIC

from clip.new_custom_clip_iptp_bas import get_coop
from clip.cocoop import get_cocoop

from data.imagnet_prompts import imagenet_classes
from data.datautils import AugMixAugmenter, build_dataset
from utils.tools import Summary, AverageMeter, ProgressMeter, load_model_weight, set_random_seed
from data.cls_to_names import *
from data.fewshot_datasets import fewshot_datasets
from data.imagenet_variants import thousand_k_to_200, imagenet_a_mask, imagenet_r_mask, imagenet_v_mask

import ipdb

model_names = sorted(
    name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name])
)

CLIP_MEAN = torch.tensor([0.48145466, 0.4578275, 0.40821073])
CLIP_STD = torch.tensor([0.26862954, 0.26130258, 0.27577711])


def clamp(X, lower_limit, upper_limit):
    return torch.max(torch.min(X, upper_limit), lower_limit)


def ECE_Loss(num_bins, predictions, confidences, correct):
    bin_boundaries = torch.linspace(0, 1, num_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]
    bin_accuracy = [0] * num_bins
    bin_confidence = [0] * num_bins
    bin_num_sample = [0] * num_bins

    for idx in range(len(predictions)):
        confidence = confidences[idx]
        bin_idx = -1
        for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
            bin_idx += 1
            bin_lower = bin_lower.item()
            bin_upper = bin_upper.item()
            if bin_lower < confidence and confidence <= bin_upper:
                bin_num_sample[bin_idx] += 1
                bin_accuracy[bin_idx] += correct[idx]
                bin_confidence[bin_idx] += confidences[idx]

    for idx in range(num_bins):
        if bin_num_sample[idx] != 0:
            bin_accuracy[idx] = bin_accuracy[idx] / bin_num_sample[idx]
            bin_confidence[idx] = bin_confidence[idx] / bin_num_sample[idx]

    ece_loss = 0.0
    for idx in range(num_bins):
        temp_abs = abs(bin_accuracy[idx] - bin_confidence[idx])
        ece_loss += (temp_abs * bin_num_sample[idx]) / len(predictions)

    return ece_loss, bin_accuracy, bin_confidence, bin_num_sample


def Calculator(result_dict):
    if len(result_dict['prediction']) == 0:
        return None, None, None, None

    list_max_confidence = result_dict['max_confidence']
    list_prediction = result_dict['prediction']
    list_label = result_dict['label']

    torch_list_prediction = torch.tensor(list_prediction).int()
    torch_list_label = torch.tensor(list_label).int()

    torch_correct = (torch_list_prediction == torch_list_label)
    list_correct = torch_correct.tolist()

    incorrect_indices = (torch_list_prediction != torch_list_label)
    torch_max_confidence = torch.tensor(list_max_confidence)
    incorrect_confidences = torch_max_confidence[incorrect_indices].tolist()

    ece_data = ECE_Loss(20, list_prediction, list_max_confidence, list_correct)
    acc = sum(list_correct) / len(list_correct)

    print('acc: ', acc * 100)
    print('ece: ', ece_data[0] * 100)

    return acc * 100, ece_data[0] * 100, ece_data[1], incorrect_confidences


def conf_acc(logits, gpu_id):
    Nb = logits.shape[0]
    prob, _ = torch.max(logits.softmax(1), dim=1)
    q_val = torch.ones(Nb).to(device=gpu_id)
    cosi = torch.nn.CosineSimilarity(dim=0)
    dw = cosi(prob, q_val)
    return dw


def select_confident_samples(logits, top):
    batch_entropy = -(logits.softmax(1) * logits.log_softmax(1)).sum(1)
    idx = torch.argsort(batch_entropy, descending=False)[:int(batch_entropy.size()[0] * top)]
    return logits[idx], idx


def avg_entropy(outputs):
    logits = outputs - outputs.logsumexp(dim=-1, keepdim=True)
    avg_logits = logits.logsumexp(dim=0) - np.log(logits.shape[0])
    min_real = torch.finfo(avg_logits.dtype).min
    avg_logits = torch.clamp(avg_logits, min=min_real)
    return -(avg_logits * torch.exp(avg_logits)).sum(dim=-1)


def accuracy(output, target, topk=(1,)):
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


def pgd_attack(model, image, target, args, cons):
    device = image.device
    mean = CLIP_MEAN.to(device).view(1, 3, 1, 1)
    std = CLIP_STD.to(device).view(1, 3, 1, 1)

    eps = (args.attack_eps / std)
    alpha = (args.attack_alpha / std)

    lower_limit = (0.0 - mean) / std
    upper_limit = (1.0 - mean) / std

    x = image.detach()

    best_adv = None
    best_loss = None

    old_flag = args.input_grad
    args.input_grad = True

    for _ in range(args.attack_restarts):
        x_adv = x.clone().detach()
        x_adv = x_adv + torch.empty_like(x_adv).uniform_(-1, 1) * eps
        x_adv = clamp(x_adv, x - eps, x + eps)
        x_adv = clamp(x_adv, lower_limit, upper_limit)

        for _ in range(args.attack_steps):
            x_adv.requires_grad_(True)
            logits = model(x_adv, cons, args)
            loss = F.cross_entropy(logits.float(), target)
            grad = torch.autograd.grad(loss, [x_adv])[0]

            x_adv = x_adv.detach() + alpha * grad.sign()
            x_adv = clamp(x_adv, x - eps, x + eps)
            x_adv = clamp(x_adv, lower_limit, upper_limit)

        with torch.no_grad():
            logits = model(x_adv, cons, args)
            loss = F.cross_entropy(logits.float(), target)

        if (best_loss is None) or (loss.item() > best_loss):
            best_loss = loss.item()
            best_adv = x_adv.detach()

    args.input_grad = old_flag
    return best_adv


def test_time_tuning(model, inputs, optimizer, scaler, args, cons):
    output = None
    output2 = None
    single_output = None

    if args.cocoop:
        image_feature, pgen_ctx = inputs
        pgen_ctx.requires_grad = True
        optimizer = torch.optim.AdamW([pgen_ctx], args.lr)

    selected_idx = None
    for j in range(args.tta_steps):
        if 'tpt' in args.run_type:
            with torch.cuda.amp.autocast():
                if args.cocoop:
                    output = model((image_feature, pgen_ctx), cons, args)
                else:
                    output = model(inputs, cons, args)

                if selected_idx is not None:
                    output = output[selected_idx]
                else:
                    output, selected_idx = select_confident_samples(output, args.selection_p)
                    softmax_out = torch.softmax(output, dim=-1)
                    soft_mean = torch.mean(softmax_out, dim=0)
                    number_of_class = output.shape[1]

                loss = avg_entropy(output)
                dw = conf_acc(output, args.gpu)
        else:
            loss = 0

        if args.two_step and 'tpt' in args.run_type:
            optimizer.zero_grad()
            scaler.scale(loss).backward(retain_graph=True)
            scaler.step(optimizer)
            scaler.update()
            loss = 0

            with torch.cuda.amp.autocast():
                if args.cocoop:
                    output2 = model((image_feature, pgen_ctx), cons, args)
                else:
                    output2, text_varience = model(inputs, cons, args)

        if 'otpt' in args.run_type:
            if output is None and output2 is None:
                single_output = model(args.image)

            lambda_ = args.lambda_term
            number_of_class = output.shape[1]

            text_feature = model.textfeatures_
            Wwt = torch.matmul(text_feature, text_feature.T)
            wwt_norm_col_HT = torch.linalg.norm(Wwt, dim=-1)
            Wwt_val_HT = wwt_norm_col_HT.mean()

            e = torch.eye(Wwt.shape[1], device=args.gpu)
            M_norm = torch.linalg.norm(Wwt, dim=0, keepdim=True)
            scaled_e = e * M_norm
            u = Wwt - scaled_e
            u_norm = torch.linalg.norm(u, dim=-1, keepdim=True)

            v = u / u_norm
            normalized_matrix_exp = v.unsqueeze(2)
            normalized_matrix_T_exp = v.unsqueeze(1)

            outer_products = normalized_matrix_exp @ normalized_matrix_T_exp
            divided_matrix = outer_products
            scaled_matrix = 2 * divided_matrix
            identity_matrix_dim = e.unsqueeze(0).expand(Wwt.shape[1], -1, -1)
            transformed_matrix = identity_matrix_dim - scaled_matrix
            Wwt_exp = Wwt.unsqueeze(2)
            Hx = torch.bmm(transformed_matrix, Wwt_exp)
            Hx = Hx.squeeze(2)

            Ht_ortho = Hx - e
            Ht_ortho_norm = torch.linalg.norm(Ht_ortho, dim=-1)
            Ht_ortho_norm_val = Ht_ortho_norm.mean()

            loss += (lambda_ * Ht_ortho_norm_val)

        if args.run_type not in ['baseline', 'baseline_cocoop', 'baseline_coop', 'baseline_ts']:
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

    if args.cocoop:
        return pgen_ctx

    return


def main(args):
    set_random_seed(args.seed)
    assert args.gpu is not None
    main_worker(args.gpu, args)


def main_worker(gpu, args):
    args.gpu = gpu
    set_random_seed(args.seed)
    print("Use GPU: {} for training".format(args.gpu))

    if args.test_sets in fewshot_datasets:
        classnames = eval("{}_classes".format(args.test_sets.lower()))
    else:
        classnames = imagenet_classes

    if args.cocoop:
        model = get_cocoop(args.arch, args.test_sets, 'cpu', args.n_ctx, args.disp_cons)
        assert args.load is not None
        load_model_weight(args.load, model, "cuda:{}".format(args.gpu), args)
        model_state = deepcopy(model.state_dict())
    else:
        model = get_coop(
            args.arch,
            args.test_sets,
            args.gpu,
            args.n_ctx,
            args.ctx_init,
            args.disp_cons,
            clip_ckpt=args.clip_ckpt
        )

        if args.load is not None:
            print("Use pre-trained soft prompt (CoOp) as initialization")
            pretrained_ctx = torch.load(args.load)['state_dict']['ctx']
            assert pretrained_ctx.size()[0] == args.n_ctx
            with torch.no_grad():
                model.prompt_learner.ctx.copy_(pretrained_ctx)
                model.prompt_learner.ctx_init_state = pretrained_ctx

        model_state = None

    for name, param in model.named_parameters():
        if not args.cocoop:
            if "prompt_learner" not in name:
                param.requires_grad_(False)
        else:
            if "text_encoder" not in name:
                param.requires_grad_(False)

    print("=> Model created: visual backbone {}".format(args.arch))

    if not torch.cuda.is_available():
        print('using CPU, this will be slow')
    else:
        assert args.gpu is not None
        torch.cuda.set_device(args.gpu)
        model = model.cuda(args.gpu)

    if args.cocoop:
        optimizer = None
        optim_state = None
    else:
        trainable_param = model.prompt_learner.parameters()
        optimizer = torch.optim.AdamW(trainable_param, args.lr)
        optim_state = deepcopy(optimizer.state_dict())

    scaler = torch.cuda.amp.GradScaler(init_scale=1000)
    print('=> Using native Torch AMP. Training in mixed precision.')

    cudnn.benchmark = True

    normalize = transforms.Normalize(
        mean=[0.48145466, 0.4578275, 0.40821073],
        std=[0.26862954, 0.26130258, 0.27577711]
    )

    datasets = args.test_sets.split("/")
    print('length of dataset', len(datasets))
    for set_id in datasets:
        print('name id of dataset:', set_id)

    results = {}

    for set_id in datasets:
        if args.tpt:
            base_transform = transforms.Compose([
                transforms.Resize(args.resolution, interpolation=BICUBIC),
                transforms.CenterCrop(args.resolution)
            ])
            preprocess = transforms.Compose([
                transforms.ToTensor(),
                normalize
            ])

            if args.I_augmix:
                data_transform = AugMixAugmenter(
                    base_transform, preprocess,
                    n_views=args.batch_size - 1,
                    augmix=len(set_id) >= 1
                )
            else:
                data_transform = AugMixAugmenter(
                    base_transform, preprocess,
                    n_views=args.batch_size - 1,
                    augmix=len(set_id) > 1
                )
            batchsize = 1
        else:
            data_transform = transforms.Compose([
                transforms.Resize(args.resolution, interpolation=BICUBIC),
                transforms.CenterCrop(args.resolution),
                transforms.ToTensor(),
                normalize,
            ])
            batchsize = args.batch_size

        print("evaluating: {}".format(set_id))

        if len(set_id) > 1:
            classnames = eval("{}_classes".format(set_id.lower()))
        else:
            assert set_id in ['A', 'R', 'K', 'V', 'I']
            classnames_all = imagenet_classes
            classnames = []
            if set_id in ['A', 'R', 'V']:
                label_mask = eval("imagenet_{}_mask".format(set_id.lower()))
                if set_id == 'R':
                    for i, m in enumerate(label_mask):
                        if m:
                            classnames.append(classnames_all[i])
                else:
                    classnames = [classnames_all[i] for i in label_mask]
            else:
                classnames = classnames_all

        if args.cocoop:
            model.prompt_generator.reset_classnames(classnames, args.arch)
            model = model.cpu()
            model_state = model.state_dict()
            model = model.cuda(args.gpu)
        else:
            model.reset_classnames(classnames, args.arch)

        val_dataset = build_dataset(set_id, data_transform, args.data, mode=args.dataset_mode)
        print("number of test samples: {}".format(len(val_dataset)))
        val_loader = torch.utils.data.DataLoader(
            val_dataset,
            batch_size=batchsize,
            shuffle=True,
            num_workers=args.workers,
            pin_memory=True
        )

        eval_out = test_time_adapt_eval(
            val_loader, model, model_state, optimizer, optim_state, scaler,
            args, args.disp_cons, classnames, set_id
        )

        clean_acc, clean_ece, _, _ = Calculator(eval_out['clean_result_dict'])

        robust_acc, robust_ece = None, None
        if len(eval_out['robust_result_dict']['prediction']) > 0:
            robust_acc, robust_ece, _, _ = Calculator(eval_out['robust_result_dict'])

        results[set_id] = {
            'clean_top1': eval_out['clean_top1'],
            'clean_top5': eval_out['clean_top5'],
            'clean_acc': clean_acc,
            'clean_ece': clean_ece,
            'robust_top1': eval_out['robust_top1'],
            'robust_top5': eval_out['robust_top5'],
            'robust_acc': robust_acc,
            'robust_ece': robust_ece,
        }

        print("=> Clean on [{}]: @1 {:.2f} / @5 {:.2f}".format(
            set_id, eval_out['clean_top1'], eval_out['clean_top5']
        ))
        if robust_acc is not None:
            print("=> Robust on [{}]: @1 {:.2f} / @5 {:.2f}".format(
                set_id, eval_out['robust_top1'], eval_out['robust_top5']
            ))

        del val_dataset, val_loader

    print("======== Result Summary ========")
    print("params: nstep lr bs")
    print("params: {} {} {}".format(args.tta_steps, args.lr, args.batch_size))

    dataset_ids = list(results.keys())
    print("\t\t [set_id]")
    for id_ in dataset_ids:
        print("{}".format(id_), end="\t")
    print("\n")

    print("Clean Top-1:")
    for id_ in dataset_ids:
        print("{:.2f}".format(results[id_]['clean_top1']), end="\t")
    print("\n")

    if any(results[id_]['robust_acc'] is not None for id_ in dataset_ids):
        print("Robust Top-1:")
        for id_ in dataset_ids:
            val = results[id_]['robust_top1']
            if results[id_]['robust_acc'] is None:
                print("N/A", end="\t")
            else:
                print("{:.2f}".format(val), end="\t")
        print("\n")

    output_csv_path = args.csv_log
    if output_csv_path is None:
        directory = os.path.dirname('/home/ashashak/VLM-calibration/C-TPT/log/test_otpt_pgd.csv')
        output_csv_path = '/home/ashashak/VLM-calibration/C-TPT/log/test_otpt_pgd.csv'
    else:
        directory = os.path.dirname(output_csv_path)

    os.makedirs(directory, exist_ok=True)
    file_exists = os.path.isfile(output_csv_path)
    current_datetime = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    with open(output_csv_path, 'a' if file_exists else 'w', newline='') as csvfile:
        csvwriter = csv.writer(csvfile)

        if not file_exists:
            csvwriter.writerow(["params: nstep", "lr", "bs", "attack", "eps", "alpha", "steps", "clip_ckpt"])
            csvwriter.writerow([
                current_datetime,
                "params: {} {} {}".format(args.tta_steps, args.lr, args.batch_size),
                args.attack,
                args.attack_eps,
                args.attack_alpha,
                args.attack_steps,
                args.clip_ckpt
            ])
            csvwriter.writerow(["", "[set_id]"])

        csvwriter.writerow([current_datetime])
        csvwriter.writerow([""] + dataset_ids)

        csvwriter.writerow(["Clean Top-1"] + ["{:.2f}".format(results[id_]['clean_top1']) for id_ in dataset_ids])
        csvwriter.writerow(["Clean Top-5"] + ["{:.2f}".format(results[id_]['clean_top5']) for id_ in dataset_ids])
        csvwriter.writerow(["Clean Accuracy"] + [
            "N/A" if results[id_]['clean_acc'] is None else "{:.2f}".format(results[id_]['clean_acc'])
            for id_ in dataset_ids
        ])
        csvwriter.writerow(["Clean ECE"] + [
            "N/A" if results[id_]['clean_ece'] is None else "{:.2f}".format(results[id_]['clean_ece'])
            for id_ in dataset_ids
        ])

        if any(results[id_]['robust_acc'] is not None for id_ in dataset_ids):
            csvwriter.writerow(["Robust Top-1"] + [
                "N/A" if results[id_]['robust_acc'] is None else "{:.2f}".format(results[id_]['robust_top1'])
                for id_ in dataset_ids
            ])
            csvwriter.writerow(["Robust Top-5"] + [
                "N/A" if results[id_]['robust_acc'] is None else "{:.2f}".format(results[id_]['robust_top5'])
                for id_ in dataset_ids
            ])
            csvwriter.writerow(["Robust Accuracy"] + [
                "N/A" if results[id_]['robust_acc'] is None else "{:.2f}".format(results[id_]['robust_acc'])
                for id_ in dataset_ids
            ])
            csvwriter.writerow(["Robust ECE"] + [
                "N/A" if results[id_]['robust_ece'] is None else "{:.2f}".format(results[id_]['robust_ece'])
                for id_ in dataset_ids
            ])


def test_time_adapt_eval(val_loader, model, model_state, optimizer, optim_state, scaler, args, cons, classnames, set_id):
    batch_time = AverageMeter('Time', ':6.3f', Summary.NONE)

    clean_top1 = AverageMeter('Clean Acc@1', ':6.2f', Summary.AVERAGE)
    clean_top5 = AverageMeter('Clean Acc@5', ':6.2f', Summary.AVERAGE)
    robust_top1 = AverageMeter('Robust Acc@1', ':6.2f', Summary.AVERAGE)
    robust_top5 = AverageMeter('Robust Acc@5', ':6.2f', Summary.AVERAGE)

    progress = ProgressMeter(
        len(val_loader),
        [batch_time, clean_top1, clean_top5],
        prefix='Test: '
    )

    model.eval()
    if not args.cocoop:
        with torch.no_grad():
            model.reset()

    end = time.time()
    softmax = torch.nn.Softmax(dim=1)

    if 'otpt' in args.run_type:
        model.l2_norm_cal = True
    else:
        model.l2_norm_cal = False

    clean_result_dict = {'max_confidence': [], 'prediction': [], 'label': []}
    robust_result_dict = {'max_confidence': [], 'prediction': [], 'label': []}

    for i, (images, target) in enumerate(val_loader):
        assert args.gpu is not None

        if isinstance(images, list):
            for k in range(len(images)):
                images[k] = images[k].cuda(args.gpu, non_blocking=True)
            image = images[0]
        else:
            if len(images.size()) > 4:
                assert images.size()[0] == 1
                images = images.squeeze(0)
            images = images.cuda(args.gpu, non_blocking=True)
            image = images

        target = target.cuda(args.gpu, non_blocking=True)

        if args.tpt:
            images = torch.cat(images, dim=0)

        if 'otpt' in args.run_type:
            args.image = image

        if not args.cocoop:
            if args.tta_steps > 0:
                with torch.no_grad():
                    model.reset()
            optimizer.load_state_dict(optim_state)
            test_time_tuning(model, images, optimizer, scaler, args, cons)
        else:
            with torch.no_grad():
                with torch.cuda.amp.autocast():
                    image_feature, pgen_ctx = model.gen_ctx(images, args.tpt)
            optimizer = None
            pgen_ctx = test_time_tuning(model, (image_feature, pgen_ctx), optimizer, scaler, args, cons)

        if args.tpt and args.cocoop:
            image_feature = image_feature[0].unsqueeze(0)

        # clean eval
        if args.eval_mode in ['clean', 'both']:
            with torch.no_grad():
                with torch.cuda.amp.autocast():
                    if args.cocoop:
                        output_clean = model((image_feature, pgen_ctx), cons, args)
                    else:
                        output_clean = model(image, cons, args)

            if 'ts' not in args.run_type:
                softmax_output_clean = softmax(output_clean)
            elif 'ViT' in args.arch:
                softmax_output_clean = softmax(output_clean / temperature_value['ViT'])
            elif 'RN' in args.arch:
                softmax_output_clean = softmax(output_clean / temperature_value['RN'])
            else:
                ipdb.set_trace()

            max_confidence, max_index = torch.max(softmax_output_clean, 1)

            clean_result_dict['max_confidence'].append(max_confidence.item())
            clean_result_dict['prediction'].append(max_index.item())
            clean_result_dict['label'].append(target.item())

            acc1, acc5 = accuracy(output_clean, target, topk=(1, 5))
            clean_top1.update(acc1[0], image.size(0))
            clean_top5.update(acc5[0], image.size(0))

        # robust eval
        if args.attack == 'pgd' and args.eval_mode in ['robust', 'both']:
            x_adv = pgd_attack(model, image, target, args, cons)

            with torch.no_grad():
                with torch.cuda.amp.autocast():
                    output_robust = model(x_adv, cons, args)

            if 'ts' not in args.run_type:
                softmax_output_robust = softmax(output_robust)
            elif 'ViT' in args.arch:
                softmax_output_robust = softmax(output_robust / temperature_value['ViT'])
            elif 'RN' in args.arch:
                softmax_output_robust = softmax(output_robust / temperature_value['RN'])
            else:
                ipdb.set_trace()

            max_confidence_r, max_index_r = torch.max(softmax_output_robust, 1)

            robust_result_dict['max_confidence'].append(max_confidence_r.item())
            robust_result_dict['prediction'].append(max_index_r.item())
            robust_result_dict['label'].append(target.item())

            acc1_r, acc5_r = accuracy(output_robust, target, topk=(1, 5))
            robust_top1.update(acc1_r[0], image.size(0))
            robust_top5.update(acc5_r[0], image.size(0))

        batch_time.update(time.time() - end)
        end = time.time()

        if (i + 1) % args.print_freq == 0:
            progress.display(i)

    progress.display_summary()

    return {
        'clean_top1': clean_top1.avg,
        'clean_top5': clean_top5.avg,
        'clean_result_dict': clean_result_dict,
        'robust_top1': robust_top1.avg,
        'robust_top5': robust_top5.avg,
        'robust_result_dict': robust_result_dict,
    }


temperature_value = {'ViT': 1.16, 'RN': 1.15}

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Test-time Prompt Tuning + O-TPT + PGD')

    parser.add_argument('data', metavar='DIR', help='path to dataset root')
    parser.add_argument('--test_sets', type=str, default='A/R/V/K/I', help='test dataset (multiple datasets split by slash)')
    parser.add_argument('--csv_log', type=str, help='path to save the CSV summary')
    parser.add_argument('--dataset_mode', type=str, default='test', help='which split to use: train/val/test')
    parser.add_argument('-a', '--arch', metavar='ARCH', default='RN50')
    parser.add_argument('--resolution', default=224, type=int, help='CLIP image resolution')
    parser.add_argument('-j', '--workers', default=4, type=int, metavar='N', help='number of data loading workers')
    parser.add_argument('-b', '--batch-size', default=64, type=int, metavar='N')
    parser.add_argument('--lr', '--learning-rate', default=5e-3, type=float, metavar='LR', help='initial learning rate', dest='lr')
    parser.add_argument('-p', '--print-freq', default=200, type=int, metavar='N', help='print frequency')
    parser.add_argument('--gpu', default=1, type=int, help='GPU id to use.')
    parser.add_argument('--tpt', action='store_true', default=False, help='run test-time prompt tuning')
    parser.add_argument('--selection_p', default=0.1, type=float, help='confidence selection percentile')
    parser.add_argument('--tta_steps', default=1, type=int, help='test-time-adapt steps')
    parser.add_argument('--n_ctx', default=4, type=int, help='number of tunable tokens')
    parser.add_argument('--ctx_init', default=None, type=str, help='init tunable prompts')
    parser.add_argument('--cocoop', action='store_true', default=False, help="use cocoop's output as prompt initialization")
    parser.add_argument('--load', default=None, type=str, help='path to a pre-trained coop/cocoop')
    parser.add_argument('--seed', type=int, default=0)

    # O-TPT args
    parser.add_argument('--lambda_term', type=float, default=0.0, help='lambda for o-tpt')
    parser.add_argument('--disp_cons', type=int, nargs='+', default=[18.0], help='List of display constants')
    parser.add_argument('--run_type', type=str, default='baseline_tpt',
                        choices=['baseline', 'tpt', 'tpt_otpt', 'tpt_ts'])
    parser.add_argument('--two_step', action='store_true', default=False, help='two step training')
    parser.add_argument('--I_augmix', action='store_true', default=False, help='augmix for I')

    # robust checkpoint
    parser.add_argument('--clip_ckpt', type=str, default=None, help='path to robust CLIP checkpoint')

    # adversarial eval
    parser.add_argument('--attack', type=str, default='none', choices=['none', 'pgd'])
    parser.add_argument('--attack_eps', type=float, default=1.0 / 255.0)
    parser.add_argument('--attack_alpha', type=float, default=0.25 / 255.0)
    parser.add_argument('--attack_steps', type=int, default=10)
    parser.add_argument('--attack_restarts', type=int, default=1)
    parser.add_argument('--eval_mode', type=str, default='both', choices=['clean', 'robust', 'both'])

    # internal flag: do not set manually in shell, PGD uses it temporarily
    parser.add_argument('--input_grad', action='store_true', default=False)

    args = parser.parse_args()

    if 'otpt' not in args.run_type:
        args.lambda_term = 0.0

    main(args)
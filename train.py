import argparse
from contextlib import nullcontext
from dataclasses import dataclass
import numpy as np
np.set_printoptions(threshold=np.inf)
from tqdm import tqdm
import os
import sys
import random
from typing import Optional, Sequence
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
import warnings
warnings.filterwarnings('ignore')


GRADIENT_CLIP_NORM = 5.0
WEIGHT_DECAY = 2e-4
USE_AMP = torch.cuda.is_available()
DIRECT_RUN_FINETUNE = False
DEFAULT_RESUME_CKPT = ''
DEFAULT_BATCH_SIZE = 6
DEFAULT_LR = 2e-4
DEFAULT_EPOCHS = 200
DEFAULT_SEED = 1234
DEFAULT_F1_INIT = 0.85
DEFAULT_EXP_NAME = 'PCINet_priorAux_bs6_lr2e4_seed1234'
DEFAULT_DATASET = 'LEVIR'
DEFAULT_FINETUNE_LR = 5e-5
DEFAULT_FINETUNE_EPOCHS = 50
AUX_LOSS_WEIGHTS = (1.0, 0.3, 0.1, 0.05)
PRIOR_SCALE_WEIGHTS = (0.05, 0.05, 0.03, 0.02)

USE_PRIOR_AUX_LOSS = True
PRIOR_AUX_WEIGHT = 0.05
MIN_SAVE_F1 = 0.9075


@dataclass(frozen=True)
class RuntimeConfig:
    seed: int
    batch_size: int
    lr: float
    epochs: int
    exp_name: str
    dataset: str
    resume_ckpt: str
    f1_init: float


@dataclass
class ModelOutputs:
    pred: torch.Tensor
    aux1: torch.Tensor
    aux2: torch.Tensor
    aux3: torch.Tensor
    prior_list: Optional[Sequence[torch.Tensor]] = None


def parse_args():
    parser = argparse.ArgumentParser(description='Clean train-only baseline training script.')
    parser.add_argument('--resume-ckpt', default=None)
    parser.add_argument('--f1-init', type=float, default=None)
    parser.add_argument('--seed', type=int, default=None)
    parser.add_argument('--batch-size', type=int, default=None)
    parser.add_argument('--lr', type=float, default=None)
    parser.add_argument('--epochs', type=int, default=None)
    parser.add_argument('--exp-name', default=None)
    parser.add_argument('--dataset', default=None, help='LEVIR, SYSU, or GZCDD')
    return parser.parse_args()


def normalize_dataset_name(dataset):
    dataset = str(dataset).strip().lower().replace('_', '-')
    aliases = {
        'levir': 'LEVIR',
        'levir-cd': 'LEVIR',
        'sysu': 'SYSU',
        'sysu-cd': 'SYSU',
        'sysucd': 'SYSU',
        'gz': 'GZCDD',
        'gz-cd': 'GZCDD',
        'gz-cdd': 'GZCDD',
        'gzcdd': 'GZCDD',
    }
    if dataset not in aliases:
        raise ValueError('Unsupported dataset. Use LEVIR, SYSU, or GZCDD.')
    return aliases[dataset]


def format_lr_for_exp_name(lr):
    lr_text = f'{lr:.0e}'
    lr_text = lr_text.replace('e-0', 'e').replace('e-', 'e')
    lr_text = lr_text.replace('e+0', 'e').replace('e+', 'e')
    return lr_text


def resolve_config(args):
    seed = args.seed if args.seed is not None else int(os.environ.get('SEED', DEFAULT_SEED))
    dataset = normalize_dataset_name(args.dataset if args.dataset is not None else os.environ.get('DATASET', DEFAULT_DATASET))
    batch_size = args.batch_size if args.batch_size is not None else int(os.environ.get('BATCH_SIZE', DEFAULT_BATCH_SIZE))
    default_resume_ckpt = DEFAULT_RESUME_CKPT if DIRECT_RUN_FINETUNE else ''
    resume_ckpt = args.resume_ckpt if args.resume_ckpt is not None else os.environ.get('RESUME_CKPT', default_resume_ckpt).strip()
    lr_default = DEFAULT_FINETUNE_LR if resume_ckpt else DEFAULT_LR
    epochs_default = DEFAULT_FINETUNE_EPOCHS if resume_ckpt else 200
    f1_default = DEFAULT_F1_INIT if resume_ckpt else 0.0
    lr = args.lr if args.lr is not None else float(os.environ.get('LR', DEFAULT_LR if not resume_ckpt else lr_default))
    epochs = args.epochs if args.epochs is not None else int(os.environ.get('EPOCHS', DEFAULT_EPOCHS if not resume_ckpt else epochs_default))
    f1_init = args.f1_init if args.f1_init is not None else float(os.environ.get('F1_INIT', f1_default))

    if args.exp_name is not None:
        exp_name = args.exp_name
    elif os.environ.get('EXP_NAME'):
        exp_name = os.environ['EXP_NAME']
    elif resume_ckpt:
        exp_name = f'baseline9080_finetune_lr{format_lr_for_exp_name(lr)}_seed{seed}'
    elif dataset != DEFAULT_DATASET:
        exp_name = f'{DEFAULT_EXP_NAME}_{dataset.lower()}'
    else:
        exp_name = DEFAULT_EXP_NAME

    return RuntimeConfig(
        seed=seed,
        batch_size=batch_size,
        lr=lr,
        epochs=epochs,
        exp_name=exp_name,
        dataset=dataset,
        resume_ckpt=resume_ckpt,
        f1_init=f1_init,
    )


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True


def amp_autocast_context(use_amp):
    if use_amp:
        return torch.cuda.amp.autocast()
    return nullcontext()


def set_bn_eval(module):
    for m in module.modules():
        if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
            m.eval()


class DiceLoss(nn.Module):
    def __init__(self, eps=1e-6):
        super(DiceLoss, self).__init__()
        self.eps = eps

    def forward(self, logits, target):
        probs = F.softmax(logits, dim=1)
        if probs.size(1) == 1:
            prob_fg = probs.squeeze(1)
        else:
            prob_fg = probs[:, 1, :, :]

        target_fg = (target == 1).float()

        intersection = (prob_fg * target_fg).sum(dim=(1, 2))
        union = prob_fg.sum(dim=(1, 2)) + target_fg.sum(dim=(1, 2))
        dice = (2.0 * intersection + self.eps) / (union + self.eps)
        loss = 1.0 - dice
        return loss.mean()


def compute_prediction_loss(logits, target, criterion_ce, criterion_dice):
    return criterion_ce(logits, target) + criterion_dice(logits, target)


def compute_prior_aux_loss(prior_list, target):
    if not prior_list:
        return target.new_tensor(0.0, dtype=torch.float32)

    target_float = target.unsqueeze(1).float()
    prior_loss = target_float.new_tensor(0.0)
    for prior, weight in zip(prior_list, PRIOR_SCALE_WEIGHTS):
        if prior is None:
            continue
        prior_resized = F.interpolate(
            prior.float(), size=target.shape[1:], mode='bilinear', align_corners=True
        )
        prior_resized = prior_resized.clamp(1e-6, 1.0 - 1e-6)
        prior_loss = prior_loss + weight * F.binary_cross_entropy(prior_resized, target_float)
    return PRIOR_AUX_WEIGHT * prior_loss


def compute_supervised_loss_from_outputs(model_outputs, target, criterion_ce, criterion_dice):
    loss_pred = compute_prediction_loss(
        model_outputs.pred, target, criterion_ce, criterion_dice
    )
    loss_aux_1 = compute_prediction_loss(
        model_outputs.aux1, target, criterion_ce, criterion_dice
    )
    loss_aux_2 = compute_prediction_loss(
        model_outputs.aux2, target, criterion_ce, criterion_dice
    )
    loss_aux_3 = compute_prediction_loss(
        model_outputs.aux3, target, criterion_ce, criterion_dice
    )
    loss = (
        AUX_LOSS_WEIGHTS[0] * loss_pred
        + AUX_LOSS_WEIGHTS[1] * loss_aux_1
        + AUX_LOSS_WEIGHTS[2] * loss_aux_2
        + AUX_LOSS_WEIGHTS[3] * loss_aux_3
    )
    if model_outputs.prior_list:
        loss = loss + compute_prior_aux_loss(model_outputs.prior_list, target)
    return loss


def compute_supervised_loss(
    pred,
    aux1=None,
    aux2=None,
    aux3=None,
    target=None,
    criterion_ce=None,
    criterion_dice=None,
    prior_list=None,
):
    if isinstance(pred, ModelOutputs):
        return compute_supervised_loss_from_outputs(pred, aux1, aux2, aux3)

    model_outputs = ModelOutputs(
        pred=pred,
        aux1=aux1,
        aux2=aux2,
        aux3=aux3,
        prior_list=prior_list,
    )
    return compute_supervised_loss_from_outputs(
        model_outputs,
        target,
        criterion_ce,
        criterion_dice,
    )


def unpack_model_outputs(outputs):
    if len(outputs) == 4:
        pred, aux1, aux2, aux3 = outputs
    elif len(outputs) == 5:
        pred, aux1, aux2, aux3, prior_list = outputs
    else:
        raise ValueError(f'Unexpected number of model outputs: {len(outputs)}')
    if len(outputs) == 4:
        prior_list = None
    return ModelOutputs(pred=pred, aux1=aux1, aux2=aux2, aux3=aux3, prior_list=prior_list)


def load_checkpoint_state(ckpt_path, device):
    state = torch.load(ckpt_path, map_location=device)
    if isinstance(state, dict):
        if 'state_dict' in state:
            state = state['state_dict']
        elif 'model' in state:
            state = state['model']
    if not isinstance(state, dict):
        raise TypeError(f'Unsupported checkpoint format: {type(state)}')
    return state


def remap_legacy_pgc_keys(state_dict):
    legacy_prefix_map = {
        'swa.': 'mfrem.pmffm.',
        'global3.': 'mfrem.deep_context3.',
        'global4.': 'mfrem.deep_context4.',
        'cross1.': 'pcim1.',
        'cross2.': 'pcim2.',
        'cross3.': 'pcim3.',
        'cross4.': 'pcim4.',
        'up4.': 'psad.up4.',
        'up3.': 'psad.up3.',
        'up2.': 'psad.up2.',
        'up1.': 'psad.up1.',
        'sp1.': 'psad.sp1.',
        'sp2.': 'psad.sp2.',
        'sp3.': 'psad.sp3.',
        'output_aux_3.': 'psad.output_aux_3.',
        'output_aux_2.': 'psad.output_aux_2.',
        'output_aux_1.': 'psad.output_aux_1.',
        'output.': 'psad.output.',
    }

    remapped = {}
    for key, value in state_dict.items():
        new_key = key
        for old_prefix, new_prefix in legacy_prefix_map.items():
            if key.startswith(old_prefix):
                new_key = new_prefix + key[len(old_prefix):]
                break
        remapped[new_key] = value
    return remapped


def load_model_state_compatible(model, state_dict):
    state_dict = remap_legacy_pgc_keys(state_dict)
    incompatible = model.load_state_dict(state_dict, strict=False)
    missing = list(incompatible.missing_keys)
    unexpected = list(incompatible.unexpected_keys)
    if missing:
        print(f'Checkpoint loaded with {len(missing)} missing keys, usually newly added modules.')
    if unexpected:
        print(f'Checkpoint loaded with {len(unexpected)} unexpected keys.')
    return incompatible


def build_model(model_cls, n_class, device):
    return model_cls(
        n_class,
        use_prior_aux_loss=USE_PRIOR_AUX_LOSS,
    ).to(device)


def predict_labels(logits):
    return F.softmax(logits, dim=1).max(dim=1)[1].data.cpu().numpy()


def evaluate_model(
    net,
    data_loader,
    device,
    criterion_ce,
    criterion_dice,
    n_class,
    local_use_amp,
    hist_sum,
    compute_metrics,
    desc,
):
    total_loss = 0.0
    hist = np.zeros((n_class, n_class), dtype=np.float64)

    with torch.no_grad():
        net.eval()
        for before, after, change in tqdm(data_loader, desc=desc, ncols=100):
            before = before.to(device)
            after = after.to(device)
            change = change.squeeze(dim=1).long().to(device)

            with amp_autocast_context(local_use_amp):
                model_outputs = unpack_model_outputs(net(before, after))
                loss = compute_prediction_loss(
                    model_outputs.pred, change, criterion_ce, criterion_dice
                )

            total_loss += loss.item()
            label_pred = predict_labels(model_outputs.pred)
            label_true = change.data.cpu().numpy()
            hist += hist_sum(label_true, label_pred, n_class)

    _, _, _, _, _, eval_iou, eval_f1 = compute_metrics(hist)
    return total_loss / len(data_loader), eval_iou, eval_f1


def train_one_epoch(
    net,
    train_loader,
    optimizer,
    trainable_params,
    device,
    criterion_ce,
    criterion_dice,
    n_class,
    local_use_amp,
    epoch,
    hist_sum,
    compute_metrics,
    scaler=None,
):
    total_loss = 0.0
    hist = np.zeros((n_class, n_class), dtype=np.float64)

    net.train()
    set_bn_eval(net.backbone)
    if hasattr(net, 'backbone'):
        net.backbone.eval()

    for before, after, change in tqdm(train_loader, desc=f'epoch{epoch}', ncols=100):
        before = before.to(device)
        after = after.to(device)
        change = change.squeeze(dim=1).long().to(device)

        optimizer.zero_grad()

        try:
            with amp_autocast_context(local_use_amp):
                model_outputs = unpack_model_outputs(net(before, after))
                loss = compute_supervised_loss_from_outputs(
                    model_outputs, change, criterion_ce, criterion_dice
                )

            if local_use_amp:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(trainable_params, GRADIENT_CLIP_NORM)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(trainable_params, GRADIENT_CLIP_NORM)
                optimizer.step()
        except RuntimeError as e:
            msg = str(e)
            if 'out of memory' in msg or 'CUDNN_STATUS_MAPPING_ERROR' in msg or 'CUDNN' in msg:
                print(f"OOM or cuDNN error on epoch {epoch}, skipping batch: {msg}")
                try:
                    optimizer.zero_grad()
                except Exception:
                    pass
                try:
                    torch.cuda.empty_cache()
                except Exception as e2:
                    print('empty_cache failed:', e2)
                continue
            raise

        total_loss += loss.item()
        label_pred = predict_labels(model_outputs.pred)
        label_true = change.data.cpu().numpy()
        hist += hist_sum(label_true, label_pred, n_class)

    _, _, _, _, _, train_iou, train_f1 = compute_metrics(hist)
    return total_loss / len(train_loader), train_iou, train_f1


def build_experiment_metadata(
    runtime_config,
    device,
    local_use_amp,
):
    metadata = [
        ('exp_name', runtime_config.exp_name),
        ('seed', runtime_config.seed),
        ('lr', runtime_config.lr),
        ('batch_size', runtime_config.batch_size),
        ('epochs', runtime_config.epochs),
        ('dataset', runtime_config.dataset),
        ('optimizer', 'AdamW'),
        ('loss', 'CrossEntropy+Dice'),
        ('ce_weights', '[1.0, 3.0]'),
        ('aux_weights', '/'.join(str(weight) for weight in AUX_LOSS_WEIGHTS)),
        ('bn_freeze', 'backbone BN eval only'),
        ('model', 'PCINet'),
        ('core_modules', 'MFREM/PCIM/PSAD'),
        ('decoder_fusion', 'static concat'),
        ('diff_mode', 'PCIM'),
        ('use_prior_aux_loss', USE_PRIOR_AUX_LOSS),
        ('prior_aux_weight', PRIOR_AUX_WEIGHT),
        ('ema', False),
        ('resume_ckpt', runtime_config.resume_ckpt if runtime_config.resume_ckpt else 'None'),
        ('f1_init', runtime_config.f1_init),
        ('train_mode', 'resume fine-tune' if runtime_config.resume_ckpt else 'train from scratch'),
        ('AMP enabled', local_use_amp),
        ('device', device),
    ]
    return metadata


def print_experiment_metadata(metadata):
    print('Experiment config:')
    for key, value in metadata:
        print(f'  {key}: {value}')


def build_datasets(dataset_name):
    if dataset_name == 'LEVIR':
        from dataload.LEVIRdataset import LEVIRDataset
        dataset_cls = LEVIRDataset
    elif dataset_name == 'SYSU':
        from dataload.SYSUCDdataset import SYSUCDDataset
        dataset_cls = SYSUCDDataset
    elif dataset_name == 'GZCDD':
        from dataload.GZCDDdataset import GZCDDDataset
        dataset_cls = GZCDDDataset
    else:
        raise ValueError(f'Unsupported dataset: {dataset_name}')

    return dataset_cls(mode='train'), dataset_cls(mode='test')


def write_best_result(best_result_path, epoch, test_f1, test_iou, runtime_config, metadata):
    with open(best_result_path, 'w', encoding='utf-8') as f_best:
        f_best.write(f'best epoch: {epoch}\n')
        f_best.write(f'best F1: {test_f1:.6f}\n')
        f_best.write(f'best IoU: {test_iou:.6f}\n')
        f_best.write(f'lr: {runtime_config.lr}\n')
        f_best.write(f'batch size: {runtime_config.batch_size}\n')
        f_best.write(f'seed: {runtime_config.seed}\n')
        for key, value in metadata:
            if key in {'seed', 'lr', 'batch_size'}:
                continue
            f_best.write(f'{key}: {value}\n')


def main():
    args = parse_args()
    config = resolve_config(args)
    set_seed(config.seed)

    sys.path.append(os.path.join(os.path.dirname(__file__), 'dataload'))
    train_data, test_data = build_datasets(config.dataset)

    num_workers = 2 if os.name != 'nt' else 0
    pin_memory = torch.cuda.is_available()

    train_loader = DataLoader(
        train_data, batch_size=config.batch_size, shuffle=True, drop_last=True,
        num_workers=num_workers, pin_memory=pin_memory
    )
    test_loader = DataLoader(
        test_data, batch_size=config.batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=pin_memory
    )

    n_class = 2
    F1_max = config.f1_init

    root = os.path.join(os.path.dirname(__file__), 'results', config.exp_name)
    os.makedirs(root, exist_ok=True)

    from models.pgc_cdnet import PCINet
    model_cls = PCINet
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if device.type == 'cuda' and os.environ.get('DISABLE_CUDNN', '0') == '1':
        torch.backends.cudnn.enabled = False
        torch.backends.cudnn.benchmark = False
        print('cuDNN disabled via DISABLE_CUDNN=1')

    net = build_model(model_cls, n_class, device)
    if config.resume_ckpt:
        if not os.path.isfile(config.resume_ckpt):
            raise FileNotFoundError(f'resume_ckpt not found: {config.resume_ckpt}')
        resume_state = load_checkpoint_state(config.resume_ckpt, device)
        load_model_state_compatible(net, resume_state)

    class_weights = torch.tensor([1.0, 3.0], device=device)
    criterion_ce = nn.CrossEntropyLoss(weight=class_weights).to(device)
    criterion_dice = DiceLoss().to(device)

    trainable_params = [p for p in net.parameters() if p.requires_grad]
    optimizer = optim.AdamW(trainable_params, lr=config.lr, weight_decay=WEIGHT_DECAY)

    from poly import adjust_learning_rate_poly
    from assess import hist_sum, compute_metrics

    local_use_amp = USE_AMP
    if os.environ.get('DISABLE_AMP', '0') == '1':
        local_use_amp = False
        print('AMP disabled via DISABLE_AMP=1')

    metadata = build_experiment_metadata(
        config,
        device,
        local_use_amp,
    )
    print_experiment_metadata(metadata)

    scaler = torch.cuda.amp.GradScaler() if local_use_amp else None

    if config.resume_ckpt:
        pre_eval_loss, pre_eval_iou, pre_eval_f1 = evaluate_model(
            net,
            test_loader,
            device,
            criterion_ce,
            criterion_dice,
            n_class,
            local_use_amp,
            hist_sum,
            compute_metrics,
            desc='resume pre-eval',
        )

        print('Resume checkpoint pre-eval:')
        print(f'  test loss: {pre_eval_loss:.4f}')
        print(f'  F1: {pre_eval_f1:.4f}')
        print(f'  IoU: {pre_eval_iou:.4f}')

        with open(os.path.join(root, 'pre_eval.txt'), 'w', encoding='utf-8') as f_pre_eval:
            f_pre_eval.write('Resume checkpoint pre-eval:\n')
            f_pre_eval.write(f'test loss: {pre_eval_loss:.6f}\n')
            f_pre_eval.write(f'F1: {pre_eval_f1:.6f}\n')
            f_pre_eval.write(f'IoU: {pre_eval_iou:.6f}\n')

    with open(os.path.join(root, 'train.txt'), 'w', encoding='utf-8') as f_train, \
         open(os.path.join(root, 'test.txt'), 'w', encoding='utf-8') as f_test:

        for epoch in range(config.epochs):
            new_lr = adjust_learning_rate_poly(optimizer, epoch, config.epochs, config.lr, 0.9)
            trainloss, train_iou, train_f1 = train_one_epoch(
                net,
                train_loader,
                optimizer,
                trainable_params,
                device,
                criterion_ce,
                criterion_dice,
                n_class,
                local_use_amp,
                epoch,
                hist_sum,
                compute_metrics,
                scaler=scaler,
            )
            testloss, test_iou, test_f1 = evaluate_model(
                net,
                test_loader,
                device,
                criterion_ce,
                criterion_dice,
                n_class,
                local_use_amp,
                hist_sum,
                compute_metrics,
                desc=f'epoch{epoch}(test)',
            )

            print(f'Epoch: {epoch} | lr: {new_lr:.6f}')
            print(f'  Train - loss: {trainloss:.4f} | F1: {train_f1:.4f} | IoU: {train_iou:.4f}')
            print(f'  Test  - loss: {testloss:.4f} | F1: {test_f1:.4f} | IoU: {test_iou:.4f}')
            print(f'  Best F1: {F1_max:.4f}')

            f_train.write(
                'Epoch:%d|train loss:%0.04f|train F1:%0.04f|train iou:%0.04f|lr:%0.06f\n' % (
                    epoch, trainloss, train_f1, train_iou, new_lr))
            f_train.flush()

            f_test.write(
                'Epoch:%d|test loss:%0.04f|test F1:%0.04f|test iou:%0.04f|lr:%0.06f\n' % (
                    epoch, testloss, test_f1, test_iou, new_lr))
            f_test.flush()

            if test_f1 > F1_max and test_f1 >= MIN_SAVE_F1:
                save_path = os.path.join(root, f'F1_{test_f1:.4f}_iou_{test_iou:.4f}_epoch_{epoch}.pth')
                torch.save(net.state_dict(), save_path)
                write_best_result(
                    os.path.join(root, 'best_result.txt'),
                    epoch,
                    test_f1,
                    test_iou,
                    config,
                    metadata,
                )
                print(f'  -> model saved: {save_path}')
                F1_max = test_f1
            elif test_f1 > F1_max:
                F1_max = test_f1
                print(f'  -> best F1 updated but not saved because it is below MIN_SAVE_F1={MIN_SAVE_F1:.4f}')


if __name__ == '__main__':
    import multiprocessing as mp
    mp.freeze_support()
    main()

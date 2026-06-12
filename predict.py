import argparse
import os
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont
import torch
import torch.nn.functional as F

from assess import compute_metrics, hist_sum
from models import MobileNetV2
from models.pgc_cdnet import (
    CBAM_Attention,
    Channel_Attention,
    Double_conv,
    LKABlock,
    PCINet,
    PMFFMCore,
    MSConv3x3,
    h_swish,
)
from train import (
    amp_autocast_context,
    load_checkpoint_state,
    load_model_state_compatible,
    normalize_dataset_name,
    unpack_model_outputs,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description='PCINet prediction and paper-style visualization script.'
    )
    parser.add_argument('--ckpt', required=True, help='Path to trained .pth checkpoint.')
    parser.add_argument('--dataset', default='LEVIR', help='LEVIR, SYSU, or GZCDD.')
    parser.add_argument('--root', default=None, help='Optional dataset root for SYSU/GZCDD.')
    parser.add_argument('--split', default='test', choices=['train', 'test', 'val'])
    parser.add_argument('--output-dir', default='results_predict')
    parser.add_argument('--max-samples', type=int, default=None)
    parser.add_argument('--device', default=None)
    parser.add_argument('--disable-amp', action='store_true')
    parser.add_argument('--no-prior', action='store_true', help='Do not request PCIM prior outputs.')
    parser.add_argument('--save-prob', action='store_true', default=True)
    parser.add_argument('--save-overlay', action='store_true', default=True)
    parser.add_argument('--save-panels', action='store_true', default=True)
    return parser.parse_args()


def build_dataset(dataset_name, split, root):
    mode = 'train' if split == 'train' else split
    if dataset_name == 'LEVIR':
        from dataload.LEVIRdataset import LEVIRDataset
        return LEVIRDataset(mode='train' if mode == 'train' else 'test')
    if dataset_name == 'SYSU':
        from dataload.SYSUCDdataset import SYSUCDDataset
        return SYSUCDDataset(mode=mode, root=root)
    if dataset_name == 'GZCDD':
        from dataload.GZCDDdataset import GZCDDDataset
        return GZCDDDataset(mode=mode, root=root)
    raise ValueError(f'Unsupported dataset: {dataset_name}')


def get_sample_paths(dataset, index):
    if hasattr(dataset, 'before_paths'):
        return (
            Path(dataset.before_paths[index]),
            Path(dataset.after_paths[index]),
            Path(dataset.change_paths[index]),
        )

    if getattr(dataset, 'mode', 'test') == 'train':
        return (
            Path(dataset.train_dataset_before[index]),
            Path(dataset.train_dataset_after[index]),
            Path(dataset.train_dataset_change[index]),
        )
    return (
        Path(dataset.test_dataset_before[index]),
        Path(dataset.test_dataset_after[index]),
        Path(dataset.test_dataset_change[index]),
    )


def ensure_dirs(root, save_prior):
    names = [
        'mask',
        'gt',
        'prob_gray',
        'prob_heatmap',
        'error_map',
        'overlay_pred',
        'compare_panel',
    ]
    if save_prior:
        for idx in range(1, 5):
            names.extend([f'prior{idx}_gray', f'prior{idx}_heatmap'])
        names.append('prior_panel')

    for name in names:
        (root / name).mkdir(parents=True, exist_ok=True)


def to_uint8_mask(mask):
    return (mask.astype(np.uint8) * 255)


def normalize_map(values):
    values = np.asarray(values, dtype=np.float32)
    values = np.nan_to_num(values, nan=0.0, posinf=1.0, neginf=0.0)
    return np.clip(values, 0.0, 1.0)


def apply_heatmap(values):
    values = normalize_map(values)
    stops = np.array(
        [
            [0, 0, 80],
            [0, 80, 255],
            [0, 220, 220],
            [255, 220, 0],
            [255, 0, 0],
        ],
        dtype=np.float32,
    )
    scaled = values * (len(stops) - 1)
    left = np.floor(scaled).astype(np.int32)
    right = np.clip(left + 1, 0, len(stops) - 1)
    alpha = (scaled - left)[..., None]
    rgb = stops[left] * (1.0 - alpha) + stops[right] * alpha
    return rgb.astype(np.uint8)


def make_error_map(gt, pred):
    gt = gt.astype(bool)
    pred = pred.astype(bool)
    canvas = np.zeros((*gt.shape, 3), dtype=np.uint8)
    canvas[(gt == 1) & (pred == 1)] = [255, 255, 255]  # TP: white
    canvas[(gt == 0) & (pred == 0)] = [0, 0, 0]        # TN: black
    canvas[(gt == 0) & (pred == 1)] = [255, 0, 0]      # FP: red
    canvas[(gt == 1) & (pred == 0)] = [0, 255, 0]      # FN: green
    return canvas


def overlay_prediction(image, pred, color=(255, 0, 0), alpha=0.45):
    base = np.asarray(image.convert('RGB')).astype(np.float32)
    mask = pred.astype(bool)
    overlay = base.copy()
    overlay[mask] = (1.0 - alpha) * overlay[mask] + alpha * np.array(color, dtype=np.float32)
    return overlay.astype(np.uint8)


def load_raw_image(path, size):
    image = Image.open(path).convert('RGB')
    if image.size != size:
        image = image.resize(size, Image.BILINEAR)
    return image


def image_with_title(image, title, title_height=28):
    image = image.convert('RGB')
    canvas = Image.new('RGB', (image.width, image.height + title_height), 'white')
    canvas.paste(image, (0, title_height))
    draw = ImageDraw.Draw(canvas)
    try:
        font = ImageFont.truetype('arial.ttf', 14)
    except OSError:
        font = ImageFont.load_default()
    bbox = draw.textbbox((0, 0), title, font=font)
    text_w = bbox[2] - bbox[0]
    draw.text(((image.width - text_w) // 2, 6), title, fill='black', font=font)
    return canvas


def make_panel(items, gap=8):
    titled = [image_with_title(image, title) for title, image in items]
    width = sum(image.width for image in titled) + gap * (len(titled) - 1)
    height = max(image.height for image in titled)
    canvas = Image.new('RGB', (width, height), 'white')
    x = 0
    for image in titled:
        canvas.paste(image, (x, 0))
        x += image.width + gap
    return canvas


def save_image(path, array_or_image):
    path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(array_or_image, Image.Image):
        image = array_or_image
    else:
        image = Image.fromarray(array_or_image)
    image.save(path)


class LegacyCoordChannelAtt(torch.nn.Module):
    def __init__(self, inp, reduction=4):
        super().__init__()
        self.cha = Channel_Attention(inp, reduction=reduction)
        self.pool_h = torch.nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = torch.nn.AdaptiveAvgPool2d((1, None))

        mip = max(8, inp // reduction)
        self.conv1 = torch.nn.Conv2d(inp, mip, kernel_size=1, stride=1, padding=0)
        self.conv1_ = torch.nn.Conv2d(mip, inp, kernel_size=1, stride=1, padding=0)
        self.conv = torch.nn.Conv2d(inp, inp, kernel_size=1, stride=1, padding=0)
        self.bn1 = torch.nn.BatchNorm2d(mip)
        self.act = h_swish()
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, x):
        identity = x
        cha = self.cha(x)

        _, _, h, w = x.size()
        x_h = self.pool_h(x)
        x_w = self.pool_w(x).permute(0, 1, 3, 2)

        y = torch.cat([x_h, x_w], dim=2)
        y = self.conv1(y)
        y = self.bn1(y)
        y = self.act(y)
        y = self.conv1_(y)

        x_h, x_w = torch.split(y, [h, w], dim=2)
        x_w = x_w.permute(0, 1, 3, 2)

        a_h = self.sigmoid(x_h)
        a_w = self.sigmoid(x_w)
        out = identity * a_w * a_h
        return out + cha


class LegacySupervisionAttentionModule(torch.nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.SA_Block = LegacyCoordChannelAtt(in_channels)
        self.conv1 = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels, in_channels, kernel_size=1),
            torch.nn.BatchNorm2d(in_channels),
            torch.nn.ReLU(inplace=True),
        )
        self.conv1_ = torch.nn.Sequential(
            torch.nn.Conv2d(2 * in_channels, in_channels, kernel_size=1),
            torch.nn.BatchNorm2d(in_channels),
            torch.nn.ReLU(inplace=True),
        )
        self.dwconv = torch.nn.Sequential(
            torch.nn.Conv2d(
                in_channels, in_channels, kernel_size=3, stride=1, padding=1,
                groups=in_channels
            ),
            torch.nn.BatchNorm2d(in_channels),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(in_channels, in_channels, kernel_size=1),
            torch.nn.BatchNorm2d(in_channels),
            torch.nn.ReLU(inplace=True),
        )

    def forward(self, x):
        mask_a = self.conv1(x)
        mask_a1 = self.dwconv(mask_a)
        mask_a_ = self.SA_Block(mask_a)
        mask_b_ = self.SA_Block(mask_a1)
        mask = torch.cat([mask_a_, mask_b_], 1)
        mask_ = self.conv1_(mask)
        mask_b1 = self.dwconv(mask_)
        deep_flow = mask_a + mask_b1
        out = self.conv1(deep_flow)
        return out, deep_flow


class LegacyMyAttention(torch.nn.Module):
    def __init__(self, in_ch, reduction=16):
        super().__init__()
        self.cbam = CBAM_Attention(in_ch, reduction=reduction)
        self.msconv = MSConv3x3(in_ch, dilation=3)
        self.conv = torch.nn.Sequential(
            torch.nn.Conv2d(in_ch * 2, in_ch, kernel_size=1),
            torch.nn.BatchNorm2d(in_ch),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(in_ch, in_ch, kernel_size=3, padding=1, stride=1, bias=False),
            torch.nn.BatchNorm2d(in_ch),
        )
        self.attention = torch.nn.Sequential(
            torch.nn.Conv2d(in_ch * 2, 1, kernel_size=1),
            torch.nn.BatchNorm2d(1),
            torch.nn.Sigmoid(),
        )
        self.conv1 = torch.nn.Sequential(
            torch.nn.Conv2d(2 * in_ch, in_ch, 1),
            torch.nn.BatchNorm2d(in_ch),
            torch.nn.ReLU(inplace=True),
        )
        self.conv1_ = torch.nn.Sequential(
            torch.nn.Conv2d(in_ch, in_ch, 1),
            torch.nn.BatchNorm2d(in_ch),
            torch.nn.ReLU(inplace=True),
        )

    def forward(self, x):
        out = self.cbam(x)
        out = torch.cat([out, x], dim=1)
        out = out * self.attention(out)
        out = self.conv1(out)
        out = self.msconv(out)
        deepflow = out + x
        out = self.conv1_(deepflow)
        return out, deepflow


class LegacyCCIM(torch.nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        mid_channels = max(8, in_channels // 4)

        self.diff_branch = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels, mid_channels, kernel_size=1, bias=False),
            torch.nn.GroupNorm(1, mid_channels),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(
                mid_channels, mid_channels, kernel_size=3, stride=1, padding=1,
                groups=mid_channels, bias=False
            ),
            torch.nn.GroupNorm(1, mid_channels),
            torch.nn.ReLU(inplace=True),
        )
        self.prior_branch = torch.nn.Sequential(
            torch.nn.Conv2d(1, mid_channels, kernel_size=1, bias=False),
            torch.nn.GroupNorm(1, mid_channels),
            torch.nn.ReLU(inplace=True),
        )
        self.fuse_gate = torch.nn.Sequential(
            torch.nn.Conv2d(mid_channels * 2, mid_channels, kernel_size=1, bias=False),
            torch.nn.GroupNorm(1, mid_channels),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(
                mid_channels, mid_channels, kernel_size=3, stride=1, padding=1,
                groups=mid_channels, bias=False
            ),
            torch.nn.GroupNorm(1, mid_channels),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(mid_channels, mid_channels, kernel_size=1, bias=False),
            torch.nn.GroupNorm(1, mid_channels),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(mid_channels, in_channels, kernel_size=1, bias=False),
            torch.nn.Sigmoid(),
        )
        self.context_conv = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels, mid_channels, kernel_size=1, bias=False),
            torch.nn.GroupNorm(1, mid_channels),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(
                mid_channels, mid_channels, kernel_size=3, stride=1, padding=1,
                groups=mid_channels, bias=False
            ),
            torch.nn.GroupNorm(1, mid_channels),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(mid_channels, in_channels, kernel_size=1, bias=False),
            torch.nn.GroupNorm(1, in_channels),
            torch.nn.ReLU(inplace=True),
        )
        self.refine = torch.nn.Sequential(
            torch.nn.Conv2d(
                in_channels, in_channels, kernel_size=3, stride=1, padding=1,
                groups=in_channels, bias=False
            ),
            torch.nn.GroupNorm(1, in_channels),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(in_channels, in_channels, kernel_size=1, bias=False),
            torch.nn.GroupNorm(1, in_channels),
        )
        self.last_prior = None

    def forward(self, f1, f2):
        diff = torch.abs(f1 - f2)
        diff_feat = self.diff_branch(diff)

        norm_f1 = F.normalize(f1, p=2, dim=1, eps=1e-6)
        norm_f2 = F.normalize(f2, p=2, dim=1, eps=1e-6)
        corr = F.cosine_similarity(norm_f1, norm_f2, dim=1).unsqueeze(1)
        prior = torch.clamp(1.0 - corr, 0.0, 2.0)
        prior = F.avg_pool2d(prior, kernel_size=3, stride=1, padding=1)
        self.last_prior = torch.clamp(prior / 2.0, 0.0, 1.0)

        prior_feat = self.prior_branch(prior)
        gate = self.fuse_gate(torch.cat([diff_feat, prior_feat], dim=1))

        ctx2 = self.context_conv(f2)
        ctx1 = self.context_conv(f1)
        out1 = f1 + gate * ctx2
        out2 = f2 + gate * ctx1
        out1 = out1 + self.refine(out1)
        out2 = out2 + self.refine(out2)
        return out1, out2


class LegacyCheckpointPCINet(torch.nn.Module):
    def __init__(self, num_classes=2, return_prior=True):
        super().__init__()
        self.return_prior = return_prior
        self.backbone = MobileNetV2.mobilenet_v2(pretrained=True)
        self.swa = PMFFMCore([16, 24, 32, 96, 320], 64)
        self.global3 = LKABlock(128)
        self.global4 = LKABlock(256)

        self.cross1 = LegacyCCIM(32)
        self.cross2 = LegacyCCIM(64)
        self.cross3 = LegacyCCIM(128)
        self.cross4 = LegacyCCIM(256)

        self.up4 = torch.nn.Sequential(
            Double_conv(256, 128),
            torch.nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
        )
        self.up3 = torch.nn.Sequential(
            Double_conv(128, 64),
            torch.nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
        )
        self.up2 = torch.nn.Sequential(
            Double_conv(64, 32),
            torch.nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
        )
        self.up1 = torch.nn.Sequential(
            Double_conv(224, 112),
            torch.nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
        )

        self.sp1 = LegacySupervisionAttentionModule(64)
        self.sp2 = LegacyMyAttention(128)
        self.sp3 = LegacyMyAttention(256)

        self.output_aux_3 = torch.nn.Sequential(
            torch.nn.Conv2d(256, 128, kernel_size=1),
            torch.nn.BatchNorm2d(128),
            torch.nn.ReLU(True),
            torch.nn.Dropout2d(0.3),
            torch.nn.Conv2d(128, num_classes, kernel_size=1),
        )
        self.output_aux_2 = torch.nn.Sequential(
            torch.nn.Conv2d(128, 64, kernel_size=1),
            torch.nn.BatchNorm2d(64),
            torch.nn.ReLU(True),
            torch.nn.Dropout2d(0.3),
            torch.nn.Conv2d(64, num_classes, kernel_size=1),
        )
        self.output_aux_1 = torch.nn.Sequential(
            torch.nn.Conv2d(64, 32, kernel_size=1),
            torch.nn.BatchNorm2d(32),
            torch.nn.ReLU(True),
            torch.nn.Dropout2d(0.3),
            torch.nn.Conv2d(32, num_classes, kernel_size=1),
        )
        self.output = torch.nn.Sequential(
            torch.nn.Conv2d(112, 56, kernel_size=1, bias=False),
            torch.nn.BatchNorm2d(56),
            torch.nn.ReLU(True),
            torch.nn.Dropout2d(0.3),
            torch.nn.Conv2d(56, num_classes, kernel_size=1, bias=False),
        )

    def forward(self, x1, x2):
        h, w = x1.shape[2:]
        x1_layer0, x1_layer1, x1_layer2, x1_layer3, x1_layer4 = self.backbone(x1)
        x2_layer0, x2_layer1, x2_layer2, x2_layer3, x2_layer4 = self.backbone(x2)

        x1_layer1, x1_layer2, x1_layer3, x1_layer4 = self.swa(
            x1_layer0, x1_layer1, x1_layer2, x1_layer3, x1_layer4
        )
        x2_layer1, x2_layer2, x2_layer3, x2_layer4 = self.swa(
            x2_layer0, x2_layer1, x2_layer2, x2_layer3, x2_layer4
        )

        x1_layer3 = self.global3(x1_layer3)
        x2_layer3 = self.global3(x2_layer3)
        x1_layer4 = self.global4(x1_layer4)
        x2_layer4 = self.global4(x2_layer4)

        inter1_a, inter1_b = self.cross1(x1_layer1, x2_layer1)
        inter2_a, inter2_b = self.cross2(x1_layer2, x2_layer2)
        inter3_a, inter3_b = self.cross3(x1_layer3, x2_layer3)
        inter4_a, inter4_b = self.cross4(x1_layer4, x2_layer4)

        sub_layer1 = torch.abs(inter1_a - inter1_b)
        sub_layer2 = torch.abs(inter2_a - inter2_b)
        sub_layer3 = torch.abs(inter3_a - inter3_b)
        sub_layer4 = torch.abs(inter4_a - inter4_b)

        sp3, aux3 = self.sp3(sub_layer4)
        up4 = self.up4(sp3)
        aux_3 = self.output_aux_3(aux3)

        add3 = sub_layer3 + up4
        sp2, aux2 = self.sp2(add3)
        up3 = self.up3(sp2)
        aux_2 = self.output_aux_2(aux2)

        add2 = sub_layer2 + up3
        sp1, aux1 = self.sp1(add2)
        up2 = self.up2(sp1)
        aux_1 = self.output_aux_1(aux1)

        add1 = sub_layer1 + up2
        out = torch.cat(
            [
                F.interpolate(add3, add1.shape[2:], mode='bilinear', align_corners=True),
                F.interpolate(add2, add1.shape[2:], mode='bilinear', align_corners=True),
                add1,
            ],
            dim=1,
        )
        out = self.up1(out)

        main = F.interpolate(self.output(out), size=(h, w), mode='bilinear', align_corners=True)
        aux_1 = F.interpolate(aux_1, size=(h, w), mode='bilinear', align_corners=True)
        aux_2 = F.interpolate(aux_2, size=(h, w), mode='bilinear', align_corners=True)
        aux_3 = F.interpolate(aux_3, size=(h, w), mode='bilinear', align_corners=True)

        if self.return_prior:
            prior_list = [
                self.cross1.last_prior,
                self.cross2.last_prior,
                self.cross3.last_prior,
                self.cross4.last_prior,
            ]
            return main, aux_1, aux_2, aux_3, prior_list
        return main, aux_1, aux_2, aux_3


def is_legacy_ccim_checkpoint(state):
    return any(key.startswith('cross1.diff_branch') for key in state)


def build_model(ckpt_path, device, save_prior):
    state = load_checkpoint_state(ckpt_path, device)
    if is_legacy_ccim_checkpoint(state):
        print('Detected legacy CCIM checkpoint; using legacy prediction model.')
        model = LegacyCheckpointPCINet(num_classes=2, return_prior=save_prior).to(device)
        incompatible = model.load_state_dict(state, strict=False)
        missing = list(incompatible.missing_keys)
        unexpected = list(incompatible.unexpected_keys)
        if missing:
            print(f'Legacy checkpoint loaded with {len(missing)} missing keys.')
        if unexpected:
            print(f'Legacy checkpoint loaded with {len(unexpected)} unexpected keys.')
    else:
        model = PCINet(num_classes=2, use_prior_aux_loss=save_prior).to(device)
        load_model_state_compatible(model, state)
    model.eval()
    return model


def write_metrics(path, hist):
    miou, oa, kappa, precision, recall, iou, f1 = compute_metrics(hist)
    lines = [
        f'mIoU: {miou:.6f}',
        f'OA: {oa:.6f}',
        f'Kappa: {kappa:.6f}',
        f'Precision: {precision:.6f}',
        f'Recall: {recall:.6f}',
        f'IoU: {iou:.6f}',
        f'F1: {f1:.6f}',
        '',
        'Percent format:',
        f'OA: {oa * 100:.4f}',
        f'Kappa: {kappa * 100:.4f}',
        f'Precision: {precision * 100:.4f}',
        f'Recall: {recall * 100:.4f}',
        f'IoU: {iou * 100:.4f}',
        f'F1: {f1 * 100:.4f}',
    ]
    path.write_text('\n'.join(lines) + '\n', encoding='utf-8')


def main():
    args = parse_args()
    dataset_name = normalize_dataset_name(args.dataset)
    save_prior = not args.no_prior
    device = torch.device(
        args.device if args.device else ('cuda' if torch.cuda.is_available() else 'cpu')
    )
    use_amp = device.type == 'cuda' and not args.disable_amp

    dataset = build_dataset(dataset_name, args.split, args.root)
    output_root = Path(args.output_dir) / dataset_name.lower() / Path(args.ckpt).stem
    ensure_dirs(output_root, save_prior)

    model = build_model(args.ckpt, device, save_prior)
    hist = np.zeros((2, 2), dtype=np.float64)
    total = len(dataset) if args.max_samples is None else min(len(dataset), args.max_samples)

    with torch.no_grad():
        for index in range(total):
            before_tensor, after_tensor, change_tensor = dataset[index]
            before_path, after_path, change_path = get_sample_paths(dataset, index)
            name = before_path.stem + '.png'

            before = before_tensor.unsqueeze(0).to(device)
            after = after_tensor.unsqueeze(0).to(device)

            with amp_autocast_context(use_amp):
                outputs = unpack_model_outputs(model(before, after))
                logits = outputs.pred
                prob = F.softmax(logits, dim=1)[0, 1].float().cpu().numpy()
                pred = logits.argmax(dim=1)[0].byte().cpu().numpy()

            gt = (change_tensor.squeeze(0).numpy() > 0.5).astype(np.uint8)
            hist += hist_sum(np.expand_dims(gt, 0), np.expand_dims(pred, 0), 2)

            h, w = pred.shape
            size = (w, h)
            raw_before = load_raw_image(before_path, size)
            raw_after = load_raw_image(after_path, size)

            gt_mask = to_uint8_mask(gt)
            pred_mask = to_uint8_mask(pred)
            prob_gray = (normalize_map(prob) * 255).astype(np.uint8)
            prob_heatmap = apply_heatmap(prob)
            error_map = make_error_map(gt, pred)
            overlay = overlay_prediction(raw_after, pred)

            save_image(output_root / 'gt' / name, gt_mask)
            save_image(output_root / 'mask' / name, pred_mask)
            save_image(output_root / 'prob_gray' / name, prob_gray)
            save_image(output_root / 'prob_heatmap' / name, prob_heatmap)
            save_image(output_root / 'error_map' / name, error_map)
            save_image(output_root / 'overlay_pred' / name, overlay)

            if args.save_panels:
                compare_items = [
                    ('T1', raw_before),
                    ('T2', raw_after),
                    ('GT', Image.fromarray(gt_mask).convert('RGB')),
                    ('PCINet', Image.fromarray(pred_mask).convert('RGB')),
                    ('TP/TN/FP/FN', Image.fromarray(error_map)),
                    ('Prob', Image.fromarray(prob_heatmap)),
                ]
                save_image(output_root / 'compare_panel' / name, make_panel(compare_items))

            if save_prior and outputs.prior_list:
                prior_items = [
                    ('T1', raw_before),
                    ('T2', raw_after),
                    ('GT', Image.fromarray(gt_mask).convert('RGB')),
                ]
                for prior_idx, prior in enumerate(outputs.prior_list, 1):
                    prior_resized = F.interpolate(
                        prior.float(), size=(h, w), mode='bilinear', align_corners=True
                    )[0, 0].clamp(0.0, 1.0).cpu().numpy()
                    prior_gray = (normalize_map(prior_resized) * 255).astype(np.uint8)
                    prior_heatmap = apply_heatmap(prior_resized)
                    save_image(output_root / f'prior{prior_idx}_gray' / name, prior_gray)
                    save_image(output_root / f'prior{prior_idx}_heatmap' / name, prior_heatmap)
                    prior_items.append((f'Prior-{prior_idx}', Image.fromarray(prior_heatmap)))

                prior_items.append(('Pred', Image.fromarray(pred_mask).convert('RGB')))
                if args.save_panels:
                    save_image(output_root / 'prior_panel' / name, make_panel(prior_items))

            if (index + 1) % 20 == 0 or index + 1 == total:
                print(f'Predicted {index + 1}/{total} samples')

    write_metrics(output_root / 'metrics.txt', hist)
    print(f'Prediction outputs saved to: {output_root}')


if __name__ == '__main__':
    main()

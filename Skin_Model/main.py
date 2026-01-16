# main.py (corrected)
# Requires: torch, torchvision, timm (optional), scikit-learn, matplotlib
# pip install torch torchvision timm scikit-learn matplotlib

import os
import argparse
from collections import Counter
import numpy as np
from PIL import Image
import copy
import time
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, WeightedRandomSampler, random_split
from torchvision import transforms, datasets, models
from sklearn.metrics import f1_score, balanced_accuracy_score, confusion_matrix, classification_report

# ---------------------------
# Utilities
# ---------------------------
def get_class_counts(folder):
    classes = sorted([d for d in os.listdir(folder) if os.path.isdir(os.path.join(folder, d))])
    counts = []
    for c in classes:
        p = os.path.join(folder, c)
        n = sum(1 for _ in os.scandir(p) if _.is_file())
        counts.append(n)
    return classes, counts

# ---------------------------
# TokenMixerBlock (Vision Mamba inspired minimal version)
# ---------------------------
class TokenMixerBlock(nn.Module):
    def __init__(self, in_channels, token_dim=128, kernel_size=3):
        super().__init__()
        self.proj = nn.Conv2d(in_channels, token_dim, kernel_size=1)
        self.token_mixer = nn.Conv1d(token_dim, token_dim, kernel_size=kernel_size, padding=kernel_size//2)
        self.channel_mlp = nn.Sequential(
            nn.Conv2d(token_dim, token_dim, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(token_dim, token_dim, kernel_size=1),
        )
        self.rescale = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        # x: (B, C, H, W)
        x = self.proj(x)  # -> (B, token_dim, H, W)
        B, D, H, W = x.shape
        tokens = x.view(B, D, H*W)                # (B, D, N)
        tokens = self.token_mixer(tokens)         # (B, D, N)
        tokens = tokens.view(B, D, H, W)
        cm = self.channel_mlp(tokens)
        out = x + self.rescale * cm
        return out

# ---------------------------
# MultiScalePooling head (FIXED total_channels calculation)
# ---------------------------
class MultiScalePoolingHead(nn.Module):
    def __init__(self, in_channels, out_dim=512, pool_sizes=(1,2,4)):
        super().__init__()
        self.pool_sizes = pool_sizes
        self.convs = nn.ModuleList([
            nn.Sequential(
                nn.AdaptiveAvgPool2d((s,s)),
                nn.Conv2d(in_channels, in_channels//2, kernel_size=1),
                nn.ReLU(inplace=True)
            )
            for s in pool_sizes
        ])
        # correct total flattened channels: for each scale s, channels = (in_channels//2) * (s*s)
        total_channels = sum([(in_channels//2) * (s * s) for s in pool_sizes])
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(total_channels, out_dim),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        feats = [conv(x) for conv in self.convs]
        feats = [f.view(f.size(0), -1) for f in feats]
        cat = torch.cat(feats, dim=1)
        return self.fc(cat)

# ---------------------------
# Student network (ResNet18 backbone + TokenMixer + MultiScale head)
# ---------------------------
class StudentNet(nn.Module):
    def __init__(self, num_classes, pretrained=True):
        super().__init__()
        backbone = models.resnet18(pretrained=pretrained)
        # remove final fc and avgpool
        self.stem = nn.Sequential(*list(backbone.children())[:-2])  # up to layer4 output (B,512,H,W)
        # token mixer reduces channels from 512 -> 256
        self.token_mixer_block = TokenMixerBlock(in_channels=512, token_dim=256, kernel_size=3)
        self.ms_head = MultiScalePoolingHead(in_channels=256, out_dim=512, pool_sizes=(1,2,4))
        self.classifier = nn.Linear(512, num_classes)

    def forward(self, x):
        x = self.stem(x)                 # (B,512,H,W)
        x = self.token_mixer_block(x)    # (B,256,H,W)
        features = self.ms_head(x)       # (B,512)
        logits = self.classifier(features)
        return logits, features

# ---------------------------
# Teacher loader helper (generic)
# ---------------------------
def build_teacher(name='resnet50', num_classes=None, pretrained=True):
    if name.lower().startswith('resnet'):
        t = getattr(models, name)(pretrained=pretrained)
        if num_classes:
            t.fc = nn.Linear(t.fc.in_features, num_classes)
    else:
        t = models.resnet50(pretrained=pretrained)
        if num_classes:
            t.fc = nn.Linear(t.fc.in_features, num_classes)
    return t

# ---------------------------
# KD Loss helpers: CE + KL
# ---------------------------
def kd_loss_fn(student_logits, teacher_logits, targets, T=4.0, alpha=0.5):
    ce = F.cross_entropy(student_logits, targets)
    p_s = F.log_softmax(student_logits / T, dim=1)
    p_t = F.softmax(teacher_logits / T, dim=1)
    kl = F.kl_div(p_s, p_t, reduction='batchmean') * (T*T)
    return alpha * kl + (1.0 - alpha) * ce, {'ce': ce.item(), 'kl': kl.item()}

def jgekd_proxy_loss(student_logits, teacher_logits):
    with torch.no_grad():
        t_probs = F.softmax(teacher_logits, dim=1)  # (B,C)
        t_corr = (t_probs.t() @ t_probs) / (t_probs.size(0)+1e-8)
    s_probs = F.softmax(student_logits, dim=1)
    s_corr = (s_probs.t() @ s_probs) / (s_probs.size(0)+1e-8)
    return F.mse_loss(s_corr, t_corr)

# ---------------------------
# Training / Eval loops
# ---------------------------
def train_one_epoch(epoch, model_s, model_t, opt, loader, device, T=4.0, alpha=0.7, use_jgekd=False, jgekd_weight=0.1):
    model_s.train()
    if model_t: model_t.eval()
    running_loss = 0.0
    all_preds, all_targets = [], []
    running_stats = {'ce':0.0, 'kl':0.0}
    for imgs, targets in loader:
        imgs = imgs.to(device)
        targets = targets.to(device)
        opt.zero_grad()
        s_logits, _ = model_s(imgs)
        if model_t:
            with torch.no_grad():
                t_logits = model_t(imgs)
            loss_kd, parts = kd_loss_fn(s_logits, t_logits, targets, T=T, alpha=alpha)
            if use_jgekd:
                jg = jgekd_proxy_loss(s_logits, t_logits) * jgekd_weight
                loss = loss_kd + jg
            else:
                loss = loss_kd
            for k,v in parts.items(): running_stats[k]+=v
        else:
            loss = F.cross_entropy(s_logits, targets)
        loss.backward()
        opt.step()
        running_loss += loss.item() * imgs.size(0)
        preds = s_logits.argmax(dim=1).cpu().numpy()
        all_preds.extend(preds.tolist())
        all_targets.extend(targets.cpu().numpy().tolist())
    avg_loss = running_loss / len(loader.dataset)
    f1 = f1_score(all_targets, all_preds, average='macro')
    bal_acc = balanced_accuracy_score(all_targets, all_preds)
    stats = {k: v/len(loader) for k,v in running_stats.items()}
    print(f"[Train] E{epoch} loss={avg_loss:.4f} f1_macro={f1:.4f} bal_acc={bal_acc:.4f} stats={stats}")
    return avg_loss, f1, bal_acc

def eval_model(model_s, loader, device):
    model_s.eval()
    all_preds, all_targets = [], []
    with torch.no_grad():
        for imgs, targets in loader:
            imgs = imgs.to(device)
            targets = targets.to(device)
            s_logits, _ = model_s(imgs)
            preds = s_logits.argmax(dim=1).cpu().numpy()
            all_preds.extend(preds.tolist())
            all_targets.extend(targets.cpu().numpy().tolist())
    f1 = f1_score(all_targets, all_preds, average='macro')
    bal_acc = balanced_accuracy_score(all_targets, all_preds)
    cm = confusion_matrix(all_targets, all_preds)
    print("[Eval] f1_macro=%.4f bal_acc=%.4f" % (f1, bal_acc))
    return f1, bal_acc, cm, classification_report(all_targets, all_preds)

# ---------------------------
# Main training setup
# ---------------------------
def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_dir = args.data_dir
    train_dir = os.path.join(data_dir, "Train")
    test_dir = os.path.join(data_dir, "Test")

    # classes and counts
    classes, counts = get_class_counts(train_dir)
    print("Classes:", classes)
    print("Counts:", counts)
    num_classes = len(classes)

    # transforms
    train_tf = transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(0.6,1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(30),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.02),
        transforms.RandomAutocontrast(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
    ])
    test_tf = transforms.Compose([
        transforms.Resize((256,256)),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
    ])

    train_dataset = datasets.ImageFolder(train_dir, transform=train_tf)

    if os.path.isdir(test_dir) and any(os.scandir(test_dir)):
        test_dataset = datasets.ImageFolder(test_dir, transform=test_tf)
    else:
        # fallback: small stratified split from train for validation if Test missing
        print("Warning: Test folder not found or empty. Creating validation split from Train (stratified approx).")
        # create a simple class-balanced split
        targets = [s[1] for s in train_dataset.samples]
        targets = np.array(targets)
        # compute indices per class
        train_idx, val_idx = [], []
        for c in np.unique(targets):
            idxs = np.where(targets == c)[0]
            np.random.shuffle(idxs)
            cut = max(1, int(0.1 * len(idxs)))  # 10% per class to val
            val_idx.extend(idxs[:cut].tolist())
            train_idx.extend(idxs[cut:].tolist())
        # create Subsets
        from torch.utils.data import Subset
        test_dataset = Subset(train_dataset, val_idx)
        train_dataset = Subset(train_dataset, train_idx)

    # Weighted sampler to handle imbalance (works whether train_dataset is Subset or ImageFolder)
    if isinstance(train_dataset, torch.utils.data.Subset):
        # build class counts from subset.targets
        subset_targets = [train_dataset.dataset.samples[i][1] for i in train_dataset.indices]
        class_counts = np.array([subset_targets.count(i) for i in range(num_classes)])
        samples_weight = np.array([1.0 / (class_counts[train_dataset.dataset.samples[i][1]] + 1e-8) for i in train_dataset.indices])
    else:
        class_counts = np.array([sum(1 for _,lbl in train_dataset.samples if lbl==i) for i in range(num_classes)])
        class_weights = 1.0 / (class_counts + 1e-8)
        samples_weight = np.array([class_weights[sample[1]] for sample in train_dataset.samples])

    samples_weight = torch.from_numpy(samples_weight.astype(np.float32))
    sampler = WeightedRandomSampler(samples_weight, num_samples=len(samples_weight), replacement=True)

    num_workers = min(8, max(0, (os.cpu_count() or 4) - 1))
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, sampler=sampler, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=num_workers)

    # Build teacher & student
    teacher = None
    if args.teacher:
        t = build_teacher(args.teacher, num_classes=num_classes, pretrained=True)
        t = t.to(device)
        t.eval()
        teacher = t

    student = StudentNet(num_classes=num_classes, pretrained=True).to(device)

    optimizer = torch.optim.AdamW(student.parameters(), lr=1e-4, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    best_metric = -1.0
    for epoch in range(1, args.epochs+1):
        train_one_epoch(epoch, student, teacher, optimizer, train_loader, device, T=args.T, alpha=args.alpha, use_jgekd=args.use_jgekd, jgekd_weight=args.jgekd_weight)
        f1, bal_acc, cm, cr = eval_model(student, test_loader, device)
        scheduler.step()
        if bal_acc > best_metric:
            best_metric = bal_acc
            torch.save({'student_state': student.state_dict(), 'epoch': epoch}, 'best_student.pth')
            print("Saved best_student.pth (bal_acc=%.4f)" % bal_acc)

    print("Training done. Best balanced-accuracy:", best_metric)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=str, required=True)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--teacher", type=str, default="resnet50", help="teacher architecture name from torchvision")
    parser.add_argument("--student", type=str, default="resnet18")
    parser.add_argument("--T", type=float, default=4.0)
    parser.add_argument("--alpha", type=float, default=0.7)
    parser.add_argument("--use_jgekd", action="store_true")
    parser.add_argument("--jgekd_weight", type=float, default=0.1)
    args = parser.parse_args()
    main(args)

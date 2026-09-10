from .data_pipeline import *
from .classic_model import RehabNet


class RehabDataset(Dataset):
    def __init__(self, samples, augment=False):
        self.samples = samples
        self.augment = augment

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s   = self.samples[idx]
        kps = safe_np(s['keypoints'], clip=5.0)
        if self.augment:
            kps = self._aug(kps)
        return {
            'keypoints':    torch.from_numpy(safe_np(kps, clip=5.0)).permute(2, 0, 1).float(),
            'adj':          torch.from_numpy(ADJ).float(),
            'label':        torch.tensor(s['label'],        dtype=torch.long),
            'quality':      torch.tensor(s['quality'],      dtype=torch.float32),
            'scalars':      torch.from_numpy(safe_np(s['scalars'], clip=10.0)).float(),
            'exercise_idx': torch.tensor(s['exercise_idx'], dtype=torch.long),
            'exercise':     s['exercise'],
            'subject':      s['subject'],
        }

    def _aug(self, kps):
        kps = safe_np(kps, clip=5.0)
        if np.random.rand() < 0.5:  # temporal jitter Â±20%
            n   = max(int(TARGET_LEN * np.random.uniform(0.8, 1.2)), MIN_FRAMES)
            tmp = np.zeros((n, N_JOINTS, 3), dtype=np.float32)
            for j in range(N_JOINTS):
                for c in range(3):
                    tmp[:, j, c] = _resample(kps[:, j, c], n)
            kps = np.zeros((TARGET_LEN, N_JOINTS, 3), dtype=np.float32)
            for j in range(N_JOINTS):
                for c in range(3):
                    kps[:, j, c] = _resample(tmp[:, j, c], TARGET_LEN)
        if np.random.rand() < 0.5:  # mirror flip Lâ†”R
            kps[:, [0, 1, 2, 3, 4, 5], :] = kps[:, [3, 4, 5, 0, 1, 2], :]
            kps[:, :, 0] *= -1.0
        if np.random.rand() < 0.5:  # Gaussian noise
            kps += np.random.randn(*kps.shape).astype(np.float32) * 0.02
        if np.random.rand() < 0.3:  # small rotation in XZ plane
            th  = np.radians(np.random.uniform(-10, 10))
            x, z = kps[:, :, 0].copy(), kps[:, :, 2].copy()
            kps[:, :, 0] = np.cos(th) * x - np.sin(th) * z
            kps[:, :, 2] = np.sin(th) * x + np.cos(th) * z
        return safe_np(kps, clip=5.0)


def collate(batch):
    return {
        k: torch.stack([b[k] for b in batch])
           if isinstance(batch[0][k], torch.Tensor)
           else [b[k] for b in batch]
        for k in batch[0]
    }

def build_loaders(train_samples, test_samples, batch_size=32):
    """Build train and validation DataLoaders with exercise-balanced sampling."""
    ex_counts = Counter(s['exercise'] for s in train_samples)
    total     = len(train_samples)
    weights   = [total / ex_counts[s['exercise']] for s in train_samples]
    sampler   = WeightedRandomSampler(weights, len(train_samples), replacement=True)
    train_loader = DataLoader(
        RehabDataset(train_samples, augment=True),
        batch_size=batch_size,
        sampler=sampler,
        collate_fn=collate,
        num_workers=2,
    )
    val_loader = DataLoader(
        RehabDataset(test_samples, augment=False),
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate,
        num_workers=2,
    )
    return train_loader, val_loader


# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# H. TRAINING
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

def compute_class_weights(train_samples, device):
    """Inverse-frequency weights for cross-entropy loss."""
    n0  = sum(s['label'] == 0 for s in train_samples)
    n1  = sum(s['label'] == 1 for s in train_samples)
    tot = n0 + n1
    return torch.tensor(
        [tot / (2 * n0 + EPS), tot / (2 * n1 + EPS)],
        dtype=torch.float32, device=device,
    )

def _train_epoch(model, loader, opt, device, cw, epoch=0):
    model.train()
    A = torch.from_numpy(ADJ).to(device)
    total, used, skipped = 0.0, 0, 0

    for b in loader:
        kps    = b['keypoints'].to(device)
        lbl    = b['label'].to(device)
        qual   = b['quality'].to(device)
        sc     = b['scalars'].to(device)
        ex_idx = b['exercise_idx'].to(device)

        good = (torch.isfinite(kps).flatten(1).all(1) &
                torch.isfinite(sc).all(1) &
                torch.isfinite(qual))
        if not good.all():
            kps, lbl, qual, sc, ex_idx = (t[good] for t in
                                           (kps, lbl, qual, sc, ex_idx))
        if kps.shape[0] == 0:
            skipped += 1
            continue

        mode = torch.zeros(kps.shape[0], dtype=torch.long, device=device)
        lg, qp, ex_lg = model(kps, A, sc, mode, ex_idx)

        if not (torch.isfinite(lg).all() and torch.isfinite(qp).all()):
            skipped += 1
            continue

        loss = (F.cross_entropy(lg, lbl, weight=cw) +
                0.3 * F.mse_loss(torch.sigmoid(qp),
                                  torch.clamp(qual, 0, 1)) +
                0.5 * F.cross_entropy(ex_lg, ex_idx))

        if not torch.isfinite(loss):
            skipped += 1
            continue

        opt.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 0.5)
        opt.step()
        total += loss.item()
        used  += 1

    if skipped:
        print(f'  ep{epoch}: skipped {skipped} batches')
    return total / max(used, 1)

@torch.no_grad()
def _evaluate(model, loader, device):
    model.eval()
    A = torch.from_numpy(ADJ).to(device)
    preds, labels, quals, qpreds = [], [], [], []

    for b in loader:
        kps    = b['keypoints'].to(device)
        sc     = b['scalars'].to(device)
        ex_idx = b['exercise_idx'].to(device)
        qual   = b['quality'].to(device)
        lbl    = b['label'].to(device)

        good = (torch.isfinite(kps).flatten(1).all(1) &
                torch.isfinite(sc).all(1))
        if good.sum() == 0:
            continue
        kps, sc, ex_idx, qual, lbl = (t[good] for t in
                                       (kps, sc, ex_idx, qual, lbl))
        mode = torch.zeros(kps.shape[0], dtype=torch.long, device=device)
        lg, qp, _ = model(kps, A, sc, mode, ex_idx)  # unpack all 3

        preds.extend(lg.argmax(1).cpu().numpy())
        labels.extend(lbl.cpu().numpy())
        quals.extend(qual.cpu().numpy())
        qpreds.extend(torch.sigmoid(qp).cpu().numpy())

    if not labels:
        return {'acc': 0.0, 'f1': 0.0, 'quality_r': 0.0}

    acc = accuracy_score(labels, preds)
    f1  = f1_score(labels, preds, average='macro', zero_division=0)
    r   = pearsonr(quals, qpreds)[0] if len(set(quals)) > 1 else 0.0
    print(classification_report(labels, preds,
          target_names=['incorrect', 'correct'], zero_division=0))
    print(f'  quality Pearson r={r:.3f}')
    return {'acc': acc, 'f1': f1, 'quality_r': r}

def run_loso(samples=None, n_epochs=N_EPOCHS,
             batch_size=32, lr=3e-4, patience=12):
    """
    Leave-One-Subject-Out cross-validation.

    Usage:
        # Option A â€” loads datasets internally
        results, model = run_loso()

        # Option B â€” pass your own sample list
        results, model = run_loso(samples=my_samples)
    """
    if samples is None:
        samples = build_master_dataset()

    device   = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')

    subjects = get_uiprmd_subjects(samples)
    if not subjects:
        subjects = sorted({s['subject'] for s in samples})

    results = []
    best_overall_f1    = 0.0
    best_overall_model = None

    for subj in subjects:
        print(f"\n{'='*55}\nLOSO held-out: {subj}\n{'='*55}")
        tr, te = loso_split(samples, subj)
        tl, vl = build_loaders(tr, te, batch_size)
        cw     = compute_class_weights(tr, device)
        print(f'  class weights: incorrect={cw[0]:.3f}  correct={cw[1]:.3f}')

        model = RehabNet().to(device)
        opt   = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

        def lr_fn(ep):
            w = 5
            if ep < w:
                return (ep + 1) / w
            return 0.5 * (1 + np.cos(np.pi * (ep - w) / max(n_epochs - w, 1)))

        sch = torch.optim.lr_scheduler.LambdaLR(opt, lr_fn)
        best_f1, best_state, no_imp = 0.0, None, 0

        for ep in range(n_epochs):
            loss = _train_epoch(model, tl, opt, device, cw, epoch=ep)
            sch.step()

            if (ep + 1) % 5 == 0:
                model.eval()
                pl, ll = [], []
                with torch.no_grad():
                    A2 = torch.from_numpy(ADJ).to(device)
                    for b in vl:
                        kps    = b['keypoints'].to(device)
                        sc     = b['scalars'].to(device)
                        ex_idx = b['exercise_idx'].to(device)
                        lbl    = b['label']
                        good   = (torch.isfinite(kps).flatten(1).all(1) &
                                  torch.isfinite(sc).all(1))
                        if good.sum() == 0:
                            continue
                        kps, sc, ex_idx = kps[good], sc[good], ex_idx[good]
                        lbl  = lbl[good.cpu()]
                        mode = torch.zeros(kps.shape[0], dtype=torch.long,
                                           device=device)
                        lg, _, _ = model(kps, A2, sc, mode, ex_idx)
                        pl.extend(lg.argmax(1).cpu().numpy())
                        ll.extend(lbl.numpy())
                model.train()

                vf1 = f1_score(ll, pl, average='macro', zero_division=0)
                print(f'  ep{ep+1:3d} loss={loss:.4f}  '
                      f'lr={sch.get_last_lr()[0]:.2e}  val_f1={vf1:.3f}')

                if vf1 > best_f1:
                    best_f1    = vf1
                    best_state = copy.deepcopy(model.state_dict())
                    no_imp     = 0
                else:
                    no_imp += 1
                    if no_imp >= patience:
                        print(f'  Early stop ep{ep+1}')
                        break

        if best_state:
            model.load_state_dict(best_state)

        if best_f1 > best_overall_f1:
            best_overall_f1    = best_f1
            best_overall_model = copy.deepcopy(model.state_dict())

        print(f'\nâ”€â”€ Eval: {subj} (best val_f1={best_f1:.3f}) â”€â”€')
        results.append(_evaluate(model, vl, device))

    # Save best checkpoint
    if best_overall_model:
        os.makedirs(os.path.dirname(MODEL_SAVE), exist_ok=True)
        torch.save(best_overall_model, MODEL_SAVE)
        print(f'\nBest model saved â†’ {MODEL_SAVE}')

    accs = [r['acc'] for r in results]
    f1s  = [r['f1']  for r in results]
    print(f"\n{'='*55}")
    print(f'LOSO RESULTS')
    print(f'  Acc = {np.mean(accs):.3f} Â± {np.std(accs):.3f}')
    print(f'  F1  = {np.mean(f1s):.3f} Â± {np.std(f1s):.3f}')
    print(f"{'='*55}")

    final_model = RehabNet().to(device)
    if best_overall_model:
        final_model.load_state_dict(best_overall_model)
    return results, final_model



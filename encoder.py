"""Contrastive training on one GPU, held-out retrieval, and exported weights."""
from collections import Counter
from contextlib import nullcontext
import gc
from pathlib import Path
import random
import time

import numpy as np
import pandas as pd

from catalog import SEED, OUT, ingredient_view, step_view, write_json

MODEL = 'sentence-transformers/all-MiniLM-L6-v2'
REVISION = '1110a243fdf4706b3f48f1d95db1a4f5529b4d41'
TOKENS = 256
TEMPERATURE = .07

def load_encoder(location=MODEL, revision=None, device=None):
    import torch
    from transformers import AutoModel, AutoTokenizer
    device = device or ('cuda:0' if torch.cuda.is_available() else 'cpu')
    local_only = isinstance(location, Path) and location.is_dir()
    tokenizer = AutoTokenizer.from_pretrained(location, revision=revision, local_files_only=local_only)
    model = AutoModel.from_pretrained(location, revision=revision, local_files_only=local_only, use_safetensors=True).to(device)
    return tokenizer, model

def forward(texts, tokenizer, model):
    import torch
    import torch.nn.functional as F
    device = next(model.parameters()).device
    batch = tokenizer(texts, padding=True, truncation=True, max_length=TOKENS, return_tensors='pt').to(device)
    context = torch.autocast('cuda', dtype=torch.float16) if device.type == 'cuda' else nullcontext()
    with context:
        tokens = model(**batch).last_hidden_state
    mask = batch['attention_mask'].unsqueeze(-1)
    pooled = (tokens.float() * mask).sum(1) / mask.sum(1).clamp_min(1)
    return F.normalize(pooled, dim=1)

def encode(texts, tokenizer, model, batch=96):
    import torch
    model.eval()
    with torch.inference_mode():
        return np.concatenate([forward(texts[i:i+batch], tokenizer, model).cpu().numpy()
                               for i in range(0, len(texts), batch)])

def stable_ranking(scores):
    return np.argsort(-scores, kind='stable', axis=-1)

def diagnostic(ingredients, steps, rows):
    scores = ingredients @ steps.T
    family = np.array([r['family'] for r in rows])
    same_family = family[:,None] == family[None,:]
    np.fill_diagonal(same_family, False)
    scores[same_family] = -np.inf
    ranking = stable_ranking(scores)
    ranks = np.argmax(ranking == np.arange(len(rows))[:,None], axis=1) + 1
    return {'queries':len(rows), 'recall@1':float(np.mean(ranks<=1)),
            'recall@5':float(np.mean(ranks<=5)), 'mrr':float(np.mean(1/ranks))}

def loss_for(indices, rows, ing, steps, tokenizer, model, rng):
    import torch
    import torch.nn.functional as F
    subset = []
    for idx in indices:
        names = rows[idx]['normalized_ingredients']
        subset.append(ingredient_view(rng.sample(names, max(1, round(len(names)*rng.uniform(.45,.80))))))
    a = forward([ing[i] for i in indices], tokenizer, model)
    b = forward([steps[i] for i in indices], tokenizer, model)
    c = forward(subset, tokenizer, model)
    families = np.array([rows[i]['family'] for i in indices])
    duplicates = np.array([rows[i]['duplicate_group'] for i in indices])
    mask = (families[:,None] == families[None,:]) | (duplicates[:,None] == duplicates[None,:])
    np.fill_diagonal(mask, False)
    mask = torch.as_tensor(mask, device=a.device)
    labels = torch.arange(len(indices), device=a.device)
    terms = []
    for query, target in [(a,b),(c,b),(a,c)]:
        logits = (query @ target.T / TEMPERATURE).masked_fill(mask, -1e4)
        terms.extend([F.cross_entropy(logits, labels), F.cross_entropy(logits.T, labels)])
    return sum(terms) / len(terms)

def train_encoder(records, out=OUT, model_location=MODEL, max_gpu_minutes=90):
    import torch
    out = Path(out)
    started = time.perf_counter()
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if not torch.cuda.is_available():
        raise RuntimeError('Kaggle GPU unavailable')
    torch.cuda.init()
    torch.cuda.set_device(0)
    torch.cuda.manual_seed_all(SEED)
    torch.cuda.reset_peak_memory_stats(0)
    torch.set_float32_matmul_precision('high')
    model_started = time.perf_counter()
    tokenizer, model = load_encoder(model_location, REVISION if model_location == MODEL else None, device='cuda:0')
    download_seconds = time.perf_counter() - model_started
    ing = [ingredient_view(r['normalized_ingredients']) for r in records]
    stp = [step_view([x['text'] for x in r['steps']], r['title']) for r in records]
    train = [i for i,r in enumerate(records) if r['split']=='train' and ing[i] and stp[i]]
    valid = [i for i,r in enumerate(records) if r['split']=='validation' and ing[i] and stp[i]]
    test = [i for i,r in enumerate(records) if r['split']=='test' and ing[i] and stp[i]]
    if min(len(train),len(valid),len(test)) < 32:
        raise ValueError('Too few valid paired recipe views')
    rng = random.Random(SEED)
    best_path = out/'encoder'
    best_path.mkdir(exist_ok=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5, weight_decay=.01)
    scaler = torch.amp.GradScaler('cuda')
    def update(indices):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        loss = loss_for(indices, records, ing, stp, tokenizer, model, rng)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
        scaler.step(optimizer)
        scaler.update()
        return float(loss.detach().cpu())
    eval_indices = valid[:min(400,len(valid))]
    def measure(indices):
        return diagnostic(encode([ing[i] for i in indices],tokenizer,model),
                          encode([stp[i] for i in indices],tokenizer,model), [records[i] for i in indices])
    # Capture baseline before any optimizer steps. Smoke uses the real data and the same loss.
    frozen_val = measure(eval_indices)
    model.save_pretrained(out/'frozen_encoder', safe_serialization=True)
    tokenizer.save_pretrained(out/'frozen_encoder')
    base_ingredients = encode(ing,tokenizer,model)
    base_steps = encode(stp,tokenizer,model)
    np.savez(out/'frozen_embeddings.npz', ingredients=base_ingredients, steps=base_steps)
    sample = rng.sample(train, 32*5)
    before = time.perf_counter()
    smoke_losses = [update(sample[j:j+32]) for j in range(0,len(sample),32)]
    torch.cuda.synchronize(0)
    smoke_seconds = time.perf_counter() - before
    smoke_validation = measure(eval_indices)
    if not np.isfinite(smoke_losses).all():
        raise RuntimeError(f'Encoder smoke test produced non-finite losses: {smoke_losses}')
    smoke = {'pairs': len(sample), 'seconds':smoke_seconds, 'pairs_per_second':len(sample)/smoke_seconds,
             'training_losses':smoke_losses, 'validation': smoke_validation,
             'peak_vram_gib': torch.cuda.max_memory_allocated(0)/2**30}
    write_json(out/'encoder_smoke.json', smoke)
    print('ENCODER SMOKE:', smoke, flush=True)
    batch = 32
    planned_epochs = 2
    budget_pairs = int((max_gpu_minutes*60*.72)/smoke_seconds*len(sample))
    planned_pairs = min(len(train), max(160,budget_pairs//planned_epochs))
    history = []
    best_mrr = -1.
    t_train = time.perf_counter()
    for epoch in range(planned_epochs):
        chosen = rng.sample(train, planned_pairs)
        losses = []
        for index in range(0,len(chosen),batch):
            chunk = chosen[index:index+batch]
            if len(chunk) >= 4:
                losses.append(update(chunk))
            if len(losses) % 100 == 0 and losses:
                print(f'Epoch {epoch+1}, {len(losses)} batches, mean recent loss {np.mean(losses[-100:]):.4f}',flush=True)
            if time.perf_counter()-t_train > max_gpu_minutes*60*.82:
                break
        metrics = measure(eval_indices)
        entry = {'epoch':epoch+1,'batches':len(losses),'mean_loss':float(np.mean(losses)),
                 'validation':metrics,'elapsed_training_seconds':time.perf_counter()-t_train}
        history.append(entry)
        print('EPOCH:',entry,flush=True)
        if metrics['mrr'] > best_mrr:
            model.save_pretrained(best_path, safe_serialization=True)
            tokenizer.save_pretrained(best_path)
            best_mrr, best_epoch = metrics['mrr'], epoch+1
        if time.perf_counter()-t_train > max_gpu_minutes*60*.82:
            break
    del optimizer, model
    gc.collect()
    torch.cuda.empty_cache()
    tokenizer,model = load_encoder(best_path, device='cuda:0')
    fine_ingredients, fine_steps = encode(ing,tokenizer,model), encode(stp,tokenizer,model)
    np.savez(out/'recipe_embeddings.npz', ingredients=fine_ingredients, steps=fine_steps,
             recipe_ids=np.array([r['recipe_id'] for r in records]))
    test_rows = [records[i] for i in test]
    report = {'model':str(model_location),'model_revision':REVISION if model_location == MODEL else 'local attachment',
        'batch_size':batch,'tokens_per_view':TOKENS,'seed':SEED,'training_pairs_per_epoch':planned_pairs,
        'availability_and_load_seconds':download_seconds,'smoke':smoke,'history':history,
        'selected_epoch':best_epoch,'selection':'highest validation paired MRR among full epochs',
        'frozen_validation':frozen_val,'frozen_test':diagnostic(base_ingredients[test],base_steps[test],test_rows),
        'fine_tuned_test':diagnostic(fine_ingredients[test],fine_steps[test],test_rows),
        'total_seconds':time.perf_counter()-started,'peak_vram_gib':torch.cuda.max_memory_allocated(0)/2**30,
        'gpu':torch.cuda.get_device_name(0),
        'limitations':'Matching CSV IDs is a diagnostic, not a similar-dish relevance label. No hard negatives were added.'}
    write_json(out/'encoder_metrics.json',report)
    print('ENCODER METRICS:', report,flush=True)
    del model,tokenizer
    gc.collect()
    torch.cuda.empty_cache()
    return report


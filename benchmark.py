#!/usr/bin/env python3
"""Reproducible BangalASR evaluation. Credentials are read only from HF_TOKEN."""
import argparse
import csv
import hashlib
import inspect
import json
import os
from pathlib import Path
import platform
import time
import unicodedata

ROOT = Path(__file__).resolve().parent

def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')

def normalize(text):
    text = unicodedata.normalize('NFC', text)
    text = ''.join(' ' if unicodedata.category(c).startswith('P') else '' if unicodedata.category(c) == 'Cf' else c for c in text)
    return ' '.join(text.split())

def distance(a, b):
    prev = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        cur = [i]
        for j, y in enumerate(b, 1):
            cur.append(min(cur[-1] + 1, prev[j] + 1, prev[j-1] + (x != y)))
        prev = cur
    return prev[-1]

def metrics(rows, normalized):
    words = chars = we = ce = 0
    for row in rows:
        ref, hyp = row['text'], row['prediction']
        if normalized:
            ref, hyp = normalize(ref), normalize(hyp)
        else:
            ref, hyp = ' '.join(ref.split()), ' '.join(hyp.split())
        words += len(ref.split()); we += distance(ref.split(), hyp.split())
        # CER excludes whitespace and counts Unicode code points, not grapheme clusters.
        ref, hyp = ''.join(ref.split()), ''.join(hyp.split())
        chars += len(ref); ce += distance(ref, hyp)
    return {'wer_percent': 100 * we / words if words else None,
            'cer_percent': 100 * ce / chars if chars else None,
            'word_edits': we, 'reference_words': words,
            'character_edits': ce, 'reference_characters': chars}

def prepare():
    import numpy as np
    import soundfile as sf
    from scipy.signal import resample_poly
    from math import gcd
    folder = ROOT / 'data'
    index = {}
    for path in (folder / 'asr_bengali_1000_wav').rglob('*.wav'):
        if path.name in index:
            raise ValueError('Duplicate audio filename: ' + path.name)
        index[path.name] = path
    rows = []; seen = set(); hashes = {}
    converted = folder / 'audio16k'; converted.mkdir(exist_ok=True)
    with (folder / 'asr_bangal_1000.csv').open(encoding='utf-8-sig', newline='') as f:
        for row in csv.DictReader(f):
            name = Path(row['audio']).name
            if name in seen or not normalize(row['text']):
                raise ValueError('Duplicate ID or empty reference: ' + name)
            seen.add(name); path = index[name]
            samples, rate = sf.read(path, dtype='float32', always_2d=True)
            if not samples.size or not np.isfinite(samples).all():
                raise ValueError('Invalid audio: ' + name)
            sha = hashlib.sha256(path.read_bytes()).hexdigest()
            hashes.setdefault(sha, []).append(name)
            if samples.shape[1] != 1 or rate != 16000:
                samples = samples.mean(axis=1)
                factor = gcd(rate, 16000)
                samples = resample_poly(samples, 16000 // factor, rate // factor)
                path = converted / name
                sf.write(path, samples, 16000, subtype='PCM_16')
            rows.append({'id': Path(name).stem, 'audio_filepath': str(path.relative_to(ROOT)),
                         'text': row['text'], 'duration': len(samples) / (16000 if path.parent == converted else rate),
                         'source_audio_sha256': sha})
    if len(rows) != 1000:
        raise ValueError(f'Expected 1000 references, got {len(rows)}')
    (folder / 'manifest.jsonl').write_text(''.join(json.dumps(r, ensure_ascii=False)+'\n' for r in rows))
    summary = {'samples': len(rows), 'duration_seconds': sum(r['duration'] for r in rows),
               'unused_audio': sorted(set(index)-seen),
               'duplicate_audio_groups': [v for v in hashes.values() if len(v)>1],
               'csv_sha256': hashlib.sha256((folder/'asr_bangal_1000.csv').read_bytes()).hexdigest(),
               'archive_sha256': hashlib.sha256((folder/'asr_bengali_1000_wav.zip').read_bytes()).hexdigest()}
    write_json(folder / 'summary.json', summary); print(json.dumps(summary, indent=2))

def extract_text(output, count):
    if isinstance(output, tuple):
        output = output[0]
    if len(output) != count:
        raise ValueError(f'Expected {count} hypotheses, got {len(output)}')
    texts = []
    for item in output:
        text = item if isinstance(item, str) else getattr(item, 'text', None)
        if not isinstance(text, str):
            raise TypeError('Unsupported transcription result: ' + type(item).__name__)
        texts.append(text)
    return texts

def run(args):
    import importlib.metadata
    import torch
    import nemo.collections.asr as asr
    from huggingface_hub import hf_hub_download
    manifest = ROOT / 'data/manifest.jsonl'
    rows = [json.loads(line) for line in manifest.read_text().splitlines()]
    if args.limit:
        rows = rows[:args.limit]
    cfg = json.loads((ROOT/'models.json').read_text())[args.model]
    device = args.device or ('cuda' if torch.cuda.is_available() else 'cpu')
    if device == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA is unavailable; use a GPU runtime or --device cpu')
    output = ROOT / args.output / (args.model + ('-'+args.decoder if cfg['backend']=='indic' else ''))
    output.mkdir(parents=True, exist_ok=False)
    metadata = {'model': cfg, 'device': device, 'decoder': args.decoder if cfg['backend']=='indic' else 'checkpoint-default',
                'samples': len(rows), 'full_dataset': len(rows)==1000, 'batch_size':args.batch_size,
                'manifest_sha256': hashlib.sha256(manifest.read_bytes()).hexdigest(),
                'python':platform.python_version(), 'torch':torch.__version__,
                'nemo':importlib.metadata.version('nemo_toolkit'),
                'hardware':torch.cuda.get_device_name() if device=='cuda' else platform.processor(),
                'status':'running'}
    write_json(output/'run.json', metadata)
    try:
        path = hf_hub_download(cfg['repo'], cfg['filename'], revision=cfg['revision'], token=os.environ.get('HF_TOKEN'))
        model = asr.models.ASRModel.restore_from(path, map_location=torch.device(device))
        model.eval(); model.to(device); model.freeze()
        kwargs = {'batch_size':args.batch_size}
        if cfg['backend']=='indic':
            if 'language_id' not in inspect.signature(model.transcribe).parameters:
                raise RuntimeError('IndicConformer requires the AI4Bharat NeMo nemo-v2 environment; see README.')
            model.cur_decoder = args.decoder
            kwargs['language_id'] = 'bn'
            if args.decoder=='ctc': kwargs['logprobs'] = False
        sync = lambda: torch.cuda.synchronize() if device=='cuda' else None
        # Untimed warmup; model load and weight download excluded from RTF.
        with torch.inference_mode():
            model.transcribe([str(ROOT/rows[0]['audio_filepath'])], **kwargs)
        sync(); start = time.perf_counter(); predictions = []
        with (output/'predictions.jsonl').open('w', encoding='utf-8') as f, torch.inference_mode():
            for offset in range(0, len(rows), args.batch_size):
                batch = rows[offset:offset+args.batch_size]
                texts = extract_text(model.transcribe([str(ROOT/r['audio_filepath']) for r in batch], **kwargs), len(batch))
                for row, text in zip(batch, texts):
                    pred = dict(row, prediction=text); predictions.append(pred)
                    f.write(json.dumps(pred, ensure_ascii=False)+'\n')
                f.flush()
        sync(); elapsed = time.perf_counter()-start
        result = {'model':args.model, 'samples':len(predictions), 'full_dataset': len(predictions)==1000,
                  'raw':metrics(predictions, False), 'normalized':metrics(predictions, True),
                  'elapsed_seconds':elapsed, 'rtf':elapsed/sum(r['duration'] for r in rows)}
        write_json(output/'metrics.json', result)
        metadata['status']='complete'; print(json.dumps(result, indent=2))
    except Exception as e:
        metadata['status']='failed'; metadata['error_type']=type(e).__name__
        raise
    finally:
        write_json(output/'run.json', metadata)

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    sub=parser.add_subparsers(dest='command', required=True)
    sub.add_parser('prepare')
    p=sub.add_parser('run')
    p.add_argument('--model', required=True, choices=json.loads((ROOT/'models.json').read_text()))
    p.add_argument('--device', choices=['cpu','cuda'])
    p.add_argument('--batch-size',type=int,default=4)
    p.add_argument('--limit',type=int)
    p.add_argument('--decoder',choices=['ctc','rnnt'],default='ctc')
    p.add_argument('--output',default='results')
    args=parser.parse_args()
    if args.command=='prepare': prepare()
    else:
        if args.batch_size<1 or (args.limit is not None and args.limit<1): parser.error('Batch size and limit must be positive')
        run(args)

if __name__=='__main__': main()

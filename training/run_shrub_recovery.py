"""Re-evaluate local checkpoints, then fine-tune the best one with shrub supervision."""
import argparse
import csv
from datetime import datetime
import json
import math
from pathlib import Path
import subprocess
import sys


def run():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--candidates', type=Path, nargs='+', default=[
        Path('outputs_training_paper/control/best_model.pth'),
        Path('outputs_training_paper/ignore_flair_shrub/best_model.pth'),
    ])
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--lr', type=float, default=1e-5)
    parser.add_argument('--class-weights', type=float, nargs=6,
                        default=[1.20, 0.80, 1.25, 1.45, 1.00, 0.90])
    parser.add_argument('--out', type=Path)
    args = parser.parse_args()
    if args.epochs < 1 or not math.isfinite(args.lr) or args.lr <= 0:
        parser.error('epochs and lr must be positive and finite')
    if not all(math.isfinite(w) and w > 0 for w in args.class_weights):
        parser.error('All class weights must be finite and positive')
    candidates = list(dict.fromkeys(p.resolve() for p in args.candidates if p.is_file()))
    for path in args.candidates:
        if not path.is_file():
            print(f'Skipping missing candidate: {path}', flush=True)
    if not candidates:
        parser.error('No checkpoint found. Supply existing files with --candidates PATH1 PATH2')
    for split in ('train', 'val'):
        if not Path(f'flair_manifests_paper/{split}.csv').is_file():
            parser.error('Run python training/prepare_flair_paper.py first')
    out = args.out or Path('outputs_training_paper') / datetime.now().strftime('recovery_%Y%m%d_%H%M%S')
    out.mkdir(parents=True, exist_ok=False)
    script = Path(__file__).with_name('train_paper.py')
    common = [sys.executable, str(script), '--run', 'control', '--class-weights',
              *map(str, args.class_weights), '--lr', str(args.lr)]
    results = []
    for index, checkpoint in enumerate(candidates, start=1):
        destination = out / f'candidate_{index}'
        subprocess.run(common + ['--checkpoint', str(checkpoint), '--eval-only',
                                 '--out', str(destination)], check=True)
        with (destination / 'history.csv').open(encoding='utf-8-sig') as stream:
            metrics = next(csv.DictReader(stream))
        score = float(metrics['miou_vegetation'])
        if not math.isfinite(score):
            raise RuntimeError(f'Invalid validation score: {checkpoint}')
        results.append({'checkpoint': str(checkpoint), 'vegetation_mIoU': score,
                        'shrub_IoU': float(metrics['shrub_iou'])})
    best = max(results, key=lambda row: row['vegetation_mIoU'])
    (out / 'selection.json').write_text(json.dumps({
        'candidates': results, 'selected': best,
        'criterion': 'Highest vegetation mIoU on the same validation loader',
        'class_weights': args.class_weights, 'lr': args.lr,
        'optimizer': 'New AdamW optimizer for fine-tuning',
    }, indent=2), encoding='utf-8')
    print(f'Selected checkpoint: {best["checkpoint"]}', flush=True)
    subprocess.run(common + ['--checkpoint', best['checkpoint'], '--epochs', str(args.epochs),
                             '--out', str(out / 'finetune')], check=True)
    print(f'Recovery finished. Results: {out}', flush=True)


if __name__ == '__main__':
    run()

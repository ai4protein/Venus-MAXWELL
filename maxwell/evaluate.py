"""Frozen MAXWELL checkpoint identity audit and matched Test12K/full-DTM evaluation."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import shutil
import socket
import time
from datetime import datetime, timezone
from pathlib import Path

PROJECT = Path('/share/home/hongmeiyao/maxwell-dev')
ROOT = PROJECT / 'experiments/checkpoint_identity_20260915'
KEYS = ['protein', 'mutation']


def guard(gpu=False):
    if not os.environ.get('SLURM_JOB_ID') or socket.gethostname().startswith(('login', 'admin')):
        raise RuntimeError('Must run on a Slurm compute node.')
    if gpu:
        import torch
        if not torch.cuda.is_available():
            raise RuntimeError('Inference requires an allocated GPU.')


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def dump(path, value):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False), encoding='utf-8')


def read_csv(path):
    import pandas as pd
    return pd.read_csv(path, float_precision='round_trip', keep_default_na=False)


def environment():
    return {'slurm_job_id': os.environ['SLURM_JOB_ID'], 'node': socket.gethostname(),
            'utc': datetime.now(timezone.utc).isoformat()}


def prepare():
    guard()
    import numpy as np
    import pandas as pd
    import torch
    from maxwell.datasets import ProteinMPNNDataset
    inventory = json.loads((ROOT / 'metadata/checkpoint_inventory.json').read_text())
    assert len(inventory['checkpoints']) == 6
    assert all(r['status'] == 'readable' for r in inventory['checkpoints'])
    for folder in ['features', 'manifests', 'results', 'logs', 'scripts']:
        (ROOT / folder).mkdir(exist_ok=True)
    definitions = {
        'Test12K': (PROJECT / 'dataset/ddG_maxwell', 'ddG', 308, 12102,
                    PROJECT / 'analysis_extension/results/phase1/tables/test12k_manifest_eligible.csv'),
        'DTM': (PROJECT / 'dataset/fireprot2_filter5/dtm', 'score', 86, 2838,
                PROJECT / 'zero_shot_unified/manifests/full_dtm_cross_dataset_evaluation/evaluation_manifest.csv'),
    }
    metadata = {'environment': environment(), 'datasets': {}, 'inputs': {}}
    for dataset_name, (data_path, target_column, n_proteins, n_mutations, source_manifest) in definitions.items():
        manifest = read_csv(source_manifest)
        assert len(manifest) == n_mutations and manifest.protein.nunique() == n_proteins
        assert not manifest.duplicated(KEYS).any()
        assert np.isfinite(manifest.target_stability_score).all()
        manifest.to_csv(ROOT / f'manifests/{dataset_name}.csv', index=False)
        proteins = sorted(manifest.protein.unique())
        list_file = ROOT / f'manifests/{dataset_name}_proteins.txt'
        list_file.write_text('\n'.join(proteins) + '\n')
        dataset = ProteinMPNNDataset(data_path, target_column, 0, dataset_list_txt=list_file, data_root=data_path)
        assert len(dataset) == len(proteins)
        features, audit = {}, []
        for protein, item in zip(proteins, dataset):
            group = manifest.loc[manifest.protein == protein]
            sequence = item['seq']
            for row in group.itertuples():
                assert sequence[int(row.position) - 1] == row.wt and row.wt != row.mut
            batch = dataset.collate_fn([item])
            # Experimental targets/masks are deliberately absent from the inference cache.
            batch = {k: batch[k] for k in ['X', 'input_ids', 'attention_mask', 'residue_idx', 'chain_encoding']}
            features[protein] = {'sequence': sequence, 'batch': batch}
            for subdir, extension in [('fasta', '.fasta'), ('pdb', '.pdb'), ('mutant', '.csv')]:
                source = data_path / subdir / (protein + extension)
                metadata['inputs'][str(source)] = digest(source)
            audit.append({'protein': protein, 'sequence_length': len(sequence), 'n_mutations': len(group),
                          'nonfinite_coordinate_values_before_collate': int((~torch.isfinite(item['X'])).sum()),
                          'coordinates_finite_after_collate': bool(torch.isfinite(batch['X']).all())})
        cache_path = ROOT / f'features/{dataset_name}.pt'
        torch.save(features, cache_path)
        pd.DataFrame(audit).to_csv(ROOT / f'manifests/{dataset_name}_feature_audit.csv', index=False)
        metadata['datasets'][dataset_name] = {
            'protein_count': n_proteins, 'mutation_count': n_mutations, 'dataset_path': str(data_path),
            'source_manifest': str(source_manifest), 'source_manifest_sha256': digest(source_manifest),
            'frozen_manifest_sha256': digest(ROOT / f'manifests/{dataset_name}.csv'),
            'feature_cache_sha256': digest(cache_path), 'max_sequence_length': max(len(v['sequence']) for v in features.values()),
            'length_cutoff': None, 'experimental_labels_used_in_inference_inputs': False,
        }
        print('PREPARED', dataset_name, n_proteins, n_mutations, flush=True)
    for file in ['maxwell/lit_model.py', 'maxwell/model_wrappers.py', 'maxwell/datasets.py', 'maxwell/protein_mpnn_utils.py']:
        metadata['inputs'][str(PROJECT / file)] = digest(PROJECT / file)
    dump(ROOT / 'metadata/preparation.json', metadata)


def metrics(frame, dataset_name):
    import numpy as np
    import pandas as pd
    from scipy.stats import pearsonr, spearmanr
    from sklearn.metrics import average_precision_score, roc_auc_score
    threshold = .3 if dataset_name == 'Test12K' else 1.0
    records = []
    for protein, group in frame.groupby('protein', sort=True):
        y = group.target_stability_score.to_numpy(float)
        s = group.predicted_stability_score.to_numpy(float)
        positive = y > threshold if dataset_name == 'Test12K' else y >= threshold
        r = {'protein': protein, 'n_mutations': len(y), 'n_positive': int(positive.sum()),
             'spearman': np.nan, 'pearson': np.nan, 'auroc': np.nan, 'auprc': np.nan, 'enrich_at_10': np.nan,
             'positive_rule': '>0.3 kcal/mol' if dataset_name == 'Test12K' else '>=1 degree C',
             'correlation_status': 'defined', 'classification_status': 'defined', 'enrich_status': 'defined'}
        if len(y) >= 3 and np.ptp(y) > 0 and np.ptp(s) > 0:
            r['spearman'] = float(spearmanr(y, s).statistic)
            r['pearson'] = float(pearsonr(y, s).statistic)
        else:
            r['correlation_status'] = 'too_few_or_constant_target_or_prediction'
        if 0 < positive.sum() < len(y):
            r['auroc'] = float(roc_auc_score(positive, s))
            r['auprc'] = float(average_precision_score(positive, s))
        else:
            r['classification_status'] = 'single_class'
        if len(y) >= 10 and positive.sum() > 0:
            order = np.argsort(-s, kind='stable')
            r['enrich_at_10'] = float(positive[order[:10]].mean() / positive.mean())
        else:
            r['enrich_status'] = 'fewer_than_10_candidates_or_no_positive'
        records.append(r)
    per_protein = pd.DataFrame(records)
    summary = {'dataset': dataset_name, 'proteins': int(frame.protein.nunique()), 'mutations': len(frame)}
    for name in ['spearman', 'pearson', 'auroc', 'auprc', 'enrich_at_10']:
        values = per_protein[name].dropna()
        summary[name + '_mean'] = float(values.mean()) if len(values) else None
        summary[name + '_sample_sd'] = float(values.std(ddof=1)) if len(values) > 1 else None
        summary[name + '_n'] = len(values)
    return per_protein, summary


def infer(index):
    guard(gpu=True)
    import numpy as np
    import pandas as pd
    import torch
    from maxwell.lit_model import LitModel
    from maxwell.model_wrappers import model_compute
    inventory = json.loads((ROOT / 'metadata/checkpoint_inventory.json').read_text())
    record = inventory['checkpoints'][index]
    checkpoint_path = Path(record['path'])
    checkpoint_id = f'C{index + 1:02d}_' + checkpoint_path.stem
    out = ROOT / 'results' / checkpoint_id
    if out.exists():
        raise RuntimeError('Output already exists, use an explicit recovery workflow: ' + str(out))
    out.mkdir(parents=True)
    assert digest(checkpoint_path) == record['file_sha256']
    with torch.serialization.safe_globals([argparse.Namespace]):
        ckpt = torch.load(checkpoint_path, map_location='cpu', weights_only=True)
    args = ckpt['hyper_parameters']['args']
    if isinstance(args, dict):
        args = argparse.Namespace(**args)
    assert args.model_type == 'proteinmpnn' and not getattr(args, 'use_lora', False)
    assert getattr(args, 'mpnn_score_mode', 'autoregressive') == 'autoregressive'
    assert not getattr(args, 'no_log_scale', False)
    random.seed(20260915)
    np.random.seed(20260915)
    torch.manual_seed(20260915)
    torch.cuda.manual_seed_all(20260915)
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    model = LitModel(args)
    loaded = model.load_state_dict(ckpt['state_dict'], strict=True)
    assert not loaded.missing_keys and not loaded.unexpected_keys
    backbone_hash = hashlib.sha256()
    for name, value in sorted(model.model.state_dict().items()):
        backbone_hash.update(name.encode() + b'\0' + value.detach().cpu().contiguous().numpy().tobytes())
    model = model.float().eval().to('cuda')
    provenance = {'checkpoint_name': checkpoint_path.name, 'checkpoint_id': checkpoint_id,
                  'checkpoint_path': str(checkpoint_path), 'checkpoint_file_sha256': record['file_sha256'],
                  'checkpoint_state_sha256': record['state_dict_sha256'], 'backbone_state_sha256': backbone_hash.hexdigest(),
                  'stored_epoch': record['epoch'], 'epoch_indexing': 'Lightning stored zero-based index',
                  'hyper_parameters': record['hyper_parameters'], 'environment': environment(),
                  'gpu': torch.cuda.get_device_name(0), 'torch_version': torch.__version__, 'precision': 'FP32',
                  'TF32': False, 'random_seed': 20260915, 'score_mode': 'natural-order autoregressive',
                  'score_formula': 'log P(mutant amino acid) - log P(wild-type amino acid)',
                  'head_used_for_scoring': 'calibrated ProteinMPNN landscape; auxiliary training heads not used for scoring',
                  'strict_state_loading': True, 'all_state_entries_loaded': len(ckpt['state_dict']),
                  'training_performed': False, 'absolute_score_calibration_performed': False}
    dump(out / 'provenance.json', provenance)
    preparations = json.loads((ROOT / 'metadata/preparation.json').read_text())
    summaries = []
    with torch.inference_mode():
        for dataset_name in ['Test12K', 'DTM']:
            source = ROOT / f'features/{dataset_name}.pt'
            assert digest(source) == preparations['datasets'][dataset_name]['feature_cache_sha256']
            manifest_path = ROOT / f'manifests/{dataset_name}.csv'
            assert digest(manifest_path) == preparations['datasets'][dataset_name]['frozen_manifest_sha256']
            features = torch.load(source, map_location='cpu', weights_only=True)
            manifest = read_csv(manifest_path)
            destination = out / dataset_name
            (destination / 'per_protein_predictions').mkdir(parents=True)
            frames, runtime = [], []
            for number, (protein, group) in enumerate(manifest.groupby('protein', sort=True), 1):
                feature = features[protein]
                batch = {k: v.to('cuda') for k, v in feature['batch'].items()}
                start = time.perf_counter()
                landscape, _ = model_compute(model.model, batch, args.model_type, getattr(args, 'no_log_scale', False))
                assert landscape.shape == (1, len(feature['sequence']), 21)
                positions = torch.tensor(group.position.to_numpy(dtype=np.int64) - 1, device='cuda')
                mutants = torch.tensor(['ACDEFGHIKLMNPQRSTVWYX'.index(a) for a in group['mut']], device='cuda')
                prediction = landscape[0, positions, mutants].cpu().numpy().astype(np.float64)
                assert np.isfinite(prediction).all()
                wt = landscape.gather(-1, batch['input_ids'].unsqueeze(-1))
                assert float(wt.abs().max()) < 1e-5
                # No target tensors enter model_compute; labels are attached after inference.
                result = group[KEYS + ['wt', 'position', 'mut', 'target_stability_score']].copy()
                result.insert(0, 'checkpoint_name', checkpoint_path.name)
                result.insert(1, 'dataset', dataset_name)
                result['predicted_stability_score'] = prediction
                result.to_csv(destination / 'per_protein_predictions' / f'{protein}.csv', index=False)
                frames.append(result)
                runtime.append({'protein': protein, 'length': len(feature['sequence']), 'seconds': time.perf_counter() - start})
                if number % 50 == 0 or number == len(features):
                    print(checkpoint_path.name, dataset_name, f'{number}/{len(features)}', flush=True)
                del batch, landscape
            frame = pd.concat(frames, ignore_index=True)
            assert len(frame) == len(manifest) and set(map(tuple, frame[KEYS].to_numpy())) == set(map(tuple, manifest[KEYS].to_numpy()))
            assert not frame.duplicated(KEYS).any()
            frame.to_csv(destination / 'combined_predictions.csv', index=False)
            per_protein, summary = metrics(frame, dataset_name)
            per_protein.to_csv(destination / 'per_protein_metrics.csv', index=False)
            pd.DataFrame(runtime).to_csv(destination / 'inference_progress.csv', index=False)
            summary.update(checkpoint_name=checkpoint_path.name, checkpoint_id=checkpoint_id, stored_epoch=record['epoch'])
            dump(destination / 'summary.json', summary)
            summaries.append(summary)
            print('SUMMARY', json.dumps(summary), flush=True)
    assert digest(checkpoint_path) == record['file_sha256']
    dump(out / 'complete.json', {'status': 'PASS', 'checkpoint_unchanged': True, 'summaries': summaries, **environment()})


def summarize():
    guard()
    import numpy as np
    import pandas as pd
    inventory = json.loads((ROOT / 'metadata/checkpoint_inventory.json').read_text())
    identities, all_summaries = [], []
    for i, record in enumerate(inventory['checkpoints']):
        checkpoint_id = f'C{i + 1:02d}_' + Path(record['name']).stem
        complete_path = ROOT / 'results' / checkpoint_id / 'complete.json'
        assert complete_path.is_file(), 'Incomplete checkpoint: ' + record['name']
        complete = json.loads(complete_path.read_text())
        assert complete['status'] == 'PASS'
        provenance = json.loads((complete_path.parent / 'provenance.json').read_text())
        identity = {'checkpoint_id': checkpoint_id, 'checkpoint_name': record['name'],
                    'stored_epoch': record['epoch'], 'global_step': record['global_step'],
                    'checkpoint_path': record['path'], 'file_sha256': record['file_sha256'],
                    'state_dict_sha256': record['state_dict_sha256'], 'backbone_state_sha256': provenance['backbone_state_sha256']}
        identity.update(record['hyper_parameters']['args'])
        identities.append(identity)
        for summary in complete['summaries']:
            dataset_name = summary['dataset']
            frame = read_csv(complete_path.parent / dataset_name / 'combined_predictions.csv')
            repeated_pp, repeated_summary = metrics(frame, dataset_name)
            for name in ['spearman', 'pearson', 'auroc', 'auprc', 'enrich_at_10']:
                for suffix in ['_mean', '_sample_sd', '_n']:
                    np.testing.assert_allclose(summary[name + suffix], repeated_summary[name + suffix], rtol=0, atol=1e-12)
            all_summaries.append(summary)
        assert digest(Path(record['path'])) == record['file_sha256']
    table = pd.DataFrame(all_summaries)
    table['spearman_rank_within_dataset'] = table.groupby('dataset').spearman_mean.rank(ascending=False, method='min').astype(int)
    table['pearson_rank_within_dataset'] = table.groupby('dataset').pearson_mean.rank(ascending=False, method='min').astype(int)
    table = table.sort_values(['dataset', 'spearman_rank_within_dataset'])
    table.to_csv(ROOT / 'macro_results_all_checkpoints.csv', index=False)
    pd.DataFrame(identities).to_csv(ROOT / 'checkpoint_identity.csv', index=False)
    winners = table.loc[table.spearman_rank_within_dataset == 1].to_dict(orient='records')
    dump(ROOT / 'comparison_summary.json', {'status': 'COMPLETE', 'ranking_metric': 'macro Spearman',
        'scope': 'six checkpoints; Test12K 308 proteins/12102 mutations; full DTM 86 proteins/2838 mutations',
        'winners': winners, 'checkpoints': 6, 'all_12_evaluations_complete': len(table) == 12,
        'all_original_checkpoints_unchanged': True,
        'metric_recalculation_verified_from_saved_per_mutation_scores': True,
        'unique_backbone_weights': len({x['backbone_state_sha256'] for x in identities}),
        'no_global_metrics': True, 'sd_ddof': 1,
        'model_selection_caution': 'Best on these observed evaluation datasets only; no checkpoint changed or selected for publication automatically. Ranking candidates on test labels is a post-hoc audit, not independent held-out model selection.',
        **environment()})
    lines = ['# MAXWELL checkpoint 核对与逐蛋白评测', '',
             '完整评测六个 checkpoint，每个均覆盖 Test12K 的308个蛋白/12,102条突变，以及完整 DTM 的86个蛋白/2,838条突变。',
             '每个蛋白分别计算指标，再等权汇总为均值 ± 样本标准差（分母n−1），不计算global指标。以Macro Spearman作为主要排名指标，两个数据集分别排序。', '',
             '## 主要结果', '', '| 数据集 | checkpoint | 文件内epoch | Macro Spearman | Macro Pearson |', '| --- | --- | ---: | ---: | ---: |']
    for row in table.itertuples():
        lines.append(f'| {row.dataset} | {row.checkpoint_name} | {row.stored_epoch} | {row.spearman_mean:.4f} ± {row.spearman_sample_sd:.4f} | {row.pearson_mean:.4f} ± {row.pearson_sample_sd:.4f} |')
    lines += ['', '## 指标口径', '',
              '- Spearman/Pearson：每个蛋白至少3条观测，且实验值和预测均非恒定。每个指标的有效蛋白数单独列出。',
              '- Test12K实验稳定性分数=−ΔΔG。按照最新筛选图约定，严格>0.3 kcal/mol为正类；保留原始浮点数，不修约。此处不等同于旧表使用的>=0.3口径。相关性不受分类阈值影响。',
              '- DTM：原score为ΔTm，不反号；>=1°C为正类。',
              '- AUROC/AUPRC：仅纳入正负两类均存在的蛋白。AUPRC采用average precision。',
              '- Enrich@10：至少10条候选且至少一个正类，前10候选中正类比例除以该蛋白正类背景比例。并列预测按冻结清单中的顺序稳定排序。',
              '- 使用相同的FP32自然顺序ProteinMPNN景观评分，TF32关闭。BCE辅助头如存在也严格加载，但不用于替代景观评分。输入为同一份CPU准备的序列/结构特征，模型不接收实验标签。',
              '- 不因checkpoint保存的训练max_seq_length过滤长蛋白；DTM最长蛋白也纳入完整测试。', '',
              '## 身份信息与局限', '',
              'checkpoint_identity.csv列出每个文件的绝对路径、SHA256、模型权重哈希、文件内epoch、训练/验证路径和实验名称。Lightning文件内epoch是从0开始的索引，报告时不额外加减。',
              '不同checkpoint的历史训练路径不同，仅凭路径名不能确认训练集实际规模或是否与当前测试集重合。没有将这些checkpoint一概重新命名为Train226K或Train412，也没有声称全部已验证无泄漏。',
              '此处“最好”指本次评测数据上的数值排名，不自动代表泛化最优。若据此选择checkpoint，两套评测数据参与了事后选择，不能再将该选择过程称为独立验证集选择。',
              '原checkpoint未修改、覆盖、移动；没有训练、调参或根据实验值反转模型分数。', '',
              '## 文件', '',
              '- macro_results_all_checkpoints.csv：全部12组评测的五项指标、样本标准差、有效蛋白数和排名。',
              '- checkpoint_identity.csv及metadata/checkpoint_inventory.json：文件身份及历史训练配置。',
              '- results/<checkpoint_id>/<Test12K或DTM>/per_protein_predictions/*.csv：逐蛋白打分。',
              '- 同目录combined_predictions.csv、per_protein_metrics.csv、summary.json：合并打分、逐蛋白指标、汇总。',
              '- manifests、metadata/preparation.json：冻结的数据清单、输入哈希、特征准备检查。',
              '- 每个checkpoint的provenance.json记录真实显卡、精度、Slurm任务、状态字典严格加载情况。',
              '全部推理和统计在Slurm计算节点完成，没有在登录节点计算。']
    (ROOT / 'README_zh.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    print(table.to_string(index=False), flush=True)
    print('WINNERS', json.dumps(winners), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices=['prepare', 'infer', 'summarize'])
    parser.add_argument('--index', type=int)
    args = parser.parse_args()
    os.chdir(PROJECT)
    if args.mode == 'prepare':
        prepare()
    elif args.mode == 'infer':
        infer(args.index)
    else:
        summarize()


if __name__ == '__main__':
    main()

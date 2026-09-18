"""Nested group cross-validation and cluster bootstrap without test-set tuning."""
from dataclasses import asdict
from pathlib import Path
import time
import numpy as np
import torch
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
from torch import nn
from torch.utils.data import TensorDataset, DataLoader
from packet_ids.data import load_prepared
from packet_ids.model import ModelConfig, PacketCNNBiLSTM
from packet_ids.training import probabilities, write_json


def metrics(truth, predicted, classes):
    precision, recall, f1, _ = precision_recall_fscore_support(
        truth, predicted, labels=np.arange(classes), average='macro', zero_division=0)
    return {'accuracy': float(accuracy_score(truth, predicted)), 'macro_precision': float(precision),
            'macro_recall': float(recall), 'macro_f1': float(f1)}


def group_bootstrap(truth, predicted, groups, classes, iterations=100, seed=42):
    """Resample whole captures; never treat correlated packets as independent."""
    if iterations < 20:
        raise ValueError('Use at least 20 bootstrap iterations.')
    groups = np.asarray(groups)
    unique = np.unique(groups)
    if len(unique) < 2:
        raise ValueError('Bootstrap requires at least two independent capture groups.')
    members = {group: np.flatnonzero(groups == group) for group in unique}
    rng = np.random.default_rng(seed)
    draws = []
    for _ in range(iterations):
        indices = np.concatenate([members[g] for g in rng.choice(unique, len(unique), replace=True)])
        draws.append(metrics(np.asarray(truth)[indices], np.asarray(predicted)[indices], classes))
    return {'method': 'capture-cluster percentile bootstrap of out-of-fold predictions',
            'iterations': iterations, 'seed': seed, 'confidence_level': .95, 'group_count': len(unique),
            'metrics': {key: {'mean': float(np.mean([d[key] for d in draws])),
                       'std': float(np.std([d[key] for d in draws], ddof=1)),
                       'lower': float(np.percentile([d[key] for d in draws], 2.5)),
                       'upper': float(np.percentile([d[key] for d in draws], 97.5))}
                        for key in draws[0]},
            'note': 'Conditional on these out-of-fold predictions; does not retrain per bootstrap draw. '
                    'All classes remain in macro averaging, including classes absent in a resampled draw.'}


def group_splits(y, groups, folds, seed):
    y, groups = np.asarray(y), np.asarray(groups)
    labels = np.unique(y)
    if folds < 2 or any(len(np.unique(groups[y == label])) < folds for label in labels):
        raise ValueError(f'{folds}-fold validation needs at least {folds} independent capture groups per class.')
    split = StratifiedGroupKFold(n_splits=folds, shuffle=True, random_state=seed)
    result = list(split.split(np.zeros(len(y)), y, groups))
    for train, test in result:
        if set(groups[train]) & set(groups[test]):
            raise ValueError('Capture overlap detected in validation fold.')
        if not np.array_equal(np.unique(y[train]), labels) or not np.array_equal(np.unique(y[test]), labels):
            raise ValueError('A group fold lacks a class. Supply more independent captures per class.')
    return result


def nested_plan(y, groups, outer_folds=5, inner_folds=3, seed=42):
    result = []
    for index, (train, test) in enumerate(group_splits(y, groups, outer_folds, seed)):
        inner = group_splits(np.asarray(y)[train], np.asarray(groups)[train], inner_folds, seed + index + 1)
        result.append((train, test, [(train[a], train[b]) for a, b in inner]))
    return result


def fit_fixed_epochs(x, y, config, epochs, batch_size, learning_rate, seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.set_num_threads(min(4, torch.get_num_threads()))
    torch.use_deterministic_algorithms(True)
    model = PacketCNNBiLSTM(config)
    counts = np.bincount(y, minlength=config.num_classes)
    if (counts == 0).any():
        raise ValueError('Every model-fitting fold must contain every class.')
    weights = torch.tensor(np.sqrt(len(y) / (config.num_classes * counts)), dtype=torch.float32)
    criterion = nn.CrossEntropyLoss(weight=weights)
    loader = DataLoader(TensorDataset(torch.from_numpy(x.astype(np.float32) / 255).unsqueeze(1),
                                     torch.from_numpy(y.astype(np.int64))), batch_size=batch_size,
                        shuffle=True, generator=torch.Generator().manual_seed(seed))
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    model.train()
    for _ in range(epochs):
        for inputs, targets in loader:
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(inputs), targets)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.)
            optimizer.step()
    return model


def run_nested_validation(data_dir, output_dir, outer_folds=5, inner_folds=3,
                          bootstrap_iterations=100, epochs=5, batch_size=128,
                          seed=42, candidates=None):
    if not 1 <= epochs <= 200 or batch_size < 1:
        raise ValueError('Invalid epochs or batch size.')
    if bootstrap_iterations < 20:
        raise ValueError('Use at least 20 bootstrap iterations.')
    output = Path(output_dir)
    if (output / 'cross_validation.json').exists() or (output / 'folds.json').exists():
        raise ValueError('Validation output already exists. Choose a new directory.')
    data, metadata = load_prepared(data_dir)
    # Keep the deployment test partition reserved. Only train + validation form
    # the development pool used by the nested experiment.
    pool = {key: np.concatenate([data[s][key] for s in ('train', 'validation')])
            for key in ('X', 'y', 'sample_ids')}
    groups = np.concatenate([data[s].get('capture_ids',
             np.asarray([str(identifier).rsplit(':', 1)[0] for identifier in data[s]['sample_ids']]))
             for s in ('train', 'validation')])
    x, y = pool['X'], pool['y']
    labels = metadata['labels']
    plans = nested_plan(y, groups, outer_folds, inner_folds, seed)
    candidates = candidates or [
        {'conv_channels': 16, 'lstm_hidden': 24, 'dropout': .2, 'learning_rate': .001},
        {'conv_channels': 32, 'lstm_hidden': 48, 'dropout': .2, 'learning_rate': .001}]
    configs = [(ModelConfig(len(labels), **{k: v for k, v in c.items() if k != 'learning_rate'}),
                float(c.get('learning_rate', .001))) for c in candidates]
    if any(lr <= 0 for _, lr in configs):
        raise ValueError('Learning rates must be positive.')
    output.mkdir(parents=True, exist_ok=True)
    prediction = np.full(len(y), -1, dtype=np.int64)
    folds, fits = [], 0
    total_fits = outer_folds * (inner_folds * len(configs) + 1)
    started = time.time()
    for fold, (outer_train, outer_test, inners) in enumerate(plans, start=1):
        candidates_report = []
        for candidate, (config, learning_rate) in enumerate(configs):
            scores = []
            for inner_index, (fit_ids, score_ids) in enumerate(inners):
                write_json(output / 'status.json', {'state': 'cross_validating', 'fold': fold,
                    'completed_fits': fits, 'total_fits': total_fits, 'elapsed_seconds': time.time() - started})
                model = fit_fixed_epochs(x[fit_ids], y[fit_ids], config, epochs, batch_size,
                                         learning_rate, seed + fold * 100 + inner_index)
                scores.append(metrics(y[score_ids], probabilities(model, x[score_ids], batch_size).argmax(1), len(labels)))
                fits += 1
            candidates_report.append({'config': asdict(config), 'learning_rate': learning_rate,
                'inner_scores': scores, 'mean_macro_f1': float(np.mean([s['macro_f1'] for s in scores]))})
        selected = int(np.argmax([c['mean_macro_f1'] for c in candidates_report]))
        config, learning_rate = configs[selected]
        model = fit_fixed_epochs(x[outer_train], y[outer_train], config, epochs, batch_size, learning_rate, seed + fold * 1000)
        prediction[outer_test] = probabilities(model, x[outer_test], batch_size).argmax(1)
        fits += 1
        folds.append({'fold': fold, 'train_groups': sorted(set(groups[outer_train].tolist())),
            'evaluation_groups': sorted(set(groups[outer_test].tolist())),
            'inner_groups': [{'train': sorted(set(groups[a].tolist())), 'validation': sorted(set(groups[b].tolist()))}
                             for a, b in inners],
            'train_count': len(outer_train), 'evaluation_count': len(outer_test),
            'candidates': candidates_report, 'selected_candidate': selected,
            'metrics': metrics(y[outer_test], prediction[outer_test], len(labels))})
        write_json(output / 'folds.json', folds)
        print(f'Outer fold {fold}/{outer_folds}; completed {fits}/{total_fits} model fits', flush=True)
    if (prediction < 0).any():
        raise RuntimeError('Not every development sample received an out-of-fold prediction.')
    result = {'dataset': metadata['dataset'], 'artifact_kind': 'test_fixture_only' if metadata.get('test_fixture') else 'ciciot2023_nested_validation',
        'outer_folds': outer_folds, 'inner_folds': inner_folds, 'epochs_per_fit': epochs,
        'candidate_count': len(configs), 'completed_model_fits': fits, 'seed': seed,
        'sample_count': len(y), 'capture_group_count': len(np.unique(groups)), 'labels': labels,
        'selection_metric': 'mean inner-fold macro F1', 'pool': 'train + validation only; reserved test excluded',
        'epoch_policy': 'Fixed epochs in every fit; no outer evaluation data used for stopping or tuning.',
        'out_of_fold_metrics': metrics(y, prediction, len(labels)),
        'confusion_matrix': confusion_matrix(y, prediction, labels=np.arange(len(labels))).tolist(),
        'bootstrap': group_bootstrap(y, prediction, groups, len(labels), bootstrap_iterations, seed),
        'elapsed_seconds': time.time() - started, 'folds': folds,
        'note': 'This experiment evaluates a selection procedure. It does not replace the deployment checkpoint.'}
    np.savez_compressed(output / 'out_of_fold.npz', y=y, predicted=prediction, capture_ids=groups, sample_ids=pool['sample_ids'])
    write_json(output / 'cross_validation.json', result)
    write_json(output / 'status.json', {'state': 'complete', 'completed_fits': fits, 'total_fits': total_fits})
    return result

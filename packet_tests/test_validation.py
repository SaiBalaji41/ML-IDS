import hashlib
import json
import numpy as np
import pytest
from packet_ids.validation import nested_plan, group_bootstrap, run_nested_validation


def development_groups():
    groups = np.repeat([f'dev-{i}' for i in range(12)], 4)
    labels = np.repeat([i % 2 for i in range(12)], 4)
    return labels, groups


def test_nested_splits_keep_outer_data_out_of_selection():
    y, groups = development_groups()
    plans = nested_plan(y, groups, outer_folds=3, inner_folds=2)
    evaluated = []
    for fit, evaluate, inners in plans:
        evaluated.extend(evaluate)
        assert set(groups[fit]).isdisjoint(groups[evaluate])
        for train, validation in inners:
            assert set(train) <= set(fit) and set(validation) <= set(fit)
            assert set(groups[train]).isdisjoint(groups[validation])
            assert set(groups[validation]).isdisjoint(groups[evaluate])
    assert sorted(evaluated) == list(range(len(y)))
    with pytest.raises(ValueError, match='independent capture'):
        nested_plan([0, 0, 1, 1], ['a', 'a', 'b', 'b'])


def test_cluster_bootstrap_reports_reproducible_intervals():
    y, groups = development_groups()
    pred = y.copy(); pred[:4] = 1 - pred[:4]
    result = group_bootstrap(y, pred, groups, 2, iterations=100)
    assert result == group_bootstrap(y, pred, groups, 2, iterations=100)
    assert result['group_count'] == 12
    assert result['metrics']['accuracy']['lower'] < result['metrics']['accuracy']['upper'] <= 1


def test_nested_training_and_bootstrap_end_to_end(trained_fixture, tmp_path):
    metadata = json.loads((trained_fixture[1]/'metadata.json').read_text())
    y, groups = development_groups()
    prepared = tmp_path/'prepared'; prepared.mkdir()
    checksums = {}
    for split, indices in [('train', np.arange(32)), ('validation', np.arange(32,48)), ('test', np.arange(8))]:
        labels = y[indices]
        ids = groups[indices] if split != 'test' else np.array([f'reserved-{i//4}' for i in range(8)])
        if split == 'test':
            labels = np.array([0]*4+[1]*4)
        x = np.zeros((len(indices), 1024), np.uint8)
        x[:, :64] = (30 + labels * 180)[:, None]
        x[:, 64] = np.arange(len(indices)) + (0 if split == 'train' else 100)
        path = prepared/f'{split}.npz'
        np.savez_compressed(path, X=x, y=labels, lengths=np.full(len(indices), 65),
                            sample_ids=np.array([f'{g}:{i}' for i,g in enumerate(ids)]), capture_ids=ids)
        checksums[split] = hashlib.sha256(path.read_bytes()).hexdigest()
    metadata['split_sha256'] = checksums
    (prepared/'metadata.json').write_text(json.dumps(metadata))
    result = run_nested_validation(prepared, tmp_path/'cv', outer_folds=3, inner_folds=2,
        bootstrap_iterations=100, epochs=1, batch_size=16,
        candidates=[{'conv_channels':2,'lstm_hidden':2,'dropout':0,'learning_rate':.001}])
    assert result['artifact_kind'] == 'test_fixture_only' and result['completed_model_fits'] == 9
    assert result['sample_count'] == 48 and np.asarray(result['confusion_matrix']).sum() == 48
    assert result['bootstrap']['iterations'] == 100
    assert all(not any(g.startswith('reserved') for g in f['train_groups']+f['evaluation_groups']) for f in result['folds'])
    assert (tmp_path/'cv/out_of_fold.npz').exists()

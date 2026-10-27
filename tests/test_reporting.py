import json
import numpy as np
import pytest
from tools.report_results import online_auc, read_run


def test_constant_curve_has_paper_integration_bounds():
    assert online_auc(np.full(1000,200.)) == pytest.approx(200.*(127000-127))


def test_trailing_mean_uses_only_previous_and_current_episodes():
    values=np.arange(1,101,dtype=float)
    expected=[values[max(0,i-49):i+1].mean() for i in range(len(values))]
    assert online_auc(values,formal=False)==pytest.approx(np.trapezoid(expected,np.arange(1,101)*127))


def test_incomplete_curve_is_rejected():
    with pytest.raises(ValueError,match='1000'):
        online_auc([1.,2.])


def test_smoke_metrics_cannot_be_published_as_formal_results(tmp_path):
    (tmp_path/'metrics.jsonl').write_text(json.dumps(dict(smoke=True,phase='train'))+'\n')
    with pytest.raises(ValueError,match='smoke'):
        read_run(tmp_path)

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest

from quality_intelligence.analysis import explain, feasible, investigate, optimize
from quality_intelligence.data import FEATURES, generate, oracle, split, validate
from quality_intelligence.modeling import anomaly, calibrate, policies, predict, train
from quality_intelligence.pipeline import run


@pytest.fixture(scope="module")
def data():
    return generate(4000, 7, 0.12, 100)


def test_reproducibility(data):
    pd.testing.assert_frame_equal(data[0], generate(4000, 7, 0.12, 100)[0])
    assert 0.09 < data[0].defect.mean() < 0.15
    assert np.all(oracle(data[0].fillna(0), data[1]) >= 0)


def test_schema(data):
    assert validate(data[0])["rows"] == 4000
    for invalid in [
        data[0].iloc[:0],
        data[0].drop(columns=FEATURES[0]),
        pd.concat([data[0], data[0].iloc[:1]]),
    ]:
        with pytest.raises(ValueError):
            validate(invalid)


@pytest.mark.parametrize("n,rate", [(0, 0.1), (100, 0), (100, 0.7)])
def test_invalid_generation(n, rate):
    with pytest.raises(ValueError):
        generate(n, rate=rate)


def test_split(data):
    sets = split(data[0])
    for i, left in enumerate(sets):
        for right in sets[i + 1 :]:
            assert set(left.batch).isdisjoint(right.batch)
            assert left.timestamp.max() < right.timestamp.min()
    assert not {"defect", "simulated_probability", "defect_category"}.intersection(FEATURES)


def test_models(data):
    fit, tune, cal, policy, test = split(data[0])
    models, trials = train(fit, tune)
    assert len(models) == 4 and len(trials) == 8
    for model in models.values():
        bundle, scores = calibrate(model, cal, policy)
        p = predict(bundle, test.assign(equipment="UNSEEN"))
        assert np.isfinite(p).all() and ((0 <= p) & (p <= 1)).all()
        assert len(scores) == 3
        assert explain(model, test.head(5)).shape == (5, len(FEATURES))
    thresholds = policies(test.defect, p)
    assert all(0 <= t <= 1 for t in thresholds.values())
    detector = anomaly(fit)
    assert set(detector.predict(test[FEATURES])).issubset({-1, 1})
    batches, report = investigate(test, p, {"welding_energy": 0.1})
    assert len(batches) and "hypotheses" in report
    optimized = optimize(
        bundle,
        test[FEATURES].fillna(fit[FEATURES].median()).head(5),
        data[1],
        seeds=(1,),
        budget=10,
    )
    assert optimized.constraint_violations.sum() == 0


def test_single_class(data):
    fit, tune, *_ = split(data[0])
    with pytest.raises(ValueError):
        train(fit.assign(defect=0), tune)
    assert all(0 <= t <= 1 for t in policies([0] * 99 + [1], np.linspace(0, 1, 100)).values())


def test_constraints():
    assert feasible([50, 10, 5, 30])
    assert not feasible([65, 7, 5, 30])
    assert not feasible([10, 10, 5, 30])


def test_end_to_end(tmp_path):
    result = run(tmp_path, n=4000, seed=7, rate=0.12, batch_size=100, quick=True)
    assert len(result["models"]) == 4
    assert (tmp_path / "models/system.joblib").exists()
    assert (tmp_path / "reports/investigation.md").exists()


def test_dashboard():
    app = AppTest.from_file(str(Path("app/dashboard.py").resolve()), default_timeout=90).run()
    assert not app.exception
    if Path("models/system.joblib").exists():
        assert len(app.tabs) == 6
        app.slider[0].set_value(0.2).run()
        assert not app.exception


def test_dashboard_missing_artifacts(tmp_path, monkeypatch):
    monkeypatch.setenv("QUALITY_ARTIFACT_ROOT", str(tmp_path))
    app = AppTest.from_file(str(Path("app/dashboard.py").resolve()), default_timeout=90).run()
    assert not app.exception
    assert any("No trained artifacts" in item.value for item in app.info)


def test_label_mechanism(data):
    from quality_intelligence.data import CENTERS

    nominal = pd.DataFrame([CENTERS])
    deviating = nominal.assign(welding_energy=65.0, assembly_pressure=6.5)
    assert oracle(deviating, data[1])[0] > oracle(nominal, data[1])[0]


def test_invalid_measurements(data):
    for invalid in [
        data[0].assign(defect=2),
        data[0].assign(current=np.inf),
        data[0].assign(current=np.nan),
    ]:
        with pytest.raises(ValueError):
            validate(invalid)


def test_calibration_and_feature_errors(data):
    fit, _tune, cal, policy, test = split(data[0])
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline

    model = make_pipeline(SimpleImputer(), LogisticRegression(max_iter=1000))
    model.fit(fit[FEATURES], fit.defect)
    with pytest.raises(ValueError):
        calibrate(model, cal.assign(defect=0), policy)
    with pytest.raises(KeyError):
        predict({"model": model, "calibrator": None}, test.drop(columns=FEATURES[0]))

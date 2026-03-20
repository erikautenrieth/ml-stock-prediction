import numpy as np

from backend.ml.training.extra_tree import ExtraTreesTrainer
from backend.workflows import train as train_workflow


def _make_dataset(n_samples: int = 220, n_features: int = 8):
    rng = np.random.default_rng(42)
    x = rng.normal(size=(n_samples, n_features))
    signal = x[:, 0] + 0.35 * x[:, 1] - 0.15 * x[:, 2]
    y = (signal > 0).astype(int)
    return x, y


def test_fit_calibrated_model_uses_isotonic_when_possible():
    x_train, y_train = _make_dataset()
    trainer = ExtraTreesTrainer()

    model, calibration_method = train_workflow._fit_calibrated_model(
        trainer=trainer,
        params=trainer.default_params(),
        x_train=x_train,
        y_train=y_train,
    )

    proba = model.predict_proba(x_train[:5])

    assert calibration_method == "isotonic"
    assert proba.shape == (5, 2)
    assert np.all((proba >= 0.0) & (proba <= 1.0))


class _FakeCalibratedClassifierCV:
    def __init__(self, estimator, method, cv):
        self.estimator = estimator
        self.method = method
        self.cv = cv
        self._fitted = False

    def fit(self, x, y, sample_weight=None):
        if self.method == "isotonic":
            raise ValueError("isotonic failed in test")
        self.estimator.fit(x, y, sample_weight=sample_weight)
        self._fitted = True
        return self

    def predict(self, x):
        return self.estimator.predict(x)

    def predict_proba(self, x):
        return self.estimator.predict_proba(x)


def test_fit_calibrated_model_falls_back_to_sigmoid(monkeypatch):
    x_train, y_train = _make_dataset()
    trainer = ExtraTreesTrainer()

    monkeypatch.setattr(
        train_workflow,
        "CalibratedClassifierCV",
        _FakeCalibratedClassifierCV,
    )

    model, calibration_method = train_workflow._fit_calibrated_model(
        trainer=trainer,
        params=trainer.default_params(),
        x_train=x_train,
        y_train=y_train,
    )

    proba = model.predict_proba(x_train[:5])

    assert calibration_method == "sigmoid"
    assert model.method == "sigmoid"
    assert proba.shape == (5, 2)

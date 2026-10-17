"""
Model registry. To add a model: write a subclass of RatingModel in this package and add it
to MODELS. Every script then accepts it via --model <name>, tunes it with backtest.py, logs
its picks with track.py and replays it with simulate.py.
"""
from common import load_params
from .elo import EloModel
from .margin import MarginModel
from .prior import PriorMarginModel
from .efficiency import EfficiencyModel
from .ensemble import EnsembleModel
from .market import MarketModel

MODELS = {EloModel.name: EloModel, MarginModel.name: MarginModel, PriorMarginModel.name: PriorMarginModel,
          EfficiencyModel.name: EfficiencyModel, EnsembleModel.name: EnsembleModel,
          MarketModel.name: MarketModel}
DEFAULT_MODEL = EloModel.name


def get_model(name=DEFAULT_MODEL, **params):
    """Instantiate a model by name with explicit parameters (defaults if none given)."""
    try:
        return MODELS[name](**params)
    except KeyError:
        raise SystemExit(f"Unknown model '{name}'. Available: {', '.join(MODELS)}")


def fitted_model(name, games):
    """A model built with its tuned parameters (data/params/<name>.json) and replayed on `games`."""
    model = get_model(name, **load_params(name))
    model.fit(games)
    return model


def add_model_argument(parser):
    parser.add_argument("--model", default=DEFAULT_MODEL, choices=sorted(MODELS),
                        help=f"rating model to use (default: {DEFAULT_MODEL})")

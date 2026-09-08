"""Reference-axis planning preserves real sample identities across input widths."""

from copy import deepcopy
from types import SimpleNamespace

import yaml

from msi_autoencoder_wrapper.binners.binners_strategies.linear_binner import LinearBinning
from msi_autoencoder_wrapper.data import TargetSchema
from msi_autoencoder_wrapper.models.datasets.splitting.partitions import SplitManifest
from msi_autoencoder_wrapper.runtime.workflows import configured


def test_range_planning_freezes_population_and_split_before_expanding_axis(tmp_path, monkeypatch):
    built = []
    assignments = {"train": (2, 7), "validation": (11,), "test": (19,)}

    def build(parameters, *, split_seed):
        built.append(deepcopy(parameters))
        binner = LinearBinning(**parameters["binning"]["parameters"])
        dataset = SimpleNamespace(
            _split_config={"seed": split_seed},
            create_partitions=lambda: SimpleNamespace(manifest=SplitManifest("fixture", split_seed, assignments)),
            get_target_schemas=lambda: {"molecule": TargetSchema("molecule", "multi_label", ("C2H4|+H",))},
        )
        wrapper = SimpleNamespace(active_context=SimpleNamespace(binner=binner),
                                  context_manager=SimpleNamespace(get_context_config=lambda: {}))
        return wrapper, dataset

    monkeypatch.setattr(configured, "_build_planning_pipeline", build)
    reference = {"strategy": "LinearBinning", "parameters": {"bin_step": .55, "x_min": 200., "x_max": 900.}}
    factory = {
        "split_reference_binning": reference,
        "dataset": {"strategy": "PixelDataset", "parameters": {"subset": {"fraction": .1, "seed": 42}}},
        "variant": {"name": "fixture", "preset": "CNNAutoencoder", "parameters": {"latent_dim": 3}},
    }
    tasks = []
    for limits in ((200., 900.), (100., 3000.)):
        parameters = deepcopy(factory)
        parameters["binning"] = deepcopy(reference)
        parameters["binning"]["parameters"].update(x_min=limits[0], x_max=limits[1])
        tasks.append({"parameters": {"factory_parameters": parameters}, "reproducibility": {"common_seeds": {"split": 42}}})
    resolved = configured.resolve_single_image_campaign(tasks, tmp_path)
    assert len(built) == 3  # One reference dataset, then one model descriptor per axis.
    for item in built[1:]:
        assert item["dataset"]["parameters"]["subset"] == {"method": "source_indices", "indices": [2, 7, 11, 19]}
    assert tasks[0]["parameters"]["factory_parameters"]["dataset"]["parameters"]["subset"] == {"fraction": .1, "seed": 42}
    assert resolved[0]["resolved"]["split_manifest"] == resolved[1]["resolved"]["split_manifest"]
    assert resolved[0]["resolved"]["model_config"] != resolved[1]["resolved"]["model_config"]
    with open(resolved[1]["resolved"]["split_manifest"]) as stream:
        assert yaml.safe_load(stream)["assignments"] == {k: list(v) for k, v in assignments.items()}

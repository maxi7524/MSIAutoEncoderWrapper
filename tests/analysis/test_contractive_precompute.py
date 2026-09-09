"""Tests for the contractive analyses' precompute runner and its settings contract."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch
import yaml

from msi_autoencoder_wrapper.analysis.autoencoder.experiments import contractive_precompute as precompute

SETTINGS_PATH = Path(
    "assets/experiments/autoencoder_architecture/notebooks/05_09_26_contractive_expaned/"
    "analysis_settings.yaml"
)


@pytest.fixture()
def settings(tmp_path: Path) -> dict:
    """A minimal settings mapping with one configured analysis."""
    path = tmp_path / "analysis_settings.yaml"
    path.write_text(
        yaml.safe_dump({
            "workspace": "data/workspace",
            "model_store": "data/workspace/models/context",
            "model_context": "context",
            "device": "cuda",
            "sample_seed": 7,
            "campaigns": {"alpha": "campaign-alpha"},
            "analyses": {
                "prediction_sweep": {
                    "campaigns_used": ["alpha"],
                    "output_directory": str(tmp_path / "results"),
                }
            },
        }),
        encoding="utf-8",
    )
    return precompute.read_settings(path)


class TestSettings:
    """Reading and merging the settings file."""

    def test_paths_are_resolved_and_the_source_is_recorded(self, settings: dict) -> None:
        assert isinstance(settings["workspace"], Path)
        assert settings["settings_path"].name == "analysis_settings.yaml"

    def test_analysis_settings_merge_shared_and_specific_entries(self, settings: dict) -> None:
        merged = precompute.analysis_settings(settings, "prediction_sweep")

        assert merged["sample_seed"] == 7                    # shared
        assert merged["campaigns_used"] == ["alpha"]          # analysis-specific
        assert isinstance(merged["output_directory"], Path)
        assert "analyses" not in merged

    def test_unknown_analysis_is_rejected(self, settings: dict) -> None:
        with pytest.raises(KeyError):
            precompute.analysis_settings(settings, "not_a_routine")

    def test_unconfigured_but_registered_analysis_is_rejected(self, settings: dict) -> None:
        # Registered in code but absent from this settings file: the failure must name
        # the settings file rather than pretend the routine does not exist.
        with pytest.raises(KeyError):
            precompute.analysis_settings(settings, "latent_geometry_sweep")

    def test_missing_settings_file_is_rejected(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            precompute.read_settings(tmp_path / "absent.yaml")


class TestDevicePolicy:
    """The configured accelerator is a requirement, not a preference."""

    def test_cuda_request_without_cuda_is_refused(self, settings: dict, monkeypatch) -> None:
        # Regression: an environment re-synced to a processor-only build once turned a
        # several-minute run into hours without any signal.
        monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

        with pytest.raises(RuntimeError, match="without CUDA"):
            precompute.resolve_device(settings)

    def test_processor_run_is_possible_but_must_be_asked_for(self, settings: dict, monkeypatch) -> None:
        monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

        assert precompute.resolve_device(settings, allow_cpu=True).type == "cpu"

    def test_configured_processor_needs_no_opt_in(self, tmp_path: Path) -> None:
        path = tmp_path / "settings.yaml"
        path.write_text(yaml.safe_dump({"device": "cpu", "analyses": {}}), encoding="utf-8")

        assert precompute.resolve_device(precompute.read_settings(path)).type == "cpu"


class TestRunCommand:
    """The command a notebook prints is generated, never hand-written."""

    def test_background_form_detaches_and_captures_a_log(self, settings: dict) -> None:
        command = precompute.run_command(settings, "prediction_sweep")

        assert command.startswith("nohup ")
        assert command.rstrip().endswith("&")
        assert "--analysis prediction_sweep" in command
        assert "prediction_sweep.log" in command

    def test_foreground_form_is_the_same_invocation(self, settings: dict) -> None:
        foreground = precompute.run_command(settings, "prediction_sweep", background=False)

        assert not foreground.startswith("nohup")
        assert "-m msi_autoencoder_wrapper.analysis.autoencoder.experiments.contractive_precompute" in foreground


class TestLoading:
    """Loading reports how to produce what is missing."""

    def test_absent_table_names_the_command_that_produces_it(self, settings: dict) -> None:
        with pytest.raises(FileNotFoundError) as failure:
            precompute.load_table(settings, "prediction_sweep", "prediction_metrics")

        assert "--analysis prediction_sweep" in str(failure.value)

    def test_absent_metadata_names_the_command_too(self, settings: dict) -> None:
        with pytest.raises(FileNotFoundError) as failure:
            precompute.load_metadata(settings, "prediction_sweep")

        assert "--analysis prediction_sweep" in str(failure.value)


class TestRoutineRegistration:
    """Every configured analysis must actually be runnable."""

    def test_registration_survives_module_execution_order(self) -> None:
        # Regression: a routine appended below the `__main__` guard was defined after
        # the command line had already been parsed, so `--analysis <name>` was rejected
        # as an invalid choice while importing the module showed it as registered.
        source = Path(precompute.__file__).read_text(encoding="utf-8")
        guard = 'if __name__ == "__main__":'
        assert source.index(guard) > source.rindex("@_routine("), (
            "the entry-point guard must be the last statement, after every routine"
        )

    @pytest.mark.skipif(not SETTINGS_PATH.is_file(), reason="contractive settings not present")
    def test_every_configured_analysis_is_registered(self) -> None:
        configured = set(precompute.read_settings(SETTINGS_PATH).get("analyses", {}))

        assert configured <= set(precompute.ROUTINES), (
            f"configured but not registered: {sorted(configured - set(precompute.ROUTINES))}"
        )

    @pytest.mark.skipif(not SETTINGS_PATH.is_file(), reason="contractive settings not present")
    def test_every_configured_analysis_has_an_output_directory(self) -> None:
        settings = precompute.read_settings(SETTINGS_PATH)

        for analysis in settings.get("analyses", {}):
            merged = precompute.analysis_settings(settings, analysis)
            assert merged["output_directory"].name.endswith("_results")

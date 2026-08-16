"""Tests for wrapper-wide floating-point dtype configuration."""

from __future__ import annotations

import torch

from msi_autoencoder_wrapper.core.wrapper import MSIAutoEncoderWrapper


def test_wrapper_defaults_to_float32(tmp_path) -> None:
    """The default numerical execution type is float32."""
    wrapper = MSIAutoEncoderWrapper(project_path=str(tmp_path))

    assert wrapper.dtype is torch.float32


def test_wrapper_accepts_a_named_floating_dtype(tmp_path) -> None:
    """A wrapper-wide dtype override is resolved once during construction."""
    wrapper = MSIAutoEncoderWrapper(project_path=str(tmp_path), dtype="float64")

    assert wrapper.dtype is torch.float64

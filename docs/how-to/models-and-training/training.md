# Train a model

Training executes ordered phases with separate freezing, losses, optimizers,
devices, DataLoader settings, and checkpoint behavior.

## Purpose and available operations

### Training lifecycle

Training partitions the dataset, fits context normalization on the training
partition, builds each phase loss graph and optimizer, processes train and
validation batches, applies early stopping, and optionally saves the best model.

### Loss groups

Autoencoder loss configuration separates reconstruction, contrastive, and named
head losses. Each loss has a registered target, constructor parameters, and an
optional weight.

## Detailed instructions

### Define a complete training configuration

```python
training = {
    "seed": 0,
    "patience": 10,
    "compute_device": "cpu",
    "preprocessing_device": "cpu",
    "checkpoint": {"enabled": True, "restore_best": True},
    "phases": [
        {
            "phase_name": "reconstruction",
            "epochs": 20,
            "freeze": [],
            "batch_size": 32,
            "dataloader": {
                "num_workers": 0,
                "shuffle": True,
                "drop_last": False,
                "pin_memory": False,
            },
            "optimizer": {
                "type": "AdamW",
                "params": {"lr": 1e-3, "weight_decay": 1e-4},
            },
            "criterions": {
                "reconstruction": {
                    "mse": {
                        "target": "MSELoss",
                        "params": {},
                        "weight": 1.0,
                    }
                }
            },
        }
    ],
}

history = wrapper.models_manager.fit(training)
```

Phase values override top-level devices. `freeze` contains model child paths.
The optimizer type must exist in `torch.optim`; an omitted optimizer uses AdamW
with `lr=1e-3` and `weight_decay=1e-4`. An incomplete optimizer block is removed
and replaced by that default.

### Configure named-head losses

```python
head_phase = {
    "criterions": {
        "reconstruction": {
            "mse": {"target": "MSELoss", "params": {}, "weight": 1.0}
        },
        "heads": {
            "molecule_primary": {
                "bce": {
                    "target": "MultiLabelBCELoss",
                    "params": {},
                    "weight": 0.2,
                }
            }
        },
    }
}
```

The head identifier must exist in the compiled model and carry a target-field
binding. Target masks suppress unavailable labels.

### Estimate resources before training

```python
estimate = wrapper.models_manager.estimate_training_resources(
    training,
    resource_limits={"memory_bytes": 8_000_000_000},
    auto_adjust_batch_size=False,
    safety_factor=1.25,
    print_return=True,
)
```

Estimation uses the same DataLoader and preprocessing configuration as training.
`auto_adjust_batch_size=True` may reduce batch size to satisfy supplied limits.

### Manage checkpoints and cache

With checkpoints enabled, an improvement writes configuration, weights, and
history. `restore_best=True` restores the best state after training. Use
`wrapper.models_manager.clear_training_cache()` when cached loss preparation
must not be reused by another run.

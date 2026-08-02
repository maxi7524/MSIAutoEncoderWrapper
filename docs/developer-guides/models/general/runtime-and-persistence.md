# Integrate model runtime and persistence

A model family requires runtime detection, trained-state handling, portable
configuration, and artifact reconstruction.

## Runtime scope and invariants

### Runtime interface

The interface wraps a Torch module and context dependencies while exposing
family operations. It must reject operations requiring trained weights when
`trained=False`.

### Artifact contract

The saved model descriptor must identify family, graph components, constructor
parameters, target bindings, and state dictionary compatibility.

## Implementation instructions

### Extend detection and attachment

Add precise family detection to `ModelRuntimeProxy._detect_model_type()` and
construct the correct interface in attachment/compilation paths. Avoid detection
based only on class-name substrings.

### Extend `ModelLoader`

Reconstruct the graph from the family registry, load weights with strictness,
and preserve trained state. Artifact fingerprints must change when relevant
configuration or weights change.

### Add schema round-trip

Save, load in a new wrapper without original runtime objects, and compare model
outputs on deterministic input.

"""Management of named cohorts without replacing the local active context."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional

from ....utils.exceptions import raise_validation_error
from ....utils.logger import get_custom_logger
from .context import CohortContext, CohortMember, ModelReference
from ....models.model_loader import ModelLoader
from ....readers.readers_manager import ReaderManager

logger = get_custom_logger(__name__)


class CohortManagerProxy:
    """Create, persist, and activate immutable cohort contexts."""

    def __init__(self, wrapper_ref: Any) -> None:
        self._wrapper = wrapper_ref
        self._contexts: Dict[str, CohortContext] = {}
        self._active_name: Optional[str] = None

    @property
    def active_context(self) -> Optional[CohortContext]:
        """Return the active cohort without resolving a default image."""
        return self._contexts.get(self._active_name) if self._active_name else None

    def create(self, name: str) -> CohortContext:
        """Create or return an empty named cohort."""
        clean = self._validate_name(name)
        context = self._contexts.setdefault(clean, CohortContext(name=clean))
        return context

    def set_images(self, images: Iterable[str], name: Optional[str] = None) -> CohortContext:
        """Replace all members of one cohort with configured local contexts."""
        context = self._resolve_context(name)
        members = tuple(self._resolve_member(image_key) for image_key in images)
        updated = context.with_members(members)
        self._contexts[updated.name] = updated
        return updated

    def add_image(self, image_key: str, name: Optional[str] = None) -> CohortContext:
        """Append one configured image to a cohort."""
        context = self._resolve_context(name)
        if any(member.image_key == image_key for member in context.members):
            return context
        updated = context.with_members((*context.members, self._resolve_member(image_key)))
        self._contexts[updated.name] = updated
        return updated

    def remove_image(self, image_key: str, name: Optional[str] = None) -> CohortContext:
        """Remove one image from a cohort."""
        context = self._resolve_context(name)
        updated = context.with_members(
            tuple(member for member in context.members if member.image_key != image_key)
        )
        self._contexts[updated.name] = updated
        return updated

    def set_latent(
        self, image_key: str, path: Path | str, name: Optional[str] = None
    ) -> CohortContext:
        """Attach one already materialized latent imzML reader to a member."""
        context = self._resolve_context(name)
        ReaderManager.discover_strategies()
        latent_reader = ReaderManager.get_reader("PyImzMLReader", file_path=path)
        members = []
        found = False
        for member in context.members:
            if member.image_key == image_key:
                found = True
                member = CohortMember(**{**member.__dict__, "latent_reader": latent_reader})
            members.append(member)
        if not found:
            raise_validation_error("Cohort", f"Unknown member '{image_key}'.")
        updated = context.with_members(tuple(members))
        self._contexts[updated.name] = updated
        return updated

    def set_autoencoder(
        self,
        *,
        policy: str,
        model: Any = None,
        models: Optional[Mapping[str, Any]] = None,
        name: Optional[str] = None,
    ) -> CohortContext:
        """Configure one common AE or one explicit AE per member."""
        context = self._resolve_context(name)
        if policy == "common":
            if model is None or models is not None:
                raise_validation_error(
                    "Cohort", "policy='common' requires exactly one model."
                )
            updated = CohortContext(
                name=context.name,
                members=tuple(
                    CohortMember(
                        **{
                            **member.__dict__,
                            "autoencoder_reference": None,
                        }
                    )
                    for member in context.members
                ),
                autoencoder_policy="common",
                common_autoencoder=self._resolve_model_reference(model),
            )
        elif policy == "per_member":
            if model is not None or models is None:
                raise_validation_error(
                    "Cohort", "policy='per_member' requires a models mapping."
                )
            missing = {member.image_key for member in context.members} - set(models)
            if missing:
                raise_validation_error(
                    "Cohort", f"Missing autoencoder references for: {sorted(missing)}."
                )
            updated = CohortContext(
                name=context.name,
                members=tuple(
                    CohortMember(
                        **{
                            **member.__dict__,
                            "autoencoder_reference": self._resolve_model_reference(
                                models[member.image_key]
                            ),
                        }
                    )
                    for member in context.members
                ),
                autoencoder_policy="per_member",
            )
        else:
            raise_validation_error(
                "Cohort", "policy must be 'common' or 'per_member'."
            )
        self._contexts[updated.name] = updated
        return updated

    def activate(self, name: str) -> CohortContext:
        """Activate a configured cohort without clearing the local image context."""
        context = self._resolve_context(name)
        if not context.members:
            raise_validation_error("Cohort", "Cannot activate an empty cohort.")
        self._active_name = context.name
        self._wrapper.workspace.execution_scope = "cohort"
        return context

    def deactivate(self) -> None:
        """Return routing to the retained local image context."""
        self._active_name = None
        self._wrapper.workspace.execution_scope = "local"

    def save(self, name: Optional[str] = None) -> Path:
        """Atomically save ``models/cohort_<name>/cohort.json``."""
        context = self._resolve_context(name)
        path = self._wrapper.workspace.get_models_root() / context.key / "cohort.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=path.parent, delete=False
        ) as temporary:
            json.dump(context.get_config(), temporary, indent=2, sort_keys=True)
            temporary.write("\n")
            temporary_path = Path(temporary.name)
        temporary_path.replace(path)
        return path

    def load_config(
        self,
        config: Mapping[str, Any],
        *,
        activate: bool = True,
        base_path: Optional[Path] = None,
    ) -> CohortContext:
        """Restore cohort members through their module-owned local contexts."""
        name = self._validate_name(str(config.get("name", "")))
        self.create(name)
        members = []
        for member_config in config.get("members", []):
            image_key = str(member_config.get("image_key", ""))
            context_config = member_config.get("context")
            if image_key not in self._wrapper.context_manager.config_ledger:
                if not isinstance(context_config, Mapping):
                    raise_validation_error(
                        "Cohort", f"Member '{image_key}' has no context snapshot."
                    )
                self._wrapper.context_manager.load_context_config(
                    dict(context_config),
                    img_name_or_path=image_key,
                    base_path=base_path,
                )
            member = self._resolve_member(image_key)
            latent_config = member_config.get("latent")
            if isinstance(latent_config, Mapping):
                parameters = dict(latent_config.get("parameters", {}))
                latent_path = Path(str(parameters.get("file_path", "")))
                if not latent_path.is_absolute() and base_path is not None:
                    latent_path = base_path / latent_path
                ReaderManager.discover_strategies()
                latent_reader = ReaderManager.get_reader(
                    latent_config["type"], file_path=latent_path
                )
                member = CohortMember(
                    **{**member.__dict__, "latent_reader": latent_reader}
                )
            reference = member_config.get("autoencoder")
            if reference is not None:
                member = CohortMember(
                    **{
                        **member.__dict__,
                        "autoencoder_reference": ModelReference.parse(reference),
                    }
                )
            members.append(member)
        autoencoder = config.get("autoencoder", {})
        policy = str(autoencoder.get("policy", "common"))
        common = autoencoder.get("model")
        context = CohortContext(
            name=name,
            members=tuple(members),
            autoencoder_policy=policy,
            common_autoencoder=(ModelReference.parse(common) if common else None),
        )
        self._contexts[name] = context
        return self.activate(name) if activate else context

    def _resolve_member(self, image_key: str) -> CohortMember:
        ledger = self._wrapper.context_manager.config_ledger
        bucket = ledger.get(image_key)
        if bucket is None:
            raise_validation_error(
                "Cohort",
                f"Image '{image_key}' has no configured local context.",
            )
        reader = bucket.get("reader")
        if reader is None or isinstance(reader, dict):
            raise_validation_error("Cohort", f"Image '{image_key}' has no reader.")
        try:
            config = self._wrapper.context_manager.get_context_config(image_key)
        except Exception:
            config = {"image_key": image_key}
        return CohortMember(
            image_key=image_key,
            reader=reader,
            binner=bucket.get("binner"),
            annotation_reader=bucket.get("annotation_reader"),
            latent_reader=bucket.get("latent_reader"),
            context_config=config,
        )

    def _resolve_model_reference(self, value: Any) -> ModelReference:
        reference = ModelReference.parse(value)
        lookup = reference.path or f"{reference.image_name}/{reference.model_name}"
        directory = ModelLoader.resolve_artifact_dir(
            lookup,
            workspace_root=self._wrapper.workspace.project_path_resolved,
        )
        return ModelReference(
            image_name=reference.image_name,
            model_name=reference.model_name,
            path=str(directory),
            fingerprint=ModelLoader.artifact_fingerprint(directory),
        )

    def _resolve_context(self, name: Optional[str]) -> CohortContext:
        resolved = name or self._active_name
        if resolved is None or resolved not in self._contexts:
            raise_validation_error("Cohort", "Create or activate a cohort first.")
        return self._contexts[resolved]

    @staticmethod
    def _validate_name(name: str) -> str:
        clean = str(name).strip()
        if not clean or "/" in clean or "\\" in clean:
            raise_validation_error("Cohort", "Cohort name must be a path-safe key.")
        return clean


class CohortMixin:
    """Inject cohort management alongside the local active context."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        self.cohorts = CohortManagerProxy(wrapper_ref=self)
        super().__init__(*args, **kwargs)

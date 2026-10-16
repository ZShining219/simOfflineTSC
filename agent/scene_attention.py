"""Scene-conditioned feature interfaces for future traffic-control variants.

The scene encoders do not know about a simulator, graph topology, actions or
rewards.  Replay receives only the immutable scene-context contract; the
``SceneGuidedAttention`` path remains an intentionally disabled reference
implementation whose exact rule can change without changing the adapters.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Mapping, Optional, Sequence, Tuple, Union

import torch
from torch import nn
import torch.nn.functional as F

from utils.text_grounding import Gnode
from utils.scene_context import SceneContext, SceneReplayState, scene_context_at


@dataclass(frozen=True)
class SceneCondition:
    """Runtime-only multi-event Z/G condition consumed by SGA.

    ``event_embeddings`` is Zscene and never contains entity IDs.  The
    optional ``node_event_mask`` is Gscene projected onto the policy node
    order; it is a soft grounding bias, not a hard action mask.
    """

    event_embeddings: torch.Tensor       # [B, M, D]
    event_mask: torch.Tensor             # [B, M]
    node_event_mask: Optional[torch.Tensor] = None  # [B, N, M]
    node_event_relation: Optional[torch.Tensor] = None  # [B, N, M, 3]

    def validate(self, batch: int, nodes: int, width: int):
        if self.event_embeddings.ndim != 3 or self.event_embeddings.shape[0] != batch:
            raise ValueError('event_embeddings must have shape [B, M, D]')
        if self.event_embeddings.shape[-1] != width:
            raise ValueError('event embedding width disagrees with traffic features')
        if self.event_mask.shape != self.event_embeddings.shape[:2]:
            raise ValueError('event_mask shape disagrees with event_embeddings')
        if self.node_event_mask is not None and self.node_event_mask.shape != (
                batch, nodes, self.event_embeddings.shape[1]):
            raise ValueError('node_event_mask must have shape [B, N, M]')
        if self.node_event_relation is not None and self.node_event_relation.shape != (
                batch, nodes, self.event_embeddings.shape[1], 3):
            raise ValueError('node_event_relation must have shape [B, N, M, 3]')


@dataclass(frozen=True)
class StructuredSceneMeta:
    """Public event semantics admitted by the structured scene branch.

    The dataclass deliberately has no simulator fields.  In particular,
    ``speed_factor``, internal closure parameters, positions and schedules are
    not accepted.  ``location`` is optional because node-level location is
    retained in :class:`Gnode` and need not be duplicated in the scene vector.
    """

    event_type: str = 'unknown'
    status: str = 'unknown'
    scope: str = 'unknown'
    direction: str = 'unknown'
    movement: str = 'unknown'
    # Kept as a compatibility field for old serialized metadata.  Location is
    # deliberately never read by the semantic encoder; it belongs to Gscene.
    location: str = 'unknown'
    severity: Optional[float] = None

    @classmethod
    def from_report(cls, report: Any) -> 'StructuredSceneMeta':
        """Extract only externally reported semantic fields at the boundary."""
        event_type = getattr(report, 'event_kind', None)
        if event_type is None:
            event_type = getattr(report, 'event_type', None)
        status = getattr(report, 'report_status', None)
        if status is None:
            status = getattr(report, 'status', None)
        scope = getattr(report, 'scope', None)
        direction = getattr(report, 'direction', None)
        movements = getattr(report, 'movements', ()) or ()
        movement = movements[0] if len(movements) == 1 else (
            'multiple' if len(movements) > 1 else None)
        severity = getattr(report, 'severity', None)
        if severity is None:
            category = getattr(report, 'severity_category', None)
            severity = {'mild': 1.0 / 3.0, 'moderate': 2.0 / 3.0,
                        'severe': 1.0}.get(category)
        return cls(
            event_type=event_type or 'unknown', status=status or 'unknown',
            scope=scope or 'unknown', direction=direction or 'unknown',
            movement=movement or 'unknown', location='unknown',
            severity=severity,
        )


@dataclass(frozen=True)
class SceneRepresentation:
    """The only scene object consumed by later policy modules."""

    z_scene: torch.Tensor
    grounding: Union[Gnode, Tuple[Gnode, ...]]

    def __post_init__(self):
        if not isinstance(self.z_scene, torch.Tensor):
            raise TypeError('SceneRepresentation.z_scene must be a torch.Tensor')
        if self.z_scene.ndim not in (1, 2) or self.z_scene.shape[-1] != 128:
            raise ValueError('SceneRepresentation.z_scene must have shape [128] or [batch, 128]')
        if isinstance(self.grounding, Gnode):
            groundings = (self.grounding,)
        else:
            groundings = tuple(self.grounding)
            if not groundings or not all(isinstance(item, Gnode) for item in groundings):
                raise TypeError('SceneRepresentation.grounding must contain Gnode objects')
        if self.z_scene.ndim == 1 and len(groundings) != 1:
            raise ValueError('single z_scene vector requires one Gnode')
        if self.z_scene.ndim == 2 and len(groundings) != self.z_scene.shape[0]:
            raise ValueError('grounding count must match z_scene batch size')


@dataclass(frozen=True)
class EventSceneRepresentation:
    """Event-level Zscene plus route-aligned Gscene for one snapshot."""

    z_events: torch.Tensor                 # [M, D]
    event_ids: Tuple[str, ...]
    node_event_mask: torch.Tensor          # [N, M]
    node_event_relation: Optional[torch.Tensor] = None  # [N, M, 3]

    def __post_init__(self):
        if not isinstance(self.z_events, torch.Tensor) or self.z_events.ndim != 2:
            raise ValueError('z_events must have shape [M, D]')
        if self.node_event_mask.ndim != 2:
            raise ValueError('node_event_mask must have shape [N, M]')
        if self.node_event_mask.shape[1] != self.z_events.shape[0]:
            raise ValueError('Gscene event dimension disagrees with Zscene')
        if self.node_event_relation is not None and self.node_event_relation.shape != (
                self.node_event_mask.shape[0], self.z_events.shape[0], 3):
            raise ValueError('Gscene route relation shape disagrees with Zscene')
        if len(self.event_ids) != self.z_events.shape[0]:
            raise ValueError('event_ids disagree with Zscene')

    @property
    def event_mask(self) -> torch.Tensor:
        return torch.ones(self.z_events.shape[0], dtype=torch.bool,
                          device=self.z_events.device)


def batch_event_scene_representations(representations: Sequence[EventSceneRepresentation],
                                      device=None) -> SceneCondition:
    """Pad variable-size event sets into a permutation-safe SGA condition."""
    reps = tuple(representations)
    if not reps:
        raise ValueError('at least one event scene representation is required')
    width = reps[0].z_events.shape[-1]
    nodes = reps[0].node_event_mask.shape[0]
    max_events = max(rep.z_events.shape[0] for rep in reps)
    device = device or reps[0].z_events.device
    z = torch.zeros((len(reps), max_events, width), device=device)
    mask = torch.zeros((len(reps), max_events), dtype=torch.bool, device=device)
    g = torch.zeros((len(reps), nodes, max_events), dtype=torch.bool, device=device)
    relation = torch.zeros((len(reps), nodes, max_events, 3), dtype=torch.bool, device=device)
    for index, rep in enumerate(reps):
        if rep.z_events.shape[-1] != width or rep.node_event_mask.shape[0] != nodes:
            raise ValueError('batched event scenes disagree on node or embedding dimensions')
        count = rep.z_events.shape[0]
        if count:
            z[index, :count] = rep.z_events.to(device)
            mask[index, :count] = True
            g[index, :, :count] = rep.node_event_mask.to(device)
            if rep.node_event_relation is not None:
                relation[index, :, :count] = rep.node_event_relation.to(device)
    return SceneCondition(z, mask, g, relation)


class StructuredSceneEncoder(nn.Module):
    """Small categorical/scalar encoder for public event semantics."""

    DEFAULT_VOCABS = {
        'event_type': ('unknown', 'multiple', 'lane_blockage', 'road_closure', 'global_rain',
                       'accident', 'construction', 'weather'),
        'status': ('unknown', 'multiple', 'pending', 'active', 'cleared'),
        'scope': ('unknown', 'multiple', 'lane', 'edge', 'network', 'global'),
        'direction': ('unknown', 'multiple', 'northbound', 'southbound', 'eastbound', 'westbound'),
        'movement': ('unknown', 'multiple', 'left-turn', 'through', 'right-turn',
                     'U-turn', 'left', 'right'),
        'location': ('unknown', 'multiple', 'global'),
    }
    # Spatial location is a Gscene concern.  Keeping it out of this list makes
    # it impossible for a report ID or junction token to leak into Zscene.
    CATEGORICAL_FIELDS = ('event_type', 'status', 'scope', 'direction', 'movement')

    def __init__(self, output_dim: int = 64, embedding_dim: int = 8,
                 hidden_dim: int = 64, vocabs: Optional[Mapping[str, Sequence[str]]] = None):
        super().__init__()
        if output_dim <= 0 or embedding_dim <= 0 or hidden_dim <= 0:
            raise ValueError('StructuredSceneEncoder dimensions must be positive')
        supplied = {} if vocabs is None else dict(vocabs)
        self.vocabs = {}
        self.embeddings = nn.ModuleDict()
        for field in self.CATEGORICAL_FIELDS:
            values = tuple(supplied.get(field, self.DEFAULT_VOCABS[field]))
            if not values or values[0] != 'unknown' or len(set(values)) != len(values):
                raise ValueError(f'{field} vocabulary must start with unique unknown')
            self.vocabs[field] = values
            self.embeddings[field] = nn.Embedding(len(values), embedding_dim)
        self.output_dim = output_dim
        self.embedding_dim = embedding_dim
        self.input_dim = len(self.CATEGORICAL_FIELDS) * embedding_dim + 2
        self.encoder = nn.Sequential(
            nn.Linear(self.input_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def _normalize(self, value: Any) -> StructuredSceneMeta:
        if isinstance(value, StructuredSceneMeta):
            return value
        if isinstance(value, Mapping):
            allowed = set(StructuredSceneMeta.__dataclass_fields__)
            unknown = set(value) - allowed
            if unknown:
                raise ValueError(f'Unsupported structured scene fields: {sorted(unknown)}')
            return StructuredSceneMeta(**dict(value))
        raise TypeError('metadata must contain StructuredSceneMeta or a mapping')

    def _batch(self, metadata: Union[StructuredSceneMeta, Mapping, Sequence]) -> Tuple[StructuredSceneMeta, ...]:
        if isinstance(metadata, (StructuredSceneMeta, Mapping)):
            return (self._normalize(metadata),)
        return tuple(self._normalize(item) for item in metadata)

    def forward(self, metadata: Union[StructuredSceneMeta, Mapping, Sequence]) -> torch.Tensor:
        rows = self._batch(metadata)
        if not rows:
            raise ValueError('metadata batch must not be empty')
        device = next(self.parameters()).device
        encoded = []
        for row in rows:
            values = []
            for field in self.CATEGORICAL_FIELDS:
                value = getattr(row, field) or 'unknown'
                try:
                    index = self.vocabs[field].index(value)
                except ValueError:
                    index = 0
                values.append(self.embeddings[field](torch.tensor(index, device=device)))
            severity = row.severity
            valid = isinstance(severity, (int, float)) and math.isfinite(float(severity))
            normalized = 0.0 if not valid else min(1.0, max(0.0, float(severity)))
            values.append(torch.tensor([normalized, float(valid)], device=device))
            encoded.append(torch.cat(values, dim=0))
        return self.encoder(torch.stack(encoded, dim=0))


class SceneContextEncoder(nn.Module):
    """Build ``SceneRepresentation`` at the event/report boundary.

    Text and report objects are accepted only here.  Every downstream module
    receives the returned ``SceneRepresentation`` and therefore cannot access
    raw text, ``PhysicalEvent`` instances, or simulator objects.
    """

    def __init__(self, text_encoder: Optional[nn.Module] = None,
                 text_dim: Optional[int] = None, text_hidden_dim: int = 128,
                 meta_dim: int = 64, fusion_hidden_dim: int = 128,
                 meta_vocabs: Optional[Mapping[str, Sequence[str]]] = None):
        super().__init__()
        self.text_branch = SceneEncoder(text_encoder, text_dim, text_hidden_dim)
        self.meta_branch = StructuredSceneEncoder(
            output_dim=meta_dim, vocabs=meta_vocabs)
        self.fusion = nn.Sequential(
            nn.Linear(text_hidden_dim + meta_dim, fusion_hidden_dim), nn.ReLU(),
            nn.Linear(fusion_hidden_dim, 128),
        )

    def _texts(self, texts: Union[str, Sequence[str]]) -> Tuple[str, ...]:
        if isinstance(texts, str):
            return (texts,)
        values = tuple(texts)
        if not values or not all(isinstance(item, str) for item in values):
            raise TypeError('texts must be a non-empty string sequence')
        return values

    def forward(self, texts: Union[str, Sequence[str]], metadata,
                grounding: Union[Gnode, Sequence[Gnode]]) -> SceneRepresentation:
        single_scene = isinstance(texts, str) and isinstance(grounding, Gnode)
        texts = self._texts(texts)
        if isinstance(grounding, Gnode):
            groundings = (grounding,)
        else:
            groundings = tuple(grounding)
        if len(groundings) != len(texts):
            raise ValueError('grounding count must match text batch size')
        z_text = self.text_branch(texts)
        z_meta = self.meta_branch(metadata)
        if z_meta.shape[0] != len(texts):
            raise ValueError('metadata count must match text batch size')
        z_scene = self.fusion(torch.cat((z_text, z_meta), dim=-1))
        if single_scene:
            return SceneRepresentation(z_scene=z_scene[0], grounding=groundings[0])
        if len(groundings) == 1:
            return SceneRepresentation(z_scene=z_scene, grounding=groundings[0])
        # Keep per-sample grounding objects together without converting their
        # variable-width masks into a false common topology.
        return SceneRepresentation(z_scene=z_scene, grounding=groundings)

    def encode_one(self, text: str, metadata, grounding: Gnode) -> SceneRepresentation:
        result = self.forward(text, metadata, grounding)
        return result

    def encode_events(self, texts: Sequence[str], metadata: Sequence[Any],
                      grounding: Gnode,
                      cached_z_text_raw: Optional[torch.Tensor] = None) -> EventSceneRepresentation:
        """Encode each active event independently and retain Gscene.

        Text and structured metadata are fused per event.  Entity IDs never
        enter the semantic branch; the immutable binder mask is returned as a
        separate node-by-event relation.
        """
        texts = tuple(texts)
        metadata = tuple(metadata)
        if len(texts) != len(metadata):
            raise ValueError('event text and metadata counts must match')
        expected = tuple(grounding.report_ids)
        if len(expected) != len(texts):
            raise ValueError('grounding report_ids must match active event count')
        if not texts:
            return EventSceneRepresentation(
                torch.zeros((0, 128), device=next(self.parameters()).device),
                (), torch.zeros((grounding.node_mask.shape[0], 0), dtype=torch.bool,
                                device=next(self.parameters()).device),
                torch.zeros((grounding.node_mask.shape[0], 0, 3), dtype=torch.bool,
                            device=next(self.parameters()).device))
        if cached_z_text_raw is None:
            z_text = self.text_branch(texts)
        else:
            raw = cached_z_text_raw
            if not isinstance(raw, torch.Tensor) or raw.shape != (len(texts), self.text_branch.text_dim):
                raise ValueError('cached_z_text_raw must contain one embedding per event')
            z_text = self.text_branch.project(raw)
        z_meta = self.meta_branch(metadata)
        z_events = self.fusion(torch.cat((z_text, z_meta), dim=-1))
        node_mask = torch.tensor(grounding.node_report_mask, dtype=torch.bool,
                                 device=z_events.device)
        if node_mask.shape != (grounding.node_mask.shape[0], len(texts)):
            raise ValueError('Gscene node/event mask has an unexpected shape')
        relation = torch.tensor(grounding.node_event_relation, dtype=torch.bool,
                                device=z_events.device)
        return EventSceneRepresentation(z_events, expected, node_mask, relation)

    def encode_context_events(self, context: SceneContext, grounding: Gnode,
                              normal_text: str = 'Normal',
                              cached_z_text_raw: Optional[torch.Tensor] = None) -> EventSceneRepresentation:
        """Build event-level Z/G without pooling multiple reports."""
        if not isinstance(context, SceneContext) or not isinstance(grounding, Gnode):
            raise TypeError('context and grounding types are invalid')
        events = tuple(context.events)
        event_ids = tuple(getattr(item, 'report_id', getattr(item, 'event_id', None))
                          for item in events)
        if event_ids != tuple(grounding.report_ids):
            raise ValueError('active event order must match Gscene report_ids')
        texts = tuple(getattr(item, 'text', normal_text) for item in events)
        metadata = tuple(StructuredSceneMeta.from_report(item) for item in events)
        return self.encode_events(texts, metadata, grounding, cached_z_text_raw)

    def encode_report(self, report: Any, grounding: Gnode) -> SceneRepresentation:
        """Consume one public/grounded report at the scene boundary."""
        text = getattr(report, 'text', None)
        if not isinstance(text, str):
            raise TypeError('report must expose public text')
        return self.encode_one(text, StructuredSceneMeta.from_report(report), grounding)

    @staticmethod
    def scene_context_at(observed_at: float, reports) -> SceneContext:
        """Expose the control-step lifecycle boundary beside the encoder."""
        return scene_context_at(observed_at, reports)

    @staticmethod
    def _aggregate_metadata(metadata: Sequence[StructuredSceneMeta]) -> StructuredSceneMeta:
        if not metadata:
            return StructuredSceneMeta()

        def categorical(field):
            values = {getattr(item, field) for item in metadata}
            return values.pop() if len(values) == 1 else 'multiple'

        severities = [item.severity for item in metadata if item.severity is not None]
        return StructuredSceneMeta(
            event_type=categorical('event_type'), status=categorical('status'),
            scope=categorical('scope'), direction=categorical('direction'),
            movement=categorical('movement'), location=categorical('location'),
            severity=max(severities) if severities else None,
        )

    def encode_context(self, context: SceneContext, grounding: Gnode,
                       normal_text: str = 'Normal',
                       cached_z_text_raw: Optional[torch.Tensor] = None
                       ) -> SceneRepresentation:
        """Encode exactly one time-indexed scene context.

        Cleared reports are intentionally absent from ``context.events`` and
        therefore cannot leak into the current control scene.  Multiple active
        events remain separate in :meth:`encode_context_events`; this method is
        retained only as a pooled compatibility view for older callers.
        """
        if not isinstance(context, SceneContext):
            raise TypeError('context must be a SceneContext')
        if not isinstance(grounding, Gnode):
            raise TypeError('grounding must be a Gnode')
        event_ids = tuple(getattr(item, 'report_id', getattr(item, 'event_id', None))
                          for item in context.events)
        if event_ids != tuple(grounding.report_ids):
            raise ValueError(
                'SceneContext event IDs must match Gnode report_ids in order')
        # Compatibility API: new callers should use encode_context_events().
        # It retains one z per event; this legacy representation is only a
        # pooled view for callers that still require a single vector.
        event_rep = self.encode_context_events(context, grounding, normal_text,
                                               cached_z_text_raw)
        if event_rep.z_events.shape[0] == 0:
            return SceneRepresentation(torch.zeros(128, device=next(self.parameters()).device), grounding)
        z_scene = event_rep.z_events.mean(dim=0, keepdim=True)
        return SceneRepresentation(z_scene=z_scene[0], grounding=grounding)

    @staticmethod
    def context_text(context: SceneContext, normal_text: str = 'Normal') -> str:
        """Return the public text visible for this context, never its cleared log."""
        if not isinstance(context, SceneContext):
            raise TypeError('context must be a SceneContext')
        texts = tuple(item.text for item in context.events)
        return ' '.join(texts) if texts else normal_text

    def cache_context_text_raw(self, context: SceneContext,
                               normal_text: str = 'Normal') -> torch.Tensor:
        """Cache one frozen text feature per active event for replay."""
        if not isinstance(context, SceneContext):
            raise TypeError('context must be a SceneContext')
        texts = tuple(item.text for item in context.events)
        if not texts:
            return torch.empty((0, self.text_branch.text_dim), dtype=torch.float32)
        return self.text_branch.encode_raw(texts).detach().cpu()

    def encode_replay_state(self, replay_state: SceneReplayState,
                            grounding: Optional[Gnode] = None,
                            normal_text: str = 'Normal') -> SceneRepresentation:
        """Rebuild ``z_scene`` from replayed context under current parameters."""
        if not isinstance(replay_state, SceneReplayState):
            raise TypeError('replay_state must be a SceneReplayState')
        if grounding is None:
            grounding = replay_state.grounding
        if not isinstance(grounding, Gnode):
            raise TypeError('replay_state must contain a Gnode or receive grounding')
        return self.encode_context(
            replay_state.context, grounding, normal_text,
            cached_z_text_raw=replay_state.cached_z_text_raw)

    def encode_grounding_batch(self, grounding_batch: Any,
                               observed_at: Optional[float] = None,
                               normal_text: str = 'Normal') -> SceneRepresentation:
        """Convert one synchronized ``GroundingBatch`` into a scene output.

        ``observed_at`` is required unless the batch explicitly exposes its
        own timestamp.  Requiring the control timestamp prevents a cleared
        report retained in the public snapshot from being mistaken for an
        active scene in a later decision step.
        """
        if observed_at is None:
            observed_at = getattr(grounding_batch, 'observed_at', None)
        if observed_at is None:
            raise ValueError('observed_at is required for scene_context_at()')
        context = grounding_batch.scene_context_at(observed_at)
        return self.encode_context(context, grounding_batch.to_gnode(), normal_text)


class SceneEncoder(nn.Module):
    """Encode event text with a frozen encoder and train a 128-D projection.

    ``text_encoder`` may be injected for tests or another project encoder.  If
    omitted, the repository's frozen BERT implementation is loaded lazily from
    ``reproduction.tarl_tsc.text``.  Replay code can store raw embeddings and
    call :meth:`project` directly, avoiding a text-encoder call during TD
    updates.
    """

    def __init__(self, text_encoder: Optional[nn.Module] = None,
                 text_dim: Optional[int] = None, hidden_dim: int = 128):
        super().__init__()
        if text_encoder is None:
            from reproduction.tarl_tsc.text.encoder import FrozenTextEncoder
            text_encoder = FrozenTextEncoder()
        if text_dim is None:
            text_dim = getattr(text_encoder, 'output_dim', None)
        if not isinstance(text_dim, int) or text_dim <= 0:
            raise ValueError('text_dim must be a positive integer or exposed by text_encoder')
        if not isinstance(hidden_dim, int) or hidden_dim <= 0:
            raise ValueError('hidden_dim must be a positive integer')
        self.text_encoder = text_encoder
        self.text_dim = text_dim
        self.hidden_dim = hidden_dim
        self.projection = nn.Linear(text_dim, hidden_dim)
        self.text_encoder.requires_grad_(False)
        self.text_encoder.eval()

    def train(self, mode: bool = True):
        # The projection follows the requested mode; the text encoder never
        # leaves eval mode and never receives gradients.
        super().train(mode)
        self.text_encoder.eval()
        return self

    @torch.no_grad()
    def encode_raw(self, texts: Sequence[str]) -> torch.Tensor:
        """Return frozen encoder output with shape ``[B, text_dim]``."""
        raw = self.text_encoder(tuple(texts))
        if not isinstance(raw, torch.Tensor) or raw.ndim != 2 or raw.shape[-1] != self.text_dim:
            raise ValueError(
                f'text_encoder must return [B, {self.text_dim}] embeddings')
        return raw

    def project(self, raw_embedding: torch.Tensor) -> torch.Tensor:
        """Project stored raw embeddings while preserving TD-loss gradients."""
        if not isinstance(raw_embedding, torch.Tensor) or raw_embedding.ndim != 2:
            raise ValueError('raw_embedding must have shape [batch, text_dim]')
        if raw_embedding.shape[-1] != self.text_dim:
            raise ValueError(f'Expected raw embedding width {self.text_dim}')
        # Frozen encoders may cache CPU tensors while the Q estimator lives on
        # another device.  Moving the stored value here keeps replay portable;
        # gradients still terminate at the frozen embedding.
        raw_embedding = raw_embedding.to(self.projection.weight.device)
        return self.projection(raw_embedding)

    def forward(self, scene: Union[Sequence[str], torch.Tensor]) -> torch.Tensor:
        if isinstance(scene, torch.Tensor):
            raw = scene
        else:
            raw = self.encode_raw(scene)
        return self.project(raw)


class SceneGuidedAttention(nn.Module):
    """Apply scene-conditioned node modulation.

    Inputs are ``H: [B, N, D]`` and ``z: [B, D]``.  The returned attention
    weights are ``alpha: [B, N]`` and sum to one over nodes.  This reference
    rule uses scene-to-node dot-product scores to form a context vector and a
    learned residual gate.  It is deliberately independent of CoLight and
    can be replaced after the SGA design review without changing callers.
    """

    def __init__(self, hidden_dim: int = 128, attention_dim: Optional[int] = None,
                 temperature: float = 1.0, dropout: float = 0.0,
                 residual_scale: float = 1.0, logit_scale: float = 4.0):
        super().__init__()
        if not isinstance(hidden_dim, int) or hidden_dim <= 0:
            raise ValueError('hidden_dim must be a positive integer')
        attention_dim = hidden_dim if attention_dim is None else attention_dim
        if not isinstance(attention_dim, int) or attention_dim <= 0:
            raise ValueError('attention_dim must be a positive integer')
        if not isinstance(temperature, (int, float)) or temperature <= 0:
            raise ValueError('temperature must be positive')
        if not isinstance(dropout, (int, float)) or not 0 <= dropout < 1:
            raise ValueError('dropout must be in [0, 1)')
        if not isinstance(residual_scale, (int, float)) or residual_scale < 0:
            raise ValueError('residual_scale must be nonnegative')
        if not isinstance(logit_scale, (int, float)) or logit_scale <= 0:
            raise ValueError('logit_scale must be positive')
        self.hidden_dim = hidden_dim
        self.attention_dim = attention_dim
        self.temperature = float(temperature)
        self.residual_scale = float(residual_scale)
        self.logit_scale = nn.Parameter(torch.tensor(float(logit_scale)).log())
        self.attention_dropout = nn.Dropout(float(dropout))
        self.traffic_norm = nn.LayerNorm(hidden_dim)
        self.scene_norm = nn.LayerNorm(hidden_dim)
        self.scene_query = nn.Linear(hidden_dim, attention_dim, bias=False)
        self.node_key = nn.Linear(hidden_dim, attention_dim, bias=False)
        self.node_value = nn.Linear(hidden_dim, hidden_dim)
        self.context_projection = nn.Linear(hidden_dim, hidden_dim)
        self.gate = nn.Linear(2 * hidden_dim, hidden_dim)
        self.output = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, traffic_features: torch.Tensor,
                scene_embedding: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if not isinstance(traffic_features, torch.Tensor) or traffic_features.ndim != 3:
            raise ValueError('traffic_features must have shape [batch, num_nodes, hidden_dim]')
        if not isinstance(scene_embedding, torch.Tensor) or scene_embedding.ndim != 2:
            raise ValueError('scene_embedding must have shape [batch, hidden_dim]')
        batch, _, width = traffic_features.shape
        if width != self.hidden_dim:
            raise ValueError(f'Expected traffic feature width {self.hidden_dim}')
        if scene_embedding.shape != (batch, self.hidden_dim):
            raise ValueError(
                f'Expected scene embedding shape [{batch}, {self.hidden_dim}]')

        normalized_scene = self.scene_norm(scene_embedding)
        normalized_traffic = self.traffic_norm(traffic_features)
        query = F.normalize(self.scene_query(normalized_scene), dim=-1).unsqueeze(1)
        key = F.normalize(self.node_key(normalized_traffic), dim=-1)
        scale = self.logit_scale.exp().clamp(1.0, 20.0)
        logits = (query * key).sum(dim=-1) * scale / self.temperature
        alpha = torch.softmax(logits, dim=1)
        alpha_for_context = self.attention_dropout(alpha)

        values = self.node_value(traffic_features)
        context = torch.sum(alpha_for_context.unsqueeze(-1) * values, dim=1)
        context = self.context_projection(context).unsqueeze(1)
        context = context.expand(-1, traffic_features.shape[1], -1)
        gate = torch.sigmoid(self.gate(torch.cat((traffic_features, context), dim=-1)))
        h_tilde = traffic_features + self.residual_scale * gate * self.output(values)
        return h_tilde, alpha


class MultiEventSceneGuidedAttention(nn.Module):
    """Multi-event SGA with independent route relevance and gates.

    Each event-node pair receives an independent sigmoid relevance. The
    grounding mask is a soft relation bias. It tells the model which nodes an
    event explicitly names, while graph propagation and the learned score can
    still represent upstream/downstream effects. No node is hard masked from
    control, and event relevances are not normalized to sum to one.
    """

    def __init__(self, hidden_dim: int = 128, attention_dim: Optional[int] = None,
                 temperature: float = 1.0, residual_scale: float = 1.0,
                 grounding_bias: float = 1.0):
        super().__init__()
        if hidden_dim <= 0 or (attention_dim is not None and attention_dim <= 0):
            raise ValueError('attention dimensions must be positive')
        if temperature <= 0 or residual_scale < 0 or grounding_bias < 0:
            raise ValueError('invalid SGA modulation parameters')
        self.hidden_dim = int(hidden_dim)
        self.attention_dim = int(attention_dim or hidden_dim)
        self.temperature = float(temperature)
        self.residual_scale = float(residual_scale)
        self.grounding_bias = float(grounding_bias)
        self.logit_scale = nn.Parameter(torch.tensor(4.0).log())
        self.traffic_norm = nn.LayerNorm(hidden_dim)
        self.event_norm = nn.LayerNorm(hidden_dim)
        self.event_query = nn.Linear(hidden_dim, self.attention_dim, bias=False)
        self.node_key = nn.Linear(hidden_dim, self.attention_dim, bias=False)
        self.event_value = nn.Linear(hidden_dim, hidden_dim)
        self.relation_projection = nn.Linear(3, 1, bias=False)
        self.context_projection = nn.Linear(hidden_dim, hidden_dim)
        self.node_gate = nn.Sequential(
            nn.Linear(2 * hidden_dim + 1, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, 1))
        self.feature_gate = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, hidden_dim))
        self.output = nn.Linear(hidden_dim, hidden_dim)
        # Diagnostic tensors from the latest forward pass. They are detached
        # by the agent logger and do not alter the replay contract.
        self.last_semantic_score = None
        self.last_direct_bias = None
        self.last_relation_bias = None
        self.last_relevance = None

    def forward(self, traffic_features: torch.Tensor,
                condition: SceneCondition):
        if traffic_features.ndim != 3:
            raise ValueError('traffic_features must have shape [B, N, D]')
        batch, nodes, width = traffic_features.shape
        if width != self.hidden_dim:
            raise ValueError('traffic feature width disagrees with SGA hidden_dim')
        condition.validate(batch, nodes, width)
        events = condition.event_embeddings
        event_count = events.shape[1]
        if event_count == 0:
            empty = traffic_features.new_zeros((batch, nodes, 0))
            self.last_semantic_score = empty.detach()
            self.last_direct_bias = empty.detach()
            self.last_relation_bias = empty.detach()
            self.last_relevance = empty.detach()
            return traffic_features, empty, traffic_features.new_zeros((batch, nodes)), \
                traffic_features.new_ones((batch, nodes, width))

        h = self.traffic_norm(traffic_features)
        z = self.event_norm(events)
        query = F.normalize(self.event_query(z), dim=-1)
        key = F.normalize(self.node_key(h), dim=-1)
        scale = self.logit_scale.exp().clamp(1.0, 20.0)
        # [B, E, N] event-to-node relevance.
        semantic_score = torch.einsum('bea,bna->ben', query, key) * scale / self.temperature
        direct = None if condition.node_event_mask is None else condition.node_event_mask.float()
        if direct is None:
            direct = semantic_score.new_zeros((batch, nodes, event_count))
        semantic_score = semantic_score.transpose(1, 2)
        direct_bias = self.grounding_bias * direct
        relation_bias = semantic_score.new_zeros((batch, nodes, event_count))
        if condition.node_event_relation is not None:
            relation_bias = self.relation_projection(
                condition.node_event_relation.to(semantic_score.dtype)).squeeze(-1)
        logits = semantic_score + direct_bias + relation_bias
        valid_events = condition.event_mask.to(dtype=torch.bool)
        valid = valid_events.unsqueeze(1).expand(batch, nodes, event_count)
        relevance = torch.sigmoid(logits) * valid.to(logits.dtype)
        self.last_semantic_score = semantic_score.detach()
        self.last_direct_bias = direct_bias.detach()
        self.last_relation_bias = relation_bias.detach()
        self.last_relevance = relevance.detach()
        # If there are no events, relevance and context are zero while the
        # normal traffic path remains an exact identity.
        has_event = valid_events.any(dim=-1).view(batch, 1, 1)
        relevance = torch.where(has_event, relevance, torch.zeros_like(relevance))
        values = self.event_value(events)
        context = torch.einsum('bne,bed->bnd', relevance, values)
        context = self.context_projection(context)
        direct_strength = direct.max(dim=-1).values.unsqueeze(-1)
        node_gate = torch.sigmoid(self.node_gate(
            torch.cat((traffic_features, context, direct_strength), dim=-1))).squeeze(-1)
        node_gate = node_gate * has_event.squeeze(-1).to(node_gate.dtype)
        feature_gate = torch.sigmoid(self.feature_gate(
            torch.cat((traffic_features, context), dim=-1)))
        # A no-event scene must be exactly the original traffic representation.
        feature_gate = torch.where(has_event, feature_gate,
                                   torch.ones_like(feature_gate))
        delta = self.output(context)
        modulated = traffic_features + self.residual_scale * node_gate.unsqueeze(-1) \
            * feature_gate * delta
        return modulated, relevance, node_gate, feature_gate


class SceneCrossAttentionResidual(nn.Module):
    """Cross-attention event injection with a zero-initialized output head.

    Traffic nodes are queries and event embeddings are keys/values:
    ``U = softmax(QK^T/sqrt(d) + grounding_bias*G + relation_bias) V`` and
    ``H' = H + P(LayerNorm(U))``.  ``P`` is zero-initialized (ControlNet /
    LoRA style), so the branch is an exact identity at init while its output
    head still receives gradient from the first update — unlike a zero scalar
    gate, which would disconnect the whole branch until the gate drifts.

    Diagnostic contract mirrors ``MultiEventSceneGuidedAttention``: the
    returned tuple is (modulated, attention_weights, node_delta_norm,
    delta_magnitude), and the ``last_*`` attributes feed the agent's
    per-decision attention log.
    """

    def __init__(self, hidden_dim: int = 128, attention_dim: Optional[int] = None,
                 heads: int = 4, grounding_bias: float = 1.0):
        super().__init__()
        attention_dim = int(attention_dim or hidden_dim)
        if hidden_dim <= 0 or attention_dim <= 0 or heads <= 0:
            raise ValueError('dimensions and heads must be positive')
        if attention_dim % heads or hidden_dim % heads:
            raise ValueError('attention_dim and hidden_dim must divide by heads')
        if grounding_bias < 0:
            raise ValueError('grounding_bias must be nonnegative')
        self.hidden_dim = int(hidden_dim)
        self.attention_dim = attention_dim
        self.heads = int(heads)
        self.head_dim = attention_dim // heads
        self.value_dim = hidden_dim // heads
        self.grounding_bias = float(grounding_bias)
        self.traffic_norm = nn.LayerNorm(hidden_dim)
        self.event_norm = nn.LayerNorm(hidden_dim)
        self.query = nn.Linear(hidden_dim, attention_dim, bias=False)
        self.key = nn.Linear(hidden_dim, attention_dim, bias=False)
        self.value = nn.Linear(hidden_dim, hidden_dim)
        self.relation_projection = nn.Linear(3, 1, bias=False)
        self.context_norm = nn.LayerNorm(hidden_dim)
        self.out = nn.Linear(hidden_dim, hidden_dim)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)
        self.last_semantic_score = None
        self.last_direct_bias = None
        self.last_relation_bias = None
        self.last_relevance = None
        self.last_delta = None

    def branch_norm(self):
        """Scalar branch-openness monitor: ||P||_Fro of the zero-init head."""
        return float(self.out.weight.norm()) + float(self.out.bias.norm())

    def forward(self, traffic_features: torch.Tensor,
                condition: SceneCondition):
        if traffic_features.ndim != 3:
            raise ValueError('traffic_features must have shape [B, N, D]')
        batch, nodes, width = traffic_features.shape
        if width != self.hidden_dim:
            raise ValueError('traffic feature width disagrees with hidden_dim')
        condition.validate(batch, nodes, width)
        events = condition.event_embeddings
        event_count = events.shape[1]
        if event_count == 0:
            empty = traffic_features.new_zeros((batch, nodes, 0))
            self.last_semantic_score = empty.detach()
            self.last_direct_bias = empty.detach()
            self.last_relation_bias = empty.detach()
            self.last_relevance = empty.detach()
            self.last_delta = traffic_features.new_zeros((batch, nodes, width)).detach()
            return traffic_features, empty, traffic_features.new_zeros((batch, nodes)), \
                traffic_features.new_zeros((batch, nodes, width))

        heads, head_dim = self.heads, self.head_dim
        q = self.query(self.traffic_norm(traffic_features))
        k = self.key(self.event_norm(events))
        v = self.value(self.event_norm(events))
        q = q.reshape(batch, nodes, heads, head_dim).transpose(1, 2)   # [B,H,N,a]
        k = k.reshape(batch, event_count, heads, head_dim).transpose(1, 2)  # [B,H,E,a]
        v = v.reshape(batch, event_count, heads, self.value_dim).transpose(1, 2)  # [B,H,E,dv]
        semantic_score = torch.einsum('bhna,bhea->bhne', q, k) / math.sqrt(head_dim)
        direct = condition.node_event_mask
        if direct is None:
            direct_bias = semantic_score.new_zeros((batch, nodes, event_count))
        else:
            direct_bias = self.grounding_bias * direct.to(semantic_score.dtype)
        if condition.node_event_relation is None:
            relation_bias = semantic_score.new_zeros((batch, nodes, event_count))
        else:
            relation_bias = self.relation_projection(
                condition.node_event_relation.to(semantic_score.dtype)).squeeze(-1)
        logits = semantic_score + (direct_bias + relation_bias).unsqueeze(1)
        valid = condition.event_mask.to(dtype=torch.bool)
        logits = logits.masked_fill(~valid.view(batch, 1, 1, event_count),
                                    torch.finfo(logits.dtype).min)
        attention = torch.softmax(logits, dim=-1)
        # Rows whose every event is invalid contain no information at all.
        attention = attention * valid.view(batch, 1, 1, event_count).to(logits.dtype)
        self.last_semantic_score = semantic_score.mean(dim=1).detach()
        self.last_direct_bias = direct_bias.detach()
        self.last_relation_bias = relation_bias.detach()
        self.last_relevance = attention.mean(dim=1).detach()
        context = torch.einsum('bhne,bhed->bhnd', attention, v)
        context = context.transpose(1, 2).reshape(batch, nodes, self.hidden_dim)
        delta = self.out(self.context_norm(context))
        modulated = traffic_features + delta
        self.last_delta = delta.detach()
        return modulated, self.last_relevance, \
            delta.detach().norm(dim=-1), delta.detach().abs()


class CoLightAdapter:
    """Expose the traffic/control split of an existing ``ColightNet``.

    The adapter only requires the two methods added to the network.  It does
    not import or inspect ``CoLightAgent`` and therefore leaves room for a
    future MPLight adapter with the same interface.
    """

    def __init__(self, network: nn.Module):
        for name in ('encode_traffic', 'control_from_features'):
            if not callable(getattr(network, name, None)):
                raise TypeError(f'network must provide {name}()')
        self.network = network

    def encode_traffic(self, observation: torch.Tensor, train: bool = True) -> torch.Tensor:
        return self.network.encode_traffic(observation, train=train)

    def control_from_features(self, traffic_features: torch.Tensor,
                              graph_context: Any, train: bool = True) -> torch.Tensor:
        return self.network.control_from_features(
            traffic_features, graph_context, train=train)


def scene_metadata_from_context(context: SceneContext) -> StructuredSceneMeta:
    """Reduce one public control-step scene to auditable metadata."""
    if not isinstance(context, SceneContext):
        raise TypeError('context must be a SceneContext')
    return SceneContextEncoder._aggregate_metadata(
        tuple(StructuredSceneMeta.from_report(item) for item in context.events))


class MPLightSGAAdapter(nn.Module):
    """State-shape adapter for the per-intersection MPLight/FRAP head.

    MPLight's original FRAP network consumes one intersection at a time,
    while SGA needs a node set.  This adapter creates a shared traffic feature
    for ``[batch, nodes, state_dim]``, applies the same SGA module used by
    CoLight, and returns an action correction.  A caller may add the
    correction to the frozen FRAP Q values, which preserves the original
    action/reward contract while making the adapter independently testable.
    """

    def __init__(self, state_dim: int, num_actions: int, hidden_dim: int = 64,
                 attention_dim: Optional[int] = None):
        super().__init__()
        if min(state_dim, num_actions, hidden_dim) <= 0:
            raise ValueError('adapter dimensions must be positive')
        self.state_dim = int(state_dim)
        self.num_actions = int(num_actions)
        self.hidden_dim = int(hidden_dim)
        self.traffic_encoder = nn.Sequential(
            nn.Linear(self.state_dim, self.hidden_dim), nn.ReLU())
        self.scene_projection = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.sga = SceneGuidedAttention(
            self.hidden_dim, attention_dim=attention_dim)
        self.action_delta = nn.Linear(self.hidden_dim, self.num_actions)
        self.last_attention = None

    def forward(self, states: torch.Tensor, scene_embedding: torch.Tensor,
                base_q_values: Optional[torch.Tensor] = None):
        if states.ndim != 3 or states.shape[-1] != self.state_dim:
            raise ValueError('states must have shape [batch, nodes, state_dim]')
        if scene_embedding.ndim != 2 or scene_embedding.shape[0] != states.shape[0]:
            raise ValueError('scene_embedding must have shape [batch, hidden_dim]')
        traffic = self.traffic_encoder(states)
        scene = self.scene_projection(scene_embedding)
        modulated, alpha = self.sga(traffic, scene)
        self.last_attention = alpha
        delta = self.action_delta(modulated)
        if base_q_values is None:
            return delta, alpha
        if base_q_values.shape != delta.shape:
            raise ValueError('base_q_values shape must match [batch, nodes, num_actions]')
        return base_q_values + delta, alpha


def scene_metadata_from_context(context: SceneContext) -> StructuredSceneMeta:
    """Reduce one public control-step scene to auditable metadata.

    Only the immutable scene snapshot crosses this boundary.  In particular,
    schedules, simulator objects and privileged targets never become policy
    features.  The helper is used by the structured SGA traffic adapters when
    a frozen text encoder is intentionally not part of a pilot.
    """
    if not isinstance(context, SceneContext):
        raise TypeError('context must be a SceneContext')
    return SceneContextEncoder._aggregate_metadata(
        tuple(StructuredSceneMeta.from_report(item) for item in context.events))

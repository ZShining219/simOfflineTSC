"""Paper-faithful TARL-TSC policy.

Implements the architecture described in Al-Quhfa, Mothana & Song,
"Text-augmented reinforcement learning for real-time and robust traffic
signal control" (EAAI 182:116083), Sections 4.3-4.6:

- Eq 9:   s_it = MLP_traffic(s_it_raw) in R^128   (per-intersection traffic feature)
- Eq 11:  e'_t = W_e e_t, s'_it = W_s s_it        (shared latent d_h = 256)
- Eqs 12-13 (TARL-ATT): bidirectional cross-attention
    o_text    = Attn(e'_t, S'_t, S'_t)            (text queries all node traffic)
    o_traffic = Attn(s'_it, E'_t, E'_t)           (each node queries text tokens)
    h_it = MLP([o_text ; o_traffic_i]) + s_it     (512 -> 256 -> 128 + residual)
- Eqs 14-15 (TARL-AGM): adaptive gating
    g_it = sigmoid(W_g [e'_t ; s'_it])
    h_it = g_it * e'_t + (1 - g_it) * s'_it, then Linear(256 -> 128)
- Eqs 16-17 (GAT): two standard GAT layers, K = 8 heads, N(i) u {i}
- Eq 18:  Q(z_it, .) = MLP_Q(z_it), lightweight two-layer head

Documented fidelity notes (paper ambiguities resolved explicitly):
- The paper writes e_t without a node index (global semantic context). We keep
  a per-node copy e'_it so each node attends its own text token:
  o_traffic_i = Attn(s'_it, e'_it, e'_it) has a single key and is
  softmax-degenerate (returns the projected text value), while the MHA
  projections are still learned, matching the figure; o_text_i =
  Attn(e'_it, S', S') lets node i's text attend all nodes' traffic features.
  Under broadcast text delivery (all nodes share the report string) e'_it is
  identical across i and the fused pathway reduces to the paper's global
  embedding semantics.
- Dropout rate is not specified in the paper; p = 0.1 is used in the fusion
  heads (figures 2 and 3 mark "ReLU + Dropout").
- Between the two GAT layers we apply ELU, following the standard GAT
  reference implementation (the paper does not specify the nonlinearity).
- 'sensor' = paper ablation without semantic input and without GAT
  (per-intersection MLP -> Q); 'gat' = paper ablation without semantic input
  (traffic features -> GAT -> Q).
"""
import torch
from torch import nn
import torch.nn.functional as F


TEXT_DIM = 768
TRAFFIC_EMB_DIM = 128   # s_it per Eq. 9
LATENT_DIM = 256        # d_h shared latent per Section 4.4
FUSED_DIM = 128         # h_it fed to GAT per figures 2-3
GAT_HEADS = 8
GAT_LAYERS = 2
GAT_HEAD_DIM = 16       # 8 heads x 16 = 128 per layer
FUSION_DROPOUT = 0.1

# Structured task-related event input (replaces the 768-dim frozen BERT slot).
# Raw per-node fields, all produced by ReportGrounder + TextEntityBinder:
#   [present, event_type, scope, direction, movement, severity]
# event_type codes: 0 none, 1 lane_blockage, 2 road_closure, 3 global_rain,
# 4 demand_surge (pairaux line).  Arms containing demand_surge events must set
# model.ztask_type_vocab=5 (embedding + semantic-aux head widen); default 4
# keeps all pre-surge checkpoints loadable.
ZTASK_FIELDS = 6
ZTASK_VOCABS = (2, 4, 4, 5, 6)   # present/type/scope/direction/movement
ZTASK_EMB = 8


class TaskEventEncoder(nn.Module):
    """f(present, type, scope, direction, movement, severity) -> e'_it (d_h).

    Field embeddings + MLP projection into the shared latent, then a hard
    present-gate: nodes with no bound event emit an exact zero vector so the
    text pathway contributes nothing on Normal steps.
    """

    def __init__(self, out_dim=LATENT_DIM, emb_dim=ZTASK_EMB, type_vocab=None):
        super().__init__()
        vocabs = list(ZTASK_VOCABS)
        if type_vocab is not None:
            vocabs[1] = int(type_vocab)
        self.vocabs = tuple(vocabs)
        self.embeddings = nn.ModuleList(
            nn.Embedding(n, emb_dim) for n in self.vocabs)
        self.encoder = nn.Sequential(
            nn.Linear(len(self.vocabs) * emb_dim + 1, 64), nn.ReLU(),
            nn.Linear(64, out_dim))

    def forward(self, raw):
        """raw: (..., ZTASK_FIELDS) float tensor of field indices + severity."""
        *lead, width = raw.shape
        if width != ZTASK_FIELDS:
            raise ValueError(f'structured event slot must have width {ZTASK_FIELDS}')
        cols = raw.unbind(-1)
        parts = [emb(cols[i].long().clamp(0, v - 1))
                 for i, (emb, v) in enumerate(zip(self.embeddings, self.vocabs))]
        parts.append(cols[-1].unsqueeze(-1))
        z = self.encoder(torch.cat(parts, dim=-1))
        return z * cols[0].unsqueeze(-1)          # present gate


class SemanticAuxHead(nn.Module):
    """Lightweight heads on h_it (fusion output, pre-GAT).

    present: per-node BCE — doubles as the location supervision, since a node
      must learn event_present=1 exactly on grounded target nodes.
    type/movement: per-node CE, computed only on present nodes by the caller.
    """

    def __init__(self, in_dim=FUSED_DIM, hidden=64, type_vocab=None):
        super().__init__()
        self.trunk = nn.Sequential(nn.Linear(in_dim, hidden), nn.ReLU())
        self.present = nn.Linear(hidden, 1)
        self.event_type = nn.Linear(
            hidden, int(type_vocab) if type_vocab is not None
            else ZTASK_VOCABS[1])
        self.movement = nn.Linear(hidden, ZTASK_VOCABS[4])

    def forward(self, h):
        z = self.trunk(h)
        return {'present': self.present(z).squeeze(-1),
                'type': self.event_type(z),
                'movement': self.movement(z)}


class GATLayer(nn.Module):
    """Standard multi-head GAT layer (Eqs 16-17).

    alpha_ij = softmax_j( LeakyReLU( a^T [W h_i || W h_j] ) )  over N(i) u {i}
    z_i = ||_k sum_j alpha_ij^(k) W^(k) h_j
    """

    def __init__(self, in_dim, head_dim, heads=GAT_HEADS):
        super().__init__()
        self.heads = heads
        self.head_dim = head_dim
        self.weight = nn.Linear(in_dim, heads * head_dim, bias=False)
        self.attn_left = nn.Parameter(torch.empty(heads, head_dim))
        self.attn_right = nn.Parameter(torch.empty(heads, head_dim))
        self.leaky = nn.LeakyReLU(0.2)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.weight.weight)
        nn.init.xavier_uniform_(self.attn_left)
        nn.init.xavier_uniform_(self.attn_right)

    def forward(self, x, adjacency):
        if adjacency is None:
            raise ValueError('GAT requires explicit adjacency')
        unbatched = x.ndim == 2
        if unbatched:
            x = x.unsqueeze(0)
        batch, nodes, _ = x.shape
        if adjacency.shape != (nodes, nodes):
            raise ValueError('Adjacency must describe graph nodes, not replay samples')
        connected = adjacency.to(device=x.device, dtype=torch.bool)
        connected = connected | torch.eye(nodes, device=x.device, dtype=torch.bool)

        h = self.weight(x).view(batch, nodes, self.heads, self.head_dim)
        # e_ij = LeakyReLU(a_left . Wh_i + a_right . Wh_j) per head
        left = torch.einsum('bnhd,hd->bnh', h, self.attn_left)
        right = torch.einsum('bnhd,hd->bnh', h, self.attn_right)
        scores = self.leaky(left.unsqueeze(2) + right.unsqueeze(1))  # (B, i, j, H)
        scores = scores.masked_fill(~connected.unsqueeze(0).unsqueeze(-1), float('-inf'))
        alpha = torch.softmax(scores, dim=2)
        agg = torch.einsum('bijh,bjhd->bihd', alpha, h)  # (B, N, H, D)
        out = agg.reshape(batch, nodes, self.heads * self.head_dim)
        return out.squeeze(0) if unbatched else out


class PaperTARLPolicy(nn.Module):
    """Whole-graph shared-parameter policy; one graph per forward call.

    forward(traffic, text_features=None, adjacency) -> Q values per node.
    traffic: (N, traffic_dim) or (B, N, traffic_dim) raw per-node observations
    text_features: (N, 768) or (B, N, 768) frozen embeddings, already aligned
      per node by the caller; the paper treats the event embedding as global,
      so identical rows are expected under broadcast delivery.
    """

    METHODS = ('sensor', 'gat', 'concat', 'selfattn', 'attention', 'gating',
               'crossq')
    TEXT_METHODS = ('concat', 'selfattn', 'attention', 'gating', 'crossq')

    def __init__(self, traffic_dim, action_dim, method='gating',
                 text_input='bert', semantic_aux=False, fusion_gate=False,
                 type_vocab=None):
        super().__init__()
        if method not in self.METHODS:
            raise ValueError(method)
        if text_input not in ('bert', 'structured'):
            raise ValueError(text_input)
        self.method = method
        self.text_input = text_input
        # Eq. 9: raw traffic -> 128-dim node feature s_it
        self.traffic_mlp = nn.Sequential(
            nn.Linear(traffic_dim, TRAFFIC_EMB_DIM), nn.ReLU())
        self.last_fusion_weights = None
        # Input-level gradient probe (ATT-ENTITY-004): when enabled, the
        # fusion inputs sp/e_p retain .grad after backward so callers can
        # compute R_t = RMS(dL/ds_p) / RMS(dL/de_p).
        self.grad_probe = False
        self._probe = {}
        self._aux = None
        self.semantic_aux = bool(semantic_aux)
        # ATT-ENTITY-004 repair: learnable scalar gate on the fused output.
        # alpha=0 at init -> text pathway starts silent and must be earned by
        # gradients (aux supervision pulls it up, TD loss can push it down).
        # Applies to the residual-fusion methods (attention/concat/selfattn).
        self.fusion_gate = bool(fusion_gate)
        if self.fusion_gate and method in self.TEXT_METHODS:
            self.fusion_alpha = nn.Parameter(torch.zeros(1))
        if method in self.TEXT_METHODS:
            # Eq. 11: projections into the shared 256-dim latent.  Structured
            # mode replaces the frozen BERT input + W_e with a learned
            # task-event encoder over grounded categorical fields.
            if text_input == 'structured':
                self.task_encoder = TaskEventEncoder(
                    LATENT_DIM, type_vocab=type_vocab)
            else:
                self.text_projection = nn.Linear(TEXT_DIM, LATENT_DIM)   # W_e
            self.state_projection = nn.Linear(TRAFFIC_EMB_DIM, LATENT_DIM)  # W_s
            if method == 'attention':
                self.attn_text_to_traffic = nn.MultiheadAttention(
                    LATENT_DIM, GAT_HEADS, dropout=0., batch_first=True)
                self.attn_traffic_to_text = nn.MultiheadAttention(
                    LATENT_DIM, GAT_HEADS, dropout=0., batch_first=True)
                # 512 -> 256 -> ReLU+Dropout -> 128 per figure 2; dropout is
                # applied via F.dropout so rollout/eval calls stay deterministic.
                self.fusion_fc1 = nn.Linear(2 * LATENT_DIM, LATENT_DIM)
                self.fusion_fc2 = nn.Linear(LATENT_DIM, FUSED_DIM)
            elif method == 'crossq':
                # ATT-ENTITY-004 crossq arm.  Module budget is IDENTICAL to
                # 'attention' (two MHA + the same 512->256->128 head); the only
                # change is the second attention block: instead of the
                # degenerate single-key Attn(s, e, e) it is a proper
                # traffic-as-Query / events-as-Key-Value cross-attention over
                # all N per-node event tokens.  With no event the event K/V are
                # exactly zero, so o_event = 0 and the block contributes nothing.
                self.attn_text_to_traffic = nn.MultiheadAttention(
                    LATENT_DIM, GAT_HEADS, dropout=0., batch_first=True)
                self.attn_traffic_to_events = nn.MultiheadAttention(
                    LATENT_DIM, GAT_HEADS, dropout=0., batch_first=True)
                self.fusion_fc1 = nn.Linear(2 * LATENT_DIM, LATENT_DIM)
                self.fusion_fc2 = nn.Linear(LATENT_DIM, FUSED_DIM)
            elif method == 'concat':
                # ATT-ENTITY-004 ablation: identical fusion head to the
                # attention arm; the only difference is replacing the
                # bidirectional cross-attention with plain feature concat.
                self.fusion_fc1 = nn.Linear(2 * LATENT_DIM, LATENT_DIM)
                self.fusion_fc2 = nn.Linear(LATENT_DIM, FUSED_DIM)
            elif method == 'selfattn':
                # ATT-ENTITY-004 ablation: vanilla-transformer arm. Joint
                # per-node tokens [s'_it || e'_it] are mixed by node-level
                # self-attention; event features ride as token content and
                # never act as a guiding query.
                self.tok_proj = nn.Linear(2 * LATENT_DIM, LATENT_DIM)
                self.node_self_attn = nn.MultiheadAttention(
                    LATENT_DIM, GAT_HEADS, dropout=0., batch_first=True)
                self.fusion_fc1 = nn.Linear(LATENT_DIM, FUSED_DIM)
            else:
                self.gate = nn.Linear(2 * LATENT_DIM, LATENT_DIM)    # W_g
                self.fusion_fc1 = nn.Linear(LATENT_DIM, FUSED_DIM)
        if method != 'sensor':
            # Eqs 16-17: two GAT layers, 8 heads, operating on h_it (128-dim)
            self.gat_layers = nn.ModuleList([
                GATLayer(FUSED_DIM, GAT_HEAD_DIM, GAT_HEADS),
                GATLayer(GAT_HEADS * GAT_HEAD_DIM, GAT_HEAD_DIM, GAT_HEADS),
            ])
            gat_out_dim = GAT_HEADS * GAT_HEAD_DIM
        else:
            self.gat_layers = None
            gat_out_dim = FUSED_DIM
        if self.semantic_aux:
            # Aux supervision taps the fused representation h_it, pre-GAT, so
            # the loss backpropagates through both attention blocks and the
            # task encoder — the full text-conditioned decision pathway.
            self.semantic_head = SemanticAuxHead(
                FUSED_DIM, type_vocab=type_vocab)
        # Eq. 18: lightweight two-layer shared Q-network
        self.q_head = nn.Sequential(
            nn.Linear(gat_out_dim, 64), nn.ReLU(), nn.Linear(64, action_dim))
        self.activation = nn.ELU()

    def _fuse_head(self, x, train):
        h = F.relu(self.fusion_fc1(x))
        h = F.dropout(h, p=FUSION_DROPOUT,
                      training=(self.training and train))
        if self.method in ('attention', 'concat', 'crossq'):
            return self.fusion_fc2(h)
        return h

    def _gated_fuse(self, fused, s_it, unbatched, train):
        """fuse_head(fused) + s_it, with the optional learnable scalar gate
        multiplying the fusion contribution (fusion_gate repair arm)."""
        fuse_out = self._fuse_head(fused, train)
        res = s_it.unsqueeze(0) if unbatched else s_it
        if self.fusion_gate:
            fuse_out = self.fusion_alpha * fuse_out
        return res + fuse_out

    def _probe_mark(self, sp, e_p):
        """Capture the two fusion inputs for input-level gradient probing.

        Frozen-upstream tensors (e.g. state_projection under a freeze) arrive
        without grad tracking; promoting them to leaves keeps the probe
        addressable by autograd.grad without restoring param grads we
        intentionally disabled.  Callers read gradients via
        ``torch.autograd.grad`` (never .grad, which partial backward passes
        would pollute).
        """
        for name, t in (('s_p', sp), ('e_p', e_p)):
            if not t.requires_grad:
                t.requires_grad_(True)
            self._probe[name] = t

    def _text_embed(self, text_features):
        if self.text_input == 'structured':
            return self.task_encoder(text_features)
        return self.text_projection(text_features)

    def _fuse(self, s_it, text_features, train=False):
        """Return h_it (..., N, 128)."""
        self._probe = {}
        if self.method == 'crossq':
            sp = self.state_projection(s_it)              # (B?, N, 256)
            unbatched = sp.ndim == 2
            if unbatched:
                sp = sp.unsqueeze(0)
            batch, nodes, dim = sp.shape
            e_p = self._text_embed(text_features)         # (B'', N, 256)
            if unbatched:
                e_p = e_p.unsqueeze(0)
            if self.grad_probe:
                self._probe_mark(sp, e_p)
            # o_event_i = Attn(s'_it, E, E): this node's traffic queries every
            # node's event token (N keys, non-degenerate).  Zero events make
            # K = V = 0, hence o_event = 0 exactly and the block adds nothing.
            E = (e_p.unsqueeze(1).expand(batch, nodes, nodes, dim)
                 .reshape(batch * nodes, nodes, dim))
            o_text, text_weights = self.attn_text_to_traffic(
                e_p.reshape(batch * nodes, 1, dim),
                sp.unsqueeze(1).expand(batch, nodes, nodes, dim)
                .reshape(batch * nodes, nodes, dim),
                sp.unsqueeze(1).expand(batch, nodes, nodes, dim)
                .reshape(batch * nodes, nodes, dim))
            o_text = o_text.reshape(batch, nodes, dim)
            o_event, ev_weights = self.attn_traffic_to_events(
                sp.reshape(batch * nodes, 1, dim), E, E)
            o_event = o_event.reshape(batch, nodes, dim)
            self.last_fusion_weights = text_weights.reshape(batch, nodes, -1).detach()
            fused = torch.cat([o_text, o_event], dim=-1)   # (B,N,512)
            h = self._gated_fuse(fused, s_it, unbatched, train)
            return h.squeeze(0) if unbatched else h
        if self.method == 'attention':
            sp = self.state_projection(s_it)            # (B?, N, 256)
            unbatched = sp.ndim == 2
            if unbatched:
                sp = sp.unsqueeze(0)
            batch, nodes, dim = sp.shape
            e_p = self._text_embed(text_features)         # e'_it (B,N,256)
            if unbatched:
                e_p = e_p.unsqueeze(0)
            if self.grad_probe:
                self._probe_mark(sp, e_p)
            # o_text_i = Attn(e'_it, S', S'): node i's text queries all node
            # traffic features (16 keys per query). The returned weights show
            # which intersections' traffic each text token focuses on.
            o_text, text_weights = self.attn_text_to_traffic(
                e_p.reshape(batch * nodes, 1, dim), sp.unsqueeze(1)
                .expand(batch, nodes, nodes, dim).reshape(batch * nodes, nodes, dim),
                sp.unsqueeze(1).expand(batch, nodes, nodes, dim)
                .reshape(batch * nodes, nodes, dim))
            o_text = o_text.reshape(batch, nodes, dim)
            # o_traffic_i = Attn(s'_it, e'_it, e'_it): each node queries its own
            # text token. Single-key attention is degenerate (weight = 1) but
            # the MHA projections are learned, matching the figure.
            o_traffic, _ = self.attn_traffic_to_text(
                sp.reshape(batch * nodes, 1, dim),
                e_p.reshape(batch * nodes, 1, dim),
                e_p.reshape(batch * nodes, 1, dim))
            o_traffic = o_traffic.reshape(batch, nodes, dim)
            self.last_fusion_weights = text_weights.reshape(batch, nodes, -1).detach()
            fused = torch.cat([o_text, o_traffic], dim=-1)         # (B,N,512)
            h = self._gated_fuse(fused, s_it, unbatched, train)    # (B,N,128)
            return h.squeeze(0) if unbatched else h
        if self.method == 'concat':
            sp = self.state_projection(s_it)
            e_p = self._text_embed(text_features)
            if self.grad_probe:
                self._probe_mark(sp, e_p)
            self.last_fusion_weights = None
            fused = torch.cat([sp, e_p], dim=-1)                   # (B?,N,512)
            return self._gated_fuse(fused, s_it, unbatched=False, train=train)
        if self.method == 'selfattn':
            sp = self.state_projection(s_it)                       # (B?,N,256)
            unbatched = sp.ndim == 2
            if unbatched:
                sp = sp.unsqueeze(0)
            e_p = self._text_embed(text_features)
            if unbatched:
                e_p = e_p.unsqueeze(0)
            if self.grad_probe:
                self._probe_mark(sp, e_p)
            tok = F.relu(self.tok_proj(                            # (B,N,256)
                torch.cat([sp, e_p], dim=-1)))
            o, node_weights = self.node_self_attn(tok, tok, tok)
            self.last_fusion_weights = node_weights.detach()       # (B,N,N)
            h = self._gated_fuse(o, s_it, unbatched, train)        # (B,N,128)
            return h.squeeze(0) if unbatched else h
        # gating
        sp = self.state_projection(s_it)
        e_p = self._text_embed(text_features)
        if self.grad_probe:
            self._probe_mark(sp, e_p)
        g = torch.sigmoid(self.gate(torch.cat([e_p, sp], dim=-1)))
        self.last_fusion_weights = g.detach()
        h_latent = g * e_p + (1 - g) * sp                          # Eq. 15
        return self._fuse_head(h_latent, train)

    def forward(self, traffic, text_features=None, adjacency=None, train=False):
        s_it = self.traffic_mlp(traffic)                           # Eq. 9
        if self.method in self.TEXT_METHODS:
            if text_features is None or text_features.shape[:-1] != traffic.shape[:-1]:
                raise ValueError('Text embeddings must align with every node and replay transition')
            h_it = self._fuse(s_it, text_features, train)
        else:
            h_it = s_it
        self._aux = None
        if self.semantic_aux and self.method in self.TEXT_METHODS:
            self._aux = self.semantic_head(h_it)
        z = h_it
        if self.gat_layers is not None:
            for index, layer in enumerate(self.gat_layers):
                z = layer(z, adjacency)
                if index < len(self.gat_layers) - 1:
                    z = self.activation(z)
        self._last_h_it = h_it.detach()
        self._last_z = z.detach()
        return self.q_head(z)

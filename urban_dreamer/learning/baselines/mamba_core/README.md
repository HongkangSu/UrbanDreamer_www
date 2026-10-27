# MAMBA network core

The discrete RSSM, agent attention, latent actor and attention critic are adapted
from jbr-ai-labs/mamba, distributed under the accompanying MIT license
(Copyright 2022 jbr-ai-labs). This is the AAMAS 2022 multi-agent model-based
algorithm.

The release adapter in `generic_mamba.py` uses the common 44-dimensional route
observation and legal-action mask. Its imagination uses learned latent
transitions, rewards, continuation and availability. Agent identities remain
stable inside each episode. Communication occurs within identity-sorted blocks
of at most 32 vehicles; no attention is computed across blocks. The critic and
actor optimize the learned mean team reward.

The production architecture has categorical `32 × 32` stochastic variables,
hidden width 256, three attention layers and eight heads. Training settings are
centralized in `GenericConfig`; smaller models are used only by explicit smoke
checks. See `docs/baselines.md` for the shared physical routing interface and
training/evaluation protocol.

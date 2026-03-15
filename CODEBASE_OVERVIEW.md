# ResFiT Codebase Overview

This document provides a detailed overview of the **Residual Off-Policy RL for Finetuning Behavior Cloning Policies** (ResFiT) codebase: structure, main packages, classes, algorithms, and how components connect.

**Paper:** [Residual Off-Policy RL for Finetuning Behavior Cloning Policies](https://arxiv.org/abs/2509.19301)  
**Website:** https://residual-offpolicy-rl.github.io/

---

## 1. High-Level Architecture

The project has two main training pipelines:

1. **BC (Behavior Cloning) policy training** – Train a base policy (ACT or Diffusion) on offline data; checkpoints are saved to W&B.
2. **Residual RL training** – Load the base policy, wrap the environment so that the RL agent outputs a *residual* on top of the base policy; train with TD3-style off-policy RL (replay buffer, critic/actor, optional offline data).

All **custom code** lives under `resfit/`. External dependencies are in `deps/` (lerobot, robosuite, mimicgen, dexmimicgen).

```
residual-offpolicy-rl/
├── resfit/                    # Main application code
│   ├── lerobot/               # BC training + policies (LeRobot-style)
│   ├── dexmg/                 # DexMimicGen/Robosuite environments
│   └── rl_finetuning/        # Residual TD3/RLPD training
├── deps/                      # External repos (lerobot, robosuite, mimicgen, dexmimicgen)
├── README.md
└── CODEBASE_OVERVIEW.md       # This file
```

---

## 2. Package: `resfit/lerobot`

Used for **BC policy training** on Hugging Face datasets and evaluation in DexMimicGen sims.

### 2.1 Scripts

| File | Purpose |
|------|--------|
| `scripts/train_bc_dexmg.py` | Main BC training script. Loads a HF dataset, builds policy via `make_policy`, trains with AdamW, optional DexMimicGen evaluation rollouts and W&B logging. |

**Important CLI options (train_bc_dexmg):**

- `--dataset`, `--policy` (act / diffusion), `--steps`, `--batch_size`
- `--eval_env`, `--rollout_freq`, `--eval_num_envs`, `--eval_num_episodes`
- `--wandb_enable`, `--wandb_project`
- `--policy_kwargs` for policy config overrides (JSON or key=value)

### 2.2 Policies

Base class and factory:

- **`policies/pretrained.py`**
  - `PreTrainedPolicy(nn.Module, HubMixin, abc.ABC)` – Base for all policies. Holds `config`, `save_pretrained`, `from_pretrained` (incl. W&B artifact `wandb:project/run_id`).

- **`policies/factory.py`**
  - `get_policy_class(name)` → policy class (`"act"` → ACTPolicy, `"diffusion"` → DiffusionPolicy; others raise NotImplementedError).
  - `make_policy_config(policy_type, **kwargs)` → config instance.
  - `make_policy(cfg, ds_meta=None, env_cfg=None)` → policy instance; uses dataset or env to set input/output features and optional normalization stats.

**ACT (Action Chunking Transformer)** – `policies/act/`

- **`configuration_act.py`**
  - `ACTConfig(PreTrainedConfig)` – `n_obs_steps`, `chunk_size`, `n_action_steps`, image/state shapes, backbone, transformer dims, VAE, temporal ensembling, etc.

- **`modeling_act.py`**
  - `ACTPolicy(PreTrainedPolicy)` – Top-level policy: normalization, `ACT` model, optional `ACTTemporalEnsembler`; `select_action(obs)`, `reset(env_ids)` for vectorized envs.
  - `ACT(nn.Module)` – Core transformer (encoder/decoder, action head).
  - `ACTEncoder`, `ACTEncoderLayer`, `ACTDecoder`, `ACTDecoderLayer` – Transformer blocks.
  - `ACTTemporalEnsembler` – Exponential weighting over action chunks for temporal ensembling.
  - `ACTSinusoidalPositionEmbedding2d` – 2D position encoding for images.

**Diffusion** – `policies/diffusion/`

- **`configuration_diffusion.py`**
  - `DiffusionConfig(PreTrainedConfig)` – Diffusion-specific hyperparameters.

- **`modeling_diffusion.py`**
  - `DiffusionPolicy(PreTrainedPolicy)` – Diffusion policy interface.
  - `DiffusionModel`, `DiffusionRgbEncoder`, `DiffusionConditionalUnet1d`, `DiffusionConditionalResidualBlock1d`, `DiffusionSinusoidalPosEmb`, `DiffusionConv1dBlock`, `SpatialSoftmax`.

### 2.3 Configs

- **`configs/policies.py`**
  - `PreTrainedConfig(draccus.ChoiceRegistry, HubMixin, abc.ABC)` – Base policy config; subclasses register via `@PreTrainedConfig.register_subclass("name")`.

### 2.4 Utils

- **`utils/load_policy.py`**
  - `download_policy_from_wandb(run_id, step=..., artifact_version=...)` → `(Path, step_str)`.
  - `load_policy(policy_dir)` → ACT (or Diffusion) from `config.json` + weights; supports W&B artifact layout.
  - `save_checkpoint`, `load_checkpoint` – Save/load policy + optimizer state.

### 2.5 Dataset

- **`dataset/convert_robomimic_to_lerobot.py`** – Conversion from Robomimic to LeRobot dataset format.

---

## 3. Package: `resfit/dexmg`

DexMimicGen/Robosuite **simulation environments** used for BC evaluation and residual RL.

### 3.1 `environments/dexmg.py`

- **`ENV_ROBOTS`** – Map from task name to list of robot models (e.g. `TwoArmCoffee` → GR1FixedLowerBody, `Lift` → Panda).

- **`RobosuiteGymWrapper`**
  - Wraps a single Robosuite env to a Gym-like API (step returns 5 elements, observations as dict).
  - Handles aliases (e.g. `Can` → `PickPlaceCan`, `Square` → `NutAssemblySquare`).
  - Observation keys include images and `observation.state`; action is delta pose + hand joints (see docstring for index layout).
  - Configurable `camera_size`, `render_size`, `horizon` per task.

- **`VectorizedEnvWrapper`**
  - Wraps `gym.vector.SyncVectorEnv` or `AsyncVectorEnv`.
  - Converts numpy obs to torch; exposes `reset()`, `step(actions)`, `render()` (video key), `device`.

- **`create_vectorized_env(env_name, num_envs, device, camera_size, render_size, debug, video_key)`**
  - Builds a list of env factory functions (`make_dexmimicgen_env`), creates Sync or Async vector env, wraps in `VectorizedEnvWrapper`, sets `video_key` for recording.

---

## 4. Package: `resfit/rl_finetuning`

Residual **TD3-style off-policy RL**: base BC policy + residual actor in a wrapped env; optional offline data; replay buffer; critic/actor updates.

### 4.1 Scripts

| File | Purpose |
|------|--------|
| `scripts/train_residual_td3.py` | Residual TD3 training: load base policy from W&B, build env + `BasePolicyVecEnvWrapper`, create `QAgent(residual_actor=True)`, fill online (and optionally offline) buffer, train with algo config (e.g. critic warmup, n_step, PER). |
| `scripts/train_rlpd_dexmg.py` | Standard RLPD (no residual): same stack but without base policy wrapper; used for non-residual baselines. |

Entry point for residual: run with Hydra, e.g. `--config-name=residual_td3_coffee_config` (see README and `resfit/rl_finetuning/README.md`).

### 4.2 Wrappers

- **`wrappers/residual_env_wrapper.py`**
  - **`BasePolicyVecEnvWrapper`**
    - Wraps `VectorizedEnvWrapper` + base policy (e.g. ACT) + `ActionScaler` + `StateStandardizer`.
    - **reset:** env reset → base policy reset → get base action → scale → augment obs with `observation.base_action` and standardized state.
    - **step(residual_naction):** `combined_naction = last_base_naction + residual_naction` → unscale → env.step → next base action → augment next obs; on `terminated`, reset base policy for those env indices.
    - Observation space: same as env but state standardized and `observation.base_action` added.

### 4.3 Config

All in **`config/`**; Hydra config store used for experiment selection.

- **`residual_td3.py`**
  - `OfflineDataConfig` – Dataset name, num_episodes, base-policy action labeling, normalization safeguards.
  - `WandBConfig`, `BasePolicyConfig` (wandb run id, artifact type/version).
  - `ResidualTD3AlgoConfig(RLPDAlgoConfig)` – critic_warmup_steps, random_action_noise_scale, use_base_policy_for_warmup, stddev schedule, progressive_clipping_steps.
  - `ResidualTD3DexmgConfig(RLPDDexmgConfig)` – algo, agent (QAgentConfig), offline_data, base_policy, wandb, eval_interval, eval_first.
  - Task-specific: `ResidualTD3CanConfig`, `ResidualTD3SquareConfig`, `ResidualTD3BoxCleanConfig`, `ResidualTD3CoffeeConfig`, `ResidualTD3TwoArmCanSortConfig`.

- **`rlpd.py`**
  - **Encoder / critic / actor:** `VitEncoderConfig`, `CriticLossCfg` (mse / hl_gauss / c51), `CriticConfig`, `ActorConfig`, `QAgentConfig` (encoder, critic, actor, LR warmup, BC regularization, freeze_encoder, etc.).
  - **Algorithm:** `RLPDAlgoConfig` – total_timesteps, batch_size, buffer_size, learning_starts, n_step, gamma, num_updates_per_iteration, prefetch_batches, offline_fraction, priority_alpha/beta, sampling_strategy.
  - **Experiment:** `RLPDDexmgConfig` and task-specific configs (e.g. `RLPDCoffeeConfig`, `RLPDTwoArmCanSortConfig`).

- **`performance.py`** – `PerformanceConfig` for tuning.

### 4.4 Agent and RL Networks

- **`off_policy/rl/q_agent.py`**
  - **`QAgent(nn.Module)`**
    - Builds per-camera encoders (ViT), critic, actor, target networks, optimizers, optional LR warmup schedulers, data augmentation.
    - `residual_actor=True`: actor gets `observation.base_action` and predicts residual; critic/actor loss use combined action when needed.
    - **Encoding:** `_encode(obs, augment)` → concat patch repr from all cameras; supports uint8 → float.
    - **Action:** `act(obs, eval_mode, stddev, cpu)` for rollout; `_act_default(..., use_target)` for current or target actor.
    - **Critic update:** `update_critic(obs, action, reward, discount, next_obs, stddev, importance_weights)` – target next action (with optional noise), combined action for residual mode, then MSE / HL-Gauss / C51 loss; optional PER weights.
    - **Actor update:** `update_actor(obs, stddev)` or `update_actor_rft(..., bc_batch, ref_agent)` with BC regularization; L2 on action magnitude; gradient clipping.
    - **Full step:** `update(batch, stddev, update_actor, bc_batch, ref_agent)` – encode, update_critic, soft_update targets, update_actor.

- **`off_policy/rl/actor.py`**
  - **`SpatialEmb`** – Fuses patch features and proprio; used when `spatial_emb > 0`.
  - **`Actor`** – Encoder repr + (optionally) base action → MLP → tanh; residual_actor adds action_dim to prop input; last layer init scale (e.g. 0 for residual).

- **`off_policy/rl/critic.py`**
  - **`Critic`** – Multi-head Q from patch repr + state + action; supports MSE, HL-Gauss, C51.
  - **`HLGaussLoss`, `C51Loss`** – Distributional losses.
  - **`SpatialEmbQNet`, `HeadMLP`, `SpatialEmbQEnsemble`** – Ensemble Q-heads with optional spatial embedding.

- **`off_policy/networks/encoder.py`**
  - **`VitEncoder`** – ViT for image observations; used by QAgent.

- **`off_policy/networks/min_vit.py`**
  - **`MinVit`** – Minimal ViT (PatchEmbed1/2, MultiHeadAttention, TransformerLayer).

### 4.5 Utils

- **`utils/normalization.py`**
  - **`ActionScaler`** – Min-max scale to [-1, 1] with expanded range and safeguards; `scale` / `unscale`; can be built from dataset stats via `from_dataset_stats`.
  - **`StateStandardizer`** – Standardize state from dataset stats; min_std safeguard; `standardize`.

- **`utils/rb_transforms.py`**
  - **`MultiStepTransform`** – TensorDict transform for n-step returns (gamma, done handling).

- **`utils/checkpoint.py`** – Checkpoint save/load helpers.

- **`utils/hugging_face.py`** – HF upload/download for replay buffer caches (`_hf_download_buffer`, `_hf_upload_buffer`, optimized_replay_buffer_dumps/loads).

- **`utils/evaluate_dexmg.py`**
  - **`run_dexmg_evaluation(...)`** – Run evaluation episodes in DexMimicGen and report metrics (e.g. success rate).

- **`utils/dtype.py`** – e.g. `to_uint8` for storing images in buffers.

### 4.6 Training Loop (residual TD3)

1. Load base policy from W&B (`download_policy_from_wandb` + `load_policy`).
2. Load offline dataset; build `ActionScaler` and `StateStandardizer` from dataset stats.
3. Create train and eval envs via `create_vectorized_env` → `BasePolicyVecEnvWrapper` (base policy + scaler + standardizer).
4. Build `QAgent(..., residual_actor=True)` with config from `cfg.agent`.
5. Replay buffers: online (and optionally offline) with `MultiStepTransform`; optional PER; optional HF cache load/save.
6. Warmup: fill buffer (e.g. base policy + noise or pure random, depending on `use_base_policy_for_warmup`).
7. Main loop: collect steps (agent.act with stddev schedule), add to buffer; sample batch; `agent.update(...)` (critic + actor); periodic eval via `run_dexmg_evaluation`; logging (e.g. W&B).

---

## 5. Key Classes Quick Reference

| Class | Module | Role |
|-------|--------|------|
| `PreTrainedPolicy` | resfit.lerobot.policies.pretrained | Base policy; save/load pretrained, W&B |
| `ACTPolicy`, `ACT` | resfit.lerobot.policies.act.modeling_act | ACT policy and core transformer |
| `DiffusionPolicy` | resfit.lerobot.policies.diffusion.modeling_diffusion | Diffusion policy |
| `make_policy`, `get_policy_class` | resfit.lerobot.policies.factory | Policy construction |
| `RobosuiteGymWrapper` | resfit.dexmg.environments.dexmg | Single Robosuite env → Gym API |
| `VectorizedEnvWrapper` | resfit.dexmg.environments.dexmg | Vector env + torch obs + render |
| `create_vectorized_env` | resfit.dexmg.environments.dexmg | Factory for vectorized DexMimicGen envs |
| `BasePolicyVecEnvWrapper` | resfit.rl_finetuning.wrappers.residual_env_wrapper | Base policy + residual action; obs augmentation |
| `QAgent` | resfit.rl_finetuning.off_policy.rl.q_agent | Encoder + critic + actor; residual or standard |
| `Actor`, `Critic` | resfit.rl_finetuning.off_policy.rl | Residual/standard actor; ensemble critic |
| `ActionScaler`, `StateStandardizer` | resfit.rl_finetuning.utils.normalization | Action/state normalization for residual RL |
| `ResidualTD3DexmgConfig`, `ResidualTD3AlgoConfig` | resfit.rl_finetuning.config.residual_td3 | Residual TD3 experiment and algo config |
| `RLPDAlgoConfig`, `QAgentConfig` | resfit.rl_finetuning.config.rlpd | Shared RLPD/Residual algo and agent config |

---

## 6. Data and Control Flow

- **BC:** HF dataset → LeRobot dataloader → policy forward → MSE on actions; optional rollout in `create_vectorized_env` + eval env for success rate.
- **Residual RL:** Env (DexMimicGen) → `BasePolicyVecEnvWrapper` (obs = state + base_action, action = residual) → QAgent (encoder + actor outputs residual) → wrapper adds base + residual → env.step. Replay buffer stores normalized (scaled) actions and standardized state; offline data can be mixed via `offline_fraction` and same normalization.

---

## 7. Config and Entry Points Summary

- **BC:**  
  `python resfit/lerobot/scripts/train_bc_dexmg.py --dataset <repo> --policy act --eval_env <EnvName> ...`

- **Residual TD3:**  
  `python resfit/rl_finetuning/scripts/train_residual_td3.py --config-name=residual_td3_<task>_config ...`  
  Task configs set `task`, `offline_data.name`, `base_policy.wandb_id`, etc. Base policy W&B id must point to a run that produced an ACT (or supported) checkpoint.

- **RLPD (no residual):**  
  `python resfit/rl_finetuning/scripts/train_rlpd_dexmg.py` with appropriate Hydra config.

---

## 8. Dependencies (high level)

- **resfit/lerobot:** torch, lerobot (deps/lerobot), wandb, huggingface_hub, safetensors, draccus, omegaconf.
- **resfit/dexmg:** gymnasium, robosuite, numpy, torch (mimicgen registered via import).
- **resfit/rl_finetuning:** torch, torchrl, tensordict, hydra, LeRobot dataset API, resfit.lerobot (policy load), resfit.dexmg (envs).

Setup: see root `README.md` and `resfit/rl_finetuning/README.md` (e.g. `./resfit/rl_finetuning/setup_rlpd_robosuite.sh`).

---

This overview should help you navigate the codebase, find where to change algorithms or add new tasks, and how BC and residual RL pipelines connect.

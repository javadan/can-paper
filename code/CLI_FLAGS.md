# CLI Reference

## Overview
This document catalogues every CLI option accepted by `src/main.ts`, including flags, aliases, defaults, config targets, and where the values are consumed.

## Parsing Entry Points
- `src/main.ts` `main()` — selects mode from `process.argv[2]`.
- `src/main.ts` `runAll()` — parses general execution flags, transition debug options, snake topology overrides, and Phase B/C config overrides.
- `src/main.ts` `buildPhaseBCTuningConfig()` — reuses Phase B/C override parsing for tuning/finetune workflows.
- `src/main.ts` `runPhaseBCProbeEntry()` — parses probe-specific flags plus shared snake and Phase B/C overrides.

## General Execution / Mode Selection

| Flag | Type / Default | Applies To | Stored In Config | Consumption |
| --- | --- | --- | --- | --- |
| positional `mode` (`run`, `tune-phaseA`, `tune-phaseB`, `tune-phaseB-finetune`, `finetune-phaseA`, `finetune:phaseA`, `finetune-phaseB`, `finetune:phaseB`, `probe-phaseBC`, `megatune-phaseBC`, `report`, `overnight`) | enum, default `run` | determines which workflow executes | n/a (dispatch only) | Dispatch map in `main()` chooses the handler function. |
| `--artifacts.dir=<path>` | string path, default `./artifacts` | all modes | n/a | Overrides the artifacts root used by `artifactPath`/`artifactsDir`; affects both reads and writes. |
| `--no-acc-history` | boolean, default `false` (history included) | run, probe | `ExperimentConfig.includeAccHistory` | Metrics history pruned in `ExperimentRunner` before emitting results. |
| `--proto-debug` | boolean, default `false` | run | `ExperimentConfig.protoDebug.enabled=true` | Proto debug logger created in `ExperimentRunner` to capture prototype traces. |
| `--proto-debug-limit=<int>` | integer, default `30` | run (only when `--proto-debug`) | `ExperimentConfig.protoDebug.limit` | Caps proto debug rows emitted by `ExperimentRunner`. |
| `--seed=<int>` | integer, default `42` | run | `ExperimentConfig.seed` | Sets the base RNG seed used when building experiment configs. |

## Activity Vector Options
| Flag | Type / Default | Applies To | Stored In Config | Consumption |
| --- | --- | --- | --- | --- |
| `--column.activityMode=<spike|ema_spike|ema_voltage>` | enum, default `spike` | all modes | `curriculum.phaseA.activityMode`, `curriculum.phaseBC.activityMode` | Chooses the activity signal driving MathColumn recurrence, successor drive, inhibition, and learning (binary spikes by default). |
| `--column.activityAlpha=<number>` | number, default `0.1` | all modes | `curriculum.phaseA.activityAlpha`, `curriculum.phaseBC.activityAlpha` | EMA smoothing factor for non-spike activity modes; ignored when `activityMode=spike`. |
| `--phaseBC.activityMode=<spike|ema_spike|ema_voltage>` | enum, default inherits `--column.activityMode`/`spike` | run, probe | `curriculum.phaseBC.activityMode` | Overrides the activity mode for Phase B/C only. |
| `--phaseBC.activityAlpha=<number>` | number, default inherits `--column.activityAlpha`/`0.1` | run, probe | `curriculum.phaseBC.activityAlpha` | Overrides Phase B/C EMA smoothing. |

Modes:
- `spike` — internal dynamics use the binary spike vector (existing behavior).
- `ema_spike` — dynamics use a low-pass filtered spike rate estimate.
- `ema_voltage` — dynamics use a low-pass filtered voltage-derived proxy (clamped to [0,1]).

Examples:
- `ts-node src/main.ts run --column.activityMode=ema_spike --column.activityAlpha=0.2`
- `ts-node src/main.ts run --phaseBC.activityMode=ema_voltage --phaseBC.activityAlpha=0.05`

## Transition Debugging Options
Parsed only when any `--debug.*` flag is present; otherwise ignored.

| Flag | Type / Default | Applies To | Stored In Config | Consumption |
| --- | --- | --- | --- | --- |
| `--debug.transitionTrace[=true|false]` | boolean, default `false` | run | `transitionDebug.transitionTrace` | Enables transition tracing in `TransitionDebugger`. |
| `--debug.traceTrials=<int>` | positive integer, default `100` | run | `transitionDebug.traceTrials` | Number of trials traced by `TransitionDebugger`. |
| `--debug.windowDefs=<list>` / `--debug.windows=<list>` | comma-list of window names, default `impulseOnly,early,mid,late,tail,mean,lateNoImpulse,tailNoImpulse,meanNoImpulse` | run | `transitionDebug.windowDefs` and `transitionDebug.windows` | Window set used by `TransitionDebugger.buildWindows`. |
| `--debug.excludeFirst=<list>` | comma-list of non-negative ints, default `0,1,2` | run | `transitionDebug.excludeFirst` | Digits excluded when evaluating transition windows. |
| `--debug.traceOutDir=<string>` | string, default `transition_traces` | run | `transitionDebug.traceOutDir` | Output directory for trace artifacts. |
| `--debug.perturb.enabled[=true|false]` | boolean, default `false` | run | `transitionDebug.perturb.enabled` | Enables perturbation/recovery evaluation during transition tracing. |
| `--debug.perturb.kind=<noise|dropout|shift>` | enum, default `noise` | run | `transitionDebug.perturb.kind` | Perturbation type applied to the activity vector. |
| `--debug.perturb.atStep=<int>` | integer, default `25` | run | `transitionDebug.perturb.atStep` | Step index (relative to free-run) where perturbation begins. |
| `--debug.perturb.durationSteps=<int>` | integer, default `1` | run | `transitionDebug.perturb.durationSteps` | Number of consecutive steps to apply the perturbation. |
| `--debug.perturb.noiseSigma=<number>` | number, default `0.03` | run | `transitionDebug.perturb.noiseSigma` | Noise standard deviation for `kind=noise`. |
| `--debug.perturb.dropoutP=<number>` | number, default `0.1` | run | `transitionDebug.perturb.dropoutP` | Dropout probability for `kind=dropout`. |
| `--debug.perturb.shiftDelta=<int>` | integer, default `1` | run | `transitionDebug.perturb.shiftDelta` | Circular shift magnitude for `kind=shift`. |
| `--debug.perturb.recoveryThreshold=<number>` | number, default `0.85` | run | `transitionDebug.perturb.recoveryThreshold` | Similarity threshold that marks recovery. |
| `--debug.perturb.maxRecoverySteps=<int>` | integer, default `120` | run | `transitionDebug.perturb.maxRecoverySteps` | Maximum steps to look for recovery. |
| `--debug.perturb.outDir=<string>` | string, default `transition_traces/perturb` | run | `transitionDebug.perturb.outDir` | Output directory for perturbation artifacts. |
| `--debug.transitionCurrents[=true|false]` | boolean, default `true` when any debug flag provided | run | `transitionDebug.transitionCurrents` | Toggles aggregate current capture in `TransitionDebugger`. |
| `--debug.ablateNext` | boolean, default `false` | run | `transitionDebug.ablateNext` | Ablates successor connections during debug runs. |
| `--debug.ablateRec` | boolean, default `false` | run | `transitionDebug.ablateRec` | Ablates recurrent connections during debug runs. |
| `--debug.ablateInhib` | boolean, default `false` | run | `transitionDebug.ablateInhib` | Ablates inhibition during debug runs. |
| `--debug.noNoise` | boolean, default `false` | run | `transitionDebug.noNoise` | Disables noise during debug runs. |

Notes:
- When `--debug.perturb.enabled=true`, transition tracing injects the perturbation into the activity vector during the free-run portion and reports recovery based on cosine similarity to the pre-perturb baseline state. Additional artifacts (`perturb_summary.json`, `perturb_trials.csv`) are written to `--debug.perturb.outDir` (default: `traceOutDir/perturb`).
- Perturbation traces require EMA/rate activity (non-`spike`) for the activity vector. If `activityMode=spike`, transition tracing temporarily switches to `ema_spike` for the perturbation run.

## Phase B/C Transition Overrides
Shared transition overrides apply to both topologies unless superseded by topology-specific flags. Precedence:
`--snake.*` / `--ring.*` > `--phaseBC.*` > defaults / artifact payloads.

| Flag | Type / Default | Applies To | Stored In Config | Consumption |
| --- | --- | --- | --- | --- |
| `--phaseBC.ttrans=<int>` | integer ≥ 5, default none (uses curriculum default) | run, probe (both topologies) | `curriculum.phaseBC.tTrans` override | Transition length used in curriculum and debug windows. |
| `--phaseBC.taillen=<int>` | integer ≥ 1 and ≤ `tTrans`, default none (uses curriculum default) | run, probe (both topologies) | `curriculum.phaseBC.tailLen` override | Tail length used in curriculum and debug windows. |

Topology-specific overrides apply only to the named topology when provided.

| Flag | Type / Default | Applies To | Stored In Config | Consumption |
| --- | --- | --- | --- | --- |
| `--snake.ttrans=<int>` | integer ≥ 5, default none (uses curriculum default) | run, probe (snake topologies) | `curriculum.phaseBC.tTrans` override | Transition length used in curriculum and debug windows. |
| `--snake.taillen=<int>` | integer ≥ 1 and ≤ `tTrans`, default none (uses curriculum default) | run, probe (snake topologies) | `curriculum.phaseBC.tailLen` override | Tail length used in curriculum and debug windows. |
| `--ring.ttrans=<int>` | integer ≥ 5, default none (uses curriculum default) | run, probe (ring topologies) | `curriculum.phaseBC.tTrans` override | Transition length used in curriculum and debug windows. |
| `--ring.taillen=<int>` | integer ≥ 1 and ≤ `tTrans`, default none (uses curriculum default) | run, probe (ring topologies) | `curriculum.phaseBC.tailLen` override | Tail length used in curriculum and debug windows. |

## Phase B/C Configuration Overrides
Usable in all modes (run, tuning, finetune, probe). Values merge into the Phase B/C config used by the chosen workflow.

### Physics source selection

| Flag | Type / Default | Applies To | Stored In Config | Consumption |
| --- | --- | --- | --- | --- |
| `--phaseBC.physicsSource=<default|phaseB_best|probe_best|megatune_best|none>` | enum, default `default` | run, probe | `resolveExperimentConfig` | Selects the artifact (if any) used to seed Phase B/C physics parameters. `default` keeps built-in defaults; `phaseB_best` loads `artifacts/phaseB_best.json`; `probe_best` loads `artifacts/probeBC_best_<topology>.json`; `megatune_best` loads `artifacts/megatune_best_<topology>.json`; `none` skips artifact loading. |
| `--phaseBC.physicsFrom=<path>` | string path | run, probe | `resolveExperimentConfig` | Explicit Phase B/C physics payload to load, overriding all other sources. Paths are resolved relative to `artifacts/` when not absolute. |

Precedence: `--phaseBC.physicsFrom` overrides all sources; otherwise `--phaseBC.physicsSource` selects between built-in defaults (`default`), single-best artifacts (`phaseB_best`, `probe_best`, `megatune_best`), or skipping artifact loading (`none`).

| Flag | Type / Default | Applies To | Stored In Config | Consumption |
| --- | --- | --- | --- | --- |
| `--phaseBC.evalWindow=<name>` | string window name, default `mean` | run, tune-phaseB, finetune-phaseB, probe | `phaseBCConfig.evalWindow` | Transition evaluation window selection in ExperimentRunner logic. |
| `--phaseBC.learnWindow=<name>` | string window name, default `mean` | run, tune-phaseB, finetune-phaseB, probe | `phaseBCConfig.learnWindow` | Learning window selection in ExperimentRunner. |
| `--phaseBC.settleWindow=<name>` | string window name, default `mean` | run, tune-phaseB, finetune-phaseB, probe | `phaseBCConfig.settleWindow` | Settle window used when running phases. |
| `--phaseBC.excludeFirstK=<int>` | integer, default `0` | run, tune-phaseB, finetune-phaseB, probe | `phaseBCConfig.excludeFirstK` | Exclusion horizon used by ExperimentRunner when validating transitions. |
| `--phaseBC.silentSpikeThreshold=<number>` | number, default `DEFAULT_SILENT_SPIKE_THRESHOLD` | run, tune-phaseB, finetune-phaseB, probe | `phaseBCConfig.silentSpikeThreshold` | Threshold consulted in ExperimentRunner diagnostics. |

### Sustain & Gating Options
| Flag | Type / Default | Applies To | Stored In Config | Consumption |
| --- | --- | --- | --- | --- |
| `--phaseBC.sustainGate[=true|false]` | boolean, default `false` | run, tune-phaseB, finetune-phaseB, probe | `phaseBCConfig.sustainGate.enabled` | Controls sustain gate checks in ExperimentRunner. |
| `--phaseBC.maxTailSilentFrac=<number>` | number, default `1.0` | run, tune-phaseB, finetune-phaseB, probe | `phaseBCConfig.sustainGate.maxTailSilentFrac` | Tail silence threshold for gating logic. |
| `--phaseBC.minTimeToSilence=<number>` | number, default `0` (disabled) | run, tune-phaseB, finetune-phaseB, probe | `phaseBCConfig.sustainGate.minTimeToSilence` | Minimum mean time-to-silence allowed before gating failure. |
| `--phaseBC.abortAfterTrials=<int>` (alias: `--phaseBC.sustainGate.abortAfterTrials`) | integer, default `0` | run, tune-phaseB, finetune-phaseB, probe | `phaseBCConfig.sustainGate.abortAfterTrials` | Warmup horizon during which sustain gate aborts are suppressed in Phase C. |
| `--phaseBC.skipUpdatesOnFail[=true|false]` | boolean, default `true` | run, tune-phaseB, finetune-phaseB, probe | `phaseBCConfig.sustainGate.skipUpdatesOnFail` | Skips parameter updates when sustain gate fails. |
| `--phaseBC.skipEpisodeOnFail[=true|false]` | boolean, default `false` | run, tune-phaseB, finetune-phaseB, probe | `phaseBCConfig.sustainGate.skipEpisodeOnFail` | Aborts episode when sustain gate fails. |
| `--phaseBC.logAbortLimit=<int>` | integer, default `10` | run, tune-phaseB, finetune-phaseB, probe | `phaseBCConfig.logAbortLimit` | Caps sustain-abort log lines in Phase C; logs a suppression notice after the limit. |
| `--phaseBC.strictSustain` | boolean flag | run, tune-phaseB, finetune-phaseB, probe | Convenience preset applied to multiple Phase B/C sustain settings | Enables sustain gate with update/episode skipping and sets eval/learn windows to `meanNoImpulse(k=1)` unless overridden. |

Notes:
- Sustain metrics and gating are computed inside `MathColumn.executeTransition`; `ExperimentRunner` aggregates counters and reports outcomes.
- Transition window names and ranges printed in logs come from the same `resolveTransitionWindows` helper used during execution, so parameterized `*NoImpulse(k=...)` windows retain their full names in logging and scoring.

### Controller Options
| Flag | Type / Default | Applies To | Stored In Config | Consumption |
| --- | --- | --- | --- | --- |
| `--controller.mode=<standard|bg>` | enum, default `standard` | run, probe, tune-phaseB, finetune-phaseB | `controller.mode` | Selects the controller policy layer: `standard` preserves existing sustain-gate semantics; `bg` enables the learned BG-style controller. |
| `--bg.actions=<comma list>` | list, default `GO,GO_NO_LEARN,WAIT,ABORT` | run, probe, tune-phaseB, finetune-phaseB | `controller.bg.actions` | Enumerates available BG actions. Unknown actions throw. |
| `--bg.epsilon=<number>` | number, default `0.05` | run, probe, tune-phaseB, finetune-phaseB | `controller.bg.epsilon` | ε-greedy exploration rate for BG policy. |
| `--bg.temperature=<number>` | number, default `1.0` | run, probe, tune-phaseB, finetune-phaseB | `controller.bg.temperature` | Softmax temperature for BG action sampling. |
| `--bg.sampleActions[=true|false]` | boolean, default `true` | run, probe, tune-phaseB, finetune-phaseB | `controller.bg.sampleActions` | When `false`, disables stochastic sampling and always picks the argmax action. |
| `--bg.eta=<number>` | number, default `0.05` | run, probe, tune-phaseB, finetune-phaseB | `controller.bg.eta` | Learning rate for BG controller weight updates. |
| `--bg.reward.correct=<number>` | number, default `1` | run, probe, tune-phaseB, finetune-phaseB | `controller.bg.reward.correct` | Reward delivered when a transition prediction is correct. |
| `--bg.reward.wrong=<number>` | number, default `-0.2` | run, probe, tune-phaseB, finetune-phaseB | `controller.bg.reward.wrong` | Reward delivered when a transition prediction is wrong. |
| `--bg.waitSteps=<int>` | integer, default `5` | run, probe, tune-phaseB, finetune-phaseB | `controller.bg.waitSteps` | Number of settle steps inserted before a transition when BG chooses the `WAIT` action. |

### Tuning / Grid Search Options
| Flag | Type / Default | Applies To | Stored In Config | Consumption |
| --- | --- | --- | --- | --- |
| `--tune.phaseA.maxSteps=<int>` | integer, default `1000` | tune-phaseA | n/a (Phase A tuning evaluator) | Caps steps in `evaluatePhaseACandidate`; applies to both topologies unless a topology-specific override is present. |
| `--snake.tune.phaseA.maxSteps=<int>` | integer, default inherits `--tune.phaseA.maxSteps`/`1000` | tune-phaseA (snake only) | n/a (Phase A tuning evaluator) | Overrides the Phase A tuning max steps for snake evaluations. |
| `--ring.tune.phaseA.maxSteps=<int>` | integer, default inherits `--tune.phaseA.maxSteps`/`1000` | tune-phaseA (ring only) | n/a (Phase A tuning evaluator) | Overrides the Phase A tuning max steps for ring evaluations. |
| `--tune.sustainWeight=<number>` | number, default `0` | tune-phaseB, finetune-phaseB | `phaseBCConfig.tuning.sustainWeight` | Weight of sustain metric in tuning fitness. |
| `--tune.useSustainFitness[=true|false]` | boolean, default `false` | tune-phaseB, finetune-phaseB | `phaseBCConfig.tuning.useSustainFitness` | Whether sustain fitness contributes to tuning score. |
| `--tune.sustainMetricWindow=late|tail` | enum, default `tail` | tune-phaseB, finetune-phaseB | `phaseBCConfig.tuning.sustainMetricWindow` | Window used for sustain fitness calculation. |

## Phase A Finetune Options
| Flag | Type / Default | Applies To | Stored In Config | Consumption |
| --- | --- | --- | --- | --- |
| `--phaseA.from=<path>` | string path, optional | finetune-phaseA | n/a | Prefers a specific Phase A snapshot directory when loading `phaseA_state_<topology>.json`; falls back to artifacts when omitted. |
| `--finetune.phaseA.targetAcc=<number>` | number, default `1.0` | finetune-phaseA | n/a (finetune criteria) | Target accuracy threshold that triggers early exit once sustained. |
| `--finetune.phaseA.windowSize=<int>` | integer, default `200` | finetune-phaseA | n/a (finetune criteria) | Rolling window size used to evaluate accuracy during finetune. |
| `--finetune.phaseA.minTrials=<int>` | integer, default `200` | finetune-phaseA | n/a (finetune criteria) | Minimum trial count before early-exit criteria are checked. |
| `--finetune.phaseA.requireAllDigits[=true|false]` | boolean, default `true` | finetune-phaseA | n/a (finetune criteria) | When true, early exit additionally requires every digit to meet `targetAcc`. |
| `--finetune.phaseA.maxSteps=<int>` | integer, default `20000` | finetune-phaseA | n/a (finetune criteria) | Global training-step cap applied unless topology-specific limits override. |
| `--snake.finetune.phaseA.maxSteps=<int>` | integer, default inherits `--finetune.phaseA.maxSteps`/`20000` | finetune-phaseA (snake only) | n/a (finetune criteria) | Topology-specific step cap for snake finetune runs. |
| `--ring.finetune.phaseA.maxSteps=<int>` | integer, default inherits `--finetune.phaseA.maxSteps`/`20000` | finetune-phaseA (ring only) | n/a (finetune criteria) | Topology-specific step cap for ring finetune runs. |
| `--finetune.phaseA.writePhaseABest[=true|false]` | boolean, default `false` | finetune-phaseA | n/a | When enabled, replaces `phaseA_best.json` with the finetune results after completion. |

## Probe Scoring Options
Parsed only in `probe-phaseBC` mode; combined with shared snake and Phase B/C overrides.

| Flag | Type / Default | Applies To | Stored In Config | Consumption |
| --- | --- | --- | --- | --- |
| `--probe.trials=<int>` | integer, default `300` | probe-phaseBC | n/a (probe config) | Trials per candidate: each sampled physics config runs this many successor transitions with random start digits. |
| `--probe.maxCandidates=<int>` | integer, default `50` | probe-phaseBC | n/a (probe config) | Candidate count: maximum number of sampled physics configs drawn from the grid and evaluated. |
| `--probe.rankTop=<int>` | integer, default `10` | probe-phaseBC | n/a (probe config) | Number of top candidates retained for deeper scoring. |
| `--probe.wSustain=<number>` | number, default `5` | probe-phaseBC | n/a (probe config) | Weight applied to sustain score in probe fitness. |
| `--probe.wAcc=<number>` | number, default `1` | probe-phaseBC | n/a (probe config) | Weight applied to accuracy score in probe fitness. |
| `--probe.seed=<int>` | integer, default `42` | probe-phaseBC | n/a (probe config) | Seed used for candidate sampling and scoring. |

## Megatune Phase B/C

Parsed only in `megatune-phaseBC` mode. The staged pipeline samples physics, semantics, and BG controller configurations while enforcing no-impulse defaults.

### Inputs and topology scope

| Flag | Type / Default | Applies To | Stored In Config | Consumption |
| --- | --- | --- | --- | --- |
| `--megatune.id=<string>` | string, default `megatune-<timestamp>` | megatune-phaseBC | n/a | Names the tuning run and seeds the default output directory. |
| `--megatune.outDir=<path>` | string path, default `artifacts/megatune/<id>` | megatune-phaseBC | n/a | Root folder for staged artifacts (guardrail, Top-K, best summaries). |
| `--megatune.topologies=<list>` | comma list, default `snake,ring` | megatune-phaseBC | n/a | Limits which topologies are tuned. |
| `--megatune.seed=<int>` | integer, default `42` | megatune-phaseBC | n/a | Base seed for sampling and evaluation RNGs. |

### Stage budgets

| Flag | Type / Default | Applies To | Stored In Config | Consumption |
| --- | --- | --- | --- | --- |
| `--megatune.guardrail.trials=<int>` | integer, default `80` | guardrail stage | n/a | Trial budget for the fast sanity check stage. |
| `--megatune.stage1.maxCandidates=<int>` | integer, default `200` | physics shortlist | n/a | Maximum sampled physics candidates. |
| `--megatune.stage1.trials=<int>` | integer, default `200` | physics shortlist | n/a | Trials per physics candidate. |
| `--megatune.stage1.topK=<int>` | integer, default `20` | physics shortlist | n/a | Top-K physics candidates retained. |
| `--megatune.stage2.maxSemantics=<int>` | integer, default `40` | semantics sweep | n/a | Max sampled semantics configurations per physics Top-K. |
| `--megatune.stage2.trials=<int>` | integer, default `150` | semantics sweep | n/a | Trials per semantics candidate. |
| `--megatune.stage2.topK=<int>` | integer, default `10` | semantics sweep | n/a | Top-K semantics candidates retained. |
| `--megatune.stage3.maxBgConfigs=<int>` | integer, default `40` | BG sweep | n/a | Max BG controller configurations sampled. |
| `--megatune.stage3.trials=<int>` | integer, default `200` | BG sweep | n/a | Trials per BG configuration. |
| `--megatune.stage3.topK=<int>` | integer, default `10` | BG sweep | n/a | Top-K BG candidates retained. |
| `--megatune.stage4.enabled[=true|false]` | boolean, default `false` | refinement | n/a | Enables the optional small physics refinement pass. |
| `--megatune.stage4.maxCandidates=<int>` | integer, default `80` | refinement | n/a | Max physics candidates in refinement. |
| `--megatune.stage4.trials=<int>` | integer, default `150` | refinement | n/a | Trials per refinement candidate. |

### Constraints and scoring

| Flag | Type / Default | Applies To | Stored In Config | Consumption |
| --- | --- | --- | --- | --- |
| `--megatune.constraints.allowImpulse[=true|false]` | boolean, default `false` | all megatune stages | n/a | Disallows impulse-only windows unless explicitly enabled. |
| `--megatune.constraints.forceLearnEqEval[=true|false]` | boolean, default `true` | all megatune stages | n/a | Forces learn/eval window equality unless relaxed. |
| `--megatune.constraints.minAcc=<number>` | number, default `0.2` | all megatune stages | n/a | Marks candidates with accuracy below this threshold as invalid (score `-Infinity`). |
| `--megatune.constraints.maxCollapseFrac=<number>` | number, default `0.6` | all megatune stages | n/a | Rejects candidates whose predictions collapse to a dominant class. |
| `--megatune.guardrail.enabled[=true|false]` | boolean, default `true` | guardrail | n/a | When false, skips the Stage 0 guardrail check entirely. |
| `--megatune.guardrail.continueOnFail[=true|false]` | boolean, default `true` | guardrail | n/a | When true, a failing guardrail logs a warning but still runs tuning; when false, the topology stops at guardrail. |
| `--megatune.guardrail.enforce` | boolean, default `false` | guardrail | n/a | Convenience flag that forces `continueOnFail=false` to yield a hard failure/exit code when guardrail blocks all topologies. |
| `--megatune.guardrail.minAcc=<number>` | number, default `0.1` | guardrail | n/a | Minimum accuracy required for the guardrail sanity check to pass. |
| `--megatune.guardrail.maxCollapseFrac=<number>` | number, default `0.9` | guardrail | n/a | Maximum collapse fraction tolerated during guardrail evaluation. |
| `--megatune.wAcc=<number>` | number, default `1.0` | scoring | n/a | Weight on accuracy term. |
| `--megatune.wSustain=<number>` | number, default `1.0` | scoring | n/a | Weight on sustain term (1 - tailSilentFrac). |
| `--megatune.wCollapse=<number>` | number, default `1.0` | scoring | n/a | Weight penalizing collapse. |

### Search spaces

| Flag | Type / Default | Applies To | Stored In Config | Consumption |
| --- | --- | --- | --- | --- |
| `--megatune.search.evalWindows=<list>` | comma list, default `meanNoImpulse,tailNoImpulse,lateNoImpulse` | semantics sweep | n/a | Eval windows explored (defaults exclude impulse-only windows). |
| `--megatune.search.settleWindows=<list>` | comma list, default `mean,mid,late` | semantics sweep | n/a | Settle window choices. |
| `--megatune.search.excludeFirstK=<list>` | comma list, default `0,1,2` | semantics sweep | n/a | Values explored for leading-digit exclusions. |
| `--megatune.search.activityModes=<list>` | comma list, default `spike,ema_spike` | semantics sweep | n/a | Activity modes tried for Phase B/C. |
| `--megatune.search.activityAlphas=<list>` | comma list, default `0.05,0.1,0.2` | semantics sweep | n/a | Activity alphas tried for Phase B/C. |
| `--megatune.search.bg.epsilons=<list>` | comma list, default `0.02,0.05,0.1` | BG sweep | n/a | BG epsilon values. |
| `--megatune.search.bg.temperatures=<list>` | comma list, default `1.0,1.4` | BG sweep | n/a | BG temperature values. |
| `--megatune.search.bg.waitSteps=<list>` | comma list, default `2,5,10` | BG sweep | n/a | BG wait-step grid. |
| `--megatune.search.bg.sampleActions=<list>` | comma list, default `true,false` | BG sweep | n/a | Whether BG samples or argmaxes actions. |

Artifacts land under `<outDir>/<topology>/` with guardrail, stage Top-K, and best files plus per-topology `megatune_best_<topology>.json` summaries.

## Report Mode
Generates publication-ready plots and a markdown index from existing run/probe artifacts.

| Flag | Type / Default | Applies To | Stored In Config | Consumption |
| --- | --- | --- | --- | --- |
| `--report.id=<string>` | string, default timestamped `report-<ts>` | report | n/a | Used as the report folder name under `artifacts/reports/` unless `--report.outDir` overrides. |
| `--report.outDir=<path>` | string path, default `artifacts/reports/<reportId>` | report | n/a | When relative, resolved under `artifacts/`; figures land in `<outDir>/figures`. |
| `--report.artifactsDir=<path>` | string path, default `artifacts/` | report | n/a | Overrides the artifact lookup root for probe/run inputs without changing output paths. |
| `--report.includeProbe[=true|false]` | boolean, default `true` | report | n/a | Toggles probe artifact inclusion. |
| `--report.includeRun[=true|false]` | boolean, default `true` | report | n/a | Toggles run artifact inclusion. |
| `--report.probeTopK=<int>` | integer, default `20` | report | n/a | Limits how many probe candidates are plotted and listed. |
| `--report.formats=<list>` | comma-list, default `png` | report | n/a | Image formats requested from `scripts/report_plot.py`. |
| `--report.topologies=<list>` | comma-list, default `snake,ring` | report | n/a | Topologies to search for artifacts. |
| `--report.input.run=<path>` | string path | report | n/a | Explicit run JSON to plot; otherwise auto-loads `run_last_<topology>_<counting>.json`. |
| `--report.input.probeTopK=<path>` | string path | report | n/a | Explicit probe Top-K payload path; otherwise auto-loads `probeBC_topK_<topology>.json`. |
| `--report.input.perturbSummary=<path>[,<path>...]` | string path list | report | n/a | Explicit perturb summary JSON(s) to plot; when multiple, entries align with `--report.input.perturbTrials` or `--report.perturbDirs`. |
| `--report.input.perturbTrials=<path>[,<path>...]` | string path list | report | n/a | Explicit perturb trials CSV(s) to plot; when multiple, entries align with `--report.input.perturbSummary` or `--report.perturbDirs`. |
| `--report.perturbDirs=<list>` | comma-list of paths | report | n/a | Explicit perturb directories to scan for `perturb_summary.json`/`perturb_trials.csv` (supports `<dir>` or `<dir>/perturb`). |
| `--report.failOnMissing[=true|false]` | boolean, default `true` | report | n/a | When true, missing artifacts abort report generation. |

Report output includes accuracy-history plots for Phase C (and Phase B when present) whenever run artifacts retain `accHistory`.
Runs invoked with `--no-acc-history` prune history; in that case the report notes the absence and omits the plots.

## Overnight Orchestration

`overnight` mode executes a plan of suites/steps sequentially, isolating artifacts per step while merging handoff files into `suite_artifacts/` for downstream steps and reporting.

| Flag | Type / Default | Applies To | Stored In Config | Consumption |
| --- | --- | --- | --- | --- |
| `--overnight.plan=<path>` | string path, **required** | overnight | n/a | JSON plan describing suites and steps to execute. |
| `--overnight.outDir=<path>` | string path, default `artifacts_overnight/<timestamp>` | overnight | n/a | Root output directory containing per-suite folders and run metadata. |
| `--overnight.cleanOutDir[=true|false]` | boolean, default `false` | overnight | n/a | Removes an existing `outDir` before running. |
| `--overnight.failFast[=true|false]` | boolean, default `true` | overnight | n/a | Stops when a step fails or exports are missing (unless `continueOnFail`). |
| `--overnight.continueOnFail[=true|false]` | boolean, default `false` | overnight | n/a | Keeps executing suites/steps even when failures occur. |
| `--overnight.resume[=true|false]` | boolean, default `true` | overnight | n/a | Skips steps whose `timing.json` already reports success. |
| `--overnight.topologies=<list>` | comma list, default `snake,ring` | overnight | n/a | Limits which topology-specific artifacts are exported per step. |
| `--overnight.formats=<list>` | comma list, default `png` | overnight | n/a | Forwarded to report generation for figure formats. |
| `--overnight.vaultOutDir=<path>` | string path, optional | overnight | n/a | Copies finished suite folders into the given vault directory. |
| `--overnight.useHtml[=true|false]` | boolean, default `false` | overnight | n/a | Emits `index.html` alongside markdown summaries when enabled. |

Notes:
- Each step runs with an isolated `--artifacts.dir=<suite>/steps/<step>/artifacts`; exported artifacts are merged into `<suite>/suite_artifacts/` for downstream consumption.
- Suite seeds in the plan are appended as `--probe.seed` and `--seed` unless the step args already override them.

## Precedence & Overrides
- Defaults originate from `DEFAULT_CURRICULUM`, `DEFAULT_LEARNING_RATES`, and `DEFAULT_PHASE_BC_CONFIG` and are copied when building an `ExperimentConfig`.
- Snapshot and tuning precedence: tuned params from `phaseA_best.json` / `phaseB_best.json` are applied atop defaults before CLI overrides.
- CLI overrides from `runAll()` and `buildPhaseBCTuningConfig()` are merged after tuned values; when both snapshot-tuned values and CLI flags exist, CLI wins for those fields.
- Phase A tuning step caps: topology-specific `--snake.tune.phaseA.maxSteps` / `--ring.tune.phaseA.maxSteps` override shared `--tune.phaseA.maxSteps`, which overrides the default of `1000`.
- Topology-specific snake overrides apply only when the selected topology is `snake`; ring topologies ignore them.
- Transition debug config is attached only when at least one `--debug.*` flag is present; otherwise no debug behavior is enabled.
- Proto debug config is attached only when `--proto-debug` is provided; the limit flag is ignored without it.

import fs from 'fs';
import path from 'path';

import {
  ActivityMode,
  Curriculum,
  DEFAULT_TAIL_LEN,
  DEFAULT_CURRICULUM,
  DEFAULT_PHASE_BC_CONFIG,
  DEFAULT_LEARNING_RATES,
  DEFAULT_T_TRANS,
  DEFAULT_CONTROLLER_CONFIG,
  ExperimentConfig,
  LearningRateSchedule,
  PhaseBCConfig,
  PhasePhysicsParams,
  ControllerConfig,
  Topology,
  TransitionWindowName,
} from './types';
import { ArtifactKey, artifactPath, resolveArtifactPath } from './artifactPaths';
import type { PhaseABestRecord, PhaseATuningRecord } from './phaseATuner';
import type { PhaseBTuningRecord } from './phaseBTuner';

interface PhaseABestPayload {
  timestamp?: string;
  bestPerTopology?: Partial<Record<Topology, PhaseABestRecord | null>>;
  allResults?: PhaseATuningRecord[];
}

interface PhaseBBestPayload {
  timestamp?: string;
  bestPerTopology?: Partial<Record<Topology, PhaseBTuningRecord | null>>;
  allResults?: PhaseBTuningRecord[];
}

interface ProbeBCBestPayload {
  version?: number;
  topology?: Topology;
  probe?: Record<string, unknown>;
  resolved?: {
    evalWindow?: TransitionWindowName;
    learnWindow?: TransitionWindowName;
    settleWindow?: TransitionWindowName;
    tTrans?: number;
    tailLen?: number;
    activityMode?: ActivityMode;
    activityAlpha?: number;
  };
  physics?: Partial<PhasePhysicsParams> & {
    eta_attr?: number;
    eta_out?: number;
  };
}

interface ExperimentOverrides {
  maxStepsPhaseA?: number;
  maxStepsPhaseB?: number;
  maxStepsPhaseC?: number;
  targetAccPhaseA?: number;
  targetAccPhaseB?: number;
  targetAccPhaseC?: number;
  includeAccHistory?: boolean;
  curriculum?: {
    phaseA?: Partial<PhasePhysicsParams>;
    phaseBC?: Partial<PhasePhysicsParams>;
  };
  learningRates?: Partial<LearningRateSchedule> & {
    phaseA?: Partial<LearningRateSchedule['phaseA']>;
    phaseBC?: Partial<LearningRateSchedule['phaseBC']>;
  };
  randomSeed?: number;
  phaseASnapshotPath?: string;
  phaseBCConfig?: Partial<PhaseBCConfig>;
  controller?: Partial<ControllerConfig>;
}

export interface ResolveExperimentConfigOptions {
  topology: Topology;
  useCountingPhase: boolean;
  seed: number;
  includeAccHistory?: boolean;
  overrides?: ExperimentOverrides;
  phaseBCConfigOverrides?: Partial<PhaseBCConfig>;
  phaseBCPhysicsSource?: PhaseBCPhysicsSource;
  phaseBCPhysicsFromPath?: string;
}

export type PhaseBCPhysicsSource = 'default' | 'phaseB_best' | 'probe_best' | 'megatune_best' | 'none' | 'file';

function cloneCurriculum(curriculum: Curriculum): Curriculum {
  return {
    phaseA: { ...curriculum.phaseA },
    phaseBC: { ...curriculum.phaseBC },
  };
}

function cloneLearningRates(schedule: LearningRateSchedule): LearningRateSchedule {
  return {
    phaseA: { ...schedule.phaseA },
    phaseBC: { ...schedule.phaseBC },
  };
}

function loadArtifactJsonIfExists<T>(key: ArtifactKey | string, options?: { topology?: Topology }): {
  payload: T;
  path: string;
} | null {
  const filePath = resolveArtifactPath({ key, topology: options?.topology });
  if (!filePath || !fs.existsSync(filePath)) return null;
  const raw = fs.readFileSync(filePath, 'utf-8');
  return { payload: JSON.parse(raw) as T, path: filePath };
}

function applyPhaseATunedParams(
  curriculum: Curriculum,
  learningRates: LearningRateSchedule,
  topology: Topology,
): void {
  const payload = loadArtifactJsonIfExists<PhaseABestPayload>('phaseA_best');
  const tuned = payload?.payload.bestPerTopology?.[topology]?.params;
  if (!tuned) return;

  curriculum.phaseA = {
    ...curriculum.phaseA,
    k_inhib: tuned.k_inhib,
    v_th: tuned.v_th,
    alpha: tuned.alpha,
    wInRowNorm: tuned.wInRowNorm,
    wAttrRowNorm: tuned.wAttrRowNorm,
    etaTrans: 0,
  };

  learningRates.phaseA = {
    ...learningRates.phaseA,
    eta_attr: tuned.eta_attr ?? learningRates.phaseA.eta_attr,
    eta_out: tuned.eta_out ?? learningRates.phaseA.eta_out,
    eta_trans: 0,
  };
}

function applyPhaseBCPhysics(
  curriculum: Curriculum,
  learningRates: LearningRateSchedule,
  physics: Partial<PhasePhysicsParams> & { eta_attr?: number; eta_out?: number },
): void {
  if (physics.k_inhib !== undefined) curriculum.phaseBC.k_inhib = physics.k_inhib;
  if (physics.v_th !== undefined) curriculum.phaseBC.v_th = physics.v_th;
  if (physics.alpha !== undefined) curriculum.phaseBC.alpha = physics.alpha;
  if (physics.wInRowNorm !== undefined) curriculum.phaseBC.wInRowNorm = physics.wInRowNorm;
  if (physics.wAttrRowNorm !== undefined) curriculum.phaseBC.wAttrRowNorm = physics.wAttrRowNorm;
  if (physics.etaTrans !== undefined) {
    curriculum.phaseBC.etaTrans = physics.etaTrans;
    learningRates.phaseBC.eta_trans = physics.etaTrans;
  }
  if (physics.tTrans !== undefined) curriculum.phaseBC.tTrans = physics.tTrans;
  if (physics.tailLen !== undefined) curriculum.phaseBC.tailLen = physics.tailLen;
  if (physics.activityMode) curriculum.phaseBC.activityMode = physics.activityMode;
  if (physics.activityAlpha !== undefined) curriculum.phaseBC.activityAlpha = physics.activityAlpha;
  if (physics.eta_attr !== undefined) learningRates.phaseBC.eta_attr = physics.eta_attr;
  if (physics.eta_out !== undefined) learningRates.phaseBC.eta_out = physics.eta_out;
}

function applyResolvedProbeFields(
  curriculum: Curriculum,
  phaseBCConfig: PhaseBCConfig,
  resolved?: ProbeBCBestPayload['resolved'],
): void {
  if (!resolved) return;
  if (resolved.tTrans !== undefined) curriculum.phaseBC.tTrans = resolved.tTrans;
  if (resolved.tailLen !== undefined) curriculum.phaseBC.tailLen = resolved.tailLen;
  if (resolved.activityMode) curriculum.phaseBC.activityMode = resolved.activityMode;
  if (resolved.activityAlpha !== undefined) curriculum.phaseBC.activityAlpha = resolved.activityAlpha;
  if (resolved.evalWindow) phaseBCConfig.evalWindow = resolved.evalWindow;
  if (resolved.learnWindow) phaseBCConfig.learnWindow = resolved.learnWindow;
  if (resolved.settleWindow) phaseBCConfig.settleWindow = resolved.settleWindow;
}

function resolvePhysicsFilePath(filePath: string): string {
  if (path.isAbsolute(filePath)) return filePath;
  return artifactPath(filePath);
}

function applyPhaseBTunedParams(
  curriculum: Curriculum,
  learningRates: LearningRateSchedule,
  topology: Topology,
): { applied: boolean; file?: string } {
  const filename = 'phaseB_best.json';
  const payload = loadArtifactJsonIfExists<PhaseBBestPayload>('phaseB_best');
  const tuned = payload?.payload.bestPerTopology?.[topology]?.params;
  if (!tuned) return { applied: false };

  applyPhaseBCPhysics(curriculum, learningRates, tuned);

  return { applied: true, file: payload?.path ?? artifactPath(filename) };
}

function applyProbeBestPhysics(
  curriculum: Curriculum,
  learningRates: LearningRateSchedule,
  phaseBCConfig: PhaseBCConfig,
  topology: Topology,
): { applied: boolean; file?: string } {
  const filename = `probeBC_best_${topology}.json`;
  const payload = loadArtifactJsonIfExists<ProbeBCBestPayload>('probeBC_best', { topology });
  if (!payload?.payload.physics) return { applied: false };

  applyPhaseBCPhysics(curriculum, learningRates, payload.payload.physics);
  applyResolvedProbeFields(curriculum, phaseBCConfig, payload.payload.resolved);
  return { applied: true, file: payload.path ?? artifactPath(filename) };
}

const NO_IMPULSE_WINDOW = /^(lateNoImpulse|tailNoImpulse|meanNoImpulse)(\(k=\d+\))?$/;

function alignNoImpulseWindows(
  phaseBCConfig: PhaseBCConfig,
  controller: ControllerConfig,
  defaults: PhaseBCConfig,
): void {
  const evalWindow = phaseBCConfig.evalWindow;
  const isNoImpulse = NO_IMPULSE_WINDOW.test(evalWindow);
  if (controller.mode !== 'bg' || !isNoImpulse) return;

  if (!NO_IMPULSE_WINDOW.test(phaseBCConfig.learnWindow)) {
    phaseBCConfig.learnWindow = evalWindow;
  }

  const settleIsDefault = phaseBCConfig.settleWindow === defaults.settleWindow;
  if (settleIsDefault && !NO_IMPULSE_WINDOW.test(phaseBCConfig.settleWindow)) {
    phaseBCConfig.settleWindow = evalWindow;
  }
}

function applyExplicitPhysicsFromFile(
  curriculum: Curriculum,
  learningRates: LearningRateSchedule,
  phaseBCConfig: PhaseBCConfig,
  filePath: string,
): { applied: boolean; file?: string } {
  const resolved = resolvePhysicsFilePath(filePath);
  if (!fs.existsSync(resolved)) {
    throw new Error(`[PhaseBC Physics] Requested physics file not found: ${resolved}`);
  }
  const payload = JSON.parse(fs.readFileSync(resolved, 'utf-8')) as ProbeBCBestPayload;
  if (!payload.physics) {
    throw new Error(`[PhaseBC Physics] File ${resolved} missing required "physics" field.`);
  }
  applyPhaseBCPhysics(curriculum, learningRates, payload.physics);
  applyResolvedProbeFields(curriculum, phaseBCConfig, payload.resolved);
  return { applied: true, file: resolved };
}

function applyMegatuneBest(
  curriculum: Curriculum,
  learningRates: LearningRateSchedule,
  phaseBCConfig: PhaseBCConfig,
  controller: ControllerConfig,
  topology: Topology,
): { applied: boolean; file?: string } {
  const filename = `megatune_best_${topology}.json`;
  const resolved = resolveArtifactPath({ key: filename, topology, logger: () => {} });
  if (!resolved || !fs.existsSync(resolved)) return { applied: false };
  const payload = JSON.parse(fs.readFileSync(resolved, 'utf-8')) as {
    phaseBC?: PhasePhysicsParams;
    semantics?: Partial<PhaseBCConfig> & {
      excludeFirstK?: number;
      activityMode?: ActivityMode;
      activityAlpha?: number;
    };
    bg?: Partial<ControllerConfig['bg']>;
  };
  if (payload.phaseBC) {
    applyPhaseBCPhysics(curriculum, learningRates, payload.phaseBC);
  }
  if (payload.semantics) {
    const semantics = payload.semantics;
    if (semantics.evalWindow) phaseBCConfig.evalWindow = semantics.evalWindow;
    if (semantics.learnWindow) phaseBCConfig.learnWindow = semantics.learnWindow;
    if (semantics.settleWindow) phaseBCConfig.settleWindow = semantics.settleWindow;
    if (semantics.excludeFirstK !== undefined) phaseBCConfig.excludeFirstK = semantics.excludeFirstK;
    if (semantics.activityMode) curriculum.phaseBC.activityMode = semantics.activityMode;
    if (semantics.activityAlpha !== undefined) curriculum.phaseBC.activityAlpha = semantics.activityAlpha;
  }
  if (payload.bg && controller.bg) {
    controller.mode = 'bg';
    controller.bg = { ...controller.bg, ...payload.bg };
  }
  return { applied: true, file: resolved };
}

function resolveSnapshotPath(topology: Topology, provided?: string): string | undefined {
  if (provided) {
    if (path.isAbsolute(provided)) return provided;
    return artifactPath(provided);
  }
  return resolveArtifactPath({ key: 'phaseA_state', topology });
}

function applyOverrides<T extends object>(target: T, overrides: Partial<T> | undefined): void {
  if (!overrides) return;
  Object.entries(overrides).forEach(([key, value]) => {
    if (value !== undefined) {
      // eslint-disable-next-line @typescript-eslint/no-explicit-any
      (target as any)[key] = value;
    }
  });
}

export function resolveExperimentConfig(options: ResolveExperimentConfigOptions): ExperimentConfig {
  const curriculum = cloneCurriculum(DEFAULT_CURRICULUM);
  const learningRates = cloneLearningRates(DEFAULT_LEARNING_RATES);
  const phaseBCConfig: PhaseBCConfig = {
    ...DEFAULT_PHASE_BC_CONFIG,
    sustainGate: { ...DEFAULT_PHASE_BC_CONFIG.sustainGate },
    tuning: { ...DEFAULT_PHASE_BC_CONFIG.tuning },
  };
  const controllerConfig: ControllerConfig = {
    ...DEFAULT_CONTROLLER_CONFIG,
    bg: { ...DEFAULT_CONTROLLER_CONFIG.bg!, minPhaseDuration: { ...DEFAULT_CONTROLLER_CONFIG.bg!.minPhaseDuration } },
  };

  const requestedPhysicsSource: PhaseBCPhysicsSource = options.phaseBCPhysicsFromPath
    ? 'file'
    : options.phaseBCPhysicsSource ?? 'default';
  let physicsFile: string | undefined;
  let physicsSourceUsed: PhaseBCPhysicsSource = requestedPhysicsSource;

  applyPhaseATunedParams(curriculum, learningRates, options.topology);

  if (requestedPhysicsSource === 'phaseB_best') {
    const result = applyPhaseBTunedParams(curriculum, learningRates, options.topology);
    physicsFile = result.file;
  } else if (requestedPhysicsSource === 'probe_best') {
    const result = applyProbeBestPhysics(curriculum, learningRates, phaseBCConfig, options.topology);
    physicsFile = result.file;
  } else if (requestedPhysicsSource === 'megatune_best') {
    const result = applyMegatuneBest(
      curriculum,
      learningRates,
      phaseBCConfig,
      controllerConfig,
      options.topology,
    );
    if (result.applied) physicsSourceUsed = 'megatune_best';
    physicsFile = result.file;
  }

  if (options.phaseBCPhysicsFromPath) {
    const result = applyExplicitPhysicsFromFile(
      curriculum,
      learningRates,
      phaseBCConfig,
      options.phaseBCPhysicsFromPath,
    );
    physicsFile = result.file;
    physicsSourceUsed = 'file';
  }

  const resolvedSnapshot = resolveSnapshotPath(options.topology, options.overrides?.phaseASnapshotPath);

  const config: ExperimentConfig = {
    topology: options.topology,
    useCountingPhase: options.useCountingPhase,
    maxStepsPhaseA: 2000,
    maxStepsPhaseB: 12000,
    maxStepsPhaseC: 5000,
    targetAccPhaseA: 0.99,
    targetAccPhaseB: 0.99,
    targetAccPhaseC: 1.0,
    curriculum,
    learningRates,
    randomSeed: options.overrides?.randomSeed ?? options.seed,
    phaseASnapshotPath: resolvedSnapshot,
    includeAccHistory: options.overrides?.includeAccHistory ?? options.includeAccHistory ?? true,
    phaseBCConfig,
    controller: controllerConfig,
  };

  const actionsOverridden = Boolean(options.overrides?.controller?.bg?.actions);

  applyOverrides(config, {
    maxStepsPhaseA: options.overrides?.maxStepsPhaseA,
    maxStepsPhaseB: options.overrides?.maxStepsPhaseB,
    maxStepsPhaseC: options.overrides?.maxStepsPhaseC,
    targetAccPhaseA: options.overrides?.targetAccPhaseA,
    targetAccPhaseB: options.overrides?.targetAccPhaseB,
    targetAccPhaseC: options.overrides?.targetAccPhaseC,
  });

  if (options.overrides?.curriculum) {
    applyOverrides(curriculum.phaseA, options.overrides.curriculum.phaseA);
    applyOverrides(curriculum.phaseBC, options.overrides.curriculum.phaseBC);
  }

  if (options.overrides?.learningRates) {
    applyOverrides(learningRates.phaseA, options.overrides.learningRates.phaseA);
    applyOverrides(learningRates.phaseBC, options.overrides.learningRates.phaseBC);
  }

  if (options.overrides?.phaseBCConfig) {
    applyOverrides(phaseBCConfig, options.overrides.phaseBCConfig);
    if (options.overrides.phaseBCConfig.sustainGate) {
      applyOverrides(phaseBCConfig.sustainGate, options.overrides.phaseBCConfig.sustainGate);
    }
    if (options.overrides.phaseBCConfig.tuning) {
      applyOverrides(phaseBCConfig.tuning, options.overrides.phaseBCConfig.tuning);
    }
  }

  if (options.overrides?.controller) {
    if (options.overrides.controller.mode !== undefined) {
      controllerConfig.mode = options.overrides.controller.mode;
    }
    if (options.overrides.controller.bg) {
      const bgOverrides = options.overrides.controller.bg;
      controllerConfig.bg = {
        ...controllerConfig.bg!,
        ...bgOverrides,
        reward: bgOverrides.reward
          ? { ...controllerConfig.bg!.reward, ...bgOverrides.reward }
          : controllerConfig.bg!.reward,
        minPhaseDuration: {
          ...controllerConfig.bg!.minPhaseDuration,
          ...(bgOverrides.minPhaseDuration ?? {}),
        },
      };
    }
  }

  if (options.phaseBCConfigOverrides) {
    applyOverrides(phaseBCConfig, options.phaseBCConfigOverrides);
    if (options.phaseBCConfigOverrides.sustainGate) {
      applyOverrides(phaseBCConfig.sustainGate, options.phaseBCConfigOverrides.sustainGate);
    }
    if (options.phaseBCConfigOverrides.tuning) {
      applyOverrides(phaseBCConfig.tuning, options.phaseBCConfigOverrides.tuning);
    }
  }

  alignNoImpulseWindows(phaseBCConfig, controllerConfig, DEFAULT_PHASE_BC_CONFIG);

  const sustainGateEnabled = phaseBCConfig.sustainGate.enabled;
  if (!actionsOverridden && !sustainGateEnabled && controllerConfig.mode === 'bg' && controllerConfig.bg) {
    controllerConfig.bg.actions = controllerConfig.bg.actions.filter((action) => action !== 'ABORT');
  }

  if (options.overrides?.phaseASnapshotPath !== undefined) {
    config.phaseASnapshotPath = resolveSnapshotPath(options.topology, options.overrides.phaseASnapshotPath);
  }

  curriculum.phaseBC.tTrans = curriculum.phaseBC.tTrans ?? DEFAULT_T_TRANS;
  const defaultTailLen = Math.min(DEFAULT_TAIL_LEN, curriculum.phaseBC.tTrans);
  curriculum.phaseBC.tailLen = curriculum.phaseBC.tailLen ?? defaultTailLen;
  if (curriculum.phaseBC.tailLen > curriculum.phaseBC.tTrans) {
    curriculum.phaseBC.tailLen = curriculum.phaseBC.tTrans;
  }
  if (curriculum.phaseBC.wAttrRowNorm === undefined) {
    curriculum.phaseBC.wAttrRowNorm = curriculum.phaseA.wAttrRowNorm;
  }

  const physicsSummary = {
    k_inhib: curriculum.phaseBC.k_inhib,
    v_th: curriculum.phaseBC.v_th,
    alpha: curriculum.phaseBC.alpha,
    wInRowNorm: curriculum.phaseBC.wInRowNorm,
    wAttrRowNorm: curriculum.phaseBC.wAttrRowNorm,
    etaTrans: curriculum.phaseBC.etaTrans,
    eta_attr: learningRates.phaseBC.eta_attr,
    eta_out: learningRates.phaseBC.eta_out,
    tTrans: curriculum.phaseBC.tTrans,
    tailLen: curriculum.phaseBC.tailLen,
    activityMode: curriculum.phaseBC.activityMode,
    activityAlpha: curriculum.phaseBC.activityAlpha,
    evalWindow: phaseBCConfig.evalWindow,
    learnWindow: phaseBCConfig.learnWindow,
    settleWindow: phaseBCConfig.settleWindow,
  };
  console.log(
    `[PhaseBC Physics] topology=${options.topology} source=${physicsSourceUsed} file=${physicsFile ?? 'n/a'} params=${JSON.stringify(
      physicsSummary,
    )}`,
  );

  return config;
}

import fs from 'fs';

import { ExperimentRunner } from './experimentRunner';
import { runPhaseATuningEntry } from './phaseATuner';
export { runPhaseATuningEntry } from './phaseATuner';
import { runPhaseAFinetuneEntry } from './phaseAFinetune';
import { parseActivityOverrides } from './activityOverrides';
import {
  phaseBIsBetter,
  PhaseBFinetuneOptions,
  PhaseBTuningRecord,
  runPhaseBFinetune,
  runPhaseBTuning,
} from './phaseBTuner';
import {
  DEFAULT_T_TRANS,
  DEFAULT_PHASE_BC_CONFIG,
  ExperimentConfig,
  ExperimentResult,
  PhaseBCConfig,
  PhasePhysicsParams,
  Topology,
  TransitionDebugConfig,
  TransitionWindowName,
  ControllerConfig,
  BGControllerAction,
  BGControllerConfig,
  ControllerPhase,
} from './types';
import { PhaseBCPhysicsSource, resolveExperimentConfig } from './configResolver';
import { artifactPath, exportToSharedArtifacts, resolveArtifactPath } from './artifactPaths';
import { runPhaseBCProbe } from './phaseBCProbe';
import { runReportEntry, ReportOptions } from './reporting/reportRunner';
import { runOvernight } from './overnight/overnightRunner';
import { runMegaTunePhaseBC } from './megatunePhaseBC';

function printResult(res: ExperimentResult): void {
  const { config } = res;

  const phaseA = res.metrics.find((m) => m.phase === 'PHASE_A_DIGITS');
  const phaseB = res.metrics.find((m) => m.phase === 'PHASE_B_COUNTING');
  const phaseC = res.metrics.find((m) => m.phase === 'PHASE_C_SUCCESSOR');

  console.log('------------------------------------------------------------');
  console.log(`Topology: ${config.topology.toUpperCase()} | Counting phase: ${config.useCountingPhase ? 'ON' : 'OFF'}`);

  const formatPhaseLine = (
    label: string,
    metrics: typeof phaseA,
    target: number,
    success: boolean,
  ) => {
    const finalAcc = metrics?.finalAccuracy ?? 0;
    const steps = metrics?.steps ?? 0;
    console.log(
      `${label}:     finalAcc=${finalAcc.toFixed(3)}  steps=${steps.toString().padEnd(4)} target=${target.toFixed(
        2,
      )}  success=${success}`,
    );
  };

  formatPhaseLine('Phase A (Digits)', phaseA, config.targetAccPhaseA, res.successPhaseA);

  if (config.useCountingPhase) {
    formatPhaseLine('Phase B (Counting)', phaseB, config.targetAccPhaseB, res.successPhaseB);
  } else {
    console.log('Phase B (Counting):   SKIPPED');
  }

  formatPhaseLine('Phase C (Successor)', phaseC, config.targetAccPhaseC, res.successPhaseC);

  const overallSuccess =
    res.successPhaseA && (config.useCountingPhase ? res.successPhaseB : true) && res.successPhaseC;
  console.log(`Overall success: ${overallSuccess ? '✅ YES' : '❌ NO'}`);
  console.log();
}

type TransitionOverrides = {
  tTrans?: number;
  tailLen?: number;
};

const DEFAULT_DEBUG_WINDOWS: TransitionWindowName[] = [
  'impulseOnly',
  'early',
  'mid',
  'late',
  'tail',
  'mean',
  'lateNoImpulse',
  'tailNoImpulse',
  'meanNoImpulse',
];

function parseReportOptions(): ReportOptions {
  const argv = process.argv;
  const getString = (prefix: string): string | undefined => {
    const raw = argv.find((arg) => arg.startsWith(prefix));
    if (!raw) return undefined;
    const idx = raw.indexOf('=');
    return idx >= 0 ? raw.slice(idx + 1) : undefined;
  };
  const getList = (prefix: string): string[] | undefined => {
    const raw = getString(prefix);
    if (!raw) return undefined;
    const list = raw
      .split(',')
      .map((v) => v.trim())
      .filter((v) => v.length > 0);
    return list.length > 0 ? list : undefined;
  };
  const parseBool = (flag: string, defaultValue: boolean): boolean => {
    const raw = argv.find((arg) => arg.startsWith(flag));
    if (!raw) return defaultValue;
    const idx = raw.indexOf('=');
    const value = idx >= 0 ? raw.slice(idx + 1) : undefined;
    if (value === undefined) return true;
    return !(value === '0' || value.toLowerCase() === 'false');
  };
  const parseIntFlag = (flag: string, defaultValue: number): number => {
    const raw = argv.find((arg) => arg.startsWith(flag));
    if (!raw) return defaultValue;
    const [, value] = raw.split('=');
    const parsed = parseInt(value, 10);
    if (!Number.isFinite(parsed)) {
      throw new Error(`[CLI] ${flag} must be numeric. Received: ${value}`);
    }
    return parsed;
  };

  const formats = getList('--report.formats=') ?? ['png'];
  const topologies = (getList('--report.topologies=') ?? ['snake', 'ring']) as Topology[];
  topologies.forEach((t) => {
    if (t !== 'snake' && t !== 'ring') {
      throw new Error(`[CLI] --report.topologies must contain snake and/or ring. Received: ${t}`);
    }
  });
  const artifactsDir = getString('--report.artifactsDir=');

  return {
    reportId: getString('--report.id='),
    outDir: getString('--report.outDir='),
    artifactsDir,
    includeProbe: parseBool('--report.includeProbe', true),
    includeRun: parseBool('--report.includeRun', true),
    probeTopK: parseIntFlag('--report.probeTopK', 20),
    formats,
    topologies,
    inputRun: getString('--report.input.run='),
    inputProbeTopK: getString('--report.input.probeTopK='),
    inputPerturbSummaries: getList('--report.input.perturbSummary='),
    inputPerturbTrials: getList('--report.input.perturbTrials='),
    perturbDirs: getList('--report.perturbDirs='),
    failOnMissing: parseBool('--report.failOnMissing', true),
  };
}

function parseControllerOverrides(argv: string[]): Partial<ControllerConfig> | undefined {
  const allowedActions: BGControllerAction[] = ['GO', 'GO_NO_LEARN', 'WAIT', 'ABORT'];
  const overrides: Partial<ControllerConfig> = {};
  const modeArg = argv.find((arg) => arg.startsWith('--controller.mode='));
  if (modeArg) {
    const value = modeArg.slice(modeArg.indexOf('=') + 1) as ControllerConfig['mode'];
    if (value !== 'standard' && value !== 'bg') {
      throw new Error(`[CLI] --controller.mode must be "standard" or "bg". Received: ${value}`);
    }
    overrides.mode = value;
  }

  const bgOverrides: Partial<BGControllerConfig> = {};
  const parseFloatFlag = (flag: string): number | undefined => {
    const raw = argv.find((arg) => arg.startsWith(flag));
    if (!raw) return undefined;
    const [, valueStr] = raw.split('=');
    const parsed = parseFloat(valueStr);
    if (!Number.isFinite(parsed)) {
      throw new Error(`[CLI] ${flag} must be numeric. Received: ${valueStr}`);
    }
    return parsed;
  };
  const parseIntFlag = (flag: string): number | undefined => {
    const raw = argv.find((arg) => arg.startsWith(flag));
    if (!raw) return undefined;
    const [, valueStr] = raw.split('=');
    const parsed = parseInt(valueStr, 10);
    if (!Number.isFinite(parsed)) {
      throw new Error(`[CLI] ${flag} must be numeric. Received: ${valueStr}`);
    }
    return parsed;
  };
  const parseBoolFlag = (flag: string): boolean | undefined => {
    const raw = argv.find((arg) => arg.startsWith(flag));
    if (!raw) return undefined;
    const idx = raw.indexOf('=');
    const value = idx >= 0 ? raw.slice(idx + 1) : undefined;
    if (value === undefined) return true;
    return !(value === '0' || value.toLowerCase() === 'false');
  };

  const actionsArg = argv.find((arg) => arg.startsWith('--bg.actions='));
  if (actionsArg) {
    const actions = actionsArg
      .slice(actionsArg.indexOf('=') + 1)
      .split(',')
      .map((v) => v.trim())
      .filter((v) => v.length > 0) as BGControllerAction[];
    actions.forEach((a) => {
      if (!allowedActions.includes(a)) {
        throw new Error(`[CLI] --bg.actions contains invalid action: ${a}`);
      }
    });
    bgOverrides.actions = actions;
  }

  const epsilon = parseFloatFlag('--bg.epsilon');
  if (epsilon !== undefined) bgOverrides.epsilon = epsilon;
  const temperature = parseFloatFlag('--bg.temperature');
  if (temperature !== undefined) {
    if (temperature <= 0) {
      throw new Error('[CLI] --bg.temperature must be a positive number.');
    }
    bgOverrides.temperature = temperature;
  }
  const eta = parseFloatFlag('--bg.eta');
  if (eta !== undefined) bgOverrides.eta = eta;
  const hysteresis = parseFloatFlag('--bg.hysteresis');
  if (hysteresis !== undefined) bgOverrides.hysteresis = hysteresis;
  const minDwell = parseIntFlag('--bg.minDwell');
  if (minDwell !== undefined) bgOverrides.minDwell = minDwell;
  const sampleActions = parseBoolFlag('--bg.sampleActions');
  if (sampleActions !== undefined) bgOverrides.sampleActions = sampleActions;
  const waitStepsArg = argv.find((arg) => arg.startsWith('--bg.waitSteps='));
  if (waitStepsArg) {
    const parsed = parseInt(waitStepsArg.slice(waitStepsArg.indexOf('=') + 1), 10);
    if (!Number.isFinite(parsed)) {
      throw new Error(`[CLI] --bg.waitSteps must be numeric. Received: ${waitStepsArg}`);
    }
    bgOverrides.waitSteps = parsed;
  }
  const enforceOrder = parseBoolFlag('--bg.enforceOrder');
  if (enforceOrder !== undefined) bgOverrides.enforceOrder = enforceOrder;
  const minPhaseDuration: Partial<Record<ControllerPhase, number>> = {};
  (['PHASE_B_COUNTING', 'PHASE_C_SUCCESSOR'] as ControllerPhase[]).forEach((phase) => {
    const arg = argv.find((value) => value.startsWith(`--bg.minPhaseDuration.${phase}=`));
    if (!arg) return;
    const parsed = parseInt(arg.slice(arg.indexOf('=') + 1), 10);
    if (!Number.isFinite(parsed)) {
      throw new Error(`[CLI] --bg.minPhaseDuration.${phase} must be numeric. Received: ${arg}`);
    }
    minPhaseDuration[phase] = parsed;
  });
  if (Object.keys(minPhaseDuration).length > 0) {
    bgOverrides.minPhaseDuration = minPhaseDuration;
  }
  const rewardCorrect = parseFloatFlag('--bg.reward.correct');
  const rewardWrong = parseFloatFlag('--bg.reward.wrong');
  const rewardAbort = parseFloatFlag('--bg.reward.abort');
  if (rewardCorrect !== undefined || rewardWrong !== undefined || rewardAbort !== undefined) {
    const reward: { correct?: number; wrong?: number; abort?: number } = bgOverrides.reward
      ? { ...bgOverrides.reward }
      : {};
    if (rewardCorrect !== undefined) reward.correct = rewardCorrect;
    if (rewardWrong !== undefined) reward.wrong = rewardWrong;
    if (rewardAbort !== undefined) reward.abort = rewardAbort;
    bgOverrides.reward = reward as BGControllerConfig['reward'];
  }

  if (Object.keys(bgOverrides).length > 0) {
    overrides.bg = bgOverrides as ControllerConfig['bg'];
  }

  return Object.keys(overrides).length > 0 ? overrides : undefined;
}

function parseTransitionDebugOptions(argv: string[]): TransitionDebugConfig | undefined {
  const hasDebugArgs = argv.some((arg) => arg.startsWith('--debug.'));
  if (!hasDebugArgs) return undefined;

  const parseBoolFlag = (flag: string, defaultValue: boolean): boolean => {
    const flagArg = argv.find((arg) => arg.startsWith(flag));
    if (!flagArg) return defaultValue;
    const eqIdx = flagArg.indexOf('=');
    const valueStr = eqIdx >= 0 ? flagArg.slice(eqIdx + 1) : undefined;
    if (valueStr === undefined) return true;
    return !(valueStr === '0' || valueStr.toLowerCase() === 'false');
  };
  const parseStringFlag = (flag: string): string | undefined => {
    const flagArg = argv.find((arg) => arg.startsWith(flag));
    if (!flagArg) return undefined;
    const eqIdx = flagArg.indexOf('=');
    return eqIdx >= 0 ? flagArg.slice(eqIdx + 1) : undefined;
  };
  const parseIntFlag = (flag: string, defaultValue: number): number => {
    const raw = parseStringFlag(flag);
    if (raw === undefined) return defaultValue;
    const parsed = parseInt(raw, 10);
    if (!Number.isFinite(parsed)) {
      throw new Error(`[CLI] ${flag} must be an integer. Received: ${raw}`);
    }
    return parsed;
  };
  const parseFloatFlag = (flag: string, defaultValue: number): number => {
    const raw = parseStringFlag(flag);
    if (raw === undefined) return defaultValue;
    const parsed = parseFloat(raw);
    if (!Number.isFinite(parsed)) {
      throw new Error(`[CLI] ${flag} must be numeric. Received: ${raw}`);
    }
    return parsed;
  };

  const ablateNext = parseBoolFlag('--debug.ablateNext', false);
  const ablateRec = parseBoolFlag('--debug.ablateRec', false);
  const ablateInhib = parseBoolFlag('--debug.ablateInhib', false);
  const noNoise = parseBoolFlag('--debug.noNoise', false);
  const transitionTrace = parseBoolFlag('--debug.transitionTrace', false);
  const perturbEnabled = parseBoolFlag('--debug.perturb.enabled', false);
  const perturbKind = (parseStringFlag('--debug.perturb.kind=') ?? 'noise') as
    | 'noise'
    | 'dropout'
    | 'shift';
  if (!['noise', 'dropout', 'shift'].includes(perturbKind)) {
    throw new Error(`[CLI] --debug.perturb.kind must be noise|dropout|shift. Received: ${perturbKind}`);
  }
  const perturbAtStep = parseIntFlag('--debug.perturb.atStep=', 25);
  const perturbDurationSteps = parseIntFlag('--debug.perturb.durationSteps=', 1);
  const perturbNoiseSigma = parseFloatFlag('--debug.perturb.noiseSigma=', 0.03);
  const perturbDropoutP = parseFloatFlag('--debug.perturb.dropoutP=', 0.1);
  const perturbShiftDelta = parseIntFlag('--debug.perturb.shiftDelta=', 1);
  const perturbRecoveryThreshold = parseFloatFlag('--debug.perturb.recoveryThreshold=', 0.85);
  const perturbMaxRecoverySteps = parseIntFlag('--debug.perturb.maxRecoverySteps=', 120);
  const perturbOutDir = parseStringFlag('--debug.perturb.outDir=');
  if (perturbAtStep < 0) {
    throw new Error(`[CLI] --debug.perturb.atStep must be >= 0. Received: ${perturbAtStep}`);
  }
  if (perturbDurationSteps < 1) {
    throw new Error(`[CLI] --debug.perturb.durationSteps must be >= 1. Received: ${perturbDurationSteps}`);
  }
  if (perturbNoiseSigma < 0) {
    throw new Error(`[CLI] --debug.perturb.noiseSigma must be >= 0. Received: ${perturbNoiseSigma}`);
  }
  if (perturbDropoutP < 0 || perturbDropoutP > 1) {
    throw new Error(`[CLI] --debug.perturb.dropoutP must be between 0 and 1. Received: ${perturbDropoutP}`);
  }
  if (perturbShiftDelta < 0) {
    throw new Error(`[CLI] --debug.perturb.shiftDelta must be >= 0. Received: ${perturbShiftDelta}`);
  }
  if (perturbRecoveryThreshold < -1 || perturbRecoveryThreshold > 1) {
    throw new Error(
      `[CLI] --debug.perturb.recoveryThreshold must be between -1 and 1. Received: ${perturbRecoveryThreshold}`,
    );
  }
  if (perturbMaxRecoverySteps < 1) {
    throw new Error(
      `[CLI] --debug.perturb.maxRecoverySteps must be >= 1. Received: ${perturbMaxRecoverySteps}`,
    );
  }

  const traceTrialsArg = argv.find((arg) => arg.startsWith('--debug.traceTrials='));
  const traceTrials = traceTrialsArg ? parseInt(traceTrialsArg.slice(traceTrialsArg.indexOf('=') + 1), 10) : 100;
  if (!Number.isFinite(traceTrials) || !Number.isInteger(traceTrials) || traceTrials <= 0) {
    throw new Error(`[CLI] --debug.traceTrials must be a positive integer. Received: ${traceTrialsArg}`);
  }

  const windowsArg =
    argv.find((arg) => arg.startsWith('--debug.windowDefs=')) ??
    argv.find((arg) => arg.startsWith('--debug.windows='));
  const requestedWindows = windowsArg
    ? windowsArg
        .slice(windowsArg.indexOf('=') + 1)
        .split(',')
        .map((w) => w.trim())
        .filter((w) => w.length > 0)
    : DEFAULT_DEBUG_WINDOWS;
  const validWindows: TransitionWindowName[] = DEFAULT_DEBUG_WINDOWS;
  const windowPattern = /(lateNoImpulse|tailNoImpulse|meanNoImpulse)\(k=\d+\)/;
  requestedWindows.forEach((w) => {
    if (!validWindows.includes(w as TransitionWindowName) && !windowPattern.test(w)) {
      throw new Error(`[CLI] Invalid debug window: ${w}`);
    }
  });
  const windows = (requestedWindows.length > 0 ? requestedWindows : DEFAULT_DEBUG_WINDOWS) as TransitionWindowName[];
  const uniqueWindows = Array.from(new Set(windows)) as TransitionWindowName[];

  const traceOutDirArg = argv.find((arg) => arg.startsWith('--debug.traceOutDir='));
  const hasWindowArg = Boolean(windowsArg);
  const excludeFirstArg = argv.find((arg) => arg.startsWith('--debug.excludeFirst='));
  const hasExcludeFirstArg = Boolean(excludeFirstArg);
  const hasTransitionCurrentsArg = argv.some((arg) => arg.startsWith('--debug.transitionCurrents'));
  const debugActive =
    transitionTrace ||
    hasWindowArg ||
    hasExcludeFirstArg ||
    Boolean(traceTrialsArg) ||
    Boolean(traceOutDirArg) ||
    perturbEnabled ||
    ablateNext ||
    ablateRec ||
    ablateInhib ||
    noNoise ||
    hasTransitionCurrentsArg;
  const transitionCurrents = parseBoolFlag('--debug.transitionCurrents', debugActive);
  if (!debugActive) return undefined;

  const traceOutDir = traceOutDirArg
    ? traceOutDirArg.slice(traceOutDirArg.indexOf('=') + 1)
    : 'transition_traces';

  const excludeFirst = excludeFirstArg
    ? excludeFirstArg
        .slice(excludeFirstArg.indexOf('=') + 1)
        .split(',')
        .map((v) => parseInt(v.trim(), 10))
        .filter((v) => Number.isFinite(v) && Number.isInteger(v) && v >= 0)
    : [0, 1, 2];

  return {
    transitionTrace,
    traceTrials,
    windows: uniqueWindows,
    windowDefs: uniqueWindows,
    excludeFirst,
    transitionCurrents,
    traceOutDir,
    perturb: perturbEnabled
      ? {
          enabled: true,
          kind: perturbKind,
          atStep: perturbAtStep,
          durationSteps: perturbDurationSteps,
          noiseSigma: perturbNoiseSigma,
          dropoutP: perturbDropoutP,
          shiftDelta: perturbShiftDelta,
          recoveryThreshold: perturbRecoveryThreshold,
          maxRecoverySteps: perturbMaxRecoverySteps,
          outDir: perturbOutDir,
        }
      : undefined,
    ablateNext,
    ablateRec,
    ablateInhib,
    noNoise,
  };
}

function parseTransitionOverrides(argv: string[], prefix: 'phaseBC' | Topology): TransitionOverrides {
  const tTransArg = argv.find((arg) => arg.startsWith(`--${prefix}.ttrans=`));
  const tailLenArg = argv.find((arg) => arg.startsWith(`--${prefix}.taillen=`));

  const parseIntArg = (arg: string, label: string): number => {
    const [, valueStr] = arg.split('=');
    const value = Number(valueStr);
    if (!Number.isFinite(value) || !Number.isInteger(value)) {
      throw new Error(`[CLI] ${label} must be an integer. Received: ${valueStr}`);
    }
    return value;
  };

  const overrides: TransitionOverrides = {};

  if (tTransArg) {
    const parsed = parseIntArg(tTransArg, `--${prefix}.ttrans`);
    if (parsed < 5) {
      throw new Error(`[CLI] --${prefix}.ttrans must be >= 5. Received: ${parsed}`);
    }
    overrides.tTrans = parsed;
  }

  if (tailLenArg) {
    const parsed = parseIntArg(tailLenArg, `--${prefix}.taillen`);
    if (parsed < 1) {
      throw new Error(`[CLI] --${prefix}.taillen must be >= 1. Received: ${parsed}`);
    }
    overrides.tailLen = parsed;
  }

  return overrides;
}

function mergeTransitionOverrides(
  shared: TransitionOverrides,
  topologySpecific: TransitionOverrides,
  topology: Topology,
): TransitionOverrides {
  const merged: TransitionOverrides = { ...shared, ...topologySpecific };
  const resolvedTTrans = merged.tTrans ?? DEFAULT_T_TRANS;
  if (merged.tailLen !== undefined && merged.tailLen > resolvedTTrans) {
    const sourceFlag = topologySpecific.tailLen !== undefined ? `--${topology}.taillen` : '--phaseBC.taillen';
    const tTransFlag = topologySpecific.tTrans !== undefined ? `--${topology}.ttrans` : '--phaseBC.ttrans';
    throw new Error(
      `[CLI] ${sourceFlag} (${merged.tailLen}) must be <= ${tTransFlag} (${resolvedTTrans}) for ${topology} topology.`,
    );
  }

  return merged;
}

function parsePhaseBCConfigOverrides(argv: string[]): Partial<PhaseBCConfig> | undefined {
  const getString = (prefix: string): string | undefined => {
    const raw = argv.find((arg) => arg.startsWith(prefix));
    if (!raw) return undefined;
    const idx = raw.indexOf('=');
    return idx >= 0 ? raw.slice(idx + 1) : undefined;
  };
  const parseIntArg = (prefix: string): number | undefined => {
    const value = getString(prefix);
    if (value === undefined) return undefined;
    const parsed = parseInt(value, 10);
    if (!Number.isFinite(parsed)) {
      throw new Error(`[CLI] ${prefix} must be numeric. Received: ${value}`);
    }
    return parsed;
  };
  const parseFloatArg = (prefix: string): number | undefined => {
    const value = getString(prefix);
    if (value === undefined) return undefined;
    const parsed = parseFloat(value);
    if (!Number.isFinite(parsed)) {
      throw new Error(`[CLI] ${prefix} must be numeric. Received: ${value}`);
    }
    return parsed;
  };
  const parseBoolFlag = (flag: string): boolean | undefined => {
    const raw = argv.find((arg) => arg.startsWith(flag));
    if (!raw) return undefined;
    const idx = raw.indexOf('=');
    const value = idx >= 0 ? raw.slice(idx + 1) : undefined;
    if (value === undefined) return true;
    return !(value === '0' || value.toLowerCase() === 'false');
  };

  const overrides: Partial<PhaseBCConfig> = { strictSustainPresetApplied: false };
  const strictSustain = argv.includes('--phaseBC.strictSustain');
  const evalWindow = getString('--phaseBC.evalWindow=');
  const learnWindow = getString('--phaseBC.learnWindow=');
  const settleWindow = getString('--phaseBC.settleWindow=');
  const excludeFirst = parseIntArg('--phaseBC.excludeFirstK=');
  const silentThreshold = parseFloatArg('--phaseBC.silentSpikeThreshold=');
  const logAbortLimit = parseIntArg('--phaseBC.logAbortLimit=');

  if (strictSustain) {
    overrides.strictSustainPresetApplied = true;
    overrides.evalWindow = 'meanNoImpulse(k=1)';
    overrides.learnWindow = 'meanNoImpulse(k=1)';
    overrides.sustainGate = {
      ...DEFAULT_PHASE_BC_CONFIG.sustainGate,
      enabled: true,
      skipUpdatesOnFail: true,
      skipEpisodeOnFail: true,
    };
  }

  if (evalWindow) overrides.evalWindow = evalWindow as TransitionWindowName;
  if (learnWindow) overrides.learnWindow = learnWindow as TransitionWindowName;
  if (settleWindow) overrides.settleWindow = settleWindow as TransitionWindowName;
  if (excludeFirst !== undefined) overrides.excludeFirstK = excludeFirst;
  if (silentThreshold !== undefined) overrides.silentSpikeThreshold = silentThreshold;

  const gateEnabled = parseBoolFlag('--phaseBC.sustainGate');
  const maxTailSilentFrac = parseFloatArg('--phaseBC.maxTailSilentFrac=');
  const minTimeToSilence = parseFloatArg('--phaseBC.minTimeToSilence=');
  const skipUpdates = parseBoolFlag('--phaseBC.skipUpdatesOnFail');
  const skipEpisode = parseBoolFlag('--phaseBC.skipEpisodeOnFail');
  const abortAfterTrialsNested = parseIntArg('--phaseBC.sustainGate.abortAfterTrials=');
  const abortAfterTrialsFlat = parseIntArg('--phaseBC.abortAfterTrials=');
  const abortAfterTrials = abortAfterTrialsFlat ?? abortAfterTrialsNested;
  if (
    gateEnabled !== undefined ||
    maxTailSilentFrac !== undefined ||
    minTimeToSilence !== undefined ||
    skipUpdates !== undefined ||
    skipEpisode !== undefined ||
    abortAfterTrials !== undefined
  ) {
    overrides.sustainGate = { ...(overrides.sustainGate ?? DEFAULT_PHASE_BC_CONFIG.sustainGate) };
    if (gateEnabled !== undefined) overrides.sustainGate.enabled = gateEnabled;
    if (maxTailSilentFrac !== undefined) overrides.sustainGate.maxTailSilentFrac = maxTailSilentFrac;
    if (minTimeToSilence !== undefined) overrides.sustainGate.minTimeToSilence = minTimeToSilence;
    if (skipUpdates !== undefined) overrides.sustainGate.skipUpdatesOnFail = skipUpdates;
    if (skipEpisode !== undefined) overrides.sustainGate.skipEpisodeOnFail = skipEpisode;
    if (abortAfterTrials !== undefined) overrides.sustainGate.abortAfterTrials = abortAfterTrials;
  }
  if (logAbortLimit !== undefined) overrides.logAbortLimit = logAbortLimit;

  const sustainWeight = parseFloatArg('--tune.sustainWeight=');
  const useSustainFitness = parseBoolFlag('--tune.useSustainFitness');
  const sustainMetricWindow = getString('--tune.sustainMetricWindow=');
  if (sustainWeight !== undefined || useSustainFitness !== undefined || sustainMetricWindow) {
    overrides.tuning = { ...DEFAULT_PHASE_BC_CONFIG.tuning };
    if (sustainWeight !== undefined) overrides.tuning.sustainWeight = sustainWeight;
    if (useSustainFitness !== undefined) overrides.tuning.useSustainFitness = useSustainFitness;
    if (sustainMetricWindow === 'late' || sustainMetricWindow === 'tail') {
      overrides.tuning.sustainMetricWindow = sustainMetricWindow;
    }
  }

  return Object.keys(overrides).length > 0 ? overrides : undefined;
}

function parsePhaseBCPhysicsFlags(
  argv: string[],
): { physicsSource: PhaseBCPhysicsSource; physicsFromPath?: string } {
  const allowedSources: PhaseBCPhysicsSource[] = ['default', 'phaseB_best', 'probe_best', 'megatune_best', 'none'];
  const sourceArg = argv.find((arg) => arg.startsWith('--phaseBC.physicsSource='));
  const physicsFromArg = argv.find((arg) => arg.startsWith('--phaseBC.physicsFrom='));

  let physicsSource: PhaseBCPhysicsSource = 'default';
  if (sourceArg) {
    const value = sourceArg.slice(sourceArg.indexOf('=') + 1) as PhaseBCPhysicsSource;
    if (!allowedSources.includes(value)) {
      throw new Error(
        `[CLI] --phaseBC.physicsSource must be one of ${allowedSources.join(', ')}. Received: ${value}`,
      );
    }
    physicsSource = value;
  }

  const physicsFromPath = physicsFromArg ? physicsFromArg.slice(physicsFromArg.indexOf('=') + 1) : undefined;
  return { physicsSource, physicsFromPath };
}

function buildPhaseBCTuningConfig(base?: Partial<PhaseBCConfig>): PhaseBCConfig {
  const overrides = parsePhaseBCConfigOverrides(process.argv);
  const config: PhaseBCConfig = {
    ...DEFAULT_PHASE_BC_CONFIG,
    ...base,
    sustainGate: { ...DEFAULT_PHASE_BC_CONFIG.sustainGate },
    tuning: { ...DEFAULT_PHASE_BC_CONFIG.tuning },
  };

  if (base?.sustainGate) config.sustainGate = { ...config.sustainGate, ...base.sustainGate };
  if (base?.tuning) config.tuning = { ...config.tuning, ...base.tuning };

  if (!overrides) return config;

  if (overrides.strictSustainPresetApplied !== undefined) {
    config.strictSustainPresetApplied = overrides.strictSustainPresetApplied;
  }
  if (overrides.settleWindow) config.settleWindow = overrides.settleWindow;
  if (overrides.excludeFirstK !== undefined) config.excludeFirstK = overrides.excludeFirstK;
  if (overrides.evalWindow) config.evalWindow = overrides.evalWindow;
  if (overrides.learnWindow) config.learnWindow = overrides.learnWindow;
  if (overrides.silentSpikeThreshold !== undefined) config.silentSpikeThreshold = overrides.silentSpikeThreshold;
  if (overrides.logAbortLimit !== undefined) config.logAbortLimit = overrides.logAbortLimit;
  if (overrides.sustainGate) config.sustainGate = { ...config.sustainGate, ...overrides.sustainGate };
  if (overrides.tuning) config.tuning = { ...config.tuning, ...overrides.tuning };

  return config;
}

function runAll(): void {
  const seedArg = process.argv.find((arg) => arg.startsWith('--seed='));
  const baseSeed = (() => {
    if (!seedArg) return 42;
    const [, seedValue] = seedArg.split('=');
    const parsed = parseInt(seedValue, 10);
    if (!Number.isFinite(parsed)) {
      throw new Error(`[CLI] --seed must be numeric. Received: ${seedValue}`);
    }
    return parsed;
  })();
  const includeAccHistory = !process.argv.includes('--no-acc-history');
  const protoDebugEnabled = process.argv.includes('--proto-debug');
  const protoDebugLimitArg = process.argv.find((arg) => arg.startsWith('--proto-debug-limit='));
  const protoDebugLimit = protoDebugLimitArg ? parseInt(protoDebugLimitArg.split('=')[1], 10) : 30;
  const protoDebugConfig = protoDebugEnabled
    ? { enabled: true, limit: Number.isFinite(protoDebugLimit) ? protoDebugLimit : 30 }
    : undefined;
  const transitionDebug = parseTransitionDebugOptions(process.argv);
  const parseBoolFlag = (flag: string, defaultValue: boolean): boolean => {
    const raw = process.argv.find((arg) => arg.startsWith(flag));
    if (!raw) return defaultValue;
    const idx = raw.indexOf('=');
    const value = idx >= 0 ? raw.slice(idx + 1) : undefined;
    if (value === undefined) return true;
    return !(value === '0' || value.toLowerCase() === 'false');
  };
  const parseStringFlag = (flag: string): string | undefined => {
    const raw = process.argv.find((arg) => arg.startsWith(flag));
    if (!raw) return undefined;
    const idx = raw.indexOf('=');
    return idx >= 0 ? raw.slice(idx + 1) : undefined;
  };
  const parseIntFlag = (flag: string, defaultValue: number): number => {
    const raw = parseStringFlag(flag);
    if (raw === undefined) return defaultValue;
    const parsed = parseInt(raw, 10);
    if (!Number.isFinite(parsed) || !Number.isInteger(parsed)) {
      throw new Error(`[CLI] ${flag} must be an integer. Received: ${raw}`);
    }
    return parsed;
  };
  const parseFloatFlag = (flag: string, defaultValue: number): number => {
    const raw = parseStringFlag(flag);
    if (raw === undefined) return defaultValue;
    const parsed = parseFloat(raw);
    if (!Number.isFinite(parsed)) {
      throw new Error(`[CLI] ${flag} must be numeric. Received: ${raw}`);
    }
    return parsed;
  };
  const parseResetStrategy = (
    flag: string,
    defaultValue: { mode: 'uniform' | 'cycle' | 'fixed'; digit?: number },
    opts: { allowCycle: boolean },
  ): { mode: 'uniform' | 'cycle' | 'fixed'; digit?: number } => {
    const raw = parseStringFlag(flag);
    if (!raw) return defaultValue;
    if (raw === 'uniform') return { mode: 'uniform' };
    if (raw === 'cycle') {
      if (!opts.allowCycle) {
        throw new Error(`[CLI] ${flag} must be "uniform" or "fixed:<digit>". Received: ${raw}`);
      }
      return { mode: 'cycle' };
    }
    if (raw.startsWith('fixed:')) {
      const digitRaw = raw.slice('fixed:'.length);
      const digit = parseInt(digitRaw, 10);
      if (!Number.isFinite(digit) || digit < 0 || digit > 9) {
        throw new Error(`[CLI] ${flag} fixed digit must be between 0 and 9. Received: ${digitRaw}`);
      }
      return { mode: 'fixed', digit };
    }
    throw new Error(`[CLI] ${flag} must be "uniform", "cycle", or "fixed:<digit>". Received: ${raw}`);
  };
  const phaseDEnabled = parseBoolFlag('--phaseD.enabled=', false);
  const phaseEEnabled = parseBoolFlag('--phaseE.enabled=', false);
  const phaseFEnabled = parseBoolFlag('--phaseF.enabled=', false);
  const phaseEOpSchedule = parseStringFlag('--phaseE.opSchedule=') ?? 'uniform';
  if (phaseEOpSchedule !== 'uniform' && phaseEOpSchedule !== 'alternate') {
    throw new Error(
      `[CLI] --phaseE.opSchedule must be "uniform" or "alternate". Received: ${phaseEOpSchedule}`,
    );
  }
  const phaseEInitFrom = parseStringFlag('--phaseE.initFrom=') ?? 'auto';
  if (!['auto', 'phaseA', 'phaseB'].includes(phaseEInitFrom)) {
    throw new Error(
      `[CLI] --phaseE.initFrom must be "auto", "phaseA", or "phaseB". Received: ${phaseEInitFrom}`,
    );
  }
  const phaseETransitionSource = parseStringFlag('--phaseE.transitionSource=') ?? 'pairs';
  if (!['pairs', 'randomWalk'].includes(phaseETransitionSource)) {
    throw new Error(
      `[CLI] --phaseE.transitionSource must be "pairs" or "randomWalk". Received: ${phaseETransitionSource}`,
    );
  }
  const phaseEWalkEpisodeLen = parseIntFlag('--phaseE.walk.episodeLen=', 20);
  const phaseEWalkEpisodesPerBatch = parseIntFlag('--phaseE.walk.episodesPerBatch=', 50);
  const phaseEWalkBias = parseFloatFlag('--phaseE.walk.bias=', 0.5);
  if (phaseEWalkBias < 0 || phaseEWalkBias > 1) {
    throw new Error(`[CLI] --phaseE.walk.bias must be between 0 and 1. Received: ${phaseEWalkBias}`);
  }
  const phaseEWalkResetStrategy = parseResetStrategy(
    '--phaseE.walk.resetStrategy=',
    { mode: 'uniform' },
    { allowCycle: true },
  );
  if (phaseEWalkEpisodeLen <= 0 || phaseEWalkEpisodesPerBatch <= 0) {
    throw new Error('[CLI] Phase E walk settings must be positive integers.');
  }
  const phaseFSplitByOp = parseBoolFlag('--phaseF.splitByOp=', phaseFEnabled);
  const phaseFIncludeBoundaryMetrics = parseBoolFlag('--phaseF.includeBoundaryMetrics=', phaseFEnabled);
  const phaseFEvalTransitionSource = parseStringFlag('--phaseF.evalTransitionSource=') ?? 'pairs';
  if (!['pairs', 'randomWalk'].includes(phaseFEvalTransitionSource)) {
    throw new Error(
      `[CLI] --phaseF.evalTransitionSource must be "pairs" or "randomWalk". Received: ${phaseFEvalTransitionSource}`,
    );
  }
  const phaseFWalkEpisodeLen = parseIntFlag('--phaseF.walk.episodeLen=', 40);
  const phaseFWalkEpisodes = parseIntFlag('--phaseF.walk.episodes=', 100);
  const phaseFWalkBias = parseFloatFlag('--phaseF.walk.bias=', 0.5);
  if (phaseFWalkBias < 0 || phaseFWalkBias > 1) {
    throw new Error(`[CLI] --phaseF.walk.bias must be between 0 and 1. Received: ${phaseFWalkBias}`);
  }
  const phaseFWalkResetStrategy = parseResetStrategy(
    '--phaseF.walk.resetStrategy=',
    { mode: 'uniform' },
    { allowCycle: false },
  );
  if (phaseFWalkEpisodeLen <= 0 || phaseFWalkEpisodes <= 0) {
    throw new Error('[CLI] Phase F walk settings must be positive integers.');
  }
  const sharedTransitionOverrides = parseTransitionOverrides(process.argv, 'phaseBC');
  const snakeTransitionOverrides = parseTransitionOverrides(process.argv, 'snake');
  const ringTransitionOverrides = parseTransitionOverrides(process.argv, 'ring');
  const phaseBCOverrides = parsePhaseBCConfigOverrides(process.argv);
  const activityOverrides = parseActivityOverrides(process.argv);
  const controllerOverrides = parseControllerOverrides(process.argv);
  const phaseBCPhysics = parsePhaseBCPhysicsFlags(process.argv);
  const hasSharedTransitionOverrides =
    sharedTransitionOverrides.tTrans !== undefined || sharedTransitionOverrides.tailLen !== undefined;
  const topologyTransitionOverrides: Record<Topology, TransitionOverrides> = {
    snake: snakeTransitionOverrides,
    ring: ringTransitionOverrides,
  };

  const buildConfig = (topology: Topology, useCountingPhase: boolean): ExperimentConfig => {
    const curriculumOverrides: { phaseA?: Partial<PhasePhysicsParams>; phaseBC?: Partial<PhasePhysicsParams> } = {};
    if (activityOverrides?.phaseA) curriculumOverrides.phaseA = { ...activityOverrides.phaseA };
    if (activityOverrides?.phaseBC) curriculumOverrides.phaseBC = { ...activityOverrides.phaseBC };
    const mergedTransitionOverrides = mergeTransitionOverrides(
      sharedTransitionOverrides,
      topologyTransitionOverrides[topology],
      topology,
    );
    const hasTransitionOverrides =
      hasSharedTransitionOverrides ||
      topologyTransitionOverrides[topology].tTrans !== undefined ||
      topologyTransitionOverrides[topology].tailLen !== undefined;
    if (hasTransitionOverrides) {
      curriculumOverrides.phaseBC = { ...curriculumOverrides.phaseBC, ...mergedTransitionOverrides };
    }

    const overrides: { curriculum?: { phaseA?: Partial<PhasePhysicsParams>; phaseBC?: Partial<PhasePhysicsParams> }; controller?: Partial<ControllerConfig> } = {};
    if (Object.keys(curriculumOverrides).length > 0) overrides.curriculum = curriculumOverrides;
    if (controllerOverrides) overrides.controller = controllerOverrides;
    const resolvedOverrides = Object.keys(overrides).length > 0 ? overrides : undefined;

    return resolveExperimentConfig({
      topology,
      useCountingPhase,
      seed: baseSeed,
      includeAccHistory,
      overrides: resolvedOverrides,
      phaseBCConfigOverrides: phaseBCOverrides,
      phaseBCPhysicsSource: phaseBCPhysics.physicsSource,
      phaseBCPhysicsFromPath: phaseBCPhysics.physicsFromPath,
    });
  };

  const configs: ExperimentConfig[] = [
    buildConfig('snake', false),
    buildConfig('snake', true),
    buildConfig('ring', false),
    buildConfig('ring', true),
  ].map((cfg) => ({
    ...cfg,
    phaseD: { enabled: phaseDEnabled },
    phaseE: {
      enabled: phaseEEnabled,
      opSchedule: phaseEOpSchedule as 'alternate' | 'uniform',
      initFrom: phaseEInitFrom as 'auto' | 'phaseA' | 'phaseB',
      transitionSource: phaseETransitionSource as 'pairs' | 'randomWalk',
      walk: {
        episodeLen: phaseEWalkEpisodeLen,
        episodesPerBatch: phaseEWalkEpisodesPerBatch,
        resetStrategy: phaseEWalkResetStrategy,
        bias: phaseEWalkBias,
      },
    },
    phaseF: {
      enabled: phaseFEnabled,
      splitByOp: phaseFSplitByOp,
      includeBoundaryMetrics: phaseFIncludeBoundaryMetrics,
      evalTransitionSource: phaseFEvalTransitionSource as 'pairs' | 'randomWalk',
      walk: {
        episodeLen: phaseFWalkEpisodeLen,
        episodes: phaseFWalkEpisodes,
        resetStrategy: phaseFWalkResetStrategy,
        bias: phaseFWalkBias,
      },
    },
  }));

  const results = configs.map((cfg) => {
    console.log('=== Running experiment ===');
    console.log(`Topology: ${cfg.topology}, counting phase: ${cfg.useCountingPhase}`);
    cfg.protoDebug = protoDebugConfig;
    if (transitionDebug) cfg.transitionDebug = transitionDebug;
    const runner = new ExperimentRunner(cfg);
    const res = runner.run();
    printResult(res);
    const countingLabel = cfg.useCountingPhase ? 'countingOn' : 'countingOff';
    const runFilename = `run_last_${cfg.topology}_${countingLabel}.json`;
    const runOutPath = artifactPath(runFilename);
    fs.writeFileSync(runOutPath, JSON.stringify(res, null, 2));
    exportToSharedArtifacts(runFilename);
    console.log(`[Run] Result written to ${runOutPath}`);
    return res;
  });

  console.log('=== Final Summary JSON ===');
  const summaryFilename = `experiment_results_${Date.now()}.json`;
  const summaryPath = artifactPath(summaryFilename);
  fs.writeFileSync(summaryPath, JSON.stringify(results, null, 2));
  console.log(`Experimental results outputted to ${summaryFilename}`);
}

function parseProbeConfig(): {
  trials: number;
  maxCandidates: number;
  rankTop: number;
  wSustain: number;
  wAcc: number;
  seed: number;
} {
  const getInt = (flag: string, defaultValue: number): number => {
    const raw = process.argv.find((arg) => arg.startsWith(flag));
    if (!raw) return defaultValue;
    const [, valueStr] = raw.split('=');
    const parsed = parseInt(valueStr, 10);
    if (!Number.isFinite(parsed)) {
      throw new Error(`[CLI] ${flag} must be numeric. Received: ${valueStr}`);
    }
    return parsed;
  };

  const getFloat = (flag: string, defaultValue: number): number => {
    const raw = process.argv.find((arg) => arg.startsWith(flag));
    if (!raw) return defaultValue;
    const [, valueStr] = raw.split('=');
    const parsed = parseFloat(valueStr);
    if (!Number.isFinite(parsed)) {
      throw new Error(`[CLI] ${flag} must be numeric. Received: ${valueStr}`);
    }
    return parsed;
  };

  return {
    trials: getInt('--probe.trials', 300),
    maxCandidates: getInt('--probe.maxCandidates', 50),
    rankTop: getInt('--probe.rankTop', 10),
    wSustain: getFloat('--probe.wSustain', 5),
    wAcc: getFloat('--probe.wAcc', 1),
    seed: getInt('--probe.seed', 42),
  };
}

function runPhaseBCProbeEntry(): void {
  const probeConfig = parseProbeConfig();
  const includeAccHistory = !process.argv.includes('--no-acc-history');
  const sharedTransitionOverrides = parseTransitionOverrides(process.argv, 'phaseBC');
  const snakeTransitionOverrides = parseTransitionOverrides(process.argv, 'snake');
  const ringTransitionOverrides = parseTransitionOverrides(process.argv, 'ring');
  const phaseBCOverrides = parsePhaseBCConfigOverrides(process.argv);
  const activityOverrides = parseActivityOverrides(process.argv);
  const controllerOverrides = parseControllerOverrides(process.argv);
  const phaseBCPhysics = parsePhaseBCPhysicsFlags(process.argv);
  const hasSharedTransitionOverrides =
    sharedTransitionOverrides.tTrans !== undefined || sharedTransitionOverrides.tailLen !== undefined;
  const topologyTransitionOverrides: Record<Topology, TransitionOverrides> = {
    snake: snakeTransitionOverrides,
    ring: ringTransitionOverrides,
  };
  const buildProbeConfig = (topology: Topology): ExperimentConfig =>
    resolveExperimentConfig({
      topology,
      useCountingPhase: true,
      seed: probeConfig.seed,
      includeAccHistory,
      overrides: (() => {
        const curriculumOverrides: { phaseA?: Partial<PhasePhysicsParams>; phaseBC?: Partial<PhasePhysicsParams> } = {};
        if (activityOverrides?.phaseA) curriculumOverrides.phaseA = { ...activityOverrides.phaseA };
        if (activityOverrides?.phaseBC) curriculumOverrides.phaseBC = { ...activityOverrides.phaseBC };
        const mergedTransitionOverrides = mergeTransitionOverrides(
          sharedTransitionOverrides,
          topologyTransitionOverrides[topology],
          topology,
        );
        const hasTransitionOverrides =
          hasSharedTransitionOverrides ||
          topologyTransitionOverrides[topology].tTrans !== undefined ||
          topologyTransitionOverrides[topology].tailLen !== undefined;
        if (hasTransitionOverrides) {
          curriculumOverrides.phaseBC = { ...curriculumOverrides.phaseBC, ...mergedTransitionOverrides };
        }
        const overrides: { curriculum?: { phaseA?: Partial<PhasePhysicsParams>; phaseBC?: Partial<PhasePhysicsParams> }; controller?: Partial<ControllerConfig> } = {};
        if (Object.keys(curriculumOverrides).length > 0) overrides.curriculum = curriculumOverrides;
        if (controllerOverrides) overrides.controller = controllerOverrides;
        return Object.keys(overrides).length > 0 ? overrides : undefined;
      })(),
      phaseBCConfigOverrides: phaseBCOverrides,
      phaseBCPhysicsSource: phaseBCPhysics.physicsSource,
      phaseBCPhysicsFromPath: phaseBCPhysics.physicsFromPath,
    });

  runPhaseBCProbe({
    ...probeConfig,
    experiments: {
      snake: buildProbeConfig('snake'),
      ring: buildProbeConfig('ring'),
    },
  });
}

function runReport(): void {
  const options = parseReportOptions();
  runReportEntry(options);
}

export function runPhaseBTuningEntry(): void {
  const phaseBCConfig = buildPhaseBCTuningConfig();
  const { bestPerTopology, allResults } = runPhaseBTuning(phaseBCConfig);
  const allResultsSorted = [...allResults].sort((a, b) => {
    if (phaseBIsBetter(a, b)) return -1;
    if (phaseBIsBetter(b, a)) return 1;
    return 0;
  });

  const timestamp = new Date().toISOString();
  const safeTimestamp = timestamp.replace(/[:.]/g, '-');
  const payload = {
    metadata: {
      timestamp,
      description: 'Phase B/C grid search from Phase A snapshot',
      phaseASnapshotFiles: [artifactPath('phaseA_state_snake.json'), artifactPath('phaseA_state_ring.json')],
    },
    bestPerTopology,
    allResults: allResultsSorted,
  };

  const filename = `phaseB_tuning_results_${safeTimestamp}.json`;
  const filePath = artifactPath(filename);
  fs.writeFileSync(filePath, JSON.stringify(payload, null, 2));
  console.log(`Phase B tuning results written to ${filePath}`);

  const stableFilePath = artifactPath('phaseB_best.json');
  fs.writeFileSync(
    stableFilePath,
    JSON.stringify({ timestamp, bestPerTopology, allResults: allResultsSorted }, null, 2),
  );
  console.log(`Phase B best-per-topology written to ${stableFilePath}`);
  exportToSharedArtifacts('phaseB_best.json');
  exportToSharedArtifacts('phaseB_tuning_progress.json');
}

export function runPhaseBFinetuneEntry(): void {
  const coarseProgressPath = resolveArtifactPath({ key: 'phaseB_tuning_progress', required: true });
  if (!coarseProgressPath || !fs.existsSync(coarseProgressPath)) {
    console.error(
      `[Phase B Finetune] Missing coarse tuning progress at ${coarseProgressPath}. Run tune-phaseB first to populate it.`,
    );
    process.exit(1);
  }

  const coarseResults = JSON.parse(fs.readFileSync(coarseProgressPath, 'utf-8')) as PhaseBTuningRecord[];
  if (coarseResults.length === 0) {
    console.error('[Phase B Finetune] No coarse tuning results found. Run tune-phaseB first.');
    process.exit(1);
  }

  console.log(`[Phase B Finetune] Loaded ${coarseResults.length} coarse results from ${coarseProgressPath}`);

  const finetuneOptions: PhaseBFinetuneOptions = {
    topN: 5,
    numCountingEpisodes: 1000,
    numSuccessorTrials: 1000,
    rngSeeds: [42, 99, 123],
  };

  const phaseBCConfig = buildPhaseBCTuningConfig();
  const { bestPerTopology, allResults } = runPhaseBFinetune(coarseResults, finetuneOptions, phaseBCConfig);
  const allResultsSorted = [...allResults].sort((a, b) => {
    if (phaseBIsBetter(a, b)) return -1;
    if (phaseBIsBetter(b, a)) return 1;
    return 0;
  });

  const timestamp = new Date().toISOString();
  const safeTimestamp = timestamp.replace(/[:.]/g, '-');
  const payload = {
    metadata: {
      timestamp,
      description: 'Phase B/C finetune rerun of top coarse candidates',
      coarseProgressFile: coarseProgressPath,
      finetuneOptions,
    },
    bestPerTopology,
    allResults: allResultsSorted,
  };

  const filename = `phaseB_finetune_results_${safeTimestamp}.json`;
  const filePath = artifactPath(filename);
  fs.writeFileSync(filePath, JSON.stringify(payload, null, 2));
  console.log(`Phase B finetune results written to ${filePath}`);

  const stableFilePath = artifactPath('phaseB_finetune_best.json');
  fs.writeFileSync(
    stableFilePath,
    JSON.stringify({ timestamp, bestPerTopology, allResults: allResultsSorted }, null, 2),
  );
  console.log(`Phase B finetune best-per-topology written to ${stableFilePath}`);
  exportToSharedArtifacts('phaseB_finetune_best.json');
  exportToSharedArtifacts('phaseB_finetune_progress.json');
}

function main(): void {
  const mode = process.argv[2] ?? 'run';

  const modeHandlers: Record<string, () => void> = {
    run: runAll,
    'tune-phaseA': runPhaseATuningEntry,
    'finetune-phaseA': runPhaseAFinetuneEntry,
    'finetune:phaseA': runPhaseAFinetuneEntry,
    'tune-phaseB': runPhaseBTuningEntry,
    'tune-phaseB-finetune': runPhaseBFinetuneEntry,
    'finetune-phaseB': runPhaseBFinetuneEntry,
    'finetune:phaseB': runPhaseBFinetuneEntry,
    'probe-phaseBC': runPhaseBCProbeEntry,
    'megatune-phaseBC': runMegaTunePhaseBC,
    report: runReport,
    overnight: runOvernight,
  };

  const handler = modeHandlers[mode];
  if (!handler) {
    console.error(`Unknown mode: ${mode}`);
    console.error('Available modes:', Object.keys(modeHandlers).join(', '));
    process.exit(1);
  }

  console.log(`[Main] Running mode: ${mode}`);
  handler();
}

main();

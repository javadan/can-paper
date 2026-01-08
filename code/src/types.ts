export type Topology = 'snake' | 'ring';

export type ActivityMode = 'spike' | 'ema_spike' | 'ema_voltage';

export type Operator = 'plus' | 'minus';

export const DEFAULT_ACTIVITY_MODE: ActivityMode = 'spike';
export const DEFAULT_ACTIVITY_ALPHA = 0.1;

export const DEFAULT_T_TRANS = 40;
export const DEFAULT_TAIL_LEN = 10;
export const DEFAULT_SILENT_SPIKE_THRESHOLD = 1e-6;

export interface NetworkParams {
  N: number; // number of neurons
  sdrDim: number; // SDR dimensionality (e.g. 256)
  numDigits: number; // typically 10
  readoutDim: number; // also 10
}

export interface NeuronParams {
  v_th: number; // threshold
  v_reset: number; // reset voltage
  alpha: number; // leak factor
  k_inhib: number; // global inhibition
  noise_std: number; // Gaussian noise std
  refractory_period: number;
}

export interface LearningParams {
  eta_attr: number; // W_attr learning rate
  eta_trans: number; // W_next/W_prev (transition) learning rate
  eta_out: number; // readout learning rate
}

export interface PhasePhysicsParams {
  k_inhib: number;
  v_th: number;
  alpha: number;
  wInRowNorm: number;
  wAttrRowNorm: number;
  etaTrans: number; // maps to eta_trans
  activityMode?: ActivityMode;
  activityAlpha?: number;
  tTrans?: number;
  tailLen?: number;
}

export interface LearningRateSchedule {
  phaseA: {
    eta_attr: number;
    eta_out: number;
    eta_trans?: number;
  };
  phaseBC: {
    eta_attr: number;
    eta_out: number;
    eta_trans: number;
  };
}

export type CurriculumPhase =
  | 'PHASE_A_DIGITS'
  | 'PHASE_B_COUNTING'
  | 'PHASE_C_SUCCESSOR'
  | 'PHASE_D_PREDECESSOR'
  | 'PHASE_E_JOINT'
  | 'PHASE_F_EVAL';

export interface Curriculum {
  phaseA: PhasePhysicsParams;
  phaseBC: PhasePhysicsParams;
}

export type ProtoDebugConfig = {
  enabled: boolean;
  limit: number; // number of trials to log
};

export type ProtoDebugRow = {
  phase: 'PHASE_B_COUNTING' | 'PHASE_C_SUCCESSOR';
  trialIdx: number;
  targetDigit: number;
  readoutPredDigit: number;
  protoPredDigit: number | null;
  protoIsNull: boolean;
  tailSpikeMass: number;
  transSpikeMass: number;
};

export type ProtoDiagCounters = {
  totalTrials: number;
  protoPredNonNull: number;
  protoPredNull: number;
  tailSilent: number;
  tailNonSilent: number;
  avgTailSpikeMass: number;
  avgTransSpikeMass: number;
  protoVsReadoutDisagreeCount: number;
};

export type TransitionWindowName =
  | 'early'
  | 'mid'
  | 'late'
  | 'tail'
  | 'mean'
  | 'lateNoImpulse'
  | 'tailNoImpulse'
  | 'meanNoImpulse'
  | 'impulseOnly'
  | `lateNoImpulse(k=${number})`
  | `tailNoImpulse(k=${number})`
  | `meanNoImpulse(k=${number})`;

export type TransitionTrialMeta = {
  topology: Topology;
  counting: boolean;
  seed: number;
  digitStart: number;
  digitTarget: number;
  operator?: Operator;
  tTrans: number;
  tailLen: number;
  alpha: number;
  v_th: number;
  th_scale: number;
  k_inhib: number;
  inhib_scale: number;
  noiseStd: number;
  wAttrRowNorm: number;
  wNextRowNorm: number;
  wInRowNorm: number;
};

export type TransitionTrialStep = {
  t: number;
  spikeMass: number;
  spikeFrac: number;
  inhibValue: number;
  recMean: number;
  recMax: number;
  nextMean: number;
  nextMax: number;
  inMean: number;
  inMax: number;
  vMean: number;
  vMax: number;
  noiseMean: number;
  noiseMax: number;
  noiseStd: number;
  readoutPred: number;
  readoutConf: number;
  protoBestDigit: number | null;
  protoBestSim: number;
  protoTargetSim: number;
  protoMargin: number;
};

export type TransitionCurrentSnapshot = {
  recMean: number;
  recMax: number;
  nextMean: number;
  nextMax: number;
  inMean: number;
  inMax: number;
  inhibValue: number;
  noiseMean: number;
  noiseMax: number;
  noiseStd: number;
  vMean: number;
  vMax: number;
  spikeFrac: number;
  spikeMass: number;
};

export type TransitionCurrentAggregate = {
  meanRecMean: number[];
  stdRecMean: number[];
  meanRecMax: number[];
  stdRecMax: number[];
  meanNextMean: number[];
  stdNextMean: number[];
  meanNextMax: number[];
  stdNextMax: number[];
  meanInMean: number[];
  stdInMean: number[];
  meanInMax: number[];
  stdInMax: number[];
  meanInhibValue: number[];
  stdInhibValue: number[];
  meanNoiseMean: number[];
  stdNoiseMean: number[];
  meanNoiseMax: number[];
  stdNoiseMax: number[];
  meanNoiseStd: number[];
  stdNoiseStd: number[];
  meanVMean: number[];
  stdVMean: number[];
  meanVMax: number[];
  stdVMax: number[];
  meanSpikeFrac: number[];
  stdSpikeFrac: number[];
  meanSpikeMass: number[];
  stdSpikeMass: number[];
};

export interface TransitionDebugConfig {
  transitionTrace: boolean;
  traceTrials: number;
  windows: TransitionWindowName[];
  windowDefs: TransitionWindowName[];
  excludeFirst: number[];
  transitionCurrents: boolean;
  traceOutDir: string;
  perturb?: TransitionPerturbConfig;
  ablateNext?: boolean;
  ablateRec?: boolean;
  ablateInhib?: boolean;
  noNoise?: boolean;
}

export type TransitionDebugAblations = {
  ablateNext?: boolean;
  ablateRec?: boolean;
  ablateInhib?: boolean;
  noNoise?: boolean;
};

export type TransitionPerturbKind = 'noise' | 'dropout' | 'shift';

export interface TransitionPerturbConfig {
  enabled: boolean;
  kind: TransitionPerturbKind;
  atStep: number;
  durationSteps: number;
  noiseSigma: number;
  dropoutP: number;
  shiftDelta: number;
  recoveryThreshold: number;
  maxRecoverySteps: number;
  outDir?: string;
}

export interface TransitionTraceEntry {
  startDigit: number;
  targetDigit: number;
  operator?: Operator;
  finalPredDigit?: number;
  correct?: boolean;
  spikeMass: number[];
  bestProtoDigit: (number | null)[];
  bestProtoSim: number[];
  targetProtoSim: number[];
  sourceProtoSim: number[];
  predDigit: number[];
  predConf: number[];
  timeOfLastSpike: number;
  timeOfPeakSpike: number;
}

export interface TransitionTrialDebug {
  meta: TransitionTrialMeta;
  steps: TransitionTrialStep[];
  timeOfLastSpike: number;
  timeOfPeakSpike: number;
  timeToSilence: number;
}

export type TransitionSettleMetrics = {
  tailSpikeMassMean: number;
  tailSpikeMassMedian: number;
  tailSilentFrac: number;
  lateSpikeMassMean: number;
  lateSilentFrac: number;
  histTimeToSilence: number[];
};

export type TransitionSustainStats = {
  lateSpikeMassMean: number;
  tailSpikeMassMean: number;
  tailSilentFrac: number;
  lateSilentFrac: number;
  timeToSilence: number;
};

export type TransitionImpulseDecomposition = {
  recVsNextRatio: { mean: number; std: number };
  driveTotal0: { mean: number; std: number };
  driveFracNext0: { mean: number; std: number };
};

export interface TransitionDebugAggregate {
  meanSpikeMass: number[];
  stdSpikeMass: number[];
  meanTargetProtoSim: number[];
  stdTargetProtoSim: number[];
  meanPredCorrect: number[];
  meanBestProtoIsTarget: number[];
  histTimeOfLastSpike: number[];
  histTimeOfPeakSpike: number[];
  acc_readout_by_window: Record<string, number>;
  acc_proto_by_window: Record<string, number>;
  acc_readout_noImpulse: Record<string, Record<string, number>>;
  acc_proto_noImpulse: Record<string, Record<string, number>>;
  impulseDominanceReadout: number;
  impulseDominanceProto: number;
  sustainDominanceReadout: number;
  sustainDominanceProto: number;
  settle: TransitionSettleMetrics;
  impulseDecomposition: TransitionImpulseDecomposition;
  currents?: TransitionCurrentAggregate;
}

export interface TransitionDebugMeta {
  topology: Topology;
  counting: boolean;
  seed: number;
  tTrans: number;
  tailLen: number;
  silentSpikeThreshold: number;
  traceTrials: number;
  excludeFirst: number[];
  windows: TransitionWindowName[];
  transitionCurrents: boolean;
  operator?: Operator;
  perturb?: TransitionPerturbConfig;
  ablateNext?: boolean;
  ablateRec?: boolean;
  ablateInhib?: boolean;
  noNoise?: boolean;
}

export interface TransitionDebugOutput {
  meta: TransitionDebugMeta;
  aggregate: TransitionDebugAggregate;
  traces: TransitionTraceEntry[];
  trialDebugs: TransitionTrialDebug[];
}

export interface PhaseMetrics {
  phase: CurriculumPhase;
  steps: number;
  accHistory?: number[]; // moving accuracy per evaluation window
  finalAccuracy: number;
  attemptedTrials?: number;
  abortedTrials?: number;
  validTrials?: number;
}

export interface ConfusionResult {
  counts: number[][];
  normalized: number[][];
}

export interface PhaseConfusionSummary {
  readout: ConfusionResult;
  proto: ConfusionResult;
}

export interface SpikeMassStats {
  mean: number;
  min: number;
  max: number;
  silentFrac: number;
}

export interface PhaseDiagnostics {
  disagreeRate: number;
  spikeMass: {
    tail: SpikeMassStats;
    trans?: SpikeMassStats;
  };
}

export type SustainCounters = {
  transitions: number;
  gateFails: number;
  updatesSkipped: number;
  tailSilentFracSum?: number;
  timeToSilenceSum?: number;
};

export type PhaseSustainSummary = {
  tailSilentFracMean: number;
  timeToSilenceMean: number;
  lateSpikeMassMean: number;
  tailSpikeMassMean: number;
  gate?: SustainCounters;
};

export type AccHistorySeries =
  | number[]
  | {
      readout?: number[];
      proto?: number[];
      samples?: number[];
      [key: string]: unknown;
    };

export interface PhaseSummary {
  finalAcc: number;
  steps: number;
  accHistory?: AccHistorySeries;
  confusion?: PhaseConfusionSummary;
  diagnostics?: PhaseDiagnostics;
  protoDiag?: ProtoDiagCounters;
  sustain?: PhaseSustainSummary;
  attemptedTrials?: number;
  abortedTrials?: number;
  abortedFrac?: number;
  validTrials?: number;
}

export interface PhaseOpSplitSummary {
  accPlus: number;
  accMinus: number;
  confusionPlus?: PhaseConfusionSummary;
  confusionMinus?: PhaseConfusionSummary;
  boundaryAccPlus?: number;
  boundaryAccMinus?: number;
}

export interface PhaseEsummary extends PhaseSummary {
  opSplit?: PhaseOpSplitSummary;
  initFromResolved: PhaseEInitResolved;
}

export interface PhaseFsummary extends PhaseOpSplitSummary {
  overallAccMean: number;
  sustainPlus?: PhaseSustainSummary;
  sustainMinus?: PhaseSustainSummary;
  diagnosticsPlus?: PhaseDiagnostics;
  diagnosticsMinus?: PhaseDiagnostics;
  protoDiagPlus?: ProtoDiagCounters;
  protoDiagMinus?: ProtoDiagCounters;
}

export type PhaseEOpSchedule = 'alternate' | 'uniform';
export type PhaseEInitFrom = 'auto' | 'phaseA' | 'phaseB';
export type PhaseEInitResolved = 'phaseA' | 'phaseB' | 'phaseD';
export type TransitionSource = 'pairs' | 'randomWalk';
export type WalkResetMode = 'uniform' | 'cycle' | 'fixed';
export type WalkResetStrategy = {
  mode: WalkResetMode;
  digit?: number;
};

export type PhaseEWalkConfig = {
  episodeLen: number;
  episodesPerBatch: number;
  resetStrategy: WalkResetStrategy;
  bias: number;
};

export type PhaseFWalkConfig = {
  episodeLen: number;
  episodes: number;
  resetStrategy: WalkResetStrategy;
  bias: number;
};

export interface PhaseDConfig {
  enabled: boolean;
}

export interface PhaseEConfig {
  enabled: boolean;
  opSchedule: PhaseEOpSchedule;
  initFrom: PhaseEInitFrom;
  transitionSource: TransitionSource;
  walk: PhaseEWalkConfig;
}

export interface PhaseFConfig {
  enabled: boolean;
  splitByOp: boolean;
  includeBoundaryMetrics: boolean;
  evalTransitionSource: TransitionSource;
  walk: PhaseFWalkConfig;
}

export interface ExperimentConfig {
  topology: Topology;
  useCountingPhase: boolean;
  maxStepsPhaseA: number;
  maxStepsPhaseB: number;
  maxStepsPhaseC: number;
  targetAccPhaseA: number;
  targetAccPhaseB: number;
  targetAccPhaseC: number;
  curriculum: Curriculum;
  learningRates: LearningRateSchedule;
  randomSeed: number;
  phaseASnapshotPath?: string;
  includeAccHistory: boolean;
  protoDebug?: ProtoDebugConfig;
  transitionDebug?: TransitionDebugConfig;
  phaseBCConfig: PhaseBCConfig;
  controller: ControllerConfig;
  phaseD?: PhaseDConfig;
  phaseE?: PhaseEConfig;
  phaseF?: PhaseFConfig;
}

export interface ExperimentResult {
  config: ExperimentConfig;
  successPhaseA: boolean;
  successPhaseB: boolean;
  successPhaseC: boolean;
  metrics: PhaseMetrics[];
  phaseB?: PhaseSummary;
  phaseC: PhaseSummary;
  phaseD?: PhaseSummary;
  phaseE?: PhaseEsummary;
  phaseF?: PhaseFsummary;
}

export type SustainGateConfig = {
  enabled: boolean;
  maxTailSilentFrac: number;
  minTimeToSilence?: number;
  abortAfterTrials?: number;
  skipUpdatesOnFail: boolean;
  skipEpisodeOnFail: boolean;
};

export type PhaseBCConfig = {
  settleWindow: TransitionWindowName;
  excludeFirstK: number;
  evalWindow: TransitionWindowName;
  learnWindow: TransitionWindowName;
  silentSpikeThreshold: number;
  strictSustainPresetApplied: boolean;
  sustainGate: SustainGateConfig;
  logAbortLimit: number;
  tuning: {
    useSustainFitness: boolean;
    sustainWeight: number;
    sustainMetricWindow: 'late' | 'tail';
    targetSustainScore?: number;
  };
};

export type ControllerMode = 'standard' | 'bg';

export type BGControllerAction = 'GO' | 'GO_NO_LEARN' | 'WAIT' | 'ABORT';

export type ControllerPhase = 'PHASE_B_COUNTING' | 'PHASE_C_SUCCESSOR';

export type ControllerPhaseDurations = Partial<Record<ControllerPhase, number>>;

export type BGControllerConfig = {
  actions: BGControllerAction[];
  epsilon: number;
  temperature: number;
  eta: number;
  sampleActions: boolean;
  reward: {
    correct: number;
    wrong: number;
    abort: number;
  };
  waitSteps: number;
  minDwell: number;
  minPhaseDuration: ControllerPhaseDurations;
  enforceOrder: boolean;
  hysteresis: number;
};

export type ControllerConfig = {
  mode: ControllerMode;
  bg?: BGControllerConfig;
};

export const DEFAULT_CURRICULUM: Curriculum = {
  phaseA: {
    k_inhib: 6.69,
    v_th: 1.8,
    alpha: 0.97,
    wInRowNorm: 3.0,
    wAttrRowNorm: 1.5,
    etaTrans: 0,
    activityMode: DEFAULT_ACTIVITY_MODE,
    activityAlpha: DEFAULT_ACTIVITY_ALPHA,
  },
  phaseBC: {
    k_inhib: 8.62,
    v_th: 1.67,
    alpha: 0.73,
    wInRowNorm: 3.0,
    wAttrRowNorm: 1.8,
    etaTrans: 0.22,
    activityMode: DEFAULT_ACTIVITY_MODE,
    activityAlpha: DEFAULT_ACTIVITY_ALPHA,
    tTrans: DEFAULT_T_TRANS,
    tailLen: DEFAULT_TAIL_LEN,
  },
};

export const DEFAULT_LEARNING_RATES: LearningRateSchedule = {
  phaseA: {
    eta_attr: 0.01,
    eta_out: 0.05,
    eta_trans: 0,
  },
  phaseBC: {
    eta_attr: 0.01,
    eta_out: 0.05,
    eta_trans: 0.02,
  },
};

export const DEFAULT_PHASE_BC_CONFIG: PhaseBCConfig = {
  settleWindow: 'mean',
  excludeFirstK: 0,
  evalWindow: 'mean',
  learnWindow: 'mean',
  silentSpikeThreshold: DEFAULT_SILENT_SPIKE_THRESHOLD,
  strictSustainPresetApplied: false,
  sustainGate: {
    enabled: false,
    maxTailSilentFrac: 1.0,
    minTimeToSilence: 0,
    abortAfterTrials: 0,
    skipUpdatesOnFail: true,
    skipEpisodeOnFail: false,
  },
  logAbortLimit: 10,
  tuning: {
    useSustainFitness: false,
    sustainWeight: 0,
    sustainMetricWindow: 'tail',
  },
};

export const DEFAULT_BG_CONTROLLER_CONFIG: BGControllerConfig = {
  actions: ['GO', 'GO_NO_LEARN', 'WAIT', 'ABORT'],
  epsilon: 0.05,
  temperature: 1.0,
  eta: 0.05,
  sampleActions: true,
  reward: {
    correct: 1,
    wrong: -0.2,
    abort: -1,
  },
  waitSteps: 5,
  minDwell: 0,
  minPhaseDuration: {},
  enforceOrder: false,
  hysteresis: 0,
};

export const DEFAULT_CONTROLLER_CONFIG: ControllerConfig = {
  mode: 'standard',
  bg: { ...DEFAULT_BG_CONTROLLER_CONFIG, minPhaseDuration: { ...DEFAULT_BG_CONTROLLER_CONFIG.minPhaseDuration } },
};

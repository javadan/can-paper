import fs from 'fs';
import path from 'path';

import { artifactsDir } from './artifactPaths';
import * as MathOps from './mathHelpers';
import {
  Topology,
  TransitionDebugAggregate,
  TransitionDebugConfig,
  TransitionDebugMeta,
  TransitionDebugOutput,
  TransitionTraceEntry,
  TransitionCurrentAggregate,
  TransitionCurrentSnapshot,
  TransitionTrialDebug,
  TransitionTrialMeta,
  TransitionTrialStep,
  TransitionSettleMetrics,
  TransitionImpulseDecomposition,
  TransitionDebugAblations,
  TransitionPerturbConfig,
  Operator,
} from './types';
import {
  accumulateWindowSums,
  buildNoImpulseTransitionWindows,
  buildTransitionWindows,
  normalizeExcludeFirstValues,
  TransitionWindowRange,
} from './transitionWindows';
import { computeTimeToSilence } from './timeToSilence';
const EPS = 1e-6;
const CURRENT_METRIC_KEYS = [
  'recMean',
  'recMax',
  'nextMean',
  'nextMax',
  'inMean',
  'inMax',
  'inhibValue',
  'noiseMean',
  'noiseMax',
  'noiseStd',
  'vMean',
  'vMax',
  'spikeFrac',
  'spikeMass',
] as const;
type CurrentMetricKey = (typeof CURRENT_METRIC_KEYS)[number];

type ScalarMoments = {
  sum: number;
  sqSum: number;
  count: number;
};

type CurrentTrialState = {
  trialId: number;
  startDigit: number;
  targetDigit: number;
  operator?: Operator;
  shouldTrace: boolean;
  trialMeta: TransitionTrialMeta;
  spikeMass: number[];
  bestProtoDigit: (number | null)[];
  bestProtoSim: number[];
  targetProtoSim: number[];
  sourceProtoSim: number[];
  predDigit: number[];
  predConf: number[];
  traceSteps: TransitionTrialStep[];
  lastSpikeTime: number;
  peakSpikeTime: number;
  peakSpikeMass: number;
  timeToSilence: number;
  settleWindows: Record<string, { sum: number; zeros: number; length: number }>;
  initialDrive?: { recMax0: number; nextMax0: number; inhib0: number };
  windowSumsPrimary: Record<string, Float32Array>;
  windowSumsNoImpulse: Record<number, Record<string, Float32Array>>;
  perturb?: PerturbTrialState;
};

type AggregateSums = {
  spikeMassSum: Float64Array;
  spikeMassSqSum: Float64Array;
  targetProtoSimSum: Float64Array;
  targetProtoSimSqSum: Float64Array;
  predCorrectSum: Float64Array;
  bestProtoIsTargetSum: Float64Array;
  histTimeOfLastSpike: Float64Array;
  histTimeOfPeakSpike: Float64Array;
  histTimeToSilence: Float64Array;
  tailMasses: number[];
  lateMasses: number[];
  tailSilentCount: number;
  lateSilentCount: number;
  driveRecVsNext: ScalarMoments;
  driveTotal0: ScalarMoments;
  driveFracNext0: ScalarMoments;
  current?: {
    sum: Record<string, Float64Array>;
    sqSum: Record<string, Float64Array>;
  };
};

type PerturbTrialState = {
  trialId: number;
  rng: MathOps.RNG;
  baseline?: Float32Array;
  baselineNorm: number;
  startStep: number | null;
  recovered: boolean;
  recoverySteps: number | null;
  collapsed: boolean;
  baselineSimilarity: number | null;
  postPerturbMinSimilarity: number;
};

type PerturbTrialSummary = {
  trialId: number;
  recovered: boolean;
  recoverySteps: number | null;
  collapsed: boolean;
  baselineSimilarity: number;
  postPerturbMinSimilarity: number;
};

export class TransitionDebugger {
  private readonly config: TransitionDebugConfig;
  private readonly tTrans: number;
  private readonly tailLen: number;
  private readonly topology: Topology;
  private readonly counting: boolean;
  private readonly seed: number;
  private readonly numDigits: number;
  private readonly readoutDim: number;
  private readonly numNeurons: number;
  private readonly prototypes: (Float32Array | null)[];
  private readonly protoNorms: number[];
  private readonly wOut: Float32Array;
  private readonly transitionParams: { alpha: number; v_th: number; k_inhib: number; noiseStd: number };
  private readonly transitionScales: { th_scale: number; inhib_scale: number };
  private readonly rowNorms: { wAttrRowNorm: number; wNextRowNorm: number; wInRowNorm: number };
  private readonly silentSpikeThreshold: number;
  private readonly excludeFirst: number[];
  private readonly operator?: Operator;
  private readonly baseNoImpulseExclude: number;
  private readonly primaryWindows: TransitionWindowRange[];
  private readonly noImpulseWindows: Record<number, TransitionWindowRange[]>;
  private readonly settleWindowRanges: Record<'tail' | 'late', TransitionWindowRange | null>;
  private readonly tmpReadout: Float32Array;
  private readonly tmpWindowMean: Float32Array;
  private readonly traces: TransitionTraceEntry[] = [];
  private readonly trialDebugs: TransitionTrialDebug[] = [];
  private readonly aggregateSums: AggregateSums;
  private readonly windowCorrectReadout: Record<string, number>;
  private readonly windowCorrectProto: Record<string, number>;
  private readonly windowCorrectReadoutNoImpulse: Record<number, Record<string, number>>;
  private readonly windowCorrectProtoNoImpulse: Record<number, Record<string, number>>;
  private readonly currentEnabled: boolean;
  private readonly ablations: TransitionDebugAblations;
  private readonly perturbConfig?: TransitionPerturbConfig;
  private readonly perturbTrials: PerturbTrialSummary[] = [];
  private readonly perturbSimilaritySum: number[] = [];
  private readonly perturbSimilarityCount: number[] = [];
  private totalTrials = 0;
  private currentTrial: CurrentTrialState | null = null;

  constructor(params: {
    config: TransitionDebugConfig;
    tTrans: number;
    tailLen: number;
    topology: Topology;
    counting: boolean;
    seed: number;
    numDigits: number;
    readoutDim: number;
    numNeurons: number;
    prototypes: (Float32Array | null)[];
    wOut: Float32Array;
    transitionParams: { alpha: number; v_th: number; k_inhib: number; noiseStd: number };
    transitionScales: { th_scale: number; inhib_scale: number };
    rowNorms: { wAttrRowNorm: number; wNextRowNorm: number; wInRowNorm: number };
    silentSpikeThreshold: number;
    operator?: Operator;
  }) {
    this.config = params.config;
    this.tTrans = params.tTrans;
    this.tailLen = params.tailLen;
    this.topology = params.topology;
    this.counting = params.counting;
    this.seed = params.seed;
    this.numDigits = params.numDigits;
    this.readoutDim = params.readoutDim;
    this.numNeurons = params.numNeurons;
    this.prototypes = params.prototypes;
    this.wOut = params.wOut;
    this.transitionParams = params.transitionParams;
    this.transitionScales = params.transitionScales;
    this.rowNorms = params.rowNorms;
    this.silentSpikeThreshold = params.silentSpikeThreshold;
    this.operator = params.operator;
    this.protoNorms = this.computeProtoNorms();
    this.excludeFirst = normalizeExcludeFirstValues(params.config.excludeFirst);
    this.baseNoImpulseExclude = this.excludeFirst.find((k) => k > 0) ?? 1;
    const windowWarnings: string[] = [];
    this.primaryWindows = buildTransitionWindows(
      params.config.windowDefs,
      this.tTrans,
      this.tailLen,
      this.baseNoImpulseExclude,
      true,
      windowWarnings,
    );
    this.noImpulseWindows = buildNoImpulseTransitionWindows(
      this.excludeFirst,
      this.tTrans,
      this.tailLen,
      windowWarnings,
    );
    this.settleWindowRanges = {
      tail: this.primaryWindows.find((w) => w.name === 'tail') ?? null,
      late: this.primaryWindows.find((w) => w.name === 'late') ?? null,
    };
    if (windowWarnings.length > 0) {
      for (const warning of windowWarnings) {
        console.warn(`[TransitionDebug] ${warning}`);
      }
    }
    this.ablations = {
      ablateNext: params.config.ablateNext,
      ablateRec: params.config.ablateRec,
      ablateInhib: params.config.ablateInhib,
      noNoise: params.config.noNoise,
    };
    this.perturbConfig = params.config.perturb?.enabled ? params.config.perturb : undefined;
    this.currentEnabled = params.config.transitionCurrents || params.config.transitionTrace;
    this.tmpReadout = new Float32Array(this.readoutDim);
    this.tmpWindowMean = new Float32Array(this.numNeurons);
    this.aggregateSums = this.createAggregateSums();
    this.windowCorrectReadout = this.initializeWindowCounters(this.primaryWindows);
    this.windowCorrectProto = this.initializeWindowCounters(this.primaryWindows);
    this.windowCorrectReadoutNoImpulse = {};
    this.windowCorrectProtoNoImpulse = {};
    for (const k of Object.keys(this.noImpulseWindows)) {
      const keyNum = Number(k);
      this.windowCorrectReadoutNoImpulse[keyNum] = this.initializeWindowCounters(this.noImpulseWindows[keyNum]);
      this.windowCorrectProtoNoImpulse[keyNum] = this.initializeWindowCounters(this.noImpulseWindows[keyNum]);
    }
  }

  wantsCurrents(): boolean {
    return this.currentEnabled;
  }

  getAblations(): TransitionDebugAblations {
    return this.ablations;
  }

  getPerturbationConfig(): TransitionPerturbConfig | undefined {
    return this.perturbConfig;
  }

  getPerturbationRng(): MathOps.RNG | undefined {
    return this.currentTrial?.perturb?.rng;
  }

  shouldApplyPerturbation(freeRunStepIndex: number): boolean {
    if (!this.perturbConfig || !this.currentTrial?.perturb) return false;
    return (
      freeRunStepIndex >= this.perturbConfig.atStep &&
      freeRunStepIndex < this.perturbConfig.atStep + this.perturbConfig.durationSteps
    );
  }

  capturePerturbationBaseline(activityVec: Float32Array, freeRunStepIndex: number): void {
    if (!this.perturbConfig || !this.currentTrial?.perturb) return;
    const state = this.currentTrial.perturb;
    if (state.baseline) return;
    const baseline = new Float32Array(activityVec);
    state.baseline = baseline;
    state.baselineNorm = this.computeVectorNorm(baseline);
    state.startStep = freeRunStepIndex;
  }

  markPerturbationApplied(freeRunStepIndex: number): void {
    if (!this.perturbConfig || !this.currentTrial?.perturb) return;
    if (this.currentTrial.perturb.startStep === null) {
      this.currentTrial.perturb.startStep = freeRunStepIndex;
    }
  }

  recordPerturbationStep(freeRunStepIndex: number, activityVec: Float32Array): void {
    if (!this.perturbConfig || !this.currentTrial?.perturb) return;
    const state = this.currentTrial.perturb;
    if (!state.baseline || state.startStep === null) return;
    if (freeRunStepIndex < state.startStep) return;
    const offset = freeRunStepIndex - state.startStep;
    if (offset > this.perturbConfig.maxRecoverySteps) return;
    const similarity = this.computeCosineSimilarity(activityVec, state.baseline, state.baselineNorm);
    if (state.baselineSimilarity === null) state.baselineSimilarity = similarity;
    state.postPerturbMinSimilarity = Math.min(state.postPerturbMinSimilarity, similarity);
    if (!state.recovered && similarity >= this.perturbConfig.recoveryThreshold) {
      state.recovered = true;
      state.recoverySteps = offset;
    }
    if (similarity < 0) state.collapsed = true;

    while (this.perturbSimilaritySum.length <= offset) {
      this.perturbSimilaritySum.push(0);
      this.perturbSimilarityCount.push(0);
    }
    this.perturbSimilaritySum[offset] += similarity;
    this.perturbSimilarityCount[offset] += 1;
  }

  startTrial(
    startDigit: number,
    targetDigit: number,
    operator?: Operator,
    overrides?: { th_scale?: number; inhib_scale?: number; noiseStd?: number },
  ): void {
    const trialId = this.totalTrials;
    const windowSumsPrimary: Record<string, Float32Array> = {};
    for (const window of this.primaryWindows) {
      windowSumsPrimary[window.name] = new Float32Array(this.numNeurons);
    }
    const windowSumsNoImpulse: Record<number, Record<string, Float32Array>> = {};
    for (const [k, windows] of Object.entries(this.noImpulseWindows)) {
      const keyNum = Number(k);
      windowSumsNoImpulse[keyNum] = {};
      for (const window of windows) {
        windowSumsNoImpulse[keyNum][window.name] = new Float32Array(this.numNeurons);
      }
    }

    const shouldTrace = this.config.transitionTrace && this.traces.length < this.config.traceTrials;
    const settleWindows: Record<string, { sum: number; zeros: number; length: number }> = {};
    for (const [name, range] of Object.entries(this.settleWindowRanges)) {
      if (range) settleWindows[name] = { sum: 0, zeros: 0, length: 0 };
    }

    const trialMeta: TransitionTrialMeta = {
      topology: this.topology,
      counting: this.counting,
      seed: this.seed,
      digitStart: startDigit,
      digitTarget: targetDigit,
      operator: operator ?? this.operator,
      tTrans: this.tTrans,
      tailLen: this.tailLen,
      alpha: this.transitionParams.alpha,
      v_th: this.transitionParams.v_th,
      th_scale: overrides?.th_scale ?? this.transitionScales.th_scale,
      k_inhib: this.transitionParams.k_inhib,
      inhib_scale: overrides?.inhib_scale ?? this.transitionScales.inhib_scale,
      noiseStd: overrides?.noiseStd ?? this.transitionParams.noiseStd,
      wAttrRowNorm: this.rowNorms.wAttrRowNorm,
      wNextRowNorm: this.rowNorms.wNextRowNorm,
      wInRowNorm: this.rowNorms.wInRowNorm,
    };

    const perturbState: PerturbTrialState | undefined = this.perturbConfig
      ? {
          trialId,
          rng: MathOps.createRng(this.seed + 1000 + trialId),
          baselineNorm: 0,
          startStep: null,
          recovered: false,
          recoverySteps: null,
          collapsed: false,
          baselineSimilarity: null,
          postPerturbMinSimilarity: Number.POSITIVE_INFINITY,
        }
      : undefined;

    this.currentTrial = {
      trialId,
      startDigit,
      targetDigit,
      operator: operator ?? this.operator,
      shouldTrace,
      trialMeta,
      spikeMass: [],
      bestProtoDigit: [],
      bestProtoSim: [],
      targetProtoSim: [],
      sourceProtoSim: [],
      predDigit: [],
      predConf: [],
      traceSteps: [],
      lastSpikeTime: -1,
      peakSpikeTime: -1,
      peakSpikeMass: -Infinity,
      timeToSilence: this.tTrans,
      settleWindows,
      windowSumsPrimary,
      windowSumsNoImpulse,
      perturb: perturbState,
    };
    this.totalTrials += 1;
  }

  recordStep(t: number, activity: Float32Array, currents?: TransitionCurrentSnapshot): void {
    if (!this.currentTrial) return;

    const spikeMass = this.sumActivity(activity);
    const spikeFrac = spikeMass / this.numNeurons;
    const vectorNorm = spikeMass > 0 ? Math.sqrt(spikeMass) : 0;

    const protoMetrics = this.computeProtoMetrics(
      activity,
      vectorNorm,
      this.currentTrial.startDigit,
      this.currentTrial.targetDigit,
    );
    const { predDigit, predConf } = this.computeReadout(activity);

    this.aggregateSums.spikeMassSum[t] += spikeMass;
    this.aggregateSums.spikeMassSqSum[t] += spikeMass * spikeMass;
    this.aggregateSums.targetProtoSimSum[t] += protoMetrics.targetSim;
    this.aggregateSums.targetProtoSimSqSum[t] += protoMetrics.targetSim * protoMetrics.targetSim;
    if (predDigit === this.currentTrial.targetDigit) this.aggregateSums.predCorrectSum[t] += 1;
    if (protoMetrics.bestDigit === this.currentTrial.targetDigit) this.aggregateSums.bestProtoIsTargetSum[t] += 1;

    if (spikeMass > this.silentSpikeThreshold) {
      this.currentTrial.lastSpikeTime = t;
      if (spikeMass > this.currentTrial.peakSpikeMass) {
        this.currentTrial.peakSpikeMass = spikeMass;
        this.currentTrial.peakSpikeTime = t;
      }
    }

    this.accumulateSettleSpikeMass(spikeMass, t, this.currentTrial.settleWindows);

    this.accumulateWindows(activity, this.primaryWindows, this.currentTrial.windowSumsPrimary, t);
    for (const [k, windows] of Object.entries(this.noImpulseWindows)) {
      this.accumulateWindows(
        activity,
        windows,
        this.currentTrial.windowSumsNoImpulse[Number(k)],
        t,
      );
    }

    if (t === 0 && currents) {
      this.currentTrial.initialDrive = {
        recMax0: currents.recMax,
        nextMax0: currents.nextMax,
        inhib0: currents.inhibValue,
      };
    }

    this.currentTrial.spikeMass.push(spikeMass);

    if (this.currentTrial.shouldTrace) {
      this.currentTrial.bestProtoDigit.push(protoMetrics.bestDigit);
      this.currentTrial.bestProtoSim.push(protoMetrics.bestSim);
      this.currentTrial.targetProtoSim.push(protoMetrics.targetSim);
      this.currentTrial.sourceProtoSim.push(protoMetrics.sourceSim);
      this.currentTrial.predDigit.push(predDigit);
      this.currentTrial.predConf.push(predConf);

      const cur = currents ?? {
        recMean: 0,
        recMax: 0,
        nextMean: 0,
        nextMax: 0,
        inMean: 0,
        inMax: 0,
        inhibValue: 0,
        noiseMean: 0,
        noiseMax: 0,
        noiseStd: 0,
        vMean: 0,
        vMax: 0,
        spikeFrac,
        spikeMass,
      };

      const margin = protoMetrics.targetSim - protoMetrics.secondBestSim;
      const step: TransitionTrialStep = {
        t,
        spikeMass,
        spikeFrac,
        inhibValue: cur.inhibValue,
        recMean: cur.recMean,
        recMax: cur.recMax,
        nextMean: cur.nextMean,
        nextMax: cur.nextMax,
        inMean: cur.inMean,
        inMax: cur.inMax,
        vMean: cur.vMean,
        vMax: cur.vMax,
        noiseMean: cur.noiseMean,
        noiseMax: cur.noiseMax,
        noiseStd: cur.noiseStd,
        readoutPred: predDigit,
        readoutConf: predConf,
        protoBestDigit: protoMetrics.bestDigit,
        protoBestSim: protoMetrics.bestSim,
        protoTargetSim: protoMetrics.targetSim,
        protoMargin: margin,
      };
      this.currentTrial.traceSteps.push(step);
    }

    if (this.aggregateSums.current && currents) {
      for (const key of CURRENT_METRIC_KEYS) {
        const val = currents[key];
        this.aggregateSums.current.sum[key][t] += val;
        this.aggregateSums.current.sqSum[key][t] += val * val;
      }
    }
  }

  finishTrial(): void {
    if (!this.currentTrial) return;

    if (this.currentTrial.peakSpikeMass <= this.silentSpikeThreshold) {
      this.currentTrial.peakSpikeTime = this.tTrans;
    }

    const lastSpikeTime = this.currentTrial.lastSpikeTime >= 0 ? this.currentTrial.lastSpikeTime : this.tTrans;
    const lastBin = Math.min(this.tTrans, lastSpikeTime);
    const peakBin = Math.min(this.tTrans, Math.max(0, this.currentTrial.peakSpikeTime));
    this.aggregateSums.histTimeOfLastSpike[lastBin] += 1;
    this.aggregateSums.histTimeOfPeakSpike[peakBin] += 1;

    const timeToSilence = computeTimeToSilence(
      this.currentTrial.spikeMass,
      this.silentSpikeThreshold,
      this.tTrans,
    );
    this.currentTrial.timeToSilence = timeToSilence;
    this.aggregateSums.histTimeToSilence[timeToSilence] += 1;

    const settleStats = this.computeSettleMetrics(this.currentTrial.settleWindows);
    if (settleStats.tailMass !== null) {
      this.aggregateSums.tailMasses.push(settleStats.tailMass);
      if (settleStats.tailSilent) this.aggregateSums.tailSilentCount += 1;
    }
    if (settleStats.lateMass !== null) {
      this.aggregateSums.lateMasses.push(settleStats.lateMass);
      if (settleStats.lateSilent) this.aggregateSums.lateSilentCount += 1;
    }

    if (this.currentTrial.initialDrive) {
      const eps = 1e-8;
      const { recMax0, nextMax0, inhib0 } = this.currentTrial.initialDrive;
      const recVsNextRatio = nextMax0 / (recMax0 + eps);
      const driveTotal0 = recMax0 + nextMax0 - inhib0;
      const driveFracNext0 = nextMax0 / (recMax0 + nextMax0 + eps);
      this.addMoment(this.aggregateSums.driveRecVsNext, recVsNextRatio);
      this.addMoment(this.aggregateSums.driveTotal0, driveTotal0);
      this.addMoment(this.aggregateSums.driveFracNext0, driveFracNext0);
    }

    for (const window of this.primaryWindows) {
      const sumVec = this.currentTrial.windowSumsPrimary[window.name];
      const { windowReadout, protoPred } = this.evaluateWindowPredictions(sumVec, window);
      if (windowReadout === this.currentTrial.targetDigit) this.windowCorrectReadout[window.name] += 1;
      if (protoPred === this.currentTrial.targetDigit) this.windowCorrectProto[window.name] += 1;
    }

    for (const [k, windows] of Object.entries(this.noImpulseWindows)) {
      const kNum = Number(k);
      for (const window of windows) {
        const sumVec = this.currentTrial.windowSumsNoImpulse[kNum][window.name];
        const { windowReadout, protoPred } = this.evaluateWindowPredictions(sumVec, window);
        if (windowReadout === this.currentTrial.targetDigit) {
          this.windowCorrectReadoutNoImpulse[kNum][window.name] += 1;
        }
        if (protoPred === this.currentTrial.targetDigit) {
          this.windowCorrectProtoNoImpulse[kNum][window.name] += 1;
        }
      }
    }

    if (this.currentTrial.shouldTrace) {
      const finalPredDigit =
        this.currentTrial.predDigit.length > 0
          ? this.currentTrial.predDigit[this.currentTrial.predDigit.length - 1]
          : undefined;
      const trace: TransitionTraceEntry = {
        startDigit: this.currentTrial.startDigit,
        targetDigit: this.currentTrial.targetDigit,
        operator: this.currentTrial.operator ?? this.operator,
        finalPredDigit,
        correct: finalPredDigit !== undefined ? finalPredDigit === this.currentTrial.targetDigit : undefined,
        spikeMass: [...this.currentTrial.spikeMass],
        bestProtoDigit: [...this.currentTrial.bestProtoDigit],
        bestProtoSim: [...this.currentTrial.bestProtoSim],
        targetProtoSim: [...this.currentTrial.targetProtoSim],
        sourceProtoSim: [...this.currentTrial.sourceProtoSim],
        predDigit: [...this.currentTrial.predDigit],
        predConf: [...this.currentTrial.predConf],
        timeOfLastSpike: lastSpikeTime,
        timeOfPeakSpike: this.currentTrial.peakSpikeTime,
      };
      this.traces.push(trace);
      this.trialDebugs.push({
        meta: this.currentTrial.trialMeta,
        steps: [...this.currentTrial.traceSteps],
        timeOfLastSpike: lastSpikeTime,
        timeOfPeakSpike: this.currentTrial.peakSpikeTime,
        timeToSilence,
      });
    }

    if (this.currentTrial.perturb?.baseline) {
      const perturb = this.currentTrial.perturb;
      this.perturbTrials.push({
        trialId: perturb.trialId,
        recovered: perturb.recovered,
        recoverySteps: perturb.recoverySteps,
        collapsed: perturb.collapsed,
        baselineSimilarity: perturb.baselineSimilarity ?? 0,
        postPerturbMinSimilarity: Number.isFinite(perturb.postPerturbMinSimilarity)
          ? perturb.postPerturbMinSimilarity
          : 0,
      });
    }

    this.currentTrial = null;
  }

  finalize(): TransitionDebugOutput {
    const totalTrials = this.totalTrials;
    const denom = totalTrials > 0 ? totalTrials : 1;
    const meanSpikeMass: number[] = [];
    const stdSpikeMass: number[] = [];
    const meanTargetProtoSim: number[] = [];
    const stdTargetProtoSim: number[] = [];
    const meanPredCorrect: number[] = [];
    const meanBestProtoIsTarget: number[] = [];

    for (let t = 0; t < this.tTrans; t++) {
      const spikeMean = totalTrials > 0 ? this.aggregateSums.spikeMassSum[t] / denom : 0;
      const spikeVar =
        totalTrials > 0 ? this.aggregateSums.spikeMassSqSum[t] / denom - spikeMean * spikeMean : 0;
      const targetMean = totalTrials > 0 ? this.aggregateSums.targetProtoSimSum[t] / denom : 0;
      const targetVar =
        totalTrials > 0 ? this.aggregateSums.targetProtoSimSqSum[t] / denom - targetMean * targetMean : 0;
      meanSpikeMass.push(spikeMean);
      stdSpikeMass.push(Math.sqrt(Math.max(spikeVar, 0)));
      meanTargetProtoSim.push(targetMean);
      stdTargetProtoSim.push(Math.sqrt(Math.max(targetVar, 0)));
      meanPredCorrect.push(totalTrials > 0 ? this.aggregateSums.predCorrectSum[t] / denom : 0);
      meanBestProtoIsTarget.push(totalTrials > 0 ? this.aggregateSums.bestProtoIsTargetSum[t] / denom : 0);
    }

    const accReadoutByWindow: Record<string, number> = {};
    const accProtoByWindow: Record<string, number> = {};
    for (const window of this.primaryWindows) {
      accReadoutByWindow[window.name] = totalTrials > 0 ? this.windowCorrectReadout[window.name] / denom : 0;
      accProtoByWindow[window.name] = totalTrials > 0 ? this.windowCorrectProto[window.name] / denom : 0;
    }

    const accReadoutNoImpulse: Record<string, Record<string, number>> = {};
    const accProtoNoImpulse: Record<string, Record<string, number>> = {};
    for (const [k, windows] of Object.entries(this.noImpulseWindows)) {
      const kNum = Number(k);
      const label = `k=${kNum}`;
      const readoutEntry: Record<string, number> = {};
      const protoEntry: Record<string, number> = {};
      for (const window of windows) {
        readoutEntry[window.name] =
          totalTrials > 0 ? this.windowCorrectReadoutNoImpulse[kNum][window.name] / denom : 0;
        protoEntry[window.name] =
          totalTrials > 0 ? this.windowCorrectProtoNoImpulse[kNum][window.name] / denom : 0;
      }
      accReadoutNoImpulse[label] = readoutEntry;
      accProtoNoImpulse[label] = protoEntry;
    }

    const impulseOnlyAcc = accReadoutByWindow['impulseOnly'] ?? 0;
    const impulseOnlyProto = accProtoByWindow['impulseOnly'] ?? 0;
    const meanNoImpulse1 = accReadoutNoImpulse['k=1']?.meanNoImpulse ?? 0;
    const protoNoImpulse1 = accProtoNoImpulse['k=1']?.meanNoImpulse ?? 0;
    const impulseDominanceReadout = impulseOnlyAcc - meanNoImpulse1;
    const impulseDominanceProto = impulseOnlyProto - protoNoImpulse1;
    const sustainDominanceReadout = (accReadoutByWindow['tail'] ?? 0) - impulseOnlyAcc;
    const sustainDominanceProto = (accProtoByWindow['tail'] ?? 0) - impulseOnlyProto;

    const currents = this.buildCurrentAggregate(denom);
    const settle: TransitionSettleMetrics = {
      tailSpikeMassMean: this.computeMean(this.aggregateSums.tailMasses),
      tailSpikeMassMedian: this.computeMedian(this.aggregateSums.tailMasses),
      tailSilentFrac: denom > 0 ? this.aggregateSums.tailSilentCount / denom : 0,
      lateSpikeMassMean: this.computeMean(this.aggregateSums.lateMasses),
      lateSilentFrac: denom > 0 ? this.aggregateSums.lateSilentCount / denom : 0,
      histTimeToSilence: Array.from(this.aggregateSums.histTimeToSilence),
    };

    const impulseDecomposition: TransitionImpulseDecomposition = {
      recVsNextRatio: this.computeMomentStats(this.aggregateSums.driveRecVsNext),
      driveTotal0: this.computeMomentStats(this.aggregateSums.driveTotal0),
      driveFracNext0: this.computeMomentStats(this.aggregateSums.driveFracNext0),
    };

    const aggregate: TransitionDebugAggregate = {
      meanSpikeMass,
      stdSpikeMass,
      meanTargetProtoSim,
      stdTargetProtoSim,
      meanPredCorrect,
      meanBestProtoIsTarget,
      histTimeOfLastSpike: Array.from(this.aggregateSums.histTimeOfLastSpike),
      histTimeOfPeakSpike: Array.from(this.aggregateSums.histTimeOfPeakSpike),
      acc_readout_by_window: accReadoutByWindow,
      acc_proto_by_window: accProtoByWindow,
      acc_readout_noImpulse: accReadoutNoImpulse,
      acc_proto_noImpulse: accProtoNoImpulse,
      impulseDominanceReadout,
      impulseDominanceProto,
      sustainDominanceReadout,
      sustainDominanceProto,
      settle,
      impulseDecomposition,
      currents,
    };

    const meta: TransitionDebugMeta = {
      topology: this.topology,
      counting: this.counting,
      seed: this.seed,
      tTrans: this.tTrans,
      tailLen: this.tailLen,
      silentSpikeThreshold: this.silentSpikeThreshold,
      traceTrials: this.config.traceTrials,
      excludeFirst: this.excludeFirst,
      windows: this.config.windowDefs,
      transitionCurrents: this.currentEnabled,
      operator: this.operator,
      perturb: this.perturbConfig,
      ablateNext: this.ablations.ablateNext,
      ablateRec: this.ablations.ablateRec,
      ablateInhib: this.ablations.ablateInhib,
      noNoise: this.ablations.noNoise,
    };

    return { meta, aggregate, traces: this.traces, trialDebugs: this.trialDebugs };
  }

  save(): string {
    const payload = this.finalize();
    const traceDir =
      this.config.traceOutDir && this.config.traceOutDir.length > 0 ? this.config.traceOutDir : 'transition_traces';
    const resolvedDir = this.resolveOutputDir(traceDir);
    if (!fs.existsSync(resolvedDir)) {
      fs.mkdirSync(resolvedDir, { recursive: true });
    }

    const operatorSuffix = this.operator ? `_${this.operator}` : '';
    const countingLabel = this.counting ? 'countingOn' : 'countingOff';
    const filenameBase = `trace_${this.topology}_${countingLabel}_${this.seed}${operatorSuffix}`;
    const outPath = path.resolve(resolvedDir, `${filenameBase}.json`);
    fs.writeFileSync(outPath, JSON.stringify(payload, null, 2));
    this.saveCompactCsv(payload, resolvedDir, filenameBase);
    this.printConsoleSummary(payload);
    this.savePerturbationArtifacts(traceDir);
    return outPath;
  }

  private saveCompactCsv(payload: TransitionDebugOutput, resolvedDir: string, filenameBase: string): void {
    const { aggregate } = payload;
    const tSteps = aggregate.meanSpikeMass.length;
    const currents = aggregate.currents;
    const recMax = currents?.meanRecMax ?? [];
    const nextMax = currents?.meanNextMax ?? [];
    const inhib = currents?.meanInhibValue ?? [];
    const header = 't,spikeMassMean,spikeMassStd,recMaxMean,nextMaxMean,inhibMean,protoTargetSimMean';
    const lines = [header];
    for (let t = 0; t < tSteps; t++) {
      const row = [
        t,
        aggregate.meanSpikeMass[t] ?? 0,
        aggregate.stdSpikeMass[t] ?? 0,
        recMax[t] ?? '',
        nextMax[t] ?? '',
        inhib[t] ?? '',
        aggregate.meanTargetProtoSim[t] ?? '',
      ];
      lines.push(row.join(','));
    }
    const csvPath = path.join(resolvedDir, `${filenameBase}.csv`);
    fs.writeFileSync(csvPath, lines.join('\n'));
  }

  private savePerturbationArtifacts(traceDir: string): void {
    if (!this.perturbConfig) return;

    const rawOutDir = this.perturbConfig.outDir ?? path.join(traceDir, 'perturb');
    const resolvedDir = this.resolveOutputDir(rawOutDir);
    if (!fs.existsSync(resolvedDir)) {
      fs.mkdirSync(resolvedDir, { recursive: true });
    }

    const trials = this.perturbTrials;
    const recoveredTrials = trials.filter((trial) => trial.recovered && trial.recoverySteps !== null);
    const collapsedTrials = trials.filter((trial) => trial.collapsed);
    const recoverySteps = recoveredTrials.map((trial) => trial.recoverySteps as number);
    const meanRecoverySteps = recoverySteps.length > 0 ? this.computeMean(recoverySteps) : null;
    const medianRecoverySteps = recoverySteps.length > 0 ? this.computeMedian(recoverySteps) : null;
    const recoveryRate = trials.length > 0 ? recoveredTrials.length / trials.length : 0;
    const collapseRate = trials.length > 0 ? collapsedTrials.length / trials.length : 0;
    const aucMean = this.computePerturbAucMean(this.perturbConfig.maxRecoverySteps);

    const similarityMeanByOffset = this.perturbSimilaritySum.map((sum, offset) => {
      const count = this.perturbSimilarityCount[offset] ?? 0;
      return count > 0 ? sum / count : null;
    });
    const similarityCountByOffset = this.perturbSimilaritySum.map(
      (_sum, offset) => this.perturbSimilarityCount[offset] ?? 0,
    );

    const summary = {
      config: {
        kind: this.perturbConfig.kind,
        seed: this.seed,
        atStep: this.perturbConfig.atStep,
        durationSteps: this.perturbConfig.durationSteps,
        noiseSigma: this.perturbConfig.noiseSigma,
        dropoutP: this.perturbConfig.dropoutP,
        shiftDelta: this.perturbConfig.shiftDelta,
        recoveryThreshold: this.perturbConfig.recoveryThreshold,
        maxRecoverySteps: this.perturbConfig.maxRecoverySteps,
      },
      totals: {
        trials: trials.length,
        recovered: recoveredTrials.length,
        collapsed: collapsedTrials.length,
      },
      metrics: {
        recoveryRate,
        meanRecoverySteps,
        medianRecoverySteps,
        collapseRate,
        aucMean,
      },
      similarityByOffset: {
        mean: similarityMeanByOffset,
        count: similarityCountByOffset,
      },
    };

    fs.writeFileSync(path.join(resolvedDir, 'perturb_summary.json'), JSON.stringify(summary, null, 2));

    const header = [
      'trialId',
      'recovered',
      'recoverySteps',
      'collapsed',
      'baselineSimilarity',
      'postPerturbMinSimilarity',
    ];
    const lines = [header.join(',')];
    for (const trial of trials) {
      lines.push(
        [
          trial.trialId,
          trial.recovered,
          trial.recoverySteps ?? '',
          trial.collapsed,
          trial.baselineSimilarity,
          trial.postPerturbMinSimilarity,
        ].join(','),
      );
    }
    fs.writeFileSync(path.join(resolvedDir, 'perturb_trials.csv'), lines.join('\n'));
  }

  private resolveOutputDir(outDir: string): string {
    const baseArtifacts = artifactsDir();
    const resolvedArtifacts = path.resolve(baseArtifacts);
    const resolvedOut = path.resolve(outDir);
    const isInArtifacts = resolvedOut === resolvedArtifacts || resolvedOut.startsWith(`${resolvedArtifacts}${path.sep}`);
    return path.isAbsolute(outDir) || isInArtifacts
      ? resolvedOut
      : path.resolve(path.join(resolvedArtifacts, outDir));
  }

  private printConsoleSummary(payload: TransitionDebugOutput): void {
    const { aggregate, meta } = payload;
    const readoutAcc = aggregate.acc_readout_by_window;
    const noImpulseK1 = aggregate.acc_readout_noImpulse['k=1'] ?? {};
    const tailNoImpulseK1 = noImpulseK1.tailNoImpulse ?? noImpulseK1['tailNoImpulse(k=1)'];
    const lateNoImpulseK1 = noImpulseK1.lateNoImpulse ?? noImpulseK1['lateNoImpulse(k=1)'];
    const meanNoImpulseK1 = noImpulseK1.meanNoImpulse ?? noImpulseK1['meanNoImpulse(k=1)'];
    const fmt = (val?: number) => (val === undefined ? 'n/a' : val.toFixed(2));

    const baseMeanNoImpulse = readoutAcc.meanNoImpulse ?? readoutAcc['meanNoImpulse(k=1)'];
    const baseTailNoImpulse = readoutAcc.tailNoImpulse ?? readoutAcc['tailNoImpulse(k=1)'];
    const baseLateNoImpulse = readoutAcc.lateNoImpulse ?? readoutAcc['lateNoImpulse(k=1)'];

    const accParts = [
      `impulseOnly=${fmt(readoutAcc.impulseOnly)}`,
      `early=${fmt(readoutAcc.early)}`,
      `mean=${fmt(readoutAcc.mean)}`,
      `late=${fmt(readoutAcc.late)}`,
      `tail=${fmt(readoutAcc.tail)}`,
      `meanNoImpulse(k=1)=${fmt(meanNoImpulseK1 ?? baseMeanNoImpulse)}`,
      `tailNoImpulse(k=1)=${fmt(tailNoImpulseK1 ?? baseTailNoImpulse)}`,
      `lateNoImpulse(k=1)=${fmt(lateNoImpulseK1 ?? baseLateNoImpulse)}`,
    ].join(' ');

    const firstSpikeMass = aggregate.meanSpikeMass[0] ?? 0;
    const lastSpikeMass = aggregate.meanSpikeMass.length > 0 ? aggregate.meanSpikeMass[aggregate.meanSpikeMass.length - 1] : 0;

    console.log(
      `[TransitionDebug] topo=${meta.topology} count=${meta.counting ? 'ON' : 'OFF'} tTrans=${meta.tTrans} tailLen=${meta.tailLen}`,
    );
    if (meta.ablateNext || meta.ablateRec || meta.ablateInhib || meta.noNoise) {
      console.log(
        `  ablations: next=${meta.ablateNext ? 'OFF' : 'ON'} rec=${meta.ablateRec ? 'OFF' : 'ON'} ` +
          `inhib=${meta.ablateInhib ? 'OFF' : 'ON'} noise=${meta.noNoise ? 'OFF' : 'ON'}`,
      );
    }
    console.log(`  acc_readout: ${accParts}`);
    console.log(
      `  impulseDominanceReadout=${aggregate.impulseDominanceReadout.toFixed(3)} ` +
        `sustainDominanceReadout=${aggregate.sustainDominanceReadout.toFixed(3)} ` +
        `sustainDominanceProto=${aggregate.sustainDominanceProto.toFixed(3)}`,
    );
    console.log(`  meanSpikeMass[0]=${firstSpikeMass.toFixed(2)} meanSpikeMass[end]=${lastSpikeMass.toFixed(2)}`);

    const settle = aggregate.settle;
    const settleCount = settle.histTimeToSilence.reduce((sum, v) => sum + v, 0);
    const timeToSilenceMean =
      settleCount > 0
        ? settle.histTimeToSilence.reduce((acc, v, idx) => acc + idx * v, 0) / settleCount
        : 0;
    console.log(
      `  settle: tailMean=${settle.tailSpikeMassMean.toFixed(3)} tailMedian=${settle.tailSpikeMassMedian.toFixed(3)} ` +
        `tailSilent=${(settle.tailSilentFrac * 100).toFixed(1)}% lateMean=${settle.lateSpikeMassMean.toFixed(3)} ` +
        `lateSilent=${(settle.lateSilentFrac * 100).toFixed(1)}% timeToSilenceMean=${timeToSilenceMean.toFixed(2)}`,
    );
    const decomp = aggregate.impulseDecomposition;
    console.log(
      `  impulseDecomp@t0: recVsNext=${decomp.recVsNextRatio.mean.toFixed(3)}±${decomp.recVsNextRatio.std.toFixed(3)} ` +
        `driveFracNext=${decomp.driveFracNext0.mean.toFixed(3)}±${decomp.driveFracNext0.std.toFixed(3)} ` +
        `driveTotal=${decomp.driveTotal0.mean.toFixed(3)}±${decomp.driveTotal0.std.toFixed(3)}`,
    );

    if (aggregate.currents) {
      const inMaxMax = aggregate.currents.meanInMax.length > 0 ? Math.max(...aggregate.currents.meanInMax) : 0;
      const nextMax0 = aggregate.currents.meanNextMax[0] ?? 0;
      const recMax0 = aggregate.currents.meanRecMax[0] ?? 0;
      const inhib0 = aggregate.currents.meanInhibValue[0] ?? 0;
      console.log(
        `  inMax during transition: ${inMaxMax.toFixed(3)}${inMaxMax === 0 ? ' (OK)' : ''}`,
      );
      console.log(`  nextMax[0]=${nextMax0.toFixed(3)} recMax[0]=${recMax0.toFixed(3)} inhibValue[0]=${inhib0.toFixed(3)}`);
      console.log(`  noiseStd=${aggregate.currents.meanNoiseStd[0]?.toFixed(3) ?? 'n/a'}`);
    }
  }

  private initializeWindowCounters(windows: TransitionWindowRange[]): Record<string, number> {
    const counters: Record<string, number> = {};
    for (const window of windows) {
      counters[window.name] = 0;
    }
    return counters;
  }

  private createAggregateSums(): AggregateSums {
    const timeHistLen = this.tTrans + 1;
    const base: AggregateSums = {
      spikeMassSum: new Float64Array(this.tTrans),
      spikeMassSqSum: new Float64Array(this.tTrans),
      targetProtoSimSum: new Float64Array(this.tTrans),
      targetProtoSimSqSum: new Float64Array(this.tTrans),
      predCorrectSum: new Float64Array(this.tTrans),
      bestProtoIsTargetSum: new Float64Array(this.tTrans),
      histTimeOfLastSpike: new Float64Array(this.tTrans + 1),
      histTimeOfPeakSpike: new Float64Array(this.tTrans + 1),
      histTimeToSilence: new Float64Array(timeHistLen),
      tailMasses: [],
      lateMasses: [],
      tailSilentCount: 0,
      lateSilentCount: 0,
      driveRecVsNext: { sum: 0, sqSum: 0, count: 0 },
      driveTotal0: { sum: 0, sqSum: 0, count: 0 },
      driveFracNext0: { sum: 0, sqSum: 0, count: 0 },
    };

    if (this.currentEnabled) {
      const sum: Record<string, Float64Array> = {};
      const sqSum: Record<string, Float64Array> = {};
      for (const key of CURRENT_METRIC_KEYS) {
        sum[key] = new Float64Array(this.tTrans);
        sqSum[key] = new Float64Array(this.tTrans);
      }
      base.current = { sum, sqSum };
    }

    return base;
  }

  private accumulateSettleSpikeMass(
    spikeMass: number,
    t: number,
    settleWindows: Record<string, { sum: number; zeros: number; length: number }>,
  ): void {
    for (const [name, range] of Object.entries(this.settleWindowRanges)) {
      if (!range) continue;
      if (t >= range.start && t < range.end) {
        const stats = settleWindows[name];
        if (!stats) continue;
        stats.sum += spikeMass;
        stats.length += 1;
        if (spikeMass <= this.silentSpikeThreshold) stats.zeros += 1;
      }
    }
  }

  private accumulateWindows(
    spikes: Float32Array,
    windows: TransitionWindowRange[],
    windowSums: Record<string, Float32Array>,
    t: number,
  ): void {
    accumulateWindowSums(spikes, windows, windowSums, t);
  }

  private evaluateWindowPredictions(
    sumVec: Float32Array,
    window: TransitionWindowRange,
  ): { windowReadout: number; protoPred: number | null } {
    this.tmpWindowMean.set(sumVec);
    const invLen = window.length > 0 ? 1 / window.length : 0;
    for (let i = 0; i < this.tmpWindowMean.length; i++) {
      this.tmpWindowMean[i] *= invLen;
    }
    const windowReadout = this.computeReadout(this.tmpWindowMean).predDigit;
    const protoPred = this.predictProtoDigit(this.tmpWindowMean);
    return { windowReadout, protoPred };
  }

  private computeSettleMetrics(settleWindows: Record<string, { sum: number; zeros: number; length: number }>): {
    tailMass: number | null;
    lateMass: number | null;
    tailSilent: boolean;
    lateSilent: boolean;
  } {
    const tailStats = settleWindows['tail'];
    const lateStats = settleWindows['late'];
    const tailMass = tailStats && tailStats.length > 0 ? tailStats.sum / tailStats.length : null;
    const lateMass = lateStats && lateStats.length > 0 ? lateStats.sum / lateStats.length : null;
    const tailSilent = Boolean(tailStats && tailStats.length > 0 && tailStats.zeros === tailStats.length);
    const lateSilent = Boolean(lateStats && lateStats.length > 0 && lateStats.zeros === lateStats.length);
    return { tailMass, lateMass, tailSilent, lateSilent };
  }

  private addMoment(target: ScalarMoments, value: number): void {
    target.sum += value;
    target.sqSum += value * value;
    target.count += 1;
  }

  private computeMomentStats(moments: ScalarMoments): { mean: number; std: number } {
    if (moments.count === 0) return { mean: 0, std: 0 };
    const mean = moments.sum / moments.count;
    const variance = moments.sqSum / moments.count - mean * mean;
    return { mean, std: Math.sqrt(Math.max(variance, 0)) };
  }

  private computeMean(values: number[]): number {
    if (values.length === 0) return 0;
    const sum = values.reduce((acc, v) => acc + v, 0);
    return sum / values.length;
  }

  private computeMedian(values: number[]): number {
    if (values.length === 0) return 0;
    const sorted = [...values].sort((a, b) => a - b);
    const mid = Math.floor(sorted.length / 2);
    if (sorted.length % 2 === 0) {
      return (sorted[mid - 1] + sorted[mid]) / 2;
    }
    return sorted[mid];
  }

  private computePerturbAucMean(maxSteps: number): number | null {
    let sum = 0;
    let count = 0;
    for (let i = 0; i <= maxSteps && i < this.perturbSimilaritySum.length; i++) {
      const denom = this.perturbSimilarityCount[i] ?? 0;
      if (denom <= 0) continue;
      sum += this.perturbSimilaritySum[i] / denom;
      count += 1;
    }
    return count > 0 ? sum / count : null;
  }

  private buildCurrentAggregate(denom: number): TransitionCurrentAggregate | undefined {
    if (!this.aggregateSums.current) return undefined;

    const agg: TransitionCurrentAggregate = {
      meanRecMean: [],
      stdRecMean: [],
      meanRecMax: [],
      stdRecMax: [],
      meanNextMean: [],
      stdNextMean: [],
      meanNextMax: [],
      stdNextMax: [],
      meanInMean: [],
      stdInMean: [],
      meanInMax: [],
      stdInMax: [],
      meanInhibValue: [],
      stdInhibValue: [],
      meanNoiseMean: [],
      stdNoiseMean: [],
      meanNoiseMax: [],
      stdNoiseMax: [],
      meanNoiseStd: [],
      stdNoiseStd: [],
      meanVMean: [],
      stdVMean: [],
      meanVMax: [],
      stdVMax: [],
      meanSpikeFrac: [],
      stdSpikeFrac: [],
      meanSpikeMass: [],
      stdSpikeMass: [],
    };

    const keyToFields: Record<
      CurrentMetricKey,
      { mean: keyof TransitionCurrentAggregate; std: keyof TransitionCurrentAggregate }
    > = {
      recMean: { mean: 'meanRecMean', std: 'stdRecMean' },
      recMax: { mean: 'meanRecMax', std: 'stdRecMax' },
      nextMean: { mean: 'meanNextMean', std: 'stdNextMean' },
      nextMax: { mean: 'meanNextMax', std: 'stdNextMax' },
      inMean: { mean: 'meanInMean', std: 'stdInMean' },
      inMax: { mean: 'meanInMax', std: 'stdInMax' },
      inhibValue: { mean: 'meanInhibValue', std: 'stdInhibValue' },
      noiseMean: { mean: 'meanNoiseMean', std: 'stdNoiseMean' },
      noiseMax: { mean: 'meanNoiseMax', std: 'stdNoiseMax' },
      noiseStd: { mean: 'meanNoiseStd', std: 'stdNoiseStd' },
      vMean: { mean: 'meanVMean', std: 'stdVMean' },
      vMax: { mean: 'meanVMax', std: 'stdVMax' },
      spikeFrac: { mean: 'meanSpikeFrac', std: 'stdSpikeFrac' },
      spikeMass: { mean: 'meanSpikeMass', std: 'stdSpikeMass' },
    };

    for (let t = 0; t < this.tTrans; t++) {
      for (const key of CURRENT_METRIC_KEYS) {
        const mean = denom > 0 ? this.aggregateSums.current.sum[key][t] / denom : 0;
        const variance = denom > 0 ? this.aggregateSums.current.sqSum[key][t] / denom - mean * mean : 0;
        const std = Math.sqrt(Math.max(variance, 0));
        agg[keyToFields[key].mean].push(mean);
        agg[keyToFields[key].std].push(std);
      }
    }

    return agg;
  }

  private computeProtoNorms(): number[] {
    return this.prototypes.map((proto) => {
      if (!proto) return 0;
      let norm2 = 0;
      for (let i = 0; i < proto.length; i++) norm2 += proto[i] * proto[i];
      return Math.sqrt(norm2);
    });
  }

  private computeVectorNorm(vec: Float32Array): number {
    let norm2 = 0;
    for (let i = 0; i < vec.length; i++) {
      norm2 += vec[i] * vec[i];
    }
    return Math.sqrt(norm2);
  }

  private computeCosineSimilarity(vec: Float32Array, baseline: Float32Array, baselineNorm: number): number {
    const vecNorm = this.computeVectorNorm(vec);
    if (baselineNorm <= EPS || vecNorm <= EPS) return 0;
    let dot = 0;
    for (let i = 0; i < vec.length; i++) {
      dot += vec[i] * baseline[i];
    }
    return dot / (baselineNorm * vecNorm);
  }

  private sumActivity(vec: Float32Array): number {
    let total = 0;
    for (let i = 0; i < vec.length; i++) {
      total += vec[i];
    }
    return total;
  }

  private computeProtoMetrics(
    vec: Float32Array,
    vecNorm: number,
    startDigit: number,
    targetDigit: number,
  ): { targetSim: number; sourceSim: number; bestDigit: number | null; bestSim: number; secondBestSim: number } {
    let targetSim = 0;
    let sourceSim = 0;
    let bestDigit: number | null = null;
    let bestSim = -Infinity;
    let secondBestSim = -Infinity;
    if (vecNorm <= EPS) {
      return { targetSim: 0, sourceSim: 0, bestDigit: null, bestSim: 0, secondBestSim: 0 };
    }

    for (let d = 0; d < this.numDigits; d++) {
      const proto = this.prototypes[d];
      const protoNorm = this.protoNorms[d];
      if (!proto || protoNorm <= EPS) continue;
      let dot = 0;
      for (let i = 0; i < this.numNeurons; i++) {
        dot += proto[i] * vec[i];
      }
      const cosine = dot / (protoNorm * vecNorm);
      if (d === targetDigit) targetSim = cosine;
      if (d === startDigit) sourceSim = cosine;
      if (bestDigit === null || cosine > bestSim) {
        secondBestSim = bestDigit === null ? secondBestSim : bestSim;
        bestDigit = d;
        bestSim = cosine;
      } else if (cosine > secondBestSim) {
        secondBestSim = cosine;
      }
    }

    const normalizedBestSim = bestDigit === null ? 0 : bestSim;
    const normalizedSecondBest = Number.isFinite(secondBestSim) ? secondBestSim : 0;
    return { targetSim, sourceSim, bestDigit, bestSim: normalizedBestSim, secondBestSim: normalizedSecondBest };
  }

  private computeReadout(vec: Float32Array): { predDigit: number; predConf: number } {
    MathOps.matVecMul(this.wOut, vec, this.readoutDim, this.numNeurons, this.tmpReadout);
    let maxLogit = this.tmpReadout[0];
    let predDigit = 0;
    for (let i = 1; i < this.tmpReadout.length; i++) {
      const v = this.tmpReadout[i];
      if (v > maxLogit) {
        maxLogit = v;
        predDigit = i;
      }
    }
    let expSum = 0;
    for (let i = 0; i < this.tmpReadout.length; i++) {
      expSum += Math.exp(this.tmpReadout[i] - maxLogit);
    }
    const predConf = expSum > 0 ? Math.exp(this.tmpReadout[predDigit] - maxLogit) / expSum : 0;
    return { predDigit, predConf };
  }

  private predictProtoDigit(vec: Float32Array): number | null {
    let bestDigit: number | null = null;
    let bestSim = -Infinity;
    let vecNorm2 = 0;
    for (let i = 0; i < vec.length; i++) vecNorm2 += vec[i] * vec[i];
    if (vecNorm2 <= EPS) return null;
    const vecNorm = Math.sqrt(vecNorm2);

    for (let d = 0; d < this.numDigits; d++) {
      const proto = this.prototypes[d];
      const protoNorm = this.protoNorms[d];
      if (!proto || protoNorm <= EPS) continue;
      let dot = 0;
      for (let i = 0; i < this.numNeurons; i++) {
        dot += proto[i] * vec[i];
      }
      const cosine = dot / (protoNorm * vecNorm);
      if (bestDigit === null || cosine > bestSim) {
        bestDigit = d;
        bestSim = cosine;
      }
    }
    return bestDigit;
  }
}

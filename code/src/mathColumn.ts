import { SDREncoder } from './sdrEncoder';
import {
  DEFAULT_TAIL_LEN,
  DEFAULT_T_TRANS,
  DEFAULT_PHASE_BC_CONFIG,
  DEFAULT_ACTIVITY_ALPHA,
  DEFAULT_ACTIVITY_MODE,
  ActivityMode,
  LearningParams,
  NetworkParams,
  NeuronParams,
  PhaseBCConfig,
  PhasePhysicsParams,
  ProtoDebugRow,
  TransitionSustainStats,
  TransitionCurrentSnapshot,
  TransitionDebugAblations,
  TransitionPerturbConfig,
  Topology,
  Operator,
} from './types';
import * as MathOps from './mathHelpers';
import { TransitionDebugger } from './transitionDebug';
import { Controller } from './controller';
import {
  accumulateWindowSums,
  createWindowSums,
  describeTransitionWindow,
  finalizeWindowMeans,
  resolveTransitionWindows,
  TransitionWindowRange,
} from './transitionWindows';
import { computeTimeToSilence } from './timeToSilence';
import { computeTargetDigit } from './operators';

const T_ENC = 40;
const T_FREE = 20;
export const W_NEXT_ROW_NORM = 1.5;
const RATES_NORM_EPS = 1e-8;
export const TRANSITION_G_ATTR = 0.3;
export const TRANSITION_G_NEXT = 2.5;
export const TRANSITION_INHIB_SCALE = 0.8;
export const TRANSITION_TH_SCALE = 0.8;

export interface ColumnSnapshot {
  topology: Topology;
  net: NetworkParams;
  neuronParams: NeuronParams;
  learnParams: LearningParams;
  activityMode?: ActivityMode;
  activityAlpha?: number;
  W_attr: number[];
  W_next: number[];
  W_prev?: number[];
  W_in: number[];
  W_out: number[];
  digitPrototypes: (number[] | null)[];
}

export type SuccessorTrialResult = {
  aborted: boolean;
  correct?: boolean;
  predDigit?: number;
  targetDigit: number;
  startDigit?: number;
  operator?: Operator;
  protoPredDigit?: number | null;
  sustain?: TransitionSustainStats;
  tailSpikeMass?: number;
  transSpikeMass?: number;
  gateFailed?: boolean;
  predictionsByWindow?: Record<string, number>;
};

export class MathColumn {
  net: NetworkParams;
  neuronParams: NeuronParams;
  learnParams: LearningParams;
  encoder: SDREncoder;

  v: Float32Array;
  s: Float32Array;
  r: Float32Array;
  refractory: Int32Array;
  W_attr: Float32Array;
  W_next: Float32Array;
  W_prev: Float32Array;
  W_in: Float32Array;
  W_out: Float32Array;

  digitPrototypes: (Float32Array | null)[];

  wInRowNorm: number;
  wAttrRowNorm: number;
  tTrans: number;
  tailLen: number;

  phaseBCConfig: PhaseBCConfig;
  sustainCounters: {
    transitions: number;
    gateFails: number;
    updatesSkipped: number;
    tailSilentFracSum: number;
    timeToSilenceSum: number;
  };
  gateFailLogCount: number;
  abortLogCount: number;
  private transitionWindowLogDone: boolean;

  rng: MathOps.RNG;

  activityMode: ActivityMode;
  activityAlpha: number;

  private tmpRec: Float32Array;
  private tmpNext: Float32Array;
  private tmpIn: Float32Array;
  private tmpReadout: Float32Array;
  private tmpCurrentStats: TransitionCurrentSnapshot;
  private tmpPerturb: Float32Array;

  private activityConfigLogged: boolean;

  constructor(topology: Topology, seed: number) {
    this.net = {
      N: 200,
      sdrDim: 256,
      numDigits: 10,
      readoutDim: 10,
    };

    this.neuronParams = {
      v_th: 1.67,
      v_reset: 0,
      alpha: 0.73,
      k_inhib: 8.62,
      noise_std: 0.1,
      refractory_period: 2,
    };

    this.learnParams = {
      eta_attr: 0.01,
      eta_trans: 0.02,
      eta_out: 0.05,
    };

    this.wInRowNorm = 3.0;
    this.wAttrRowNorm = 1.8;
    this.tTrans = DEFAULT_T_TRANS;
    this.tailLen = Math.min(DEFAULT_TAIL_LEN, this.tTrans);
    this.phaseBCConfig = {
      ...DEFAULT_PHASE_BC_CONFIG,
      sustainGate: { ...DEFAULT_PHASE_BC_CONFIG.sustainGate },
      tuning: { ...DEFAULT_PHASE_BC_CONFIG.tuning },
    };
    this.sustainCounters = {
      transitions: 0,
      gateFails: 0,
      updatesSkipped: 0,
      tailSilentFracSum: 0,
      timeToSilenceSum: 0,
    };
    this.gateFailLogCount = 0;
    this.abortLogCount = 0;
    this.transitionWindowLogDone = false;

    this.encoder = new SDREncoder(this.net.sdrDim, this.net.numDigits, topology);
    this.rng = MathOps.createRng(seed);

    this.v = new Float32Array(this.net.N);
    this.s = new Float32Array(this.net.N);
    this.r = new Float32Array(this.net.N);
    this.refractory = new Int32Array(this.net.N);
    this.W_attr = new Float32Array(this.net.N * this.net.N);
    this.W_next = new Float32Array(this.net.N * this.net.N);
    this.W_prev = new Float32Array(this.net.N * this.net.N);
    this.W_in = new Float32Array(this.net.N * this.net.sdrDim);
    this.W_out = new Float32Array(this.net.readoutDim * this.net.N);

    this.tmpRec = new Float32Array(this.net.N);
    this.tmpNext = new Float32Array(this.net.N);
    this.tmpIn = new Float32Array(this.net.N);
    this.tmpReadout = new Float32Array(this.net.readoutDim);
    this.tmpCurrentStats = {
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
      spikeFrac: 0,
      spikeMass: 0,
    };
    this.tmpPerturb = new Float32Array(this.net.N);

    this.digitPrototypes = Array.from({ length: this.net.numDigits }, () => null);

    this.activityMode = DEFAULT_ACTIVITY_MODE;
    this.activityAlpha = DEFAULT_ACTIVITY_ALPHA;
    this.activityConfigLogged = false;

    this.initWeights();
  }

  setRngSeed(seed: number): void {
    this.rng = MathOps.createRng(seed);
  }

  exportSnapshot(): ColumnSnapshot {
    return {
      topology: this.encoder.topology,
      net: { ...this.net },
      neuronParams: { ...this.neuronParams },
      learnParams: { ...this.learnParams },
      activityMode: this.activityMode,
      activityAlpha: this.activityAlpha,
      W_attr: Array.from(this.W_attr),
      W_next: Array.from(this.W_next),
      W_prev: Array.from(this.W_prev),
      W_in: Array.from(this.W_in),
      W_out: Array.from(this.W_out),
      digitPrototypes: this.digitPrototypes.map((p) => (p ? Array.from(p) : null)),
    };
  }

  static fromSnapshot(snapshot: ColumnSnapshot): MathColumn {
    const col = new MathColumn(snapshot.topology, 0);
    col.net = { ...snapshot.net };
    col.neuronParams = { ...snapshot.neuronParams };
    col.learnParams = { ...snapshot.learnParams };
    col.v = new Float32Array(col.net.N);
    col.s = new Float32Array(col.net.N);
    col.r = new Float32Array(col.net.N);
    col.refractory = new Int32Array(col.net.N);
    col.W_attr = new Float32Array(snapshot.W_attr);
    col.W_next = new Float32Array(snapshot.W_next);
    col.W_prev = snapshot.W_prev
      ? new Float32Array(snapshot.W_prev)
      : new Float32Array(snapshot.W_next);
    col.W_in = new Float32Array(snapshot.W_in);
    col.W_out = new Float32Array(snapshot.W_out);
    col.tmpRec = new Float32Array(col.net.N);
    col.tmpNext = new Float32Array(col.net.N);
    col.tmpIn = new Float32Array(col.net.N);
    col.tmpReadout = new Float32Array(col.net.readoutDim);
    col.tmpPerturb = new Float32Array(col.net.N);
    col.digitPrototypes = snapshot.digitPrototypes.map((p) => (p ? Float32Array.from(p) : null));
    col.activityMode = snapshot.activityMode ?? DEFAULT_ACTIVITY_MODE;
    col.activityAlpha = snapshot.activityAlpha ?? DEFAULT_ACTIVITY_ALPHA;
    col.resetState();
    return col;
  }

  applyPhysicsParams(phase: PhasePhysicsParams): void {
    this.neuronParams.k_inhib = phase.k_inhib;
    this.neuronParams.v_th = phase.v_th;
    this.neuronParams.alpha = phase.alpha;
    this.wInRowNorm = phase.wInRowNorm;
    this.wAttrRowNorm = phase.wAttrRowNorm;
    this.learnParams.eta_trans = phase.etaTrans;
    this.setTransitionWindow(phase.tTrans, phase.tailLen);
    this.setActivityConfig(
      phase.activityMode ?? this.activityMode ?? DEFAULT_ACTIVITY_MODE,
      phase.activityAlpha ?? this.activityAlpha ?? DEFAULT_ACTIVITY_ALPHA,
    );
    MathOps.normalizeRows(this.W_in, this.net.N, this.net.sdrDim, this.wInRowNorm);
    MathOps.normalizeRows(this.W_attr, this.net.N, this.net.N, this.wAttrRowNorm);
    this.normalizeWNext();
    this.normalizeWPrev();
  }

  setPhaseBCConfig(config: PhaseBCConfig): void {
    this.phaseBCConfig = {
      ...config,
      sustainGate: { ...config.sustainGate },
      tuning: { ...config.tuning },
    };
    this.gateFailLogCount = 0;
    this.abortLogCount = 0;
  }

  resetPhaseBCSustainCounters(): void {
    this.sustainCounters = {
      transitions: 0,
      gateFails: 0,
      updatesSkipped: 0,
      tailSilentFracSum: 0,
      timeToSilenceSum: 0,
    };
    this.gateFailLogCount = 0;
    this.abortLogCount = 0;
  }

  getPhaseBCSustainCounters(): {
    transitions: number;
    gateFails: number;
    updatesSkipped: number;
    tailSilentFracSum: number;
    timeToSilenceSum: number;
  } {
    return { ...this.sustainCounters };
  }

  resetState(): void {
    this.v.fill(0);
    this.s.fill(0);
    this.r.fill(0);
    this.refractory.fill(0);
  }

  private setActivityConfig(mode: ActivityMode, alpha: number): void {
    const prevMode = this.activityMode;
    const prevAlpha = this.activityAlpha;
    this.activityMode = mode;
    this.activityAlpha = alpha;
    if (!this.activityConfigLogged || prevMode !== mode || prevAlpha !== alpha) {
      console.log(`[MathColumn] Activity mode=${mode} alpha=${alpha}`);
      this.activityConfigLogged = true;
    }
  }

  private getActivityVector(): Float32Array {
    return this.activityMode === 'spike' ? this.s : this.r;
  }

  private applyPerturbation(
    activityVec: Float32Array,
    cfg: TransitionPerturbConfig,
    rng: MathOps.RNG,
  ): void {
    if (cfg.kind === 'noise') {
      for (let i = 0; i < activityVec.length; i++) {
        const nextVal = activityVec[i] + MathOps.randNormal(0, cfg.noiseSigma, rng);
        activityVec[i] = Math.min(1, Math.max(0, nextVal));
      }
      return;
    }

    if (cfg.kind === 'dropout') {
      if (cfg.dropoutP <= 0) return;
      for (let i = 0; i < activityVec.length; i++) {
        if (rng() < cfg.dropoutP) activityVec[i] = 0;
      }
      return;
    }

    if (cfg.kind === 'shift') {
      const delta = Math.abs(cfg.shiftDelta);
      if (delta === 0) return;
      const shift = rng() < 0.5 ? -delta : delta;
      const len = activityVec.length;
      for (let i = 0; i < len; i++) {
        const idx = (i + shift + len * 1000) % len;
        this.tmpPerturb[idx] = activityVec[i];
      }
      activityVec.set(this.tmpPerturb);
    }
  }

  private computeVoltageProxy(vNext: number, effTh: number): number {
    if (vNext >= effTh) return 1;
    const normalized = effTh > 0 ? vNext / effTh : 0;
    if (!Number.isFinite(normalized)) return 0;
    if (normalized < 0) return 0;
    if (normalized > 1) return 1;
    return normalized;
  }

  private computeSustainMetrics(spikeMasses: number[]): TransitionSustainStats {
    const tailStart = Math.max(0, this.tTrans - this.tailLen);
    const lateStart = Math.max(0, this.tTrans - 6);
    const tailSlice = spikeMasses.slice(tailStart);
    const lateSlice = spikeMasses.slice(lateStart);
    const tailLen = tailSlice.length || 1;
    const lateLen = lateSlice.length || 1;
    const spikeThreshold = this.phaseBCConfig.silentSpikeThreshold;

    const tailSilent = tailSlice.filter((v) => v <= spikeThreshold).length;
    const tailSilentFrac = tailSilent / tailLen;
    const tailSpikeMassMean = tailSlice.reduce((acc, v) => acc + v, 0) / tailLen;
    const lateSpikeMassMean = lateSlice.reduce((acc, v) => acc + v, 0) / lateLen;
    const lateSilent = lateSlice.filter((v) => v <= spikeThreshold).length;
    const lateSilentFrac = lateSilent / lateLen;

    const timeToSilence = computeTimeToSilence(spikeMasses, spikeThreshold, this.tTrans);
    return { lateSpikeMassMean, tailSpikeMassMean, tailSilentFrac, lateSilentFrac, timeToSilence };
  }

  private computePreTransitionFeatures(preStateWindow: Float32Array[]): number[] {
    if (preStateWindow.length === 0) return [0, 0, 0, 0, 0];

    const spikeMasses = preStateWindow.map((snapshot) => this.sumActivity(snapshot));
    const silentThreshold = this.phaseBCConfig.silentSpikeThreshold;
    const tailLen = Math.max(1, Math.min(this.tailLen, spikeMasses.length));
    const lateLen = Math.max(1, Math.min(6, spikeMasses.length));
    const tailSlice = spikeMasses.slice(-tailLen);
    const lateSlice = spikeMasses.slice(-lateLen);
    const tailSilentFrac = tailSlice.filter((v) => v <= silentThreshold).length / tailSlice.length;
    const lateSilentFrac = lateSlice.filter((v) => v <= silentThreshold).length / lateSlice.length;
    const tailSpikeMass = tailSlice.reduce((acc, v) => acc + v, 0) / tailSlice.length;
    const transSpikeMass = spikeMasses.reduce((acc, v) => acc + v, 0) / spikeMasses.length;
    const timeToSilence = computeTimeToSilence(spikeMasses, silentThreshold, preStateWindow.length);

    return [tailSpikeMass, transSpikeMass, tailSilentFrac, lateSilentFrac, timeToSilence];
  }

  private predictReadoutDigit(rates: Float32Array): number {
    MathOps.matVecMul(this.W_out, rates, this.net.readoutDim, this.net.N, this.tmpReadout);
    return MathOps.argmax(this.tmpReadout);
  }

  private executeTransition(
    startDigit: number,
    targetDigit: number,
    preState: Float32Array,
    transitionDebugger?: TransitionDebugger,
    suppressStopEpisode = false,
    policyOverrides?: { applyUpdates?: boolean; allowStopEpisode?: boolean },
    operator?: Operator,
  ): {
    predDigit: number;
    protoPredDigit: number | null;
    tailSpikeMass: number;
    transSpikeMass: number;
    sustain: TransitionSustainStats;
    gateFailed: boolean;
    updatesSkipped: boolean;
    stopEpisode: boolean;
    predictionsByWindow: Record<string, number>;
  } {
    const windows: TransitionWindowRange[] = resolveTransitionWindows(
      this.phaseBCConfig,
      this.tTrans,
      this.tailLen,
    );
    if (windows.length === 0) {
      throw new Error(
        `[Transition windows] No valid transition windows resolved (tTrans=${this.tTrans}, tailLen=${this.tailLen})`,
      );
    }
    if (!this.transitionWindowLogDone) {
      const evalWindow = describeTransitionWindow(windows, this.phaseBCConfig.evalWindow);
      const learnWindow = describeTransitionWindow(windows, this.phaseBCConfig.learnWindow);
      console.log(
        `[Transition windows] evalWindow=${evalWindow.label} evalRange=${evalWindow.range} ` +
          `learnWindow=${learnWindow.label} learnRange=${learnWindow.range}`,
      );
      this.transitionWindowLogDone = true;
    }
    const windowSums = createWindowSums(windows, this.net.N);
    const spikeMasses: number[] = [];

    const debugAblations = transitionDebugger?.getAblations();
    const currentStats = transitionDebugger?.wantsCurrents() ? this.tmpCurrentStats : undefined;
    const perturbConfig = transitionDebugger?.getPerturbationConfig();
    transitionDebugger?.startTrial(startDigit, targetDigit, operator, {
      th_scale: TRANSITION_TH_SCALE,
      inhib_scale: debugAblations?.ablateInhib ? 0 : TRANSITION_INHIB_SCALE,
      noiseStd: debugAblations?.noNoise ? 0 : this.neuronParams.noise_std,
    });

    const prevActivityMode = this.activityMode;
    const prevActivityAlpha = this.activityAlpha;
    const forceEmaSpike = !!perturbConfig && this.activityMode === 'spike';
    if (forceEmaSpike) {
      this.setActivityConfig('ema_spike', this.activityAlpha);
    }

    try {
      const freeRunStart = Math.max(0, this.tTrans - this.tailLen);
      for (let t = 0; t < this.tTrans; t++) {
        this.step(null, true, currentStats, debugAblations, operator);
        const activityVec = this.getActivityVector();
        const freeRunStepIndex = t - freeRunStart;
        if (
          t >= freeRunStart &&
          perturbConfig &&
          transitionDebugger?.shouldApplyPerturbation(freeRunStepIndex)
        ) {
          transitionDebugger.capturePerturbationBaseline(activityVec, freeRunStepIndex);
          const rng = transitionDebugger.getPerturbationRng() ?? this.rng;
          this.applyPerturbation(activityVec, perturbConfig, rng);
          transitionDebugger.markPerturbationApplied(freeRunStepIndex);
        }
        accumulateWindowSums(activityVec, windows, windowSums, t);
        spikeMasses.push(this.sumActivity(activityVec));
        transitionDebugger?.recordStep(t, activityVec, currentStats);
        if (t >= freeRunStart) {
          transitionDebugger?.recordPerturbationStep(freeRunStepIndex, activityVec);
        }
      }
    } finally {
      if (forceEmaSpike) {
        this.setActivityConfig(prevActivityMode, prevActivityAlpha);
      }
    }

    const windowMeans = finalizeWindowMeans(windows, windowSums);
    const evalRates = windowMeans[this.phaseBCConfig.evalWindow];
    const learnRates = windowMeans[this.phaseBCConfig.learnWindow];
    const settleRates = windowMeans[this.phaseBCConfig.settleWindow];
    const missingWindows: string[] = [];
    if (!evalRates) missingWindows.push(`evalWindow=${this.phaseBCConfig.evalWindow}`);
    if (!learnRates) missingWindows.push(`learnWindow=${this.phaseBCConfig.learnWindow}`);
    if (!settleRates) missingWindows.push(`settleWindow=${this.phaseBCConfig.settleWindow}`);
    if (missingWindows.length > 0) {
      const available = Object.keys(windowMeans).length > 0 ? Object.keys(windowMeans).join(', ') : 'none';
      const diag =
        `[Transition windows] Missing configured windows (${missingWindows.join(', ')}). ` +
        `Available windows: ${available}. Config excludeFirstK=${this.phaseBCConfig.excludeFirstK} ` +
        `tTrans=${this.tTrans} tailLen=${this.tailLen}`;
      throw new Error(diag);
    }
    const tailRates = windowMeans.tail ?? settleRates;
    const meanRates = windowMeans.mean ?? settleRates;

    const sustain = this.computeSustainMetrics(spikeMasses);
    const transSpikeMass =
      spikeMasses.length > 0 ? spikeMasses.reduce((acc, v) => acc + v, 0) / spikeMasses.length : 0;
    const tailSpikeMass = this.sumActivity(tailRates);

    const gateCfg = this.phaseBCConfig.sustainGate;
    const timeGateEnabled = (gateCfg.minTimeToSilence ?? 0) > 0;
    const timeGateFailed = timeGateEnabled && sustain.timeToSilence < (gateCfg.minTimeToSilence ?? 0);
    const gateFailed =
      gateCfg.enabled && (sustain.tailSilentFrac > gateCfg.maxTailSilentFrac || timeGateFailed);
    const updatesSkipped = gateFailed && gateCfg.skipUpdatesOnFail;
    const gateAbortRequested = gateFailed && gateCfg.skipEpisodeOnFail;
    const stopEpisode = gateAbortRequested && !suppressStopEpisode;
    if (gateFailed && gateCfg.enabled && this.gateFailLogCount < 5) {
      const trialIdx = this.sustainCounters.transitions + 1;
      console.log(
        `[SustainGate FAIL] trial=${trialIdx} tailSilentFrac=${sustain.tailSilentFrac.toFixed(3)} ` +
          `timeToSilence=${sustain.timeToSilence.toFixed(2)} thresholds: ` +
          `maxTailSilentFrac=${gateCfg.maxTailSilentFrac} minTimeToSilence=${gateCfg.minTimeToSilence ?? 0} ` +
          `actions: skipUpdates=${gateCfg.skipUpdatesOnFail} stopEpisode=${gateCfg.skipEpisodeOnFail}`,
      );
      this.gateFailLogCount += 1;
    }
    const applyUpdates = policyOverrides?.applyUpdates ?? !(updatesSkipped || gateAbortRequested);
    const finalUpdatesSkipped = !applyUpdates;
    this.sustainCounters.transitions += 1;
    this.sustainCounters.tailSilentFracSum += sustain.tailSilentFrac;
    this.sustainCounters.timeToSilenceSum += sustain.timeToSilence;
    if (gateFailed) this.sustainCounters.gateFails += 1;
    if (finalUpdatesSkipped) this.sustainCounters.updatesSkipped += 1;

    const predDigit = this.predictReadoutDigit(evalRates);
    const meanPred = meanRates ? this.predictReadoutDigit(meanRates) : predDigit;
    const impulsePred = windowMeans.impulseOnly ? this.predictReadoutDigit(windowMeans.impulseOnly) : undefined;
    const protoPredDigit = this.protoPredict(learnRates);

    if (!finalUpdatesSkipped) {
      const targetProto = this.digitPrototypes[targetDigit];
      if (!targetProto) {
        throw new Error(
          `Missing digit prototype for ${targetDigit}. Ensure Phase A pretraining ran and prototypes are built before Phase B/C.`,
        );
      }
      this.assertLength(targetProto, this.net.N, 'targetProto');
      for (let i = 0; i < this.net.N; i++) {
        this.tmpRec[i] = targetProto[i] - learnRates[i];
      }
      MathOps.outerUpdate(
        this.getTransitionWeights(operator),
        this.tmpRec,
        preState,
        this.net.N,
        this.net.N,
        this.learnParams.eta_trans,
      );
      this.normalizeTransitionWeights(operator);

      MathOps.matVecMul(this.W_out, learnRates, this.net.readoutDim, this.net.N, this.tmpReadout);
      const targetOut = MathOps.oneHot(targetDigit, this.net.readoutDim);
      for (let i = 0; i < this.tmpReadout.length; i++) {
        this.tmpReadout[i] = targetOut[i] - this.tmpReadout[i];
      }
      MathOps.outerUpdate(
        this.W_out,
        this.tmpReadout,
        learnRates,
        this.net.readoutDim,
        this.net.N,
        this.learnParams.eta_out,
      );
    }

    transitionDebugger?.finishTrial();

    const predictionsByWindow: Record<string, number> = { eval: predDigit };
    if (windowMeans.mean) predictionsByWindow.mean = meanPred;
    if (impulsePred !== undefined) predictionsByWindow.impulseOnly = impulsePred;

    return {
      predDigit,
      protoPredDigit,
      tailSpikeMass,
      transSpikeMass,
      sustain,
      gateFailed,
      updatesSkipped: finalUpdatesSkipped,
      stopEpisode: policyOverrides?.allowStopEpisode === false ? false : stopEpisode,
      predictionsByWindow,
    };
  }

  step(
    inputSDR: Float32Array | null,
    useTransition: boolean,
    currentStats?: TransitionCurrentSnapshot,
    debugAblations?: TransitionDebugAblations,
    operator?: Operator,
  ): void {
    const activityVec = this.getActivityVector();

    MathOps.matVecMul(this.W_attr, activityVec, this.net.N, this.net.N, this.tmpRec);
    if (useTransition) {
      MathOps.matVecMul(this.getTransitionWeights(operator), activityVec, this.net.N, this.net.N, this.tmpNext);
      if (debugAblations?.ablateNext) {
        this.tmpNext.fill(0);
      }
      if (debugAblations?.ablateRec) {
        this.tmpRec.fill(0);
      }
    } else {
      this.tmpNext.fill(0);
    }
    if (inputSDR) {
      MathOps.matVecMul(this.W_in, inputSDR, this.net.N, this.net.sdrDim, this.tmpIn);
    } else {
      this.tmpIn.fill(0);
    }

    const captureCurrents = Boolean(currentStats);
    let recSum = 0;
    let recMax = Number.NEGATIVE_INFINITY;
    let nextSum = 0;
    let nextMax = Number.NEGATIVE_INFINITY;
    let inSum = 0;
    let inMax = Number.NEGATIVE_INFINITY;
    for (let i = 0; i < this.net.N; i++) {
      if (captureCurrents) {
        const recVal = this.tmpRec[i];
        const nextVal = this.tmpNext[i];
        const inVal = this.tmpIn[i];
        recSum += recVal;
        nextSum += nextVal;
        inSum += inVal;
        if (recVal > recMax) recMax = recVal;
        if (nextVal > nextMax) nextMax = nextVal;
        if (inVal > inMax) inMax = inVal;
      }
    }

    let g_attr = 1.0;
    let g_next = 0.0;
    let inhib_scale = 1.0;
    let th_scale = 1.0;
    if (useTransition) {
      g_attr = TRANSITION_G_ATTR;
      g_next = TRANSITION_G_NEXT;
      inhib_scale = debugAblations?.ablateInhib ? 0 : TRANSITION_INHIB_SCALE;
      th_scale = TRANSITION_TH_SCALE;
    }

    let meanActivity = 0;
    for (let i = 0; i < this.net.N; i++) meanActivity += activityVec[i];
    meanActivity /= this.net.N;
    let inhibition = this.neuronParams.k_inhib * inhib_scale * meanActivity;
    if (debugAblations?.ablateInhib) inhibition = 0;

    let vSum = 0;
    let vMax = Number.NEGATIVE_INFINITY;
    let spikeCount = 0;
    let noiseSum = 0;
    let noiseMax = Number.NEGATIVE_INFINITY;
    const noiseStd = debugAblations?.noNoise ? 0 : this.neuronParams.noise_std;

    for (let i = 0; i < this.net.N; i++) {
      if (this.refractory[i] > 0) {
        this.refractory[i] -= 1;
        this.s[i] = 0;
        this.v[i] = this.neuronParams.v_reset;
        vSum += this.v[i];
        if (this.v[i] > vMax) vMax = this.v[i];
        continue;
      }

      const noise = MathOps.randNormal(0, noiseStd, this.rng);
      const vNext =
        this.neuronParams.alpha * this.v[i] +
        g_attr * this.tmpRec[i] +
        g_next * this.tmpNext[i] +
        this.tmpIn[i] -
        inhibition +
        noise;
      const effTh = this.neuronParams.v_th * th_scale;

      if (vNext >= effTh) {
        this.s[i] = 1;
        this.v[i] = this.neuronParams.v_reset;
        this.refractory[i] = this.neuronParams.refractory_period;
        spikeCount += 1;
      } else {
        this.s[i] = 0;
        this.v[i] = vNext;
      }
      if (this.activityMode !== 'spike') {
        const x = this.activityMode === 'ema_spike' ? this.s[i] : this.computeVoltageProxy(vNext, effTh);
        this.r[i] = (1 - this.activityAlpha) * this.r[i] + this.activityAlpha * x;
      }
      vSum += this.v[i];
      if (this.v[i] > vMax) vMax = this.v[i];

      if (captureCurrents) {
        noiseSum += noise;
        if (noise > noiseMax) noiseMax = noise;
      }
    }

    if (currentStats) {
      const invN = 1 / this.net.N;
      this.tmpCurrentStats.recMean = recSum * invN;
      this.tmpCurrentStats.recMax = recMax;
      this.tmpCurrentStats.nextMean = nextSum * invN;
      this.tmpCurrentStats.nextMax = nextMax;
      this.tmpCurrentStats.inMean = inSum * invN;
      this.tmpCurrentStats.inMax = inMax;
      this.tmpCurrentStats.inhibValue = inhibition;
      this.tmpCurrentStats.noiseMean = captureCurrents ? noiseSum * invN : 0;
      this.tmpCurrentStats.noiseMax = captureCurrents ? noiseMax : 0;
      this.tmpCurrentStats.noiseStd = noiseStd;
      this.tmpCurrentStats.vMean = vSum * invN;
      this.tmpCurrentStats.vMax = vMax;
      this.tmpCurrentStats.spikeFrac = spikeCount * invN;
      this.tmpCurrentStats.spikeMass = spikeCount;
      Object.assign(currentStats, this.tmpCurrentStats);
    }
  }

  runPhaseA(digit: number): { correct: boolean } {
    this.resetState();
    const inputSDR = this.encoder.encode(digit);
    const encRates = new Float32Array(this.net.N);
    const freeRates = new Float32Array(this.net.N);
    const totalSpikes = new Float32Array(this.net.N);

    const prev = new Float32Array(this.net.N);
    for (let t = 0; t < T_ENC; t++) {
      this.step(inputSDR, false);
      const activityVec = this.getActivityVector();
      for (let i = 0; i < this.net.N; i++) {
        encRates[i] += activityVec[i];
        totalSpikes[i] += activityVec[i];
      }
      MathOps.outerUpdate(
        this.W_attr,
        activityVec,
        prev,
        this.net.N,
        this.net.N,
        this.learnParams.eta_attr,
      );
      prev.set(activityVec);
    }
    MathOps.normalizeRows(this.W_attr, this.net.N, this.net.N, this.wAttrRowNorm);

    for (let t = 0; t < T_FREE; t++) {
      this.step(null, false);
      const activityVec = this.getActivityVector();
      for (let i = 0; i < this.net.N; i++) {
        freeRates[i] += activityVec[i];
        totalSpikes[i] += activityVec[i];
      }
    }

    const duration = T_ENC + T_FREE;
    const rates = new Float32Array(this.net.N);
    for (let i = 0; i < this.net.N; i++) {
      freeRates[i] /= T_FREE;
      rates[i] = totalSpikes[i] / duration;
    }
    if (!this.digitPrototypes[digit]) {
      this.digitPrototypes[digit] = new Float32Array(rates);
    } else {
      const proto = this.digitPrototypes[digit]!;
      for (let i = 0; i < this.net.N; i++) {
        proto[i] = 0.95 * proto[i] + 0.05 * rates[i];
      }
    }

    MathOps.matVecMul(this.W_out, rates, this.net.readoutDim, this.net.N, this.tmpReadout);
    const predDigit = MathOps.argmax(this.tmpReadout);
    const target = MathOps.oneHot(digit, this.net.readoutDim);

    for (let i = 0; i < this.tmpReadout.length; i++) {
      this.tmpReadout[i] = target[i] - this.tmpReadout[i];
    }
    MathOps.outerUpdate(
      this.W_out,
      this.tmpReadout,
      rates,
      this.net.readoutDim,
      this.net.N,
      this.learnParams.eta_out,
    );

    return { correct: predDigit === digit };
  }

  ensureDigitPrototypes(minUpdatesPerDigit: number): void {
    for (let epoch = 0; epoch < minUpdatesPerDigit; epoch++) {
      for (let digit = 0; digit < this.net.numDigits; digit++) {
        this.runPhaseA(digit);
      }
    }

    const missing: number[] = [];
    let minVal = Number.POSITIVE_INFINITY;
    let maxVal = Number.NEGATIVE_INFINITY;
    let sum = 0;
    let count = 0;

    for (let d = 0; d < this.net.numDigits; d++) {
      const proto = this.digitPrototypes[d];
      if (!proto) {
        missing.push(d);
        continue;
      }
      for (let i = 0; i < proto.length; i++) {
        const val = proto[i];
        minVal = Math.min(minVal, val);
        maxVal = Math.max(maxVal, val);
        sum += val;
        count += 1;
      }
    }

    if (missing.length > 0) {
      console.warn(`[MathColumn] Missing digit prototypes after ensureDigitPrototypes: ${missing.join(', ')}`);
    } else {
      console.log(
        `[MathColumn] All digit prototypes present after ensureDigitPrototypes (${minUpdatesPerDigit} updates per digit).`,
      );
    }

    if (count > 0) {
      const mean = sum / count;
      console.log(`[MathColumn] Prototype stats: min=${minVal.toFixed(3)} max=${maxVal.toFixed(3)} mean=${mean.toFixed(3)}`);
    }
  }

  runCountingEpisode(
    minLen: number,
    maxLen: number,
    opts?: {
      recordPrediction?: (target: number, pred: number) => void;
      recordProtoPrediction?: (target: number, pred: number) => void;
      recordTrialStats?: (stats: {
        target: number;
        readoutPred: number;
        protoPred: number | null;
        tailSpikeMass: number;
        transSpikeMass: number;
        sustain?: TransitionSustainStats;
        predictionsByWindow?: Record<string, number>;
        aborted?: boolean;
      }) => void;
    },
    debugCb?: (row: ProtoDebugRow) => void,
    controller?: Controller,
  ): { avgAcc: number } {
    this.resetState();
    const startDigit = Math.floor(this.rng() * this.net.numDigits) % this.net.numDigits;
    const length =
      minLen + Math.floor(this.rng() * Math.max(1, maxLen - minLen + 1));
    let curr = startDigit;
    let correct = 0;
    let trials = 0;

    for (let stepIdx = 0; stepIdx < length - 1; stepIdx++) {
      const nextDigit = (curr + 1) % this.net.numDigits;
      const inputSDR = this.encoder.encode(curr);
      const preStateWindow: Float32Array[] = [];
      const recordPreState = () => {
        const activityVec = this.getActivityVector();
        const snapshot = new Float32Array(activityVec.length);
        snapshot.set(activityVec);
        if (preStateWindow.length === 10) {
          preStateWindow.shift();
        }
        preStateWindow.push(snapshot);
      };
      for (let t = 0; t < T_ENC; t++) {
        this.step(inputSDR, false);
        recordPreState();
      }

      const trialFeatures = this.computePreTransitionFeatures(preStateWindow);
      const decision = controller?.beforeTransition({
        phase: 'PHASE_B_COUNTING',
        trialIdx: stepIdx,
        features: trialFeatures,
      });
      if (decision?.abortEpisode) {
        controller?.afterTransition({
          phase: 'PHASE_B_COUNTING',
          trialIdx: stepIdx,
          correct: false,
          aborted: true,
          sustain: {
            lateSpikeMassMean: 0,
            tailSpikeMassMean: 0,
            tailSilentFrac: 0,
            lateSilentFrac: 0,
            timeToSilence: 0,
          },
          tailSpikeMass: 0,
          transSpikeMass: 0,
          gateFailed: false,
          updatesSkipped: true,
          stopEpisode: true,
        });
        break;
      }
      if (decision?.settleSteps) {
        for (let i = 0; i < decision.settleSteps; i++) {
          this.step(null, false);
          recordPreState();
        }
      }
      const preState = new Float32Array(this.net.N);

      this.assertLength(preState, this.net.N, 'preState');
      const windowLen = preStateWindow.length || 1;
      for (const snapshot of preStateWindow) {
        for (let i = 0; i < this.net.N; i++) {
          preState[i] += snapshot[i];
        }
      }
      for (let i = 0; i < this.net.N; i++) {
        preState[i] /= windowLen;
      }
      const outcome = this.executeTransition(
        curr,
        nextDigit,
        preState,
        undefined,
        Boolean(decision?.suppressStopEpisode),
        { applyUpdates: decision?.applyUpdates, allowStopEpisode: decision?.allowStopEpisode },
      );
      if (outcome.predDigit === nextDigit) correct++;
      if (opts?.recordPrediction) opts.recordPrediction(nextDigit, outcome.predDigit);

      const needsDiagnostics = Boolean(opts?.recordProtoPrediction || opts?.recordTrialStats || debugCb);
      if (needsDiagnostics) {
        if (outcome.protoPredDigit !== null && opts?.recordProtoPrediction) {
          opts.recordProtoPrediction(nextDigit, outcome.protoPredDigit);
        }

        if (opts?.recordTrialStats) {
          opts.recordTrialStats({
            target: nextDigit,
            readoutPred: outcome.predDigit,
            protoPred: outcome.protoPredDigit,
            tailSpikeMass: outcome.tailSpikeMass,
            transSpikeMass: outcome.transSpikeMass,
            sustain: outcome.sustain,
            predictionsByWindow: outcome.predictionsByWindow,
            aborted: false,
          });
        }

        if (debugCb) {
          debugCb({
            phase: 'PHASE_B_COUNTING',
            trialIdx: trials,
            targetDigit: nextDigit,
            readoutPredDigit: outcome.predDigit,
            protoPredDigit: outcome.protoPredDigit,
            protoIsNull: outcome.protoPredDigit === null,
            tailSpikeMass: outcome.tailSpikeMass,
            transSpikeMass: outcome.transSpikeMass,
          });
        }
      }
      const correctPrediction = outcome.predDigit === nextDigit;
      const controllerOutcome = {
        phase: 'PHASE_B_COUNTING' as const,
        trialIdx: stepIdx,
        correct: correctPrediction,
        sustain: outcome.sustain,
        tailSpikeMass: outcome.tailSpikeMass,
        transSpikeMass: outcome.transSpikeMass,
        gateFailed: outcome.gateFailed,
        updatesSkipped: outcome.updatesSkipped ?? false,
        stopEpisode: outcome.stopEpisode,
      };
      controller?.afterTransition(controllerOutcome);
      trials++;
      curr = nextDigit;
      if (outcome.stopEpisode) break;
    }

    const avgAcc = trials > 0 ? correct / trials : 0;
    return { avgAcc };
  }

  runSuccessorTrial(
    digit: number,
    trialIdx = 0,
    opts?: {
      recordPrediction?: (target: number, pred: number) => void;
      recordProtoPrediction?: (target: number, pred: number) => void;
      recordTrialStats?: (stats: {
        target: number;
        readoutPred: number;
        protoPred: number | null;
        tailSpikeMass: number;
        transSpikeMass: number;
        sustain?: TransitionSustainStats;
        predictionsByWindow?: Record<string, number>;
        aborted?: boolean;
      }) => void;
    },
    debugCb?: (row: ProtoDebugRow) => void,
    transitionDebugger?: TransitionDebugger,
    controller?: Controller,
  ): SuccessorTrialResult {
    return this.runTransitionTrial(
      digit,
      'plus',
      trialIdx,
      opts,
      debugCb,
      transitionDebugger,
      controller,
    );
  }

  runTransitionTrial(
    digit: number,
    operator: Operator,
    trialIdx = 0,
    opts?: {
      recordPrediction?: (target: number, pred: number) => void;
      recordProtoPrediction?: (target: number, pred: number) => void;
      recordTrialStats?: (stats: {
        startDigit: number;
        target: number;
        operator: Operator;
        readoutPred: number;
        protoPred: number | null;
        tailSpikeMass: number;
        transSpikeMass: number;
        sustain?: TransitionSustainStats;
        predictionsByWindow?: Record<string, number>;
        aborted?: boolean;
      }) => void;
    },
    debugCb?: (row: ProtoDebugRow) => void,
    transitionDebugger?: TransitionDebugger,
    controller?: Controller,
    policyOverrides?: { applyUpdates?: boolean; allowStopEpisode?: boolean },
  ): SuccessorTrialResult {
    this.resetState();
    const targetDigit = computeTargetDigit(digit, operator, this.net.numDigits);
    const inputSDR = this.encoder.encode(digit);
    const preStateWindow: Float32Array[] = [];
    const recordPreState = () => {
      const activityVec = this.getActivityVector();
      const snapshot = new Float32Array(activityVec.length);
      snapshot.set(activityVec);
      if (preStateWindow.length === 10) {
        preStateWindow.shift();
      }
      preStateWindow.push(snapshot);
    };
    for (let t = 0; t < T_ENC; t++) {
      this.step(inputSDR, false);
      recordPreState();
    }

    const trialFeatures = this.computePreTransitionFeatures(preStateWindow);
    const decision = controller?.beforeTransition({
      phase: 'PHASE_C_SUCCESSOR',
      trialIdx,
      features: trialFeatures,
    });
    const suppressStopEpisode =
      decision?.suppressStopEpisode ??
      ((this.phaseBCConfig.sustainGate.abortAfterTrials ?? 0) > trialIdx &&
        (this.phaseBCConfig.sustainGate.abortAfterTrials ?? 0) > 0);
    if (decision?.abortEpisode) {
      controller?.afterTransition({
        phase: 'PHASE_C_SUCCESSOR',
        trialIdx,
        correct: false,
        aborted: true,
        sustain: {
          lateSpikeMassMean: 0,
          tailSpikeMassMean: 0,
          tailSilentFrac: 0,
          lateSilentFrac: 0,
          timeToSilence: 0,
        },
        tailSpikeMass: 0,
        transSpikeMass: 0,
        gateFailed: false,
        updatesSkipped: true,
        stopEpisode: true,
      });
      return { aborted: true, targetDigit, startDigit: digit, operator };
    }
    if (decision?.settleSteps) {
      for (let i = 0; i < decision.settleSteps; i++) {
        this.step(null, false);
        recordPreState();
      }
    }
    const preState = new Float32Array(this.net.N);

    this.assertLength(preState, this.net.N, 'preState');
    const windowLen = preStateWindow.length || 1;
    for (const snapshot of preStateWindow) {
      for (let i = 0; i < this.net.N; i++) {
        preState[i] += snapshot[i];
      }
    }
    for (let i = 0; i < this.net.N; i++) {
      preState[i] /= windowLen;
    }
    const outcome = this.executeTransition(
      digit,
      targetDigit,
      preState,
      transitionDebugger,
      suppressStopEpisode,
      {
        applyUpdates: policyOverrides?.applyUpdates ?? decision?.applyUpdates,
        allowStopEpisode: policyOverrides?.allowStopEpisode ?? decision?.allowStopEpisode,
      },
      operator,
    );
    const aborted =
      (decision?.abortEpisode ?? false) ||
      this.phaseBCConfig.sustainGate.skipEpisodeOnFail && outcome.stopEpisode;
    const baseResult: SuccessorTrialResult = {
      aborted,
      targetDigit,
      startDigit: digit,
      operator,
      sustain: outcome.sustain,
      gateFailed: outcome.gateFailed,
      tailSpikeMass: outcome.tailSpikeMass,
      transSpikeMass: outcome.transSpikeMass,
      predDigit: outcome.predDigit,
      protoPredDigit: outcome.protoPredDigit,
      predictionsByWindow: outcome.predictionsByWindow,
    };

    const needsDiagnostics = Boolean(opts?.recordProtoPrediction || opts?.recordTrialStats || debugCb);
    if (needsDiagnostics && opts?.recordTrialStats) {
      opts.recordTrialStats({
        startDigit: digit,
        target: targetDigit,
        operator,
        readoutPred: outcome.predDigit,
        protoPred: outcome.protoPredDigit,
        tailSpikeMass: outcome.tailSpikeMass,
        transSpikeMass: outcome.transSpikeMass,
        sustain: outcome.sustain,
        predictionsByWindow: outcome.predictionsByWindow,
        aborted,
      });
    }

    if (aborted) {
      if (debugCb || transitionDebugger) {
        const limit = this.phaseBCConfig.logAbortLimit ?? 10;
        if (this.abortLogCount < limit) {
          console.log(
            `[PhaseC] Trial aborted by sustain gate: start=${digit} target=${targetDigit} tailSilent=${outcome.sustain.tailSilentFrac.toFixed(3)}`,
          );
        } else if (this.abortLogCount === limit) {
          console.log('... further abort logs suppressed');
        }
      }
      this.abortLogCount += 1;
      return baseResult;
    }

    if (opts?.recordPrediction) opts.recordPrediction(targetDigit, outcome.predDigit);

    if (needsDiagnostics && outcome.protoPredDigit !== null && opts?.recordProtoPrediction) {
      opts.recordProtoPrediction(targetDigit, outcome.protoPredDigit);
    }

    if (needsDiagnostics && debugCb) {
      debugCb({
        phase: 'PHASE_C_SUCCESSOR',
        trialIdx,
        targetDigit,
        readoutPredDigit: outcome.predDigit,
        protoPredDigit: outcome.protoPredDigit,
        protoIsNull: outcome.protoPredDigit === null,
        tailSpikeMass: outcome.tailSpikeMass,
        transSpikeMass: outcome.transSpikeMass,
      });
    }

    const correct = outcome.predDigit === targetDigit;
    controller?.afterTransition({
      phase: 'PHASE_C_SUCCESSOR',
      trialIdx,
      correct,
      sustain: outcome.sustain,
      tailSpikeMass: outcome.tailSpikeMass,
      transSpikeMass: outcome.transSpikeMass,
      gateFailed: outcome.gateFailed,
      updatesSkipped: outcome.updatesSkipped,
      stopEpisode: outcome.stopEpisode,
    });

    return {
      ...baseResult,
      correct,
      predDigit: outcome.predDigit,
      protoPredDigit: outcome.protoPredDigit,
      tailSpikeMass: outcome.tailSpikeMass,
      transSpikeMass: outcome.transSpikeMass,
      predictionsByWindow: outcome.predictionsByWindow,
    };
  }

  private protoPredict(rates: Float32Array): number | null {
    let bestDigit: number | null = null;
    let bestCosine = -Infinity;

    let ratesNorm2 = 0;
    for (let i = 0; i < rates.length; i++) {
      ratesNorm2 += rates[i] * rates[i];
    }
    if (ratesNorm2 <= RATES_NORM_EPS) return null;
    const ratesNorm = Math.sqrt(ratesNorm2);

    for (let digit = 0; digit < this.digitPrototypes.length; digit++) {
      const proto = this.digitPrototypes[digit];
      if (!proto) continue;

      let dot = 0;
      let protoNorm2 = 0;
      for (let i = 0; i < proto.length; i++) {
        dot += proto[i] * rates[i];
        protoNorm2 += proto[i] * proto[i];
      }
      if (protoNorm2 <= RATES_NORM_EPS) continue;
      const cosine = dot / (Math.sqrt(protoNorm2) * ratesNorm);
      if (cosine > bestCosine) {
        bestCosine = cosine;
        bestDigit = digit;
      }
    }

    return bestDigit;
  }

  private sumActivity(vec: Float32Array): number {
    let total = 0;
    for (let i = 0; i < vec.length; i++) {
      total += vec[i];
    }
    return total;
  }

  private assertLength(vec: Float32Array, expected: number, name: string): void {
    if (vec.length !== expected) {
      throw new Error(`Unexpected length for ${name}: got ${vec.length}, expected ${expected}`);
    }
  }

  private getTransitionWeights(operator?: Operator): Float32Array {
    return operator === 'minus' ? this.W_prev : this.W_next;
  }

  private normalizeTransitionWeights(operator?: Operator): void {
    if (operator === 'minus') {
      this.normalizeWPrev();
    } else {
      this.normalizeWNext();
    }
  }

  private initWeights(): void {
    const { N, sdrDim } = this.net;
    for (let r = 0; r < N; r++) {
      for (let c = 0; c < N; c++) {
        const idx = r * N + c;
        const noise = MathOps.randNormal(0, 0.01, this.rng);
        this.W_attr[idx] = r === c ? 0 : noise;
        this.W_next[idx] = noise + (r === c ? 0.1 : 0);
        this.W_prev[idx] = noise + (r === c ? 0.1 : 0);
      }
    }
    MathOps.normalizeRows(this.W_attr, N, N, this.wAttrRowNorm);
    this.normalizeWNext();
    this.normalizeWPrev();

    const neuronsPerDigit = Math.max(1, Math.floor(N / this.net.numDigits));
    for (let r = 0; r < N; r++) {
      const prefDigit = Math.floor(r / neuronsPerDigit) % this.net.numDigits;
      const proto = this.encoder.encode(prefDigit);
      const base = r * sdrDim;
      for (let c = 0; c < sdrDim; c++) {
        if (proto[c]) {
          this.W_in[base + c] = MathOps.randNormal(0.2, 0.05, this.rng);
        } else {
          this.W_in[base + c] = Math.max(0, MathOps.randNormal(0.0, 0.01, this.rng));
        }
      }
    }
    MathOps.normalizeRows(this.W_in, N, sdrDim, this.wInRowNorm);
    this.W_out.fill(0);
  }

  // W_next uses a fixed row norm (1.5) to keep transition drive comparable and untuned.
  private normalizeWNext(): void {
    MathOps.normalizeRows(this.W_next, this.net.N, this.net.N, W_NEXT_ROW_NORM);
  }

  private normalizeWPrev(): void {
    MathOps.normalizeRows(this.W_prev, this.net.N, this.net.N, W_NEXT_ROW_NORM);
  }

  private setTransitionWindow(tTrans?: number, tailLen?: number): void {
    if (tTrans !== undefined) {
      this.tTrans = tTrans;
    }
    if (tailLen !== undefined) {
      this.tailLen = tailLen;
    }
    this.tailLen = Math.min(this.tailLen, this.tTrans);
  }
}

import fs from 'fs';
import path from 'path';

import { ColumnSnapshot, MathColumn, TRANSITION_INHIB_SCALE, TRANSITION_TH_SCALE, W_NEXT_ROW_NORM } from './mathColumn';
import { formatConfusion, makeConfusionMatrix, normalizeConfusion, addConfusion } from './metrics';
import {
  DEFAULT_TAIL_LEN,
  DEFAULT_T_TRANS,
  DEFAULT_SILENT_SPIKE_THRESHOLD,
  ExperimentConfig,
  ExperimentResult,
  Operator,
  PhaseConfusionSummary,
  PhaseDiagnostics,
  PhaseMetrics,
  PhaseOpSplitSummary,
  PhaseSummary,
  PhaseEsummary,
  PhaseFsummary,
  PhaseEInitResolved,
  ProtoDiagCounters,
  ProtoDebugRow,
  PhaseSustainSummary,
  SpikeMassStats,
  SustainCounters,
  TransitionWindowName,
} from './types';
import * as MathOps from './mathHelpers';
import { artifactPath } from './artifactPaths';
import { TransitionDebugger } from './transitionDebug';
import { isBoundaryStartDigit } from './operators';
import {
  describeTransitionWindow,
  resolveTransitionWindows,
  TransitionWindowRange,
} from './transitionWindows';
import { Controller, ControllerStats, createController } from './controller';

type ProtoDiagAccumulator = {
  totalTrials: number;
  protoPredNonNull: number;
  protoPredNull: number;
  tailSilent: number;
  tailNonSilent: number;
  tailSpikeMassSum: number;
  transSpikeMassSum: number;
  protoVsReadoutDisagreeCount: number;
};

type WindowAccuracy = Record<string, { correct: number; total: number }>;

type OpCounters = {
  correct: number;
  total: number;
  boundaryCorrect: number;
  boundaryTotal: number;
  confusion: number[][];
  protoConfusion: number[][];
};

export class ExperimentRunner {
  config: ExperimentConfig;
  column: MathColumn;
  rng: MathOps.RNG;
  snapshotLoaded: boolean;
  controller: Controller;

  constructor(config: ExperimentConfig) {
    this.config = config;
    this.rng = MathOps.createRng(config.randomSeed);
    const controllerRng = MathOps.createRng(config.randomSeed + 1);
    if (config.phaseASnapshotPath) {
      const snapshot = ExperimentRunner.loadSnapshot(config.phaseASnapshotPath);
      this.column = MathColumn.fromSnapshot(snapshot);
      this.column.setRngSeed(config.randomSeed);
      this.snapshotLoaded = true;
    } else {
      this.column = new MathColumn(config.topology, config.randomSeed);
      this.snapshotLoaded = false;
    }

    this.column.setPhaseBCConfig(this.config.phaseBCConfig);
    this.controller = createController(this.config.controller, this.config.phaseBCConfig, controllerRng);
  }

  run(): ExperimentResult {
    this.validateTransitionWindow();
    const metrics: PhaseMetrics[] = [];

    const protoDebugLogger = this.createProtoDebugLogger();

    this.logResolvedConfig();

    this.column.resetPhaseBCSustainCounters();

    this.column.learnParams.eta_attr = this.config.learningRates.phaseA.eta_attr;
    this.column.learnParams.eta_out = this.config.learningRates.phaseA.eta_out;
    this.column.learnParams.eta_trans = this.config.learningRates.phaseA.eta_trans ?? 0;
    this.column.applyPhysicsParams(this.config.curriculum.phaseA);
    this.column.resetState();
    const phaseA = this.runPhase(
      'PHASE_A_DIGITS',
      () => {
        const digit = Math.floor(this.rng() * this.column.net.numDigits);
        return this.column.runPhaseA(digit).correct ? 1 : 0;
      },
      this.config.targetAccPhaseA,
      this.config.maxStepsPhaseA,
    );
    metrics.push(phaseA.metrics);

    this.column.ensureDigitPrototypes(50);
    const phaseASnapshot = this.column.exportSnapshot();

    const numDigits = this.column.net.numDigits;
    const tTrans = this.config.curriculum.phaseBC.tTrans ?? DEFAULT_T_TRANS;
    const tailLen = this.config.curriculum.phaseBC.tailLen ?? Math.min(DEFAULT_TAIL_LEN, tTrans);
    const buildTransitionDebugger = (operator?: Operator): TransitionDebugger | undefined =>
      this.config.transitionDebug
        ? new TransitionDebugger({
            config: this.config.transitionDebug,
            tTrans,
            tailLen,
            topology: this.config.topology,
            counting: this.config.useCountingPhase,
            seed: this.config.randomSeed,
            numDigits,
            readoutDim: this.column.net.readoutDim,
            numNeurons: this.column.net.N,
            prototypes: this.column.digitPrototypes,
            wOut: this.column.W_out,
            transitionParams: {
              alpha: this.config.curriculum.phaseBC.alpha,
              v_th: this.config.curriculum.phaseBC.v_th,
              k_inhib: this.config.curriculum.phaseBC.k_inhib,
              noiseStd: this.column.neuronParams.noise_std,
            },
            transitionScales: { th_scale: TRANSITION_TH_SCALE, inhib_scale: TRANSITION_INHIB_SCALE },
            rowNorms: {
              wAttrRowNorm: this.config.curriculum.phaseBC.wAttrRowNorm,
              wNextRowNorm: W_NEXT_ROW_NORM,
              wInRowNorm: this.config.curriculum.phaseBC.wInRowNorm,
            },
            silentSpikeThreshold: this.config.phaseBCConfig.silentSpikeThreshold,
            operator,
          })
        : undefined;
    const transitionDebugger = buildTransitionDebugger();
    let phaseBResult: { success: boolean; metrics: PhaseMetrics; confusion?: number[][] } = {
      success: false,
      metrics: {
        phase: 'PHASE_B_COUNTING',
        steps: 0,
        accHistory: [],
        finalAccuracy: 0,
      },
    };
    let phaseBConfusionSummary: PhaseConfusionSummary | undefined;
    let phaseBDiagnostics: PhaseDiagnostics | undefined;
    let phaseBProtoDiagSummary: ProtoDiagCounters | undefined;
    const phaseBConfusion = makeConfusionMatrix(numDigits);
    const phaseBProtoConfusion = makeConfusionMatrix(numDigits);
    const phaseBSpikeTail: number[] = [];
    const phaseBSpikeTrans: number[] = [];
    const phaseBTailSilent: number[] = [];
    const phaseBTimeToSilence: number[] = [];
    const phaseBWindowAcc: WindowAccuracy = {};
    let phaseBDisagreeCount = 0;
    let phaseBProtoTrials = 0;
    let gateCountersAfterPhaseB: SustainCounters = this.column.getPhaseBCSustainCounters();
    let phaseBSustainSummary: PhaseSustainSummary | undefined;

    if (this.config.useCountingPhase) {
      const phaseBProtoDiag = this.createProtoDiagAccumulator();
      let phaseBTrialIdx = 0;
      const phaseBDebugCb = protoDebugLogger
        ? (row: ProtoDebugRow) => protoDebugLogger({ ...row, trialIdx: phaseBTrialIdx++ })
        : undefined;
      this.column.learnParams.eta_attr = this.config.learningRates.phaseBC.eta_attr;
      this.column.learnParams.eta_out = this.config.learningRates.phaseBC.eta_out;
      this.column.learnParams.eta_trans = this.config.learningRates.phaseBC.eta_trans;
      this.column.applyPhysicsParams(this.config.curriculum.phaseBC);
      this.column.resetState();
      this.controller.resetEpisode();
      phaseBResult = this.runPhase(
        'PHASE_B_COUNTING',
        () =>
          this.column.runCountingEpisode(2, 6, {
            recordPrediction: (target, pred) => addConfusion(phaseBConfusion, target, pred),
            recordProtoPrediction: (target, pred) => addConfusion(phaseBProtoConfusion, target, pred),
          recordTrialStats: ({
            target,
            readoutPred,
            protoPred,
            tailSpikeMass,
            transSpikeMass,
            sustain,
            predictionsByWindow,
            aborted,
          }) => {
            phaseBSpikeTail.push(tailSpikeMass);
            phaseBSpikeTrans.push(transSpikeMass);
            if (sustain) {
              phaseBTailSilent.push(sustain.tailSilentFrac);
              phaseBTimeToSilence.push(sustain.timeToSilence);
            }
            if (!aborted) {
              if (protoPred !== null) {
                phaseBProtoTrials += 1;
                if (protoPred !== readoutPred) phaseBDisagreeCount += 1;
              }
              this.updateProtoDiagCounters(phaseBProtoDiag, {
                protoPred,
                readoutPred,
                tailSpikeMass,
                transSpikeMass,
              });
              if (predictionsByWindow) {
                this.updateWindowAccuracy(phaseBWindowAcc, predictionsByWindow, target);
              }
            }
          },
        },
          phaseBDebugCb,
          this.controller,
        ).avgAcc,
        this.config.targetAccPhaseB,
        this.config.maxStepsPhaseB,
      );
      phaseBResult.confusion = phaseBConfusion;
      phaseBDiagnostics = this.buildDiagnostics(
        phaseBDisagreeCount,
        phaseBProtoTrials,
        phaseBSpikeTail,
        phaseBSpikeTrans,
      );
      phaseBProtoDiagSummary = this.finalizeProtoDiagCounters(phaseBProtoDiag);
      phaseBConfusionSummary = this.buildConfusionSummary(phaseBConfusion, phaseBProtoConfusion);
      gateCountersAfterPhaseB = this.column.getPhaseBCSustainCounters();
      phaseBSustainSummary = this.buildSustainSummary(
        phaseBTailSilent,
        phaseBTimeToSilence,
        phaseBSpikeTrans,
        phaseBSpikeTail,
        gateCountersAfterPhaseB,
      );
      metrics.push(phaseBResult.metrics);
    } else {
      metrics.push(phaseBResult.metrics);
    }

    const phaseCConfusion = makeConfusionMatrix(numDigits);
    const phaseCProtoConfusion = makeConfusionMatrix(numDigits);
    const phaseCSpikeTail: number[] = [];
    const phaseCSpikeTrans: number[] = [];
    const phaseCTailSilent: number[] = [];
    const phaseCTimeToSilence: number[] = [];
    const phaseCWindowAcc: WindowAccuracy = {};
    let phaseCDisagreeCount = 0;
    let phaseCProtoTrials = 0;
    const phaseCProtoDiag = this.createProtoDiagAccumulator();
    let phaseCTrialIdx = 0;
    let phaseCSustainSummary: PhaseSustainSummary | undefined;
    const phaseCLogEvery = 100;
    let intervalAttempted = 0;
    let intervalAborted = 0;
    let intervalValid = 0;
    let intervalGateFails = 0;
    const intervalTailSilent: number[] = [];
    const intervalTimeToSilence: number[] = [];
    const flushPhaseCIntervalLog = () => {
      if (intervalAttempted === 0) return;
      const tailMean = this.computeMean(intervalTailSilent);
      const timeMean = this.computeMean(intervalTimeToSilence);
      console.log(
        `[PhaseC Interval] attempted=${intervalAttempted} aborted=${intervalAborted} valid=${intervalValid} ` +
          `gateFailed=${intervalGateFails} tailSilentMean=${tailMean.toFixed(3)} timeToSilenceMean=${timeMean.toFixed(2)}`,
      );
      intervalAttempted = 0;
      intervalAborted = 0;
      intervalValid = 0;
      intervalGateFails = 0;
      intervalTailSilent.length = 0;
      intervalTimeToSilence.length = 0;
    };
    this.column.learnParams.eta_attr = this.config.learningRates.phaseBC.eta_attr;
    this.column.learnParams.eta_out = this.config.learningRates.phaseBC.eta_out;
    this.column.learnParams.eta_trans = this.config.learningRates.phaseBC.eta_trans;
    this.column.applyPhysicsParams(this.config.curriculum.phaseBC);
    this.column.resetState();
    this.controller.resetEpisode();
    const phaseBCBaseSnapshot = this.column.exportSnapshot();
    const phaseC = this.runPhase(
      'PHASE_C_SUCCESSOR',
      () => {
        const digit = Math.floor(this.rng() * this.column.net.numDigits);
        const trialIdx = phaseCTrialIdx;
        const phaseCDebugCb = protoDebugLogger
          ? (row: ProtoDebugRow) => protoDebugLogger({ ...row, trialIdx })
          : undefined;
        const trial = this.column.runSuccessorTrial(
          digit,
          trialIdx,
          {
            recordPrediction: (target, pred) => addConfusion(phaseCConfusion, target, pred),
            recordProtoPrediction: (target, pred) => addConfusion(phaseCProtoConfusion, target, pred),
          recordTrialStats: ({
            target,
            readoutPred,
            protoPred,
            tailSpikeMass,
            transSpikeMass,
            sustain,
            predictionsByWindow,
            aborted,
          }) => {
            phaseCSpikeTail.push(tailSpikeMass);
            phaseCSpikeTrans.push(transSpikeMass);
            if (sustain) {
              phaseCTailSilent.push(sustain.tailSilentFrac);
              phaseCTimeToSilence.push(sustain.timeToSilence);
            }
            if (!aborted) {
              if (protoPred !== null) {
                phaseCProtoTrials += 1;
                if (protoPred !== readoutPred) phaseCDisagreeCount += 1;
              }
              this.updateProtoDiagCounters(phaseCProtoDiag, {
                protoPred,
                readoutPred,
                tailSpikeMass,
                transSpikeMass,
              });
              if (predictionsByWindow) {
                this.updateWindowAccuracy(phaseCWindowAcc, predictionsByWindow, target);
              }
            }
          },
        },
          phaseCDebugCb,
          transitionDebugger,
          this.controller,
        );
        intervalAttempted += 1;
        if (trial.aborted) intervalAborted += 1;
        if (!trial.aborted) intervalValid += 1;
        if (trial.gateFailed) intervalGateFails += 1;
        if (trial.sustain) {
          intervalTailSilent.push(trial.sustain.tailSilentFrac);
          intervalTimeToSilence.push(trial.sustain.timeToSilence);
        }
        if (intervalAttempted % phaseCLogEvery === 0) flushPhaseCIntervalLog();
        phaseCTrialIdx += 1;

        return { value: trial.correct ? 1 : 0, aborted: trial.aborted };
      },
      this.config.targetAccPhaseC,
      this.config.maxStepsPhaseC,
    );
    flushPhaseCIntervalLog();
    const phaseCAttempted = phaseC.metrics.attemptedTrials ?? phaseC.metrics.steps;
    const phaseCAborted = phaseC.metrics.abortedTrials ?? 0;
    const phaseCValidTrials = phaseC.metrics.validTrials ?? phaseCAttempted - phaseCAborted;
    const phaseCAbortedFrac = phaseCAttempted > 0 ? phaseCAborted / phaseCAttempted : 0;
    metrics.push(phaseC.metrics);
    const phaseCSnapshot = this.column.exportSnapshot();

    console.log('Phase C confusion saved (rows=target, cols=pred)');
    if (process.env.PRINT_CONFUSION === '1') {
      console.log(formatConfusion(normalizeConfusion(phaseCConfusion)));
    }

    const phaseCConfusionSummary = this.buildConfusionSummary(phaseCConfusion, phaseCProtoConfusion);
    const phaseCDiagnostics = this.buildDiagnostics(
      phaseCDisagreeCount,
      phaseCProtoTrials,
      phaseCSpikeTail,
      phaseCSpikeTrans,
    );
    const phaseCProtoDiagSummary = this.finalizeProtoDiagCounters(phaseCProtoDiag);
    const finalGateCounters = this.column.getPhaseBCSustainCounters();
    const phaseCGate: SustainCounters = {
      transitions: finalGateCounters.transitions - gateCountersAfterPhaseB.transitions,
      gateFails: finalGateCounters.gateFails - gateCountersAfterPhaseB.gateFails,
      updatesSkipped: finalGateCounters.updatesSkipped - gateCountersAfterPhaseB.updatesSkipped,
      tailSilentFracSum:
        (finalGateCounters.tailSilentFracSum ?? 0) - (gateCountersAfterPhaseB.tailSilentFracSum ?? 0),
      timeToSilenceSum:
        (finalGateCounters.timeToSilenceSum ?? 0) - (gateCountersAfterPhaseB.timeToSilenceSum ?? 0),
    };
    phaseCSustainSummary = this.buildSustainSummary(
      phaseCTailSilent,
      phaseCTimeToSilence,
      phaseCSpikeTrans,
      phaseCSpikeTail,
      phaseCGate,
    );
    if (!phaseBSustainSummary && this.config.useCountingPhase) {
      phaseBSustainSummary = this.buildSustainSummary(
        phaseBTailSilent,
        phaseBTimeToSilence,
        phaseBSpikeTrans,
        phaseBSpikeTail,
        gateCountersAfterPhaseB,
      );
    }

    if (transitionDebugger) {
      const outPath = transitionDebugger.save();
      console.log(`[TransitionDebug] Saved transition traces to ${outPath}`);
    }

    const evalAccEntry = phaseCWindowAcc.eval ?? phaseCWindowAcc['eval'];
    const meanAccEntry = phaseCWindowAcc.mean ?? phaseCWindowAcc['mean'];
    const impulseAccEntry = phaseCWindowAcc.impulseOnly ?? phaseCWindowAcc['impulseOnly'];
    const resolveAcc = (entry: { correct: number; total: number } | undefined) => {
      if (!entry || entry.total === 0) return { value: null as number | null, total: 0 };
      return { value: entry.correct / entry.total, total: entry.total };
    };
    const evalAccPhaseC = resolveAcc(evalAccEntry);
    const meanAccPhaseC = resolveAcc(meanAccEntry);
    const impulseAccPhaseC = resolveAcc(impulseAccEntry);
    const tailSilentMean = this.computeMean(phaseCTailSilent);
    const timeToSilenceMean = this.computeMean(phaseCTimeToSilence);
    const sustainSamples = phaseCTailSilent.length;
    const gateSummary = this.column.getPhaseBCSustainCounters();
    const gateCfg = this.config.phaseBCConfig.sustainGate;
    const controllerStats: ControllerStats | undefined = this.controller.getStats();
    const abortActionCount = controllerStats?.actions?.ABORT ?? 0;
    const totalActionCount = controllerStats?.actions
      ? Object.values(controllerStats.actions).reduce((acc, v) => acc + (v ?? 0), 0)
      : 0;
    const abortActionFrac = totalActionCount > 0 ? abortActionCount / totalActionCount : null;
    const resolvedWindows = resolveTransitionWindows(this.config.phaseBCConfig, tTrans, tailLen);
    const formatWindowLabel = (name: TransitionWindowName) => describeTransitionWindow(resolvedWindows, name).label;
    const formatAcc = (label: string, acc: { value: number | null; total: number }) =>
      acc.value === null ? `${label}=n/a` : `${label}=${acc.value.toFixed(3)} (${acc.total})`;
    const formatMaybeMean = (value: number, digits: number) =>
      Number.isFinite(value) && sustainSamples > 0 ? value.toFixed(digits) : 'n/a';

    console.log(
      `[PhaseBC Summary] topo=${this.config.topology} counting=${this.config.useCountingPhase ? 'on' : 'off'} ` +
        `evalWindow="${formatWindowLabel(this.config.phaseBCConfig.evalWindow)}" ` +
        `learnWindow="${formatWindowLabel(this.config.phaseBCConfig.learnWindow)}" ` +
        `excludeFirst=${this.config.phaseBCConfig.excludeFirstK} attempted=${phaseCAttempted} ` +
        `aborted=${phaseCAborted} valid=${phaseCValidTrials} ` +
        `abortActionFrac=${abortActionFrac === null ? 'n/a' : abortActionFrac.toFixed(3)} ` +
        `${formatAcc('acc_eval', evalAccPhaseC)} ${formatAcc('acc_mean', meanAccPhaseC)} ` +
        `${formatAcc('acc_impulse', impulseAccPhaseC)} tailSilent=${formatMaybeMean(tailSilentMean, 3)} ` +
        `timeToSilence=${formatMaybeMean(timeToSilenceMean, 2)} gateFails=${gateSummary.gateFails}/${gateSummary.transitions} ` +
        `gateEnabled=${gateCfg.enabled} maxTailSilentFrac=${gateCfg.maxTailSilentFrac} ` +
        `minTimeToSilence=${gateCfg.minTimeToSilence ?? 0} ` +
        `updatesSkipped=${gateSummary.updatesSkipped}`,
    );
    if (controllerStats?.actions) {
      const actions = Object.entries(controllerStats.actions)
        .map(([k, v]) => `${k}:${v}`)
        .join(' ');
      const rewardLabel = controllerStats.totalReward !== undefined ? controllerStats.totalReward.toFixed(3) : 'n/a';
      console.log(
        `[Controller Summary] actions=${actions} rewardSum=${rewardLabel} trials=${controllerStats.trials ?? 'n/a'}`,
      );
    }

    const phaseDEnabled = this.config.phaseD?.enabled ?? false;
    const phaseEEnabled = this.config.phaseE?.enabled ?? false;
    const phaseFEnabled = this.config.phaseF?.enabled ?? false;
    const includeBoundaryMetrics = this.config.phaseF?.includeBoundaryMetrics ?? true;
    let phaseD: PhaseSummary | undefined;
    let phaseE: PhaseEsummary | undefined;
    let phaseF: PhaseFsummary | undefined;
    let phaseDSnapshot: ColumnSnapshot | undefined;
    let phaseESnapshot: ColumnSnapshot | undefined;

    if (phaseDEnabled) {
      this.resetColumnFromSnapshot(phaseBCBaseSnapshot);
      this.column.resetPhaseBCSustainCounters();
      this.controller.resetEpisode();
      const phaseDResult = this.runOperatorPhase({
        phase: 'PHASE_D_PREDECESSOR',
        operatorSelector: () => 'minus',
        targetAcc: this.config.targetAccPhaseC,
        maxSteps: this.config.maxStepsPhaseC,
        includeBoundaryMetrics: false,
        trackOpSplit: false,
      });
      metrics.push(phaseDResult.metrics);
      phaseD = phaseDResult.summary;
      phaseDSnapshot = this.column.exportSnapshot();
    }

    if (phaseEEnabled) {
      const phaseEInitFrom = this.config.phaseE?.initFrom ?? 'auto';
      let phaseEInitResolved: PhaseEInitResolved;
      const phaseEBase = (() => {
        if (phaseEInitFrom === 'phaseA') {
          phaseEInitResolved = 'phaseA';
          return phaseASnapshot;
        }
        if (phaseEInitFrom === 'phaseB') {
          if (this.config.useCountingPhase) {
            phaseEInitResolved = 'phaseB';
            return phaseBCBaseSnapshot;
          }
          phaseEInitResolved = 'phaseA';
          return phaseASnapshot;
        }
        if (phaseDSnapshot) {
          phaseEInitResolved = 'phaseD';
          return phaseDSnapshot;
        }
        phaseEInitResolved = 'phaseB';
        return phaseBCBaseSnapshot;
      })();
      this.resetColumnFromSnapshot(phaseEBase);
      if (phaseEInitFrom === 'phaseA' || (phaseEInitFrom === 'phaseB' && !this.config.useCountingPhase)) {
        this.column.learnParams.eta_attr = this.config.learningRates.phaseBC.eta_attr;
        this.column.learnParams.eta_out = this.config.learningRates.phaseBC.eta_out;
        this.column.learnParams.eta_trans = this.config.learningRates.phaseBC.eta_trans;
        this.column.applyPhysicsParams(this.config.curriculum.phaseBC);
      }
      this.column.resetPhaseBCSustainCounters();
      this.controller.resetEpisode();
      const opSchedule = this.config.phaseE?.opSchedule ?? 'uniform';
      const transitionSource = this.config.phaseE?.transitionSource ?? 'pairs';
      const walkConfig = this.config.phaseE?.walk ?? {
        episodeLen: 20,
        episodesPerBatch: 50,
        resetStrategy: { mode: 'uniform' as const },
        bias: 0.5,
      };
      const phaseEMaxSteps =
        transitionSource === 'randomWalk'
          ? Math.ceil(this.config.maxStepsPhaseC / (walkConfig.episodeLen || 1))
          : this.config.maxStepsPhaseC;
      const opBias = transitionSource === 'randomWalk' ? walkConfig.bias : 0.5;
      const phaseEResult = this.runOperatorPhase({
        phase: 'PHASE_E_JOINT',
        operatorSelector: (idx) =>
          opSchedule === 'alternate'
            ? idx % 2 === 0
              ? 'plus'
              : 'minus'
            : this.rng() < opBias
              ? 'plus'
              : 'minus',
        targetAcc: this.config.targetAccPhaseC,
        maxSteps: phaseEMaxSteps,
        includeBoundaryMetrics,
        trackOpSplit: true,
        transitionSource,
        walkConfig: {
          episodeLen: walkConfig.episodeLen,
          resetStrategy: walkConfig.resetStrategy,
        },
      });
      metrics.push(phaseEResult.metrics);
      phaseE = { ...phaseEResult.summary, opSplit: phaseEResult.opSplit, initFromResolved: phaseEInitResolved };
      phaseESnapshot = this.column.exportSnapshot();
      console.log(
        `[PhaseE Summary] initFrom=${phaseEInitFrom} resolved=${phaseEInitResolved} ` +
          `opSchedule=${opSchedule} transitionSource=${transitionSource}`,
      );
    }

    if (phaseFEnabled) {
      const evalSnapshot = phaseESnapshot ?? phaseDSnapshot ?? phaseCSnapshot;
      const splitByOp = this.config.phaseF?.splitByOp ?? true;
      const evalTransitionSource = this.config.phaseF?.evalTransitionSource ?? 'pairs';
      const walkConfig = this.config.phaseF?.walk ?? {
        episodeLen: 40,
        episodes: 100,
        resetStrategy: { mode: 'uniform' as const },
        bias: 0.5,
      };
      if (evalTransitionSource === 'pairs') {
        const runEval = (operator: Operator) => {
          this.resetColumnFromSnapshot(evalSnapshot);
          this.column.resetPhaseBCSustainCounters();
          this.controller.resetEpisode();
          const opDebugger = splitByOp ? buildTransitionDebugger(operator) : undefined;
          const result = this.runOperatorPhase({
            phase: 'PHASE_F_EVAL',
            operatorSelector: () => operator,
            targetAcc: 2,
            maxSteps: this.config.maxStepsPhaseC,
            includeBoundaryMetrics,
            trackOpSplit: true,
            transitionDebugger: opDebugger,
            policyOverrides: { applyUpdates: false, allowStopEpisode: false },
          });
          if (opDebugger) {
            const outPath = opDebugger.save();
            console.log(`[TransitionDebug] Saved transition traces (${operator}) to ${outPath}`);
          }
          return result;
        };
        const plusEval = runEval('plus');
        const minusEval = runEval('minus');
        const accPlus = plusEval.opSplit?.accPlus ?? plusEval.summary.finalAcc;
        const accMinus = minusEval.opSplit?.accMinus ?? minusEval.summary.finalAcc;
        phaseF = {
          accPlus,
          accMinus,
          overallAccMean: (accPlus + accMinus) / 2,
          confusionPlus: splitByOp ? plusEval.opSplit?.confusionPlus : undefined,
          confusionMinus: splitByOp ? minusEval.opSplit?.confusionMinus : undefined,
          boundaryAccPlus: includeBoundaryMetrics ? plusEval.opSplit?.boundaryAccPlus : undefined,
          boundaryAccMinus: includeBoundaryMetrics ? minusEval.opSplit?.boundaryAccMinus : undefined,
          sustainPlus: plusEval.summary.sustain,
          sustainMinus: minusEval.summary.sustain,
          diagnosticsPlus: plusEval.summary.diagnostics,
          diagnosticsMinus: minusEval.summary.diagnostics,
          protoDiagPlus: plusEval.summary.protoDiag,
          protoDiagMinus: minusEval.summary.protoDiag,
        };
      } else {
        this.resetColumnFromSnapshot(evalSnapshot);
        this.column.resetPhaseBCSustainCounters();
        this.controller.resetEpisode();
        const opDebuggers = splitByOp
          ? {
              plus: buildTransitionDebugger('plus'),
              minus: buildTransitionDebugger('minus'),
            }
          : undefined;
        const result = this.runOperatorPhase({
          phase: 'PHASE_F_EVAL',
          operatorSelector: () => (this.rng() < walkConfig.bias ? 'plus' : 'minus'),
          targetAcc: 2,
          maxSteps: walkConfig.episodes,
          includeBoundaryMetrics,
          trackOpSplit: true,
          transitionDebuggers: opDebuggers,
          transitionSource: 'randomWalk',
          walkConfig: {
            episodeLen: walkConfig.episodeLen,
            resetStrategy: walkConfig.resetStrategy,
          },
          policyOverrides: { applyUpdates: false, allowStopEpisode: false },
        });
        if (opDebuggers?.plus) {
          const outPath = opDebuggers.plus.save();
          console.log(`[TransitionDebug] Saved transition traces (plus) to ${outPath}`);
        }
        if (opDebuggers?.minus) {
          const outPath = opDebuggers.minus.save();
          console.log(`[TransitionDebug] Saved transition traces (minus) to ${outPath}`);
        }
        const accPlus = result.opSplit?.accPlus ?? result.summary.finalAcc;
        const accMinus = result.opSplit?.accMinus ?? result.summary.finalAcc;
        phaseF = {
          accPlus,
          accMinus,
          overallAccMean: (accPlus + accMinus) / 2,
          confusionPlus: splitByOp ? result.opSplit?.confusionPlus : undefined,
          confusionMinus: splitByOp ? result.opSplit?.confusionMinus : undefined,
          boundaryAccPlus: includeBoundaryMetrics ? result.opSplit?.boundaryAccPlus : undefined,
          boundaryAccMinus: includeBoundaryMetrics ? result.opSplit?.boundaryAccMinus : undefined,
        };
      }
    }

    const metricsForOutput = this.config.includeAccHistory
      ? metrics
      : metrics.map(({ accHistory, ...rest }) => rest);

    return {
      config: this.config,
      successPhaseA: phaseA.success,
      successPhaseB: phaseBResult.success,
      successPhaseC: phaseC.success,
      metrics: metricsForOutput,
      phaseB: this.config.useCountingPhase
        ? {
            finalAcc: phaseBResult.metrics.finalAccuracy,
            steps: phaseBResult.metrics.steps,
            accHistory: this.config.includeAccHistory ? phaseBResult.metrics.accHistory : undefined,
            confusion: phaseBResult.confusion
              ? phaseBConfusionSummary
              : undefined,
            diagnostics: phaseBResult.confusion ? phaseBDiagnostics : undefined,
            protoDiag: phaseBResult.confusion ? phaseBProtoDiagSummary : undefined,
            sustain: phaseBSustainSummary,
          }
        : undefined,
      phaseC: {
        finalAcc: phaseC.metrics.finalAccuracy,
        steps: phaseC.metrics.steps,
        accHistory: this.config.includeAccHistory ? phaseC.metrics.accHistory : undefined,
        attemptedTrials: phaseCAttempted,
        abortedTrials: phaseCAborted,
        abortedFrac: phaseCAbortedFrac,
        validTrials: phaseCValidTrials,
        confusion: phaseCConfusionSummary,
        diagnostics: phaseCDiagnostics,
        protoDiag: phaseCProtoDiagSummary,
        sustain: phaseCSustainSummary,
      },
      phaseD,
      phaseE,
      phaseF,
    };
  }

  private createProtoDebugLogger(): ((row: ProtoDebugRow) => void) | undefined {
    const cfg = this.config.protoDebug;
    if (!cfg?.enabled) return undefined;

    const limit = cfg.limit ?? 30;
    let logged = 0;

    return (row: ProtoDebugRow) => {
      if (logged >= limit) return;
      const phaseLabel = row.phase === 'PHASE_B_COUNTING' ? 'B' : 'C';
      const protoLabel = row.protoIsNull ? 'null' : `${row.protoPredDigit}`;
      const disagree = !row.protoIsNull && row.protoPredDigit !== row.readoutPredDigit;
      const parts = [
        `[proto] phase=${phaseLabel}`,
        `trial=${row.trialIdx}`,
        `target=${row.targetDigit}`,
        `readout=${row.readoutPredDigit}`,
        `proto=${protoLabel}`,
        `tailMass=${row.tailSpikeMass.toFixed(4)}`,
        `transMass=${row.transSpikeMass.toFixed(4)}`,
      ];
      if (!row.protoIsNull) {
        parts.push(`disagree=${disagree}`);
      }

      console.log(parts.join(' '));
      logged += 1;
    };
  }

  private logResolvedConfig(): void {
    const { curriculum, learningRates, randomSeed, topology, useCountingPhase, phaseASnapshotPath } = this.config;
    const snapshotPath = phaseASnapshotPath ?? 'none';
    console.log('=== Config (resolved) ===');
    console.log(`topology=${topology} counting=${useCountingPhase ? 'ON' : 'OFF'} seed=${randomSeed}`);
    console.log(`phaseASnapshot=${snapshotPath} (loaded=${this.snapshotLoaded}) columnRngReseeded=${this.snapshotLoaded}`);
    console.log(
      `phaseA physics: k_inhib=${curriculum.phaseA.k_inhib}, v_th=${curriculum.phaseA.v_th}, ` +
        `alpha=${curriculum.phaseA.alpha}, wInRowNorm=${curriculum.phaseA.wInRowNorm}, ` +
        `wAttrRowNorm=${curriculum.phaseA.wAttrRowNorm}, eta_attr=${learningRates.phaseA.eta_attr}, ` +
        `eta_out=${learningRates.phaseA.eta_out}`,
    );
    console.log(
      `phaseBC physics: k_inhib=${curriculum.phaseBC.k_inhib}, v_th=${curriculum.phaseBC.v_th}, ` +
        `alpha=${curriculum.phaseBC.alpha}, wInRowNorm=${curriculum.phaseBC.wInRowNorm}, ` +
        `wAttrRowNorm=${curriculum.phaseBC.wAttrRowNorm}, etaTrans=${curriculum.phaseBC.etaTrans}, ` +
        `eta_attr=${learningRates.phaseBC.eta_attr}, eta_out=${learningRates.phaseBC.eta_out}, ` +
        `tTrans=${curriculum.phaseBC.tTrans}, tailLen=${curriculum.phaseBC.tailLen}`,
    );
    const phaseBC = this.config.phaseBCConfig;
    if (
      phaseBC.evalWindow.includes('NoImpulse') &&
      !/(lateNoImpulse|tailNoImpulse|meanNoImpulse)\(k=\d+\)/.test(phaseBC.evalWindow)
    ) {
      console.warn(`WARNING: evalWindow looks parameterized but failed parse; raw=${phaseBC.evalWindow}`);
    }
    const tTrans = curriculum.phaseBC.tTrans ?? DEFAULT_T_TRANS;
    const tailLen = curriculum.phaseBC.tailLen ?? Math.min(DEFAULT_TAIL_LEN, tTrans);
    const windows: TransitionWindowRange[] = resolveTransitionWindows(phaseBC, tTrans, tailLen);
    const formatWindow = (name: TransitionWindowName): { label: string; range: string } =>
      describeTransitionWindow(windows, name);
    const gateCfg = phaseBC.sustainGate;
    const controller = this.config.controller;
    const controllerMode = controller.mode ?? 'standard';
    console.log(
      `[PhaseBC Resolved] strictSustainPresetApplied=${phaseBC.strictSustainPresetApplied} ` +
        `evalWindow=${formatWindow(phaseBC.evalWindow).label} learnWindow=${formatWindow(phaseBC.learnWindow).label} ` +
        `settleWindow=${formatWindow(phaseBC.settleWindow).label} ` +
        `excludeFirstK=${phaseBC.excludeFirstK} silentSpikeThreshold=${phaseBC.silentSpikeThreshold} ` +
        `sustainGate.enabled=${gateCfg.enabled} sustainGate.maxTailSilentFrac=${gateCfg.maxTailSilentFrac} ` +
        `sustainGate.minTimeToSilence=${gateCfg.minTimeToSilence ?? 0} ` +
        `sustainGate.skipUpdatesOnFail=${gateCfg.skipUpdatesOnFail} sustainGate.skipEpisodeOnFail=${gateCfg.skipEpisodeOnFail} ` +
        `sustainGate.abortAfterTrials=${gateCfg.abortAfterTrials ?? 0} logAbortLimit=${phaseBC.logAbortLimit} ` +
        `evalRange=${formatWindow(phaseBC.evalWindow).range} learnRange=${formatWindow(phaseBC.learnWindow).range} ` +
        `settleRange=${formatWindow(phaseBC.settleWindow).range}`,
    );
    if (controllerMode === 'bg' && controller.bg) {
      const minPhaseDurationEntries = Object.entries(controller.bg.minPhaseDuration ?? {})
        .map(([phase, duration]) => `${phase}:${duration}`)
        .join(',');
      const minPhaseDurationLabel = minPhaseDurationEntries.length > 0 ? minPhaseDurationEntries : 'none';
      console.log(
        `[Controller] mode=bg actions=${controller.bg.actions.join(',')} epsilon=${controller.bg.epsilon} ` +
          `temperature=${controller.bg.temperature} eta=${controller.bg.eta} waitSteps=${controller.bg.waitSteps} ` +
          `reward.correct=${controller.bg.reward.correct} reward.wrong=${controller.bg.reward.wrong} ` +
          `reward.abort=${controller.bg.reward.abort} ` +
          `minDwell=${controller.bg.minDwell} minPhaseDuration=${minPhaseDurationLabel} ` +
          `enforceOrder=${controller.bg.enforceOrder} hysteresis=${controller.bg.hysteresis}`,
      );
    } else {
      console.log(`[Controller] mode=standard`);
    }
    console.log('=========================');
  }

  private static loadSnapshot(snapshotPath: string): ColumnSnapshot {
    const resolvedPath = path.isAbsolute(snapshotPath) ? snapshotPath : artifactPath(snapshotPath);
    const raw = fs.readFileSync(resolvedPath, 'utf-8');
    return JSON.parse(raw) as ColumnSnapshot;
  }

  private runPhase(
    phase: PhaseMetrics['phase'],
    stepFn: () =>
      | number
      | { value: number; aborted?: boolean; attempted?: number; abortedTrials?: number; validTrials?: number },
    targetAcc: number,
    maxSteps: number,
  ): { success: boolean; metrics: PhaseMetrics } {
    const windowSize = 200;
    const history: number[] = [];
    const accHistory: number[] = [];
    let sum = 0;
    let attemptedTrials = 0;
    let abortedTrials = 0;
    let validTrials = 0;

    for (let step = 0; step < maxSteps; step++) {
      const rawResult = stepFn();
      const normalized =
        typeof rawResult === 'number'
          ? { value: rawResult, aborted: false, attempted: 1, abortedTrials: 0, validTrials: 1 }
          : {
              value: rawResult.value,
              aborted: rawResult.aborted ?? false,
              attempted: rawResult.attempted ?? 1,
              abortedTrials: rawResult.abortedTrials ?? (rawResult.aborted ? rawResult.attempted ?? 1 : 0),
              validTrials: rawResult.validTrials ?? (rawResult.aborted ? 0 : rawResult.attempted ?? 1),
            };

      attemptedTrials += normalized.attempted;
      abortedTrials += normalized.abortedTrials;
      validTrials += normalized.validTrials;

      const windowValue = Number.isFinite(normalized.value) ? normalized.value : 0;
      if (normalized.aborted && normalized.validTrials === 0) {
        const windowAcc = history.length > 0 ? sum / history.length : 0;
        accHistory.push(windowAcc);
        continue;
      }

      history.push(windowValue);
      sum += windowValue;
      if (history.length > windowSize) {
        sum -= history.shift() ?? 0;
      }
      const windowAcc = history.length > 0 ? sum / history.length : 0;
      accHistory.push(windowAcc);
      if (windowAcc >= targetAcc && history.length >= windowSize / 2) {
        return {
          success: true,
          metrics: {
            phase,
            steps: step + 1,
            accHistory,
            finalAccuracy: windowAcc,
            attemptedTrials,
            abortedTrials,
            validTrials,
          },
        };
      }
    }

    const finalAccuracy = history.length > 0 ? sum / history.length : 0;
    return {
      success: finalAccuracy >= targetAcc,
      metrics: { phase, steps: maxSteps, accHistory, finalAccuracy, attemptedTrials, abortedTrials, validTrials },
    };
  }

  private resetColumnFromSnapshot(snapshot: ColumnSnapshot): void {
    this.column = MathColumn.fromSnapshot(snapshot);
    this.column.setRngSeed(this.config.randomSeed);
    this.column.setPhaseBCConfig(this.config.phaseBCConfig);
  }

  private runOperatorPhase(params: {
    phase: PhaseMetrics['phase'];
    operatorSelector: (trialIdx: number) => Operator;
    targetAcc: number;
    maxSteps: number;
    includeBoundaryMetrics: boolean;
    trackOpSplit: boolean;
    transitionDebugger?: TransitionDebugger;
    transitionDebuggers?: Partial<Record<Operator, TransitionDebugger>>;
    transitionSource?: 'pairs' | 'randomWalk';
    walkConfig?: {
      episodeLen: number;
      resetStrategy: { mode: 'uniform' | 'cycle' | 'fixed'; digit?: number };
    };
    policyOverrides?: { applyUpdates?: boolean; allowStopEpisode?: boolean };
  }): { success: boolean; metrics: PhaseMetrics; summary: PhaseSummary; opSplit?: PhaseOpSplitSummary } {
    const numDigits = this.column.net.numDigits;
    const phaseConfusion = makeConfusionMatrix(numDigits);
    const phaseProtoConfusion = makeConfusionMatrix(numDigits);
    const spikeTail: number[] = [];
    const spikeTrans: number[] = [];
    const tailSilent: number[] = [];
    const timeToSilence: number[] = [];
    const phaseWindowAcc: WindowAccuracy = {};
    let disagreeCount = 0;
    let protoTrials = 0;
    const protoDiag = this.createProtoDiagAccumulator();
    let trialIdx = 0;
    const gateCountersBefore = this.column.getPhaseBCSustainCounters();

    const initOpCounters = (): OpCounters => ({
      correct: 0,
      total: 0,
      boundaryCorrect: 0,
      boundaryTotal: 0,
      confusion: makeConfusionMatrix(numDigits),
      protoConfusion: makeConfusionMatrix(numDigits),
    });
    const opCounters: Record<Operator, OpCounters> | undefined = params.trackOpSplit
      ? { plus: initOpCounters(), minus: initOpCounters() }
      : undefined;

    const transitionSource = params.transitionSource ?? 'pairs';
    const resolveDebugger = (operator: Operator): TransitionDebugger | undefined =>
      params.transitionDebuggers?.[operator] ?? params.transitionDebugger;
    const runTransition = (digit: number, operator: Operator) => {
      const phaseDebugCb = undefined;
      const trial = this.column.runTransitionTrial(
        digit,
        operator,
        trialIdx,
        {
          recordPrediction: (target, pred) => {
            addConfusion(phaseConfusion, target, pred);
            if (opCounters) addConfusion(opCounters[operator].confusion, target, pred);
          },
          recordProtoPrediction: (target, pred) => {
            addConfusion(phaseProtoConfusion, target, pred);
            if (opCounters) addConfusion(opCounters[operator].protoConfusion, target, pred);
          },
          recordTrialStats: ({
            target,
            readoutPred,
            protoPred,
            tailSpikeMass,
            transSpikeMass,
            sustain,
            predictionsByWindow,
            aborted,
            startDigit,
          }) => {
            spikeTail.push(tailSpikeMass);
            spikeTrans.push(transSpikeMass);
            if (sustain) {
              tailSilent.push(sustain.tailSilentFrac);
              timeToSilence.push(sustain.timeToSilence);
            }
            if (!aborted) {
              if (protoPred !== null) {
                protoTrials += 1;
                if (protoPred !== readoutPred) disagreeCount += 1;
              }
              this.updateProtoDiagCounters(protoDiag, {
                protoPred,
                readoutPred,
                tailSpikeMass,
                transSpikeMass,
              });
              if (predictionsByWindow) {
                this.updateWindowAccuracy(phaseWindowAcc, predictionsByWindow, target);
              }
              if (opCounters) {
                const counters = opCounters[operator];
                counters.total += 1;
                if (readoutPred === target) counters.correct += 1;
                if (params.includeBoundaryMetrics && isBoundaryStartDigit(startDigit, operator, numDigits)) {
                  counters.boundaryTotal += 1;
                  if (readoutPred === target) counters.boundaryCorrect += 1;
                }
              }
            }
          },
        },
        phaseDebugCb,
        resolveDebugger(operator),
        this.controller,
        params.policyOverrides,
      );
      trialIdx += 1;
      return trial;
    };
    const walkCfg = params.walkConfig ?? { episodeLen: 1, resetStrategy: { mode: 'uniform' as const } };
    let cycleIdx = 0;
    const pickStartDigit = () => {
      if (walkCfg.resetStrategy.mode === 'fixed') {
        return walkCfg.resetStrategy.digit ?? 0;
      }
      if (walkCfg.resetStrategy.mode === 'cycle') {
        const digit = cycleIdx % numDigits;
        cycleIdx += 1;
        return digit;
      }
      return Math.floor(this.rng() * numDigits);
    };
    const phaseResult = this.runPhase(
      params.phase,
      () => {
        if (transitionSource === 'pairs') {
          const digit = Math.floor(this.rng() * this.column.net.numDigits);
          const operator = params.operatorSelector(trialIdx);
          const trial = runTransition(digit, operator);
          return { value: trial.correct ? 1 : 0, aborted: trial.aborted };
        }
        let curr = pickStartDigit();
        let attempted = 0;
        let abortedTrials = 0;
        let validTrials = 0;
        let correct = 0;
        for (let step = 0; step < walkCfg.episodeLen; step++) {
          const operator = params.operatorSelector(trialIdx);
          const trial = runTransition(curr, operator);
          attempted += 1;
          if (trial.aborted) {
            abortedTrials += 1;
          } else {
            validTrials += 1;
            if (trial.correct) correct += 1;
          }
          curr = trial.targetDigit;
        }
        const value = validTrials > 0 ? correct / validTrials : 0;
        return { value, attempted, abortedTrials, validTrials };
      },
      params.targetAcc,
      params.maxSteps,
    );

    const phaseConfusionSummary = this.buildConfusionSummary(phaseConfusion, phaseProtoConfusion);
    const phaseDiagnostics = this.buildDiagnostics(disagreeCount, protoTrials, spikeTail, spikeTrans);
    const phaseProtoDiagSummary = this.finalizeProtoDiagCounters(protoDiag);
    const gateCountersAfter = this.column.getPhaseBCSustainCounters();
    const phaseGate: SustainCounters = {
      transitions: gateCountersAfter.transitions - gateCountersBefore.transitions,
      gateFails: gateCountersAfter.gateFails - gateCountersBefore.gateFails,
      updatesSkipped: gateCountersAfter.updatesSkipped - gateCountersBefore.updatesSkipped,
      tailSilentFracSum:
        (gateCountersAfter.tailSilentFracSum ?? 0) - (gateCountersBefore.tailSilentFracSum ?? 0),
      timeToSilenceSum: (gateCountersAfter.timeToSilenceSum ?? 0) - (gateCountersBefore.timeToSilenceSum ?? 0),
    };
    const phaseSustainSummary = this.buildSustainSummary(
      tailSilent,
      timeToSilence,
      spikeTrans,
      spikeTail,
      phaseGate,
    );

    const summary: PhaseSummary = {
      finalAcc: phaseResult.metrics.finalAccuracy,
      steps: phaseResult.metrics.steps,
      accHistory: this.config.includeAccHistory ? phaseResult.metrics.accHistory : undefined,
      attemptedTrials: phaseResult.metrics.attemptedTrials,
      abortedTrials: phaseResult.metrics.abortedTrials,
      abortedFrac:
        phaseResult.metrics.attemptedTrials && phaseResult.metrics.abortedTrials !== undefined
          ? phaseResult.metrics.abortedTrials / phaseResult.metrics.attemptedTrials
          : undefined,
      validTrials: phaseResult.metrics.validTrials,
      confusion: phaseConfusionSummary,
      diagnostics: phaseDiagnostics,
      protoDiag: phaseProtoDiagSummary,
      sustain: phaseSustainSummary,
    };

    let opSplit: PhaseOpSplitSummary | undefined;
    if (opCounters) {
      const accPlus = opCounters.plus.total > 0 ? opCounters.plus.correct / opCounters.plus.total : 0;
      const accMinus = opCounters.minus.total > 0 ? opCounters.minus.correct / opCounters.minus.total : 0;
      const boundaryAccPlus =
        opCounters.plus.boundaryTotal > 0 ? opCounters.plus.boundaryCorrect / opCounters.plus.boundaryTotal : 0;
      const boundaryAccMinus =
        opCounters.minus.boundaryTotal > 0 ? opCounters.minus.boundaryCorrect / opCounters.minus.boundaryTotal : 0;
      opSplit = {
        accPlus,
        accMinus,
        confusionPlus: this.buildConfusionSummary(opCounters.plus.confusion, opCounters.plus.protoConfusion),
        confusionMinus: this.buildConfusionSummary(opCounters.minus.confusion, opCounters.minus.protoConfusion),
        boundaryAccPlus: params.includeBoundaryMetrics ? boundaryAccPlus : undefined,
        boundaryAccMinus: params.includeBoundaryMetrics ? boundaryAccMinus : undefined,
      };
    }

    return { success: phaseResult.success, metrics: phaseResult.metrics, summary, opSplit };
  }

  private createProtoDiagAccumulator(): ProtoDiagAccumulator {
    return {
      totalTrials: 0,
      protoPredNonNull: 0,
      protoPredNull: 0,
      tailSilent: 0,
      tailNonSilent: 0,
      tailSpikeMassSum: 0,
      transSpikeMassSum: 0,
      protoVsReadoutDisagreeCount: 0,
    };
  }

  private updateProtoDiagCounters(
    acc: ProtoDiagAccumulator,
    stats: { protoPred: number | null; readoutPred: number; tailSpikeMass: number; transSpikeMass: number },
  ): void {
    const eps = this.config.phaseBCConfig.silentSpikeThreshold ?? DEFAULT_SILENT_SPIKE_THRESHOLD;
    acc.totalTrials += 1;
    acc.tailSpikeMassSum += stats.tailSpikeMass;
    acc.transSpikeMassSum += stats.transSpikeMass;

    if (stats.tailSpikeMass <= eps) {
      acc.tailSilent += 1;
    } else {
      acc.tailNonSilent += 1;
    }

    if (stats.protoPred === null) {
      acc.protoPredNull += 1;
      return;
    }

    acc.protoPredNonNull += 1;
    if (stats.protoPred !== stats.readoutPred) {
      acc.protoVsReadoutDisagreeCount += 1;
    }
  }

  private updateWindowAccuracy(acc: WindowAccuracy, predictions: Record<string, number>, target: number): void {
    Object.entries(predictions).forEach(([window, pred]) => {
      if (!acc[window]) acc[window] = { correct: 0, total: 0 };
      acc[window].total += 1;
      if (pred === target) acc[window].correct += 1;
    });
  }

  private finalizeProtoDiagCounters(acc: ProtoDiagAccumulator): ProtoDiagCounters {
    const hasTrials = acc.totalTrials > 0;
    return {
      totalTrials: acc.totalTrials,
      protoPredNonNull: acc.protoPredNonNull,
      protoPredNull: acc.protoPredNull,
      tailSilent: acc.tailSilent,
      tailNonSilent: acc.tailNonSilent,
      avgTailSpikeMass: hasTrials ? acc.tailSpikeMassSum / acc.totalTrials : 0,
      avgTransSpikeMass: hasTrials ? acc.transSpikeMassSum / acc.totalTrials : 0,
      protoVsReadoutDisagreeCount: acc.protoVsReadoutDisagreeCount,
    };
  }

  private buildSustainSummary(
    tailSilentFrac: number[],
    timeToSilence: number[],
    transSpikeMass: number[],
    tailSpikeMass: number[],
    gate?: SustainCounters,
  ): PhaseSustainSummary {
    return {
      tailSilentFracMean: this.computeMean(tailSilentFrac),
      timeToSilenceMean: this.computeMean(timeToSilence),
      lateSpikeMassMean: this.computeMean(transSpikeMass),
      tailSpikeMassMean: this.computeMean(tailSpikeMass),
      gate,
    };
  }

  private computeMean(values: number[]): number {
    if (values.length === 0) return 0;
    return values.reduce((acc, v) => acc + v, 0) / values.length;
  }

  private buildConfusionSummary(readout: number[][], proto: number[][]): PhaseConfusionSummary {
    return {
      readout: { counts: readout, normalized: normalizeConfusion(readout) },
      proto: { counts: proto, normalized: normalizeConfusion(proto) },
    };
  }

  private buildDiagnostics(
    disagreeCount: number,
    protoTrials: number,
    tailMasses: number[],
    transMasses: number[],
  ): PhaseDiagnostics {
    const summarize = (values: number[]): SpikeMassStats => {
      if (values.length === 0) {
        return { mean: 0, min: 0, max: 0, silentFrac: 0 };
      }
      let min = Number.POSITIVE_INFINITY;
      let max = Number.NEGATIVE_INFINITY;
      let sum = 0;
      let silent = 0;
      const eps = this.config.phaseBCConfig.silentSpikeThreshold ?? DEFAULT_SILENT_SPIKE_THRESHOLD;
      for (const v of values) {
        if (v < min) min = v;
        if (v > max) max = v;
        sum += v;
        if (v <= eps) silent += 1;
      }
      return { mean: sum / values.length, min, max, silentFrac: silent / values.length };
    };

    return {
      disagreeRate: protoTrials > 0 ? disagreeCount / protoTrials : 0,
      spikeMass: {
        tail: summarize(tailMasses),
        trans: transMasses.length > 0 ? summarize(transMasses) : undefined,
      },
    };
  }

  private validateTransitionWindow(): void {
    const tTrans = this.config.curriculum.phaseBC.tTrans ?? DEFAULT_T_TRANS;
    const tailLenFallback = Math.min(DEFAULT_TAIL_LEN, tTrans);
    const tailLen = this.config.curriculum.phaseBC.tailLen ?? tailLenFallback;

    const isValidNumber = (value: number) => Number.isFinite(value) && Number.isInteger(value);
    if (!isValidNumber(tTrans) || tTrans < 5) {
      throw new Error(`[Config] phaseBC.tTrans must be an integer >= 5. Received: ${tTrans}`);
    }
    if (!isValidNumber(tailLen) || tailLen <= 0) {
      throw new Error(`[Config] phaseBC.tailLen must be a positive integer. Received: ${tailLen}`);
    }
    if (tailLen > tTrans) {
      throw new Error(
        `[Config] phaseBC.tailLen (${tailLen}) must be less than or equal to phaseBC.tTrans (${tTrans}).`,
      );
    }

    this.config.curriculum.phaseBC.tTrans = tTrans;
    this.config.curriculum.phaseBC.tailLen = tailLen;
  }
}

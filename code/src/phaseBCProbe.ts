import fs from 'fs';
import path from 'path';

import { artifactPath, exportToSharedArtifacts, resolveArtifactPath } from './artifactPaths';
import { MathColumn, ColumnSnapshot } from './mathColumn';
import * as MathOps from './mathHelpers';
import { ControllerStats, controllerMode, createController } from './controller';
import { describeTransitionWindow, resolveTransitionWindows, TransitionWindowRange } from './transitionWindows';
import {
  DEFAULT_CURRICULUM,
  DEFAULT_TAIL_LEN,
  DEFAULT_T_TRANS,
  ExperimentConfig,
  PhasePhysicsParams,
  Topology,
  TransitionWindowName,
} from './types';

type ProbeConfig = {
  trials: number;
  maxCandidates: number;
  rankTop: number;
  wSustain: number;
  wAcc: number;
  seed: number;
  experiments: Record<Topology, ExperimentConfig>;
};

type ProbeCandidateResult = {
  sampleOrder: number;
  params: PhasePhysicsParams;
  accuracy: number;
  sustainScore: number;
  tailSilentFrac: number;
  tailSpikeMass: number;
  transSpikeMass: number;
  abortedFrac: number;
  totalTrials: number;
  evaluatedTrials: number;
  abortedTrials: number;
  score: number;
  columnSeed: number;
  controllerSeed: number;
  startDigitSeed?: number;
  startDigits: number[];
  controllerMode: ReturnType<typeof controllerMode>;
  controllerStats?: ControllerStats;
};

export type TrialStartContext = {
  startDigits?: number[];
  startDigitSeed?: number;
};

const paramsSortKey = (params: PhasePhysicsParams): string =>
  [
    params.k_inhib,
    params.v_th,
    params.alpha,
    params.wInRowNorm,
    params.wAttrRowNorm,
    params.etaTrans,
    params.tTrans,
    params.tailLen,
  ].join('|');

const compareProbeResults = (a: ProbeCandidateResult, b: ProbeCandidateResult): number => {
  if (b.score !== a.score) return b.score - a.score;
  if (a.columnSeed !== b.columnSeed) return a.columnSeed - b.columnSeed;
  if (a.controllerSeed !== b.controllerSeed) return a.controllerSeed - b.controllerSeed;
  const aKey = paramsSortKey(a.params);
  const bKey = paramsSortKey(b.params);
  return aKey.localeCompare(bKey);
};

const loadSnapshot = (config: ExperimentConfig): ColumnSnapshot => {
  const explicit = config.phaseASnapshotPath
    ? path.isAbsolute(config.phaseASnapshotPath)
      ? config.phaseASnapshotPath
      : artifactPath(config.phaseASnapshotPath)
    : undefined;
  const snapshotPath =
    explicit ?? resolveArtifactPath({ key: 'phaseA_state', topology: config.topology, required: true });
  if (!snapshotPath || !fs.existsSync(snapshotPath)) {
    throw new Error(`Missing Phase A snapshot for ${config.topology} at ${snapshotPath}`);
  }
  const raw = fs.readFileSync(snapshotPath, 'utf-8');
  return JSON.parse(raw) as ColumnSnapshot;
};

const cloneExperimentConfig = (config: ExperimentConfig): ExperimentConfig => ({
  ...config,
  curriculum: {
    phaseA: { ...config.curriculum.phaseA },
    phaseBC: { ...config.curriculum.phaseBC },
  },
  learningRates: {
    phaseA: { ...config.learningRates.phaseA },
    phaseBC: { ...config.learningRates.phaseBC },
  },
  phaseBCConfig: {
    ...config.phaseBCConfig,
    sustainGate: { ...config.phaseBCConfig.sustainGate },
    tuning: { ...config.phaseBCConfig.tuning },
  },
  controller: {
    ...config.controller,
    bg: config.controller.bg ? { ...config.controller.bg } : undefined,
  },
  protoDebug: config.protoDebug ? { ...config.protoDebug } : undefined,
  transitionDebug: config.transitionDebug
    ? {
        ...config.transitionDebug,
        windows: [...config.transitionDebug.windows],
        windowDefs: [...config.transitionDebug.windowDefs],
        excludeFirst: [...config.transitionDebug.excludeFirst],
      }
    : undefined,
});

const applyCandidatePhysics = (config: ExperimentConfig, params: PhasePhysicsParams): void => {
  config.curriculum.phaseBC = {
    ...config.curriculum.phaseBC,
    k_inhib: params.k_inhib,
    v_th: params.v_th,
    alpha: params.alpha,
    wInRowNorm: params.wInRowNorm,
    wAttrRowNorm: params.wAttrRowNorm,
    etaTrans: params.etaTrans,
  };
};

const logResolvedPhaseBCConfig = (config: ExperimentConfig): void => {
  const { curriculum, phaseBCConfig } = config;
  const tTrans = curriculum.phaseBC.tTrans ?? DEFAULT_T_TRANS;
  const tailLen = curriculum.phaseBC.tailLen ?? Math.min(DEFAULT_TAIL_LEN, tTrans);
  const windows = resolveTransitionWindows(phaseBCConfig, tTrans, tailLen);
  const formatWindow = (name: TransitionWindowName) => describeTransitionWindow(windows, name);
  const gateCfg = phaseBCConfig.sustainGate;
  console.log(
    `phaseBC physics: k_inhib=${curriculum.phaseBC.k_inhib}, v_th=${curriculum.phaseBC.v_th}, ` +
      `alpha=${curriculum.phaseBC.alpha}, wInRowNorm=${curriculum.phaseBC.wInRowNorm}, ` +
      `wAttrRowNorm=${curriculum.phaseBC.wAttrRowNorm}, etaTrans=${curriculum.phaseBC.etaTrans}, ` +
      `eta_attr=${config.learningRates.phaseBC.eta_attr}, eta_out=${config.learningRates.phaseBC.eta_out}, ` +
      `tTrans=${tTrans}, tailLen=${tailLen}`,
  );
  console.log(
    `[PhaseBC Resolved] strictSustainPresetApplied=${phaseBCConfig.strictSustainPresetApplied} ` +
      `evalWindow=${formatWindow(phaseBCConfig.evalWindow).label} learnWindow=${formatWindow(phaseBCConfig.learnWindow).label} ` +
      `settleWindow=${formatWindow(phaseBCConfig.settleWindow).label} ` +
      `excludeFirstK=${phaseBCConfig.excludeFirstK} silentSpikeThreshold=${phaseBCConfig.silentSpikeThreshold} ` +
      `sustainGate.enabled=${gateCfg.enabled} sustainGate.maxTailSilentFrac=${gateCfg.maxTailSilentFrac} ` +
      `sustainGate.minTimeToSilence=${gateCfg.minTimeToSilence ?? 0} ` +
      `sustainGate.skipUpdatesOnFail=${gateCfg.skipUpdatesOnFail} sustainGate.skipEpisodeOnFail=${gateCfg.skipEpisodeOnFail} ` +
      `sustainGate.abortAfterTrials=${gateCfg.abortAfterTrials ?? 0} logAbortLimit=${phaseBCConfig.logAbortLimit} ` +
      `evalRange=${formatWindow(phaseBCConfig.evalWindow).range} learnRange=${formatWindow(phaseBCConfig.learnWindow).range} ` +
      `settleRange=${formatWindow(phaseBCConfig.settleWindow).range}`,
  );
};

const generateCandidateGrid = (topology: Topology): PhasePhysicsParams[] => {
  const wAttrRowNorm = DEFAULT_CURRICULUM.phaseBC.wAttrRowNorm;
  const tTrans = DEFAULT_T_TRANS;
  const tailLen = DEFAULT_TAIL_LEN;

  if (topology === 'ring') {
    const k_inhib = [1, 2];
    const v_th = [0.8, 1.0];
    const alpha = [0.9, 0.95];
    const wInRowNorm = [3.0, 4.0];
    const etaTrans = [0.005, 0.01];

    const grid: PhasePhysicsParams[] = [];
    for (const k of k_inhib) {
      for (const v of v_th) {
        for (const a of alpha) {
          for (const wIn of wInRowNorm) {
            for (const eta of etaTrans) {
              grid.push({
                k_inhib: k,
                v_th: v,
                alpha: a,
                wInRowNorm: wIn,
                wAttrRowNorm,
                etaTrans: eta,
                tTrans,
                tailLen,
              });
            }
          }
        }
      }
    }
    return grid;
  }

  const k_inhib = [1, 2, 3, 4];
  const v_th = [0.6, 0.8, 1.0, 1.2];
  const alpha = [0.85, 0.9, 0.95, 0.98];
  const wInRowNorm = [3.0, 3.5, 4.0, 4.5, 5.0];
  const etaTrans = [0.005, 0.01, 0.02];

  const grid: PhasePhysicsParams[] = [];
  for (const k of k_inhib) {
    for (const v of v_th) {
      for (const a of alpha) {
        for (const wIn of wInRowNorm) {
          for (const eta of etaTrans) {
            grid.push({
              k_inhib: k,
              v_th: v,
              alpha: a,
              wInRowNorm: wIn,
              wAttrRowNorm,
              etaTrans: eta,
              tTrans,
              tailLen,
            });
          }
        }
      }
    }
  }
  return grid;
};

const sampleCandidates = (
  candidates: PhasePhysicsParams[],
  maxCandidates: number,
  rng: MathOps.RNG,
): PhasePhysicsParams[] => {
  if (candidates.length <= maxCandidates) return candidates;
  const shuffled = [...candidates];
  for (let i = shuffled.length - 1; i > 0; i--) {
    const j = Math.floor(rng() * (i + 1));
    [shuffled[i], shuffled[j]] = [shuffled[j], shuffled[i]];
  }
  return shuffled.slice(0, maxCandidates);
};

const evaluateCandidate = (
  snapshot: ColumnSnapshot,
  params: PhasePhysicsParams,
  config: ProbeConfig,
  baseExperiment: ExperimentConfig,
  rng: MathOps.RNG,
  sampleOrder: number,
  trialStartContext?: TrialStartContext,
  seedOverrides?: { column?: number; controller?: number },
): ProbeCandidateResult => {
  const col = MathColumn.fromSnapshot(snapshot);
  const candidateSeed = seedOverrides?.column ?? Math.floor(rng() * 1_000_000);
  col.setRngSeed(candidateSeed);
  const experimentConfig = cloneExperimentConfig(baseExperiment);
  const phasePhysics: PhasePhysicsParams = {
    ...params,
    tTrans: experimentConfig.curriculum.phaseBC.tTrans,
    tailLen: experimentConfig.curriculum.phaseBC.tailLen,
  };
  applyCandidatePhysics(experimentConfig, phasePhysics);
  col.setPhaseBCConfig(experimentConfig.phaseBCConfig);
  col.applyPhysicsParams(experimentConfig.curriculum.phaseBC);
  col.learnParams.eta_trans = 0;
  col.learnParams.eta_out = 0;
  col.learnParams.eta_attr = 0;
  const controllerSeed = seedOverrides?.controller ?? candidateSeed + 1;
  const controllerRng = MathOps.createRng(controllerSeed);
  const controller = createController(experimentConfig.controller, experimentConfig.phaseBCConfig, controllerRng);
  const mode = controllerMode(experimentConfig.controller);
  if (mode === 'bg' && experimentConfig.controller.bg) {
    const bg = experimentConfig.controller.bg;
    console.log(
      `[Probe Controller] sample=${sampleOrder} mode=bg actions=${bg.actions.join(',')} ` +
        `epsilon=${bg.epsilon} temperature=${bg.temperature} waitSteps=${bg.waitSteps} ` +
        `reward(correct=${bg.reward.correct},wrong=${bg.reward.wrong}) eta=${bg.eta} sampleActions=${bg.sampleActions}`,
    );
  } else {
    console.log(`[Probe Controller] sample=${sampleOrder} mode=${mode}`);
  }
  controller.resetEpisode();

  let evaluatedTrials = 0;
  let correct = 0;
  let tailSilentSum = 0;
  let tailSpikeSum = 0;
  let transSpikeSum = 0;
  let aborted = 0;

  const numDigits = col.net.numDigits;

  const startDigits: number[] = trialStartContext?.startDigits
    ? trialStartContext.startDigits.slice(0, config.trials)
    : [];
  const trialStartSeed = trialStartContext?.startDigitSeed;
  const trialStartRng =
    trialStartSeed !== undefined ? MathOps.createRng(trialStartSeed) : undefined;

  if (trialStartContext?.startDigits && trialStartContext.startDigits.length < config.trials) {
    throw new Error(
      `[Probe] Provided startDigits length (${trialStartContext.startDigits.length}) is less than trials (${config.trials}).`,
    );
  }

  for (let i = 0; i < config.trials; i++) {
    let startDigit: number;
    if (trialStartContext?.startDigits) {
      startDigit = trialStartContext.startDigits[i];
    } else if (trialStartRng) {
      startDigit = Math.floor(trialStartRng() * numDigits);
      startDigits.push(startDigit);
    } else {
      startDigit = Math.floor(rng() * numDigits);
      startDigits.push(startDigit);
    }
    const trial = col.runSuccessorTrial(startDigit, i, undefined, undefined, undefined, controller);
    if (trial.aborted) {
      aborted += 1;
      tailSilentSum += trial.sustain?.tailSilentFrac ?? 1;
      continue;
    }
    evaluatedTrials += 1;
    tailSilentSum += trial.sustain?.tailSilentFrac ?? 1;
    tailSpikeSum += trial.tailSpikeMass ?? 0;
    transSpikeSum += trial.transSpikeMass ?? 0;
    const pred = trial.predictionsByWindow?.eval ?? trial.predDigit;
    if (pred === trial.targetDigit) correct += 1;
  }

  const denom = evaluatedTrials > 0 ? evaluatedTrials : 1;
  const tailSilentFrac = tailSilentSum / config.trials;
  const sustainScore = 1 - tailSilentFrac;
  const accuracy = correct / denom;
  const score = config.wSustain * sustainScore + config.wAcc * accuracy;
  const controllerStats = controller.getStats();
  return {
    params,
    accuracy,
    sustainScore,
    tailSilentFrac,
    tailSpikeMass: tailSpikeSum / denom,
    transSpikeMass: transSpikeSum / denom,
    abortedFrac: aborted / config.trials,
    totalTrials: config.trials,
    evaluatedTrials,
    abortedTrials: aborted,
    score,
    sampleOrder,
    columnSeed: candidateSeed,
    controllerSeed,
    controllerMode: mode,
    controllerStats,
    startDigitSeed: trialStartSeed,
    startDigits,
  };
};

const formatParams = (params: PhasePhysicsParams): string =>
  `k_inhib=${params.k_inhib} v_th=${params.v_th} alpha=${params.alpha} wIn=${params.wInRowNorm} etaTrans=${params.etaTrans}`;

export type ProbeReplayCandidate = {
  sampleOrder?: number;
  physics: PhasePhysicsParams;
  seeds?: { column?: number; controller?: number };
  trialStarts?: TrialStartContext;
};

export const replayProbeCandidate = (
  snapshot: ColumnSnapshot,
  candidate: ProbeReplayCandidate,
  config: ProbeConfig,
  baseExperiment: ExperimentConfig,
): ProbeCandidateResult =>
  evaluateCandidate(
    snapshot,
    candidate.physics,
    config,
    baseExperiment,
    MathOps.createRng(config.seed),
    candidate.sampleOrder ?? 0,
    candidate.trialStarts,
    candidate.seeds,
  );

const logTopResults = (
  topology: Topology,
  sortedResults: ProbeCandidateResult[],
  rankTop: number,
  trials: number,
): void => {
  console.log(`[Probe] Top ${Math.min(rankTop, sortedResults.length)} candidates for ${topology}:`);
  sortedResults
    .slice(0, rankTop)
    .forEach((res, idx) => {
      console.log(
        `  #${idx + 1} score=${res.score.toFixed(3)} sustain=${res.sustainScore.toFixed(3)} acc=${res.accuracy.toFixed(3)} ` +
          `tailSilent=${res.tailSilentFrac.toFixed(3)} tailSpike=${res.tailSpikeMass.toFixed(3)} transSpike=${res.transSpikeMass.toFixed(3)} ` +
          `aborted=${(res.abortedFrac * 100).toFixed(1)}% trials={total:${res.totalTrials} eval:${res.evaluatedTrials} aborted:${res.abortedTrials}} ` +
          `seeds={col:${res.columnSeed} ctrl:${res.controllerSeed}} params={${formatParams(res.params)}}`,
      );
    });
  const best = sortedResults[0];
  if (best) {
    console.log(
      `[Probe] Best for ${topology}: score=${best.score.toFixed(3)} sustain=${best.sustainScore.toFixed(3)} acc=${best.accuracy.toFixed(3)} ` +
        `over ${trials} trials (eval=${best.evaluatedTrials}, aborted=${best.abortedTrials})`,
    );
  }
};

const buildResolvedProbeFields = (experimentConfig: ExperimentConfig) => {
  const tTrans = experimentConfig.curriculum.phaseBC.tTrans ?? DEFAULT_T_TRANS;
  const tailLen = Math.min(experimentConfig.curriculum.phaseBC.tailLen ?? DEFAULT_TAIL_LEN, tTrans);
  return {
    evalWindow: experimentConfig.phaseBCConfig.evalWindow,
    learnWindow: experimentConfig.phaseBCConfig.learnWindow,
    settleWindow: experimentConfig.phaseBCConfig.settleWindow,
    tTrans,
    tailLen,
    activityMode: experimentConfig.curriculum.phaseBC.activityMode,
    activityAlpha: experimentConfig.curriculum.phaseBC.activityAlpha,
  };
};

const writeProbeArtifacts = (
  topology: Topology,
  sortedResults: ProbeCandidateResult[],
  sampledResults: ProbeCandidateResult[],
  experimentConfig: ExperimentConfig,
  probeConfig: ProbeConfig,
): void => {
  if (sortedResults.length === 0) return;

  const best = sortedResults[0];
  const resolved = buildResolvedProbeFields(experimentConfig);
  const physics = best.params;
  const metrics = {
    score: best.score,
    sustain: best.sustainScore,
    acc: best.accuracy,
    tailSilent: best.tailSilentFrac,
    tailSpike: best.tailSpikeMass,
    transSpike: best.transSpikeMass,
    abortedFrac: best.abortedFrac,
    trials: best.totalTrials,
    trialsEvaluated: best.evaluatedTrials,
    trialsAborted: best.abortedTrials,
  };

  const bestPayload = {
    version: 1,
    topology,
    probe: {
      seed: probeConfig.seed,
      maxCandidates: probeConfig.maxCandidates,
      trialsPerCandidate: probeConfig.trials,
      rankTop: probeConfig.rankTop,
      wSustain: probeConfig.wSustain,
      wAcc: probeConfig.wAcc,
    },
    resolved,
    physics,
    metrics,
    controller: { mode: best.controllerMode, stats: best.controllerStats },
    seeds: { column: best.columnSeed, controller: best.controllerSeed },
    trialStarts: { seed: best.startDigitSeed, digits: best.startDigits },
    sampledCandidates: sampledResults
      .slice()
      .sort((a, b) => a.sampleOrder - b.sampleOrder)
      .map((entry) => ({
        sampleOrder: entry.sampleOrder,
        physics: entry.params,
        controller: { mode: entry.controllerMode, stats: entry.controllerStats },
        seeds: { column: entry.columnSeed, controller: entry.controllerSeed },
        trialStarts: { seed: entry.startDigitSeed, digits: entry.startDigits },
      })),
  };
    const bestFilename = `probeBC_best_${topology}.json`;
    const bestPath = artifactPath(bestFilename);
    fs.writeFileSync(bestPath, JSON.stringify(bestPayload, null, 2));
    exportToSharedArtifacts(bestFilename);

  const topKPayload = {
    version: 1,
    topology,
    probe: bestPayload.probe,
    resolved,
    candidates: sortedResults.slice(0, probeConfig.rankTop).map((entry) => ({
      physics: entry.params,
      metrics: {
        score: entry.score,
        sustain: entry.sustainScore,
        acc: entry.accuracy,
        tailSilent: entry.tailSilentFrac,
        tailSpike: entry.tailSpikeMass,
        transSpike: entry.transSpikeMass,
        abortedFrac: entry.abortedFrac,
        trials: entry.totalTrials,
        trialsEvaluated: entry.evaluatedTrials,
        trialsAborted: entry.abortedTrials,
      },
      controller: { mode: entry.controllerMode, stats: entry.controllerStats },
      seeds: { column: entry.columnSeed, controller: entry.controllerSeed },
      trialStarts: { seed: entry.startDigitSeed, digits: entry.startDigits },
    })),
    sampledCandidates: sampledResults
      .slice()
      .sort((a, b) => a.sampleOrder - b.sampleOrder)
      .map((entry) => ({
        sampleOrder: entry.sampleOrder,
        physics: entry.params,
        controller: { mode: entry.controllerMode, stats: entry.controllerStats },
        seeds: { column: entry.columnSeed, controller: entry.controllerSeed },
        trialStarts: { seed: entry.startDigitSeed, digits: entry.startDigits },
      })),
  };
    const topKFilename = `probeBC_topK_${topology}.json`;
    const topKPath = artifactPath(topKFilename);
    fs.writeFileSync(topKPath, JSON.stringify(topKPayload, null, 2));
    exportToSharedArtifacts(topKFilename);

  console.log(`[Probe] Artifacts written: best=${bestPath}, topK=${topKPath}`);
};

export function runPhaseBCProbe(config: ProbeConfig): void {
  const rng = MathOps.createRng(config.seed);
  const topologies: Topology[] = ['snake', 'ring'];

  for (const topology of topologies) {
    const experimentConfig = config.experiments[topology];
    const snapshot = loadSnapshot(experimentConfig);
    const grid = generateCandidateGrid(topology);
    const sampled = sampleCandidates(grid, config.maxCandidates, rng);
    console.log(
      `[Probe] ${topology}: params maxCandidates=${config.maxCandidates} trialsPerCandidate=${config.trials} ` +
        `rankTop=${config.rankTop} wSustain=${config.wSustain} wAcc=${config.wAcc} seed=${config.seed}`,
    );
    logResolvedPhaseBCConfig(experimentConfig);
    console.log(
      `[Probe] ${topology}: evaluating ${sampled.length}/${grid.length} candidates | trials=${config.trials} | evalWindow=${experimentConfig.phaseBCConfig.evalWindow}`,
    );
    const sampledResults = sampled.map((params, idx) =>
      evaluateCandidate(snapshot, params, config, experimentConfig, rng, idx),
    );
    const sortedResults = [...sampledResults].sort(compareProbeResults);
    logTopResults(topology, sortedResults, config.rankTop, config.trials);
    writeProbeArtifacts(topology, sortedResults, sampledResults, experimentConfig, config);
  }
}

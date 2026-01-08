import fs from 'fs';

import { MathColumn, ColumnSnapshot } from './mathColumn';
import * as MathOps from './mathHelpers';
import { createController } from './controller';
import { DEFAULT_CONTROLLER_CONFIG, DEFAULT_CURRICULUM, DEFAULT_PHASE_BC_CONFIG, PhaseBCConfig, PhasePhysicsParams, Topology } from './types';
import { artifactPath, resolveArtifactPath } from './artifactPaths';
import type { PhaseABestRecord, PhaseATuningRecord } from './phaseATuner';

interface PhaseBEvalResult {
  avgCountingAcc: number;
  avgSuccessorAcc: number;
  accFitness: number;
  sustainFitness: number;
  sustainMetrics: PhaseBSustainMetrics;
  fitness: number;
}

interface PhaseBSustainMetrics {
  tailSilentFrac?: number;
  lateSilentFrac?: number;
  timeToSilenceMean?: number;
}

export interface PhaseBTunePoint {
  k_inhib: number;
  v_th: number;
  alpha: number;
  wInRowNorm: number;
  etaTrans: number;
  eta_out: number;
}

export interface PhaseBTuningRecord {
  topology: Topology;
  params: PhaseBTunePoint;
  avgCountingAcc: number;
  avgSuccessorAcc: number;
  accFitness?: number;
  sustainFitness?: number;
  sustainMetrics?: PhaseBSustainMetrics;
  fitness: number;
}

export const phaseBIsBetter = (candidate: PhaseBTuningRecord, incumbent: PhaseBTuningRecord | null) => {
  if (!incumbent) return true;
  if (candidate.fitness !== incumbent.fitness) return candidate.fitness > incumbent.fitness;
  if (candidate.avgSuccessorAcc !== incumbent.avgSuccessorAcc)
    return candidate.avgSuccessorAcc > incumbent.avgSuccessorAcc;
  return candidate.avgCountingAcc > incumbent.avgCountingAcc;
};

const comparePhaseBRecords = <T extends PhaseBTuningRecord>(a: T, b: T) => {
  if (phaseBIsBetter(a, b)) return -1;
  if (phaseBIsBetter(b, a)) return 1;
  return 0;
};

const clonePhaseBCConfig = (base?: PhaseBCConfig): PhaseBCConfig => {
  const source = base ?? DEFAULT_PHASE_BC_CONFIG;
  return {
    ...source,
    sustainGate: { ...source.sustainGate },
    tuning: { ...source.tuning },
  };
};

const logTopCandidates = (
  topology: Topology,
  records: PhaseBTuningRecord[],
  config: PhaseBCConfig,
  stage: string,
  limit = 3,
): void => {
  const sorted = records
    .filter((r) => r.topology === topology)
    .sort(comparePhaseBRecords)
    .slice(0, limit);

  if (sorted.length === 0) return;
  console.log(`[Phase B] Top ${sorted.length} ${stage} candidates for ${topology}:`);
  for (let i = 0; i < sorted.length; i++) {
    const rec = sorted[i];
    const accFitness = rec.accFitness ?? rec.avgCountingAcc + rec.avgSuccessorAcc;
    const sustainFitness = rec.sustainFitness ?? 0;
    const tailSilentFrac = rec.sustainMetrics?.tailSilentFrac ?? rec.sustainMetrics?.lateSilentFrac;
    console.log(
      `  #${i + 1}: accFitness=${accFitness.toFixed(3)} sustainFitness=${sustainFitness.toFixed(3)} ` +
        `sustainWeight=${config.tuning.sustainWeight} fitness=${rec.fitness.toFixed(3)} ` +
        `tailSilentFrac=${tailSilentFrac !== undefined ? tailSilentFrac.toFixed(3) : 'n/a'}`,
    );
  }
};

export interface PhaseBFinetuneRecord extends PhaseBTuningRecord {
  numCountingEpisodes: number;
  numSuccessorTrials: number;
  rngSeeds: number[];
}

export interface PhaseBFinetuneOptions {
  topN: number;
  numCountingEpisodes: number;
  numSuccessorTrials: number;
  rngSeeds?: number[];
  progressFilename?: string;
}

function evaluatePhaseBCandidate(
  col: MathColumn,
  numCountingEpisodes: number,
  numSuccessorTrials: number,
  rngSeed: number,
  phaseBCConfig: PhaseBCConfig,
): PhaseBEvalResult {
  const rng = MathOps.createRng(rngSeed);
  const controllerRng = MathOps.createRng(rngSeed + 1);
  let sumCountAcc = 0;
  let succCorrect = 0;
  let validSuccessorTrials = 0;
  const numDigits = col.net.numDigits;
  const sustainStats: PhaseBSustainMetrics[] = [];
  const controller = createController(DEFAULT_CONTROLLER_CONFIG, phaseBCConfig, controllerRng);
  controller.resetEpisode();

  for (let i = 0; i < numCountingEpisodes; i++) {
    const { avgAcc } = col.runCountingEpisode(2, 6, undefined, undefined, controller);
    sumCountAcc += avgAcc;
  }

  for (let i = 0; i < numSuccessorTrials; i++) {
    const d = Math.floor(rng() * numDigits);
    const { correct, aborted } = col.runSuccessorTrial(d, i, {
      recordTrialStats: (stats) => {
        if (stats.sustain) {
          sustainStats.push({
            tailSilentFrac: stats.sustain.tailSilentFrac,
            lateSilentFrac: stats.sustain.lateSilentFrac,
            timeToSilenceMean: stats.sustain.timeToSilence,
          });
        }
      },
    }, undefined, undefined, controller);
    if (aborted) continue;
    validSuccessorTrials += 1;
    if (correct) succCorrect += 1;
  }

  const avgCountingAcc = numCountingEpisodes > 0 ? sumCountAcc / numCountingEpisodes : 0;
  const avgSuccessorAcc = validSuccessorTrials > 0 ? succCorrect / validSuccessorTrials : 0;
  const accFitness = avgCountingAcc + avgSuccessorAcc;
  const sustainMetrics = summarizeSustainMetrics(sustainStats);
  const sustainFitness = computeSustainFitness(phaseBCConfig, sustainMetrics);
  const fitness = phaseBCConfig.tuning.useSustainFitness
    ? accFitness + phaseBCConfig.tuning.sustainWeight * sustainFitness
    : accFitness;
  return { avgCountingAcc, avgSuccessorAcc, accFitness, sustainFitness, sustainMetrics, fitness };
}

let sustainMetricMissingWarned = false;

const clamp01 = (value: number) => Math.max(0, Math.min(1, value));

function summarizeSustainMetrics(stats: PhaseBSustainMetrics[]): PhaseBSustainMetrics {
  if (stats.length === 0) return {};
  const sumTail = stats.reduce((acc, v) => acc + (v.tailSilentFrac ?? 0), 0);
  const sumLate = stats.reduce((acc, v) => acc + (v.lateSilentFrac ?? 0), 0);
  const sumTime = stats.reduce((acc, v) => acc + (v.timeToSilenceMean ?? 0), 0);
  return {
    tailSilentFrac: sumTail / stats.length,
    lateSilentFrac: sumLate / stats.length,
    timeToSilenceMean: sumTime / stats.length,
  };
}

function computeSustainFitness(config: PhaseBCConfig, metrics: PhaseBSustainMetrics): number {
  if (!config.tuning.useSustainFitness) return 0;
  const metricValue =
    config.tuning.sustainMetricWindow === 'late'
      ? metrics.lateSilentFrac ?? metrics.tailSilentFrac
      : metrics.tailSilentFrac;

  if (metricValue === undefined) {
    if (!sustainMetricMissingWarned) {
      console.warn('[Phase B] Sustain fitness requested but no sustain metrics were recorded.');
      sustainMetricMissingWarned = true;
    }
    return 0;
  }

  const rawFitness = clamp01(1 - metricValue);
  if (config.tuning.targetSustainScore !== undefined) {
    const target = clamp01(config.tuning.targetSustainScore);
    const deviation = Math.abs(rawFitness - target);
    // When a target is set, reward closeness to that sustain score instead of the raw value.
    return clamp01(1 - deviation);
  }

  return rawFitness;
}

  function loadSnapshot(topology: Topology): ColumnSnapshot {
    const filePath = resolveArtifactPath({ key: 'phaseA_state', topology, required: true });
    if (!filePath) {
      throw new Error(`[Phase B] Missing Phase A snapshot for ${topology}`);
    }
    const raw = fs.readFileSync(filePath, 'utf-8');
    return JSON.parse(raw) as ColumnSnapshot;
  }

  interface PhaseABestPayload {
    bestPerTopology?: Partial<Record<Topology, PhaseABestRecord | null>>;
  }

  function loadPhaseAWAttrRowNorm(): Record<Topology, number> {
    const defaultValue = DEFAULT_CURRICULUM.phaseA.wAttrRowNorm;
    const payloadPath = resolveArtifactPath({ key: 'phaseA_best' });

    if (!payloadPath || !fs.existsSync(payloadPath)) {
      return { snake: defaultValue, ring: defaultValue };
    }

    const raw = fs.readFileSync(payloadPath, 'utf-8');
    const payload = JSON.parse(raw) as PhaseABestPayload;
  const result: Record<Topology, number> = { snake: defaultValue, ring: defaultValue };

  for (const topology of ['snake', 'ring'] as Topology[]) {
    const tuned = payload.bestPerTopology?.[topology]?.params?.wAttrRowNorm;
    if (tuned !== undefined) result[topology] = tuned;
  }

  return result;
}
function generatePhaseBGrid(topology: Topology): PhaseBTunePoint[] {
  const grid: PhaseBTunePoint[] = [];
  
  const k_inhib    = topology === 'ring' ? [7, 8] : [4, 5, 6];
  const v_th       = topology === 'ring' ? [1.6, 1.8] : [1.2, 1.4, 1.6];
  const alpha      = topology === 'ring' ? [0.7, 0.8] : [0.85, 0.9, 0.95];
  const wInRowNorm = topology === 'ring' ? [2.0, 2.5, 3.0] : [2.0, 2.5, 3.0];
  const etaTrans   = topology === 'ring' ? [0.05, 0.1, 0.2] : [0.002, 0.005, 0.01];
  const eta_out    = [0.05];


  for (const k of k_inhib) {
    for (const v of v_th) {
      for (const a of alpha) {
        for (const win of wInRowNorm) {
          for (const etaT of etaTrans) {
            for (const etaOut of eta_out) {
              grid.push({
                k_inhib: k,
                v_th: v,
                alpha: a,
                wInRowNorm: win,
                etaTrans: etaT,
                eta_out: etaOut,
              });
            }
          }
        }
      }
    }
  }

  return grid;
}

export function runPhaseBTuning(phaseBCConfigOverride?: PhaseBCConfig): {
  bestPerTopology: Record<Topology, PhaseBTuningRecord | null>;
  allResults: PhaseBTuningRecord[];
} {
  const phaseBCConfig = clonePhaseBCConfig(phaseBCConfigOverride);
  const progressFilePath = artifactPath('phaseB_tuning_progress.json');
    const snapshots: Record<Topology, ColumnSnapshot> = {
      snake: loadSnapshot('snake'),
      ring: loadSnapshot('ring'),
    };

  const topologies: Topology[] = ['snake', 'ring'];
  const candidatesByTopology: Record<Topology, PhaseBTunePoint[]> = {
    snake: generatePhaseBGrid('snake'),
    ring: generatePhaseBGrid('ring'),
  };
  const wAttrRowNorm = loadPhaseAWAttrRowNorm();
  const allResults: PhaseBTuningRecord[] = fs.existsSync(progressFilePath)
    ? JSON.parse(fs.readFileSync(progressFilePath, 'utf-8'))
    : [];

  if (allResults.length > 0) {
    console.log(`[Phase B] Loaded ${allResults.length} previous results from ${progressFilePath}`);
  }

  const numCountingEpisodes = 200;
  const numSuccessorTrials = 200;
  const rngSeed = 42;

  const totalEvals = Object.values(candidatesByTopology).reduce((acc, arr) => acc + arr.length, 0);
  console.log(
    `[Phase B] Tuning plan: ${topologies.length} topologies (${topologies.join(', ')}) = ${totalEvals} total candidates`,
  );
  for (const topology of topologies) {
    console.log(`[Phase B] ${topology} candidates=${candidatesByTopology[topology].length}`);
  }
  console.log(`[Phase B] Using wAttrRowNorm inherited from Phase A: snake=${wAttrRowNorm.snake}, ring=${wAttrRowNorm.ring}`);
  console.log(
    `[Phase B] Episodes per candidate: counting=${numCountingEpisodes}, successor trials=${numSuccessorTrials}, rngSeed=${rngSeed}`,
  );
  console.log(
    `[Phase B] Tuning config: useSustainFitness=${phaseBCConfig.tuning.useSustainFitness} sustainWeight=${phaseBCConfig.tuning.sustainWeight} sustainMetricWindow=${phaseBCConfig.tuning.sustainMetricWindow}`,
  );

  let completed = allResults.length;
  let evalIndex = 0;
  const bestSoFar: Record<Topology, PhaseBTuningRecord | null> = { snake: null, ring: null };

  for (const record of allResults) {
    const currentBest = bestSoFar[record.topology];
    if (!currentBest) {
      bestSoFar[record.topology] = record;
      continue;
    }
    if (record.fitness > currentBest.fitness) {
      bestSoFar[record.topology] = record;
    } else if (
      record.fitness === currentBest.fitness &&
      (record.avgSuccessorAcc > currentBest.avgSuccessorAcc ||
        (record.avgSuccessorAcc === currentBest.avgSuccessorAcc && record.avgCountingAcc > currentBest.avgCountingAcc))
    ) {
      bestSoFar[record.topology] = record;
    }
  }

  const saveProgress = () => {
    fs.writeFileSync(progressFilePath, JSON.stringify(allResults, null, 2));
  };

  const paramsEqual = (a: PhaseBTunePoint, b: PhaseBTunePoint): boolean =>
    a.k_inhib === b.k_inhib &&
    a.v_th === b.v_th &&
    a.alpha === b.alpha &&
    a.wInRowNorm === b.wInRowNorm &&
    a.etaTrans === b.etaTrans &&
    a.eta_out === b.eta_out;

  const findExisting = (topology: Topology, point: PhaseBTunePoint) =>
    allResults.find((r) => r.topology === topology && paramsEqual(r.params, point));

  for (const topology of topologies) {
    const snapshot = snapshots[topology];
    const candidates = candidatesByTopology[topology];
    for (const point of candidates) {
      evalIndex += 1;
      const col = MathColumn.fromSnapshot(snapshot);
      col.setRngSeed(rngSeed);
      col.setPhaseBCConfig(phaseBCConfig);
      const physics: PhasePhysicsParams = {
        k_inhib: point.k_inhib,
        v_th: point.v_th,
        alpha: point.alpha,
        wInRowNorm: point.wInRowNorm,
        wAttrRowNorm: wAttrRowNorm[topology],
        etaTrans: point.etaTrans,
      };

      col.applyPhysicsParams(physics);
      col.learnParams.eta_trans = point.etaTrans;
      col.learnParams.eta_out = point.eta_out;
      col.resetState();

      const paramDetails =
        `k_inhib=${point.k_inhib}, v_th=${point.v_th}, alpha=${point.alpha}, ` +
        `wInRowNorm=${point.wInRowNorm}, etaTrans=${point.etaTrans}, eta_out=${point.eta_out}`;

      const existing = findExisting(topology, point);
      if (existing) {
        const best = bestSoFar[topology];
        const bestFitness = best ? best.fitness.toFixed(3) : 'n/a';
        const bestSucc = best ? best.avgSuccessorAcc.toFixed(3) : 'n/a';
        const bestCount = best ? best.avgCountingAcc.toFixed(3) : 'n/a';
        console.log(
          `[Phase B] [${evalIndex}/${totalEvals}] topology=${topology} params={${paramDetails}} SKIPPED (already evaluated) | best so far fitness=${bestFitness} (succ=${bestSucc}, count=${bestCount}) | progress=${completed}/${totalEvals}`,
        );
        continue;
      }

      const result = evaluatePhaseBCandidate(
        col,
        numCountingEpisodes,
        numSuccessorTrials,
        rngSeed,
        phaseBCConfig,
      );

      allResults.push({
        topology,
        params: point,
        avgCountingAcc: result.avgCountingAcc,
        avgSuccessorAcc: result.avgSuccessorAcc,
        accFitness: result.accFitness,
        sustainFitness: result.sustainFitness,
        sustainMetrics: result.sustainMetrics,
        fitness: result.fitness,
      });

      const record = allResults[allResults.length - 1];
      if (phaseBIsBetter(record, bestSoFar[topology])) {
        bestSoFar[topology] = record;
      }

      completed += 1;
      saveProgress();
      const best = bestSoFar[topology];
      const bestFitness = best ? best.fitness.toFixed(3) : 'n/a';
      const bestSucc = best ? best.avgSuccessorAcc.toFixed(3) : 'n/a';
      const bestCount = best ? best.avgCountingAcc.toFixed(3) : 'n/a';
      console.log(
        `[Phase B] [${evalIndex}/${totalEvals}] topology=${topology} params={${paramDetails}} fitness=${result.fitness.toFixed(
          3,
        )} succ=${result.avgSuccessorAcc.toFixed(3)} count=${result.avgCountingAcc.toFixed(
          3,
        )} | best so far fitness=${bestFitness} (succ=${bestSucc}, count=${bestCount}) | progress=${completed}/${totalEvals}`,
      );
      logTopCandidates(topology, allResults, phaseBCConfig, 'coarse');
    }
  }

  const bestPerTopology: Record<Topology, PhaseBTuningRecord | null> = {
    snake: null,
    ring: null,
  };

  for (const topology of topologies) {
    const records = allResults.filter((r) => r.topology === topology);
    records.sort(comparePhaseBRecords);
    bestPerTopology[topology] = records[0] ?? null;
  }

  return { bestPerTopology, allResults };
}

export function runPhaseBFinetune(
  coarseResults: PhaseBTuningRecord[],
  options: PhaseBFinetuneOptions,
  phaseBCConfigOverride?: PhaseBCConfig,
): {
  bestPerTopology: Record<Topology, PhaseBFinetuneRecord | null>;
  allResults: PhaseBFinetuneRecord[];
} {
  const phaseBCConfig = clonePhaseBCConfig(phaseBCConfigOverride);
  const progressFilePath = artifactPath(options.progressFilename ?? 'phaseB_finetune_progress.json');
  const rngSeeds = options.rngSeeds ?? [42];
    const snapshots: Record<Topology, ColumnSnapshot> = {
      snake: loadSnapshot('snake'),
      ring: loadSnapshot('ring'),
    };

  const topologies: Topology[] = ['snake', 'ring'];
  const candidatesByTopology: Record<Topology, PhaseBTunePoint[]> = {
    snake: [],
    ring: [],
  };

  const wAttrRowNorm = loadPhaseAWAttrRowNorm();

  for (const topology of topologies) {
    const records = coarseResults.filter((r) => r.topology === topology);
    const sorted = [...records].sort(comparePhaseBRecords);
    candidatesByTopology[topology] = sorted.map((r) => r.params).slice(0, options.topN);
  }

  const selectedCount = Object.values(candidatesByTopology).reduce((acc, arr) => acc + arr.length, 0);
  const totalEvals = selectedCount;
  console.log(
    `[Phase B Finetune] Plan: ${totalEvals} candidates across ${topologies.length} topologies (up to top ${options.topN} each)`,
  );
  console.log(
    `[Phase B Finetune] Using wAttrRowNorm inherited from Phase A: snake=${wAttrRowNorm.snake}, ring=${wAttrRowNorm.ring}`,
  );
  console.log(
    `[Phase B Finetune] Episodes per candidate: counting=${options.numCountingEpisodes}, successor trials=${options.numSuccessorTrials}, rngSeeds=[${rngSeeds.join(', ')}]`,
  );
  console.log(
    `[Phase B Finetune] Tuning config: useSustainFitness=${phaseBCConfig.tuning.useSustainFitness} sustainWeight=${phaseBCConfig.tuning.sustainWeight} sustainMetricWindow=${phaseBCConfig.tuning.sustainMetricWindow}`,
  );

  const allResults: PhaseBFinetuneRecord[] = fs.existsSync(progressFilePath)
    ? (JSON.parse(fs.readFileSync(progressFilePath, 'utf-8')) as PhaseBFinetuneRecord[])
    : [];

  if (allResults.length > 0) {
    console.log(`[Phase B Finetune] Loaded ${allResults.length} previous results from ${progressFilePath}`);
  }

  let completed = allResults.length;
  let evalIndex = 0;
  const bestSoFar: Record<Topology, PhaseBFinetuneRecord | null> = { snake: null, ring: null };

  const arraysEqual = (a: number[], b: number[]): boolean => a.length === b.length && a.every((v, idx) => v === b[idx]);

  for (const record of allResults) {
    const currentBest = bestSoFar[record.topology];
    if (phaseBIsBetter(record, currentBest)) {
      bestSoFar[record.topology] = record;
    }
  }

  const paramsEqual = (a: PhaseBTunePoint, b: PhaseBTunePoint): boolean =>
    a.k_inhib === b.k_inhib &&
    a.v_th === b.v_th &&
    a.alpha === b.alpha &&
    a.wInRowNorm === b.wInRowNorm &&
    a.etaTrans === b.etaTrans &&
    a.eta_out === b.eta_out;

  const findExisting = (topology: Topology, point: PhaseBTunePoint) =>
    allResults.find(
      (r) =>
        r.topology === topology &&
        paramsEqual(r.params, point) &&
        r.numCountingEpisodes === options.numCountingEpisodes &&
        r.numSuccessorTrials === options.numSuccessorTrials &&
        arraysEqual(r.rngSeeds, rngSeeds),
    );

  const evaluateWithSeeds = (
    snapshot: ColumnSnapshot,
    point: PhaseBTunePoint,
    seeds: number[],
    wAttrRowNormValue: number,
  ): PhaseBEvalResult => {
    let sumCounting = 0;
    let sumSuccessor = 0;
    let sumAccFitness = 0;
    let sumSustainFitness = 0;
    let sumFitness = 0;
    const sustainMetrics: PhaseBSustainMetrics[] = [];

    for (const seed of seeds) {
      const col = MathColumn.fromSnapshot(snapshot);
      col.setRngSeed(seed);
      col.setPhaseBCConfig(phaseBCConfig);
      const physics: PhasePhysicsParams = {
        k_inhib: point.k_inhib,
        v_th: point.v_th,
        alpha: point.alpha,
        wInRowNorm: point.wInRowNorm,
        wAttrRowNorm: wAttrRowNormValue,
        etaTrans: point.etaTrans,
      };

      col.applyPhysicsParams(physics);
      col.learnParams.eta_trans = point.etaTrans;
      col.learnParams.eta_out = point.eta_out;
      col.resetState();

      const result = evaluatePhaseBCandidate(
        col,
        options.numCountingEpisodes,
        options.numSuccessorTrials,
        seed,
        phaseBCConfig,
      );

      sumCounting += result.avgCountingAcc;
      sumSuccessor += result.avgSuccessorAcc;
      sumAccFitness += result.accFitness;
      sumSustainFitness += result.sustainFitness;
      sumFitness += result.fitness;
      sustainMetrics.push(result.sustainMetrics);
    }

    const denom = seeds.length === 0 ? 1 : seeds.length;
    return {
      avgCountingAcc: sumCounting / denom,
      avgSuccessorAcc: sumSuccessor / denom,
      accFitness: sumAccFitness / denom,
      sustainFitness: sumSustainFitness / denom,
      sustainMetrics: summarizeSustainMetrics(sustainMetrics),
      fitness: sumFitness / denom,
    };
  };

  const saveProgress = () => {
    fs.writeFileSync(progressFilePath, JSON.stringify(allResults, null, 2));
  };

  for (const topology of topologies) {
    const snapshot = snapshots[topology];
    for (const point of candidatesByTopology[topology]) {
      evalIndex += 1;

      const paramDetails =
        `k_inhib=${point.k_inhib}, v_th=${point.v_th}, alpha=${point.alpha}, ` +
        `wInRowNorm=${point.wInRowNorm}, etaTrans=${point.etaTrans}, eta_out=${point.eta_out}`;

      const existing = findExisting(topology, point);
      if (existing) {
        const best = bestSoFar[topology];
        const bestFitness = best ? best.fitness.toFixed(3) : 'n/a';
        const bestSucc = best ? best.avgSuccessorAcc.toFixed(3) : 'n/a';
        const bestCount = best ? best.avgCountingAcc.toFixed(3) : 'n/a';
        console.log(
          `[Phase B Finetune] [${evalIndex}/${totalEvals}] topology=${topology} params={${paramDetails}} SKIPPED (already evaluated) | best so far fitness=${bestFitness} (succ=${bestSucc}, count=${bestCount}) | progress=${completed}/${totalEvals}`,
        );
        continue;
      }

      const result = evaluateWithSeeds(snapshot, point, rngSeeds, wAttrRowNorm[topology]);

      allResults.push({
        topology,
        params: point,
        avgCountingAcc: result.avgCountingAcc,
        avgSuccessorAcc: result.avgSuccessorAcc,
        accFitness: result.accFitness,
        sustainFitness: result.sustainFitness,
        sustainMetrics: result.sustainMetrics,
        fitness: result.fitness,
        numCountingEpisodes: options.numCountingEpisodes,
        numSuccessorTrials: options.numSuccessorTrials,
        rngSeeds,
      });

      const record = allResults[allResults.length - 1];
      if (phaseBIsBetter(record, bestSoFar[topology])) {
        bestSoFar[topology] = record;
      }

      completed += 1;
      saveProgress();
      const best = bestSoFar[topology];
      const bestFitness = best ? best.fitness.toFixed(3) : 'n/a';
      const bestSucc = best ? best.avgSuccessorAcc.toFixed(3) : 'n/a';
      const bestCount = best ? best.avgCountingAcc.toFixed(3) : 'n/a';
      console.log(
        `[Phase B Finetune] [${evalIndex}/${totalEvals}] topology=${topology} params={${paramDetails}} fitness=${result.fitness.toFixed(
          3,
        )} succ=${result.avgSuccessorAcc.toFixed(3)} count=${result.avgCountingAcc.toFixed(
          3,
        )} | best so far fitness=${bestFitness} (succ=${bestSucc}, count=${bestCount}) | progress=${completed}/${totalEvals}`,
      );
      logTopCandidates(topology, allResults, phaseBCConfig, 'finetune');
    }
  }

  const bestPerTopology: Record<Topology, PhaseBFinetuneRecord | null> = {
    snake: null,
    ring: null,
  };

  for (const topology of topologies) {
    const records = allResults.filter((r) => r.topology === topology);
    records.sort(comparePhaseBRecords);
    bestPerTopology[topology] = records[0] ?? null;
  }

  return { bestPerTopology, allResults };
}

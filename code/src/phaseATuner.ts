import fs from 'fs';

import { PhasePhysicsParams, Topology } from './types';
import { evaluatePhaseACandidate, generatePhaseAGrid, PhaseATunePoint, trainPhaseAUntilTarget } from './tunerUtils';
import { artifactPath, exportToSharedArtifacts } from './artifactPaths';
import { parseActivityOverrides } from './activityOverrides';

export interface PhaseATuningRecord {
  topology: Topology;
  params: PhaseATunePoint;
  finalAccuracy: number;
  steps: number;
  maxSteps: number;
}

export interface PhaseASnapshotSummary {
  targetAcc: number;
  windowSize: number;
  minFill: number;
  maxSteps: number;
  stepsTrained: number;
  finalWindowAcc: number;
}

export type PhaseABestRecord = PhaseATuningRecord & { snapshot?: PhaseASnapshotSummary };

type PhaseATuningOptions = {
  maxSteps?: Partial<Record<Topology, number>>;
  activityMode?: PhasePhysicsParams['activityMode'];
  activityAlpha?: PhasePhysicsParams['activityAlpha'];
};

export function runPhaseATuning(options?: PhaseATuningOptions): {
  bestPerTopology: Record<Topology, PhaseATuningRecord | PhaseABestRecord | null>;
  allResults: PhaseATuningRecord[];
} {
  const progressFilePath = artifactPath('phaseA_tuning_progress.json');
  const baseSeed = 42;
  const defaultMaxStepsPhaseA = 1000;
  const topologies: Topology[] = ['snake', 'ring'];
  const candidates = generatePhaseAGrid();

  const loadProgress = (): PhaseATuningRecord[] => {
    if (!fs.existsSync(progressFilePath)) return [];
    const raw = fs.readFileSync(progressFilePath, 'utf-8');
    const parsed = JSON.parse(raw) as (PhaseATuningRecord & { maxSteps?: number })[];
    return parsed.map((record) => ({
      ...record,
      maxSteps: record.maxSteps ?? defaultMaxStepsPhaseA,
    }));
  };

  const saveProgress = (records: PhaseATuningRecord[]): void => {
    fs.writeFileSync(progressFilePath, JSON.stringify(records, null, 2));
  };

  const paramsEqual = (a: PhaseATunePoint, b: PhaseATunePoint): boolean =>
    a.k_inhib === b.k_inhib &&
    a.v_th === b.v_th &&
    a.alpha === b.alpha &&
    a.wInRowNorm === b.wInRowNorm &&
    a.wAttrRowNorm === b.wAttrRowNorm &&
    a.eta_attr === b.eta_attr &&
    a.eta_out === b.eta_out;

  const findExisting = (topology: Topology, point: PhaseATunePoint, records: PhaseATuningRecord[], maxSteps: number) =>
    records.find((r) => r.topology === topology && r.maxSteps === maxSteps && paramsEqual(r.params, point));

  const totalEvals = topologies.length * candidates.length;
  console.log(
    `[Phase A] Tuning plan: ${topologies.length} topologies (${topologies.join(', ')}) × ${candidates.length} candidates = ${totalEvals} evaluations`,
  );

  const allResults: PhaseATuningRecord[] = loadProgress();
  if (allResults.length > 0) {
    console.log(`[Phase A] Loaded ${allResults.length} previous results from ${progressFilePath}`);
  }

  let completed = allResults.length;
  let evalIndex = 0;
  const bestSoFar: Record<Topology, PhaseATuningRecord | null> = { snake: null, ring: null };

  for (const record of allResults) {
    const currentBest = bestSoFar[record.topology];
    if (!currentBest) {
      bestSoFar[record.topology] = record;
      continue;
    }
    if (record.finalAccuracy > currentBest.finalAccuracy) {
      bestSoFar[record.topology] = record;
    } else if (record.finalAccuracy === currentBest.finalAccuracy && record.steps < currentBest.steps) {
      bestSoFar[record.topology] = record;
    }
  }

  const isBetter = (candidate: PhaseATuningRecord, incumbent: PhaseATuningRecord | null) => {
    if (!incumbent) return true;
    if (candidate.finalAccuracy !== incumbent.finalAccuracy)
      return candidate.finalAccuracy > incumbent.finalAccuracy;
    return candidate.steps < incumbent.steps;
  };

  for (const topology of topologies) {
    for (const point of candidates) {
      evalIndex += 1;
      const physics: PhasePhysicsParams = {
        k_inhib: point.k_inhib,
        v_th: point.v_th,
        alpha: point.alpha,
        wInRowNorm: point.wInRowNorm,
        wAttrRowNorm: point.wAttrRowNorm,
        etaTrans: 0,
        activityMode: options?.activityMode,
        activityAlpha: options?.activityAlpha,
      };

      const paramDetails =
        `k_inhib=${point.k_inhib}, v_th=${point.v_th}, alpha=${point.alpha}, ` +
        `wInRowNorm=${point.wInRowNorm}, wAttrRowNorm=${point.wAttrRowNorm}, ` +
        `eta_attr=${point.eta_attr}, eta_out=${point.eta_out}`;

      const maxStepsPhaseA = options?.maxSteps?.[topology] ?? defaultMaxStepsPhaseA;

      const existing = findExisting(topology, point, allResults, maxStepsPhaseA);
      if (existing) {
        const best = bestSoFar[topology];
        const bestAcc = best ? best.finalAccuracy.toFixed(3) : 'n/a';
        const bestSteps = best ? best.steps : 'n/a';
        console.log(
          `[Phase A] [${evalIndex}/${totalEvals}] topology=${topology} params={${paramDetails}} maxSteps=${existing.maxSteps} SKIPPED (already evaluated) | best so far acc=${bestAcc} steps=${bestSteps} | progress=${completed}/${totalEvals}`,
        );
        continue;
      }

      const result = evaluatePhaseACandidate(topology, physics, point.eta_attr, point.eta_out, baseSeed, maxStepsPhaseA);

      allResults.push({
        topology,
        params: point,
        finalAccuracy: result.finalAccuracy,
        steps: result.steps,
        maxSteps: maxStepsPhaseA,
      });

      const record = allResults[allResults.length - 1];
      if (isBetter(record, bestSoFar[topology])) {
        bestSoFar[topology] = record;
      }

      completed += 1;
      saveProgress(allResults);
      const best = bestSoFar[topology];
      const bestAcc = best ? best.finalAccuracy.toFixed(3) : 'n/a';
      const bestSteps = best ? best.steps : 'n/a';
      console.log(
        `[Phase A] [${evalIndex}/${totalEvals}] topology=${topology} params={${paramDetails}} acc=${result.finalAccuracy.toFixed(
          3,
        )} steps=${result.steps} | best so far acc=${bestAcc} steps=${bestSteps} | progress=${completed}/${totalEvals}`,
      );
    }
  }

  const bestPerTopology: Record<Topology, PhaseATuningRecord | null> = {
    snake: null,
    ring: null,
  };

  for (const topology of topologies) {
    const records = allResults.filter((r) => r.topology === topology);
    records.sort((a, b) => {
      if (b.finalAccuracy !== a.finalAccuracy) return b.finalAccuracy - a.finalAccuracy;
      return a.steps - b.steps;
    });
    bestPerTopology[topology] = records[0] ?? null;
  }

  return { bestPerTopology, allResults };
}

const parseIntFlag = (flag: string): number | undefined => {
  const raw = process.argv.find((arg) => arg.startsWith(flag));
  if (!raw) return undefined;
  const [, value] = raw.split('=');
  if (!value) return undefined;
  const parsed = parseInt(value, 10);
  if (!Number.isFinite(parsed)) {
    throw new Error(`[CLI] ${flag.slice(2, -1)} must be numeric. Received: ${value}`);
  }
  return parsed;
};

const parsePhaseATuningCli = (): { maxSteps: Record<Topology, number> } => {
  const defaultMaxSteps = 1000;
  const shared = parseIntFlag('--tune.phaseA.maxSteps=');
  const snake = parseIntFlag('--snake.tune.phaseA.maxSteps=');
  const ring = parseIntFlag('--ring.tune.phaseA.maxSteps=');

  return {
    maxSteps: {
      snake: snake ?? shared ?? defaultMaxSteps,
      ring: ring ?? shared ?? defaultMaxSteps,
    },
  };
};

export function runPhaseATuningEntry(): void {
  const tuningCli = parsePhaseATuningCli();
  const activityOverrides = parseActivityOverrides(process.argv);
  const phaseAOverrides = activityOverrides?.phaseA;

  const { bestPerTopology, allResults } = runPhaseATuning({
    maxSteps: tuningCli.maxSteps,
    activityMode: phaseAOverrides?.activityMode,
    activityAlpha: phaseAOverrides?.activityAlpha,
  });
  const allResultsSorted = [...allResults].sort((a, b) => {
    if (b.finalAccuracy !== a.finalAccuracy) return b.finalAccuracy - a.finalAccuracy;
    return a.steps - b.steps;
  });

  const timestamp = new Date().toISOString();
  const safeTimestamp = timestamp.replace(/[:.]/g, '-');
  const snapshotConfigBase: Omit<PhaseASnapshotSummary, 'stepsTrained' | 'finalWindowAcc'> = {
    targetAcc: 0.99,
    windowSize: 200,
    minFill: 150,
    maxSteps: 20000,
  };

  const bestWithSnapshot: Record<Topology, PhaseABestRecord | null> = {
    snake: bestPerTopology.snake ? { ...bestPerTopology.snake } : null,
    ring: bestPerTopology.ring ? { ...bestPerTopology.ring } : null,
  };

  const baseSeed = 42;
  const snapshotFiles: Record<Topology, string> = {
    snake: artifactPath('phaseA_state_snake.json'),
    ring: artifactPath('phaseA_state_ring.json'),
  };

  for (const topology of Object.keys(bestPerTopology) as Topology[]) {
    const best = bestPerTopology[topology];
    if (!best) continue;

    const physics: PhasePhysicsParams = {
      k_inhib: best.params.k_inhib,
      v_th: best.params.v_th,
      alpha: best.params.alpha,
      wInRowNorm: best.params.wInRowNorm,
      wAttrRowNorm: best.params.wAttrRowNorm,
      etaTrans: 0,
      activityMode: phaseAOverrides?.activityMode,
      activityAlpha: phaseAOverrides?.activityAlpha,
    };

    const trained = trainPhaseAUntilTarget(
      topology,
      physics,
      best.params.eta_attr,
      best.params.eta_out,
      baseSeed,
      snapshotConfigBase.targetAcc,
      snapshotConfigBase.windowSize,
      snapshotConfigBase.minFill,
      snapshotConfigBase.maxSteps,
    );

    console.log(
      `[Phase A] Snapshot training for ${topology}: steps=${trained.steps} windowAcc=${trained.finalWindowAcc.toFixed(
        3,
      )} target=${snapshotConfigBase.targetAcc}`,
    );

    bestWithSnapshot[topology] = {
      ...best,
      snapshot: {
        ...snapshotConfigBase,
        stepsTrained: trained.steps,
        finalWindowAcc: trained.finalWindowAcc,
      },
    };

    const snapshot = trained.col.exportSnapshot();
    fs.writeFileSync(snapshotFiles[topology], JSON.stringify(snapshot));
    exportToSharedArtifacts(`phaseA_state_${topology}.json`);
    console.log(`Phase A snapshot for ${topology} written to ${snapshotFiles[topology]}`);
  }

  const resultsPayload = {
    metadata: {
      timestamp,
      description: 'Phase A grid search',
    },
    bestPerTopology: bestWithSnapshot,
    allResults: allResultsSorted,
  };

  const filename = `phaseA_tuning_results_${safeTimestamp}.json`;
  const filePath = artifactPath(filename);
  fs.writeFileSync(filePath, JSON.stringify(resultsPayload, null, 2));
  console.log(`Phase A tuning results written to ${filePath}`);

  const stableFilePath = artifactPath('phaseA_best.json');
  fs.writeFileSync(stableFilePath, JSON.stringify(resultsPayload, null, 2));
  console.log(`Phase A best-per-topology written to ${stableFilePath}`);
  exportToSharedArtifacts('phaseA_best.json');
}

import fs from 'fs';
import path from 'path';

import { artifactPath, resolveArtifactPath, exportToSharedArtifacts, artifactsDir } from './artifactPaths';
import { MathColumn, ColumnSnapshot } from './mathColumn';
import * as MathOps from './mathHelpers';
import { PhaseATunePoint } from './tunerUtils';
import { PhasePhysicsParams, Topology } from './types';

type PhaseABestRecord = {
  topology: Topology;
  params: PhaseATunePoint;
  finalAccuracy?: number;
  snapshot?: unknown;
};

type PhaseABestPayload = {
  bestPerTopology?: Partial<Record<Topology, PhaseABestRecord | null>>;
};

type PhaseAFinetuneCriteria = {
  targetAcc: number;
  windowSize: number;
  minTrials: number;
  requireAllDigits: boolean;
  maxSteps: number;
};

type PhaseAFinetuneProgress = {
  step: number;
  meanAcc: number;
  minDigitAcc: number;
  accuracyByDigit: number[];
  trials: number;
};

type PhaseAFinetuneResult = {
  topology: Topology;
  startingParams?: PhaseATunePoint;
  criteria: PhaseAFinetuneCriteria;
  stepsTrained: number;
  meanAcc: number;
  minDigitAcc: number;
  accuracyByDigit: number[];
  validationTrials: number;
  snapshotFile: string;
};

type PhaseAFinetunePayload = {
  timestamp: string;
  source: {
    phaseAFrom?: string;
    resolvedPhaseABest?: string;
  };
  results: Partial<Record<Topology, PhaseAFinetuneResult | null>>;
};

function parseBoolFlag(flag: string, defaultValue: boolean): boolean {
  const raw = process.argv.find((arg) => arg.startsWith(flag));
  if (!raw) return defaultValue;
  const idx = raw.indexOf('=');
  const value = idx >= 0 ? raw.slice(idx + 1) : undefined;
  if (value === undefined) return true;
  return !(value === '0' || value.toLowerCase() === 'false');
}

function parseFloatFlag(flag: string, defaultValue: number): number {
  const raw = process.argv.find((arg) => arg.startsWith(flag));
  if (!raw) return defaultValue;
  const [, value] = raw.split('=');
  const parsed = parseFloat(value);
  if (!Number.isFinite(parsed)) {
    throw new Error(`[Phase A Finetune] ${flag} must be numeric. Received: ${value}`);
  }
  return parsed;
}

function parseIntFlag(flag: string, defaultValue: number): number {
  const raw = process.argv.find((arg) => arg.startsWith(flag));
  if (!raw) return defaultValue;
  const [, value] = raw.split('=');
  const parsed = parseInt(value, 10);
  if (!Number.isFinite(parsed)) {
    throw new Error(`[Phase A Finetune] ${flag} must be numeric. Received: ${value}`);
  }
  return parsed;
}

function parseFinetuneCli(): {
  phaseAFrom?: string;
  criteria: PhaseAFinetuneCriteria;
  perTopologyMaxSteps: Partial<Record<Topology, number>>;
  updatePhaseABest: boolean;
} {
  const phaseAFromArg = process.argv.find((arg) => arg.startsWith('--phaseA.from='));
  const phaseAFrom = phaseAFromArg ? phaseAFromArg.slice(13) : undefined;

  const targetAcc = parseFloatFlag('--finetune.phaseA.targetAcc=', 1.0);
  const windowSize = parseIntFlag('--finetune.phaseA.windowSize=', 200);
  const minTrials = parseIntFlag('--finetune.phaseA.minTrials=', 200);
  const requireAllDigits = parseBoolFlag('--finetune.phaseA.requireAllDigits', true);
  const maxSteps = parseIntFlag('--finetune.phaseA.maxSteps=', 20000);
  const snakeMax = parseIntFlag('--snake.finetune.phaseA.maxSteps=', NaN);
  const ringMax = parseIntFlag('--ring.finetune.phaseA.maxSteps=', NaN);
  const updatePhaseABest = parseBoolFlag('--finetune.phaseA.writePhaseABest', false);

  const perTopologyMaxSteps: Partial<Record<Topology, number>> = {};
  if (Number.isFinite(snakeMax)) perTopologyMaxSteps.snake = snakeMax;
  if (Number.isFinite(ringMax)) perTopologyMaxSteps.ring = ringMax;

  return {
    phaseAFrom,
    criteria: { targetAcc, windowSize, minTrials, requireAllDigits, maxSteps },
    perTopologyMaxSteps,
    updatePhaseABest,
  };
}

function resolvePhaseABestPath(explicit: string | undefined): string | undefined {
  if (explicit) {
    const resolved = path.isAbsolute(explicit) ? explicit : artifactPath(explicit);
    if (fs.existsSync(resolved)) return resolved;
    console.warn(`[Phase A Finetune] Explicit phaseA.from path not found: ${resolved}`);
    return undefined;
  }

  const resolved = resolveArtifactPath({ key: 'phaseA_best', artifactsDirOverride: artifactsDir(), required: false });
  if (resolved) return resolved;
  return undefined;
}

function loadPhaseABest(resolvedPath: string | undefined): PhaseABestPayload | undefined {
  if (!resolvedPath) return undefined;
  if (!fs.existsSync(resolvedPath)) return undefined;
  try {
    const raw = fs.readFileSync(resolvedPath, 'utf-8');
    return JSON.parse(raw) as PhaseABestPayload;
  } catch (err) {
    console.warn(`[Phase A Finetune] Failed to load phaseA_best from ${resolvedPath}: ${err}`);
    return undefined;
  }
}

function findSnapshotPath(topology: Topology, baseDir: string | undefined): string | undefined {
  const candidates: string[] = [];
  if (baseDir) {
    candidates.push(path.join(baseDir, `phaseA_state_${topology}.json`));
  }
  candidates.push(artifactPath(`phaseA_state_${topology}.json`));
  const resolved = resolveArtifactPath({ key: 'phaseA_state', topology, required: false });
  if (resolved) candidates.push(resolved);

  for (const candidate of candidates) {
    if (candidate && fs.existsSync(candidate)) return candidate;
  }
  return undefined;
}

function prepareColumn(topology: Topology, params: PhaseATunePoint, snapshotPath?: string): MathColumn {
  const physics: PhasePhysicsParams = {
    k_inhib: params.k_inhib,
    v_th: params.v_th,
    alpha: params.alpha,
    wInRowNorm: params.wInRowNorm,
    wAttrRowNorm: params.wAttrRowNorm,
    etaTrans: 0,
  };

  if (snapshotPath && fs.existsSync(snapshotPath)) {
    const snapshot = JSON.parse(fs.readFileSync(snapshotPath, 'utf-8')) as ColumnSnapshot;
    const col = MathColumn.fromSnapshot(snapshot);
    col.applyPhysicsParams(physics);
    col.learnParams.eta_attr = params.eta_attr;
    col.learnParams.eta_out = params.eta_out;
    col.setRngSeed(42);
    return col;
  }

  const col = new MathColumn(topology, 42);
  col.learnParams.eta_attr = params.eta_attr;
  col.learnParams.eta_out = params.eta_out;
  col.applyPhysicsParams(physics);
  col.resetState();
  return col;
}

function evaluateColumn(
  col: MathColumn,
  trials: number,
  seed: number,
): { accuracyByDigit: number[]; meanAcc: number; minDigitAcc: number } {
  const snapshot = col.exportSnapshot();
  const evalCol = MathColumn.fromSnapshot(snapshot);
  evalCol.learnParams.eta_attr = 0;
  evalCol.learnParams.eta_out = 0;
  evalCol.learnParams.eta_trans = 0;
  evalCol.resetState();

  const rng = MathOps.createRng(seed);
  const numDigits = evalCol.net.numDigits;
  const counts = Array.from({ length: numDigits }, () => 0);
  const correct = Array.from({ length: numDigits }, () => 0);

  for (let i = 0; i < trials; i += 1) {
    const digit = trials >= numDigits ? i % numDigits : Math.floor(rng() * numDigits);
    const { correct: isCorrect } = evalCol.runPhaseA(digit);
    counts[digit] += 1;
    if (isCorrect) correct[digit] += 1;
  }

  const accuracyByDigit = counts.map((c, idx) => (c > 0 ? correct[idx] / c : 0));
  const meanAcc = accuracyByDigit.reduce((sum, acc) => sum + acc, 0) / accuracyByDigit.length;
  const minDigitAcc = accuracyByDigit.length > 0 ? Math.min(...accuracyByDigit) : 0;
  return { accuracyByDigit, meanAcc, minDigitAcc };
}

function trainUntilCriterion(
  topology: Topology,
  params: PhaseATunePoint,
  criteria: PhaseAFinetuneCriteria,
  maxStepsOverride?: number,
  snapshotBaseDir?: string,
): { result: PhaseAFinetuneResult; progress: PhaseAFinetuneProgress[]; validation: PhaseAFinetuneProgress } {
  const col = prepareColumn(topology, params, findSnapshotPath(topology, snapshotBaseDir));
  const rng = MathOps.createRng(1337 + (topology === 'snake' ? 0 : 1));
  const numDigits = col.net.numDigits;
  const maxSteps = maxStepsOverride ?? criteria.maxSteps;

  const progress: PhaseAFinetuneProgress[] = [];
  let lastValidation: PhaseAFinetuneProgress = {
    step: 0,
    meanAcc: 0,
    minDigitAcc: 0,
    accuracyByDigit: Array.from({ length: numDigits }, () => 0),
    trials: criteria.minTrials,
  };

  for (let step = 1; step <= maxSteps; step += 1) {
    const digit = Math.floor(rng() * numDigits);
    col.runPhaseA(digit);

    const shouldValidate = step === 1 || step % criteria.windowSize === 0 || step === maxSteps;
    if (!shouldValidate) continue;

    const { accuracyByDigit, meanAcc, minDigitAcc } = evaluateColumn(col, criteria.minTrials, 9000 + step);
    lastValidation = {
      step,
      meanAcc,
      minDigitAcc,
      accuracyByDigit,
      trials: criteria.minTrials,
    };
    progress.push(lastValidation);

    const pass = criteria.requireAllDigits ? minDigitAcc >= criteria.targetAcc : meanAcc >= criteria.targetAcc;
    if (pass) break;
  }

  const snapshotFile = artifactPath(`phaseA_state_${topology}.json`);
  const snapshot = col.exportSnapshot();
  fs.writeFileSync(snapshotFile, JSON.stringify(snapshot));
  exportToSharedArtifacts(`phaseA_state_${topology}.json`);

  const result: PhaseAFinetuneResult = {
    topology,
    startingParams: params,
    criteria,
    stepsTrained: lastValidation.step,
    meanAcc: lastValidation.meanAcc,
    minDigitAcc: lastValidation.minDigitAcc,
    accuracyByDigit: lastValidation.accuracyByDigit,
    validationTrials: lastValidation.trials,
    snapshotFile: path.basename(snapshotFile),
  };

  return { result, progress, validation: lastValidation };
}

export function runPhaseAFinetuneEntry(): void {
  const cli = parseFinetuneCli();
  const resolvedBestPath = resolvePhaseABestPath(cli.phaseAFrom);
  const bestPayload = loadPhaseABest(resolvedBestPath);
  const timestamp = new Date().toISOString();
  const results: Partial<Record<Topology, PhaseAFinetuneResult | null>> = {};
  const progressLog: Partial<Record<Topology, PhaseAFinetuneProgress[]>> = {};
  const validations: Partial<Record<Topology, PhaseAFinetuneProgress | null>> = {};

  (['snake', 'ring'] as Topology[]).forEach((topology) => {
    const bestRecord = bestPayload?.bestPerTopology?.[topology];
    if (!bestRecord?.params) {
      console.warn(`[Phase A Finetune] No starting params found for ${topology}; skipping.`);
      results[topology] = null;
      validations[topology] = null;
      return;
    }

    const maxSteps = cli.perTopologyMaxSteps[topology] ?? cli.criteria.maxSteps;
    console.log(
      `[Phase A Finetune] Starting ${topology} with targetAcc=${cli.criteria.targetAcc} requireAllDigits=${cli.criteria.requireAllDigits}` +
        ` maxSteps=${maxSteps}`,
    );

    const { result, progress, validation } = trainUntilCriterion(
      topology,
      bestRecord.params,
      { ...cli.criteria, maxSteps },
      maxSteps,
      resolvedBestPath ? path.dirname(resolvedBestPath) : undefined,
    );

    results[topology] = result;
    progressLog[topology] = progress;
    validations[topology] = validation;
  });

  const bestPayloadOut: PhaseAFinetunePayload = {
    timestamp,
    source: { phaseAFrom: cli.phaseAFrom, resolvedPhaseABest: resolvedBestPath },
    results,
  };

  const bestPath = artifactPath('phaseA_finetune_best.json');
  fs.writeFileSync(bestPath, JSON.stringify(bestPayloadOut, null, 2));
  console.log(`[Phase A Finetune] Summary written to ${bestPath}`);
  exportToSharedArtifacts('phaseA_finetune_best.json');

  const progressPath = artifactPath('phaseA_finetune_progress.json');
  fs.writeFileSync(
    progressPath,
    JSON.stringify({ timestamp, progress: progressLog, criteria: cli.criteria }, null, 2),
  );
  exportToSharedArtifacts('phaseA_finetune_progress.json');

  const validationPath = artifactPath('phaseA_validation.json');
  fs.writeFileSync(validationPath, JSON.stringify({ timestamp, validations }, null, 2));
  exportToSharedArtifacts('phaseA_validation.json');

  if (cli.updatePhaseABest) {
    const phaseABestPath = artifactPath('phaseA_best.json');
    fs.writeFileSync(phaseABestPath, JSON.stringify(bestPayloadOut, null, 2));
    console.log(`[Phase A Finetune] phaseA_best.json overwritten at ${phaseABestPath}`);
    exportToSharedArtifacts('phaseA_best.json');
  }
}

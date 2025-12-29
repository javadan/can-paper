import fs from 'fs';
import path from 'path';

import { artifactsDir, exportToSharedArtifacts } from './artifactPaths';
import { MathColumn, ColumnSnapshot } from './mathColumn';
import * as MathOps from './mathHelpers';
import { createController } from './controller';
import { resolveExperimentConfig } from './configResolver';
import { describeTransitionWindow, resolveTransitionWindows } from './transitionWindows';
import {
  DEFAULT_CURRICULUM,
  DEFAULT_TAIL_LEN,
  DEFAULT_T_TRANS,
  ExperimentConfig,
  PhasePhysicsParams,
  Topology,
  TransitionWindowName,
  ActivityMode,
  BGControllerConfig,
} from './types';

type MegaTuneConstraints = {
  allowImpulse: boolean;
  forceLearnEqEval: boolean;
  minAcc: number;
  maxCollapseFrac: number;
};

type MegaTuneGuardrailConfig = {
  enabled: boolean;
  minAcc: number;
  maxCollapseFrac: number;
  continueOnFail: boolean;
};

type MegaTuneSearchSpaces = {
  evalWindows: TransitionWindowName[];
  settleWindows: TransitionWindowName[];
  excludeFirstK: number[];
  activityModes: ActivityMode[];
  activityAlphas: number[];
  tTrans: number;
  tailLen: number;
  bg: {
    epsilons: number[];
    temperatures: number[];
    waitSteps: number[];
    sampleActions: boolean[];
  };
};

type MegaTuneBudgets = {
  guardrailTrials: number;
  stage1MaxCandidates: number;
  stage1Trials: number;
  stage1TopK: number;
  stage2MaxSemantics: number;
  stage2Trials: number;
  stage2TopK: number;
  stage3MaxBg: number;
  stage3Trials: number;
  stage3TopK: number;
  stage4Enabled: boolean;
  stage4MaxCandidates: number;
  stage4Trials: number;
};

type MegaTuneConfig = {
  id: string;
  outDir: string;
  seed: number;
  topologies: Topology[];
  wAcc: number;
  wSustain: number;
  wCollapse: number;
  constraints: MegaTuneConstraints;
  guardrail: MegaTuneGuardrailConfig;
  budgets: MegaTuneBudgets;
  search: MegaTuneSearchSpaces;
};

type PhysicsCandidate = PhasePhysicsParams;

type SemanticsCandidate = {
  evalWindow: TransitionWindowName;
  learnWindow: TransitionWindowName;
  settleWindow: TransitionWindowName;
  excludeFirstK: number;
  activityMode?: ActivityMode;
  activityAlpha?: number;
  tTrans?: number;
  tailLen?: number;
};

type BgCandidate = Pick<BGControllerConfig, 'epsilon' | 'temperature' | 'waitSteps' | 'sampleActions'>;

type MegaTuneCandidate = {
  physics: PhysicsCandidate;
  semantics: SemanticsCandidate;
  bg?: BgCandidate;
};

type MegaTuneScore = {
  accuracy: number;
  sustain: number;
  collapse: number;
  score: number;
  invalidReason?: string;
};

function isNoImpulseWindow(window: TransitionWindowName): boolean {
  return /noimpulse/i.test(window);
}

type MegaTuneResultEntry = {
  topology: Topology;
  candidate: MegaTuneCandidate;
  score: MegaTuneScore;
  trials: number;
  evaluatedTrials: number;
  seed: number;
  confusion: number[][];
};

function ensureDir(dir: string): void {
  if (!fs.existsSync(dir)) {
    fs.mkdirSync(dir, { recursive: true });
  }
}

function formatBgConfig(bg: BgCandidate): string {
  return `mode=bg epsilon=${bg.epsilon} temperature=${bg.temperature} waitSteps=${bg.waitSteps} sampleActions=${bg.sampleActions}`;
}

function parseListFlag<T>(flag: string, defaultValue: T[], mapper: (v: string) => T): T[] {
  const raw = process.argv.find((arg) => arg.startsWith(flag));
  if (!raw) return defaultValue;
  const [, value] = raw.split('=');
  if (!value) return defaultValue;
  return value
    .split(',')
    .map((v) => v.trim())
    .filter((v) => v.length > 0)
    .map(mapper);
}

function parseBoolFlag(flag: string, defaultValue: boolean): boolean {
  const raw = process.argv.find((arg) => arg.startsWith(flag));
  if (!raw) return defaultValue;
  const idx = raw.indexOf('=');
  const value = idx >= 0 ? raw.slice(idx + 1) : undefined;
  if (value === undefined) return true;
  return !(value === '0' || value.toLowerCase() === 'false');
}

function parseNumberFlag(flag: string, defaultValue: number): number {
  const raw = process.argv.find((arg) => arg.startsWith(flag));
  if (!raw) return defaultValue;
  const [, value] = raw.split('=');
  const parsed = parseFloat(value);
  if (!Number.isFinite(parsed)) {
    throw new Error(`[MegaTune] ${flag} must be numeric. Received: ${value}`);
  }
  return parsed;
}

function parseIntFlag(flag: string, defaultValue: number): number {
  const raw = process.argv.find((arg) => arg.startsWith(flag));
  if (!raw) return defaultValue;
  const [, value] = raw.split('=');
  const parsed = parseInt(value, 10);
  if (!Number.isFinite(parsed)) {
    throw new Error(`[MegaTune] ${flag} must be numeric. Received: ${value}`);
  }
  return parsed;
}

function defaultOutDir(id: string): string {
  return path.join(artifactsDir(), 'megatune', id);
}

function parseTopologies(): Topology[] {
  const parsed = parseListFlag<Topology>(
    '--megatune.topologies=',
    ['snake', 'ring'] as Topology[],
    (v) => v as Topology,
  );
  parsed.forEach((t) => {
    if (t !== 'snake' && t !== 'ring') {
      throw new Error(`[MegaTune] Invalid topology ${t}; must be snake or ring.`);
    }
  });
  return parsed;
}

function parseMegaTuneConfig(): MegaTuneConfig {
  const idArg = process.argv.find((arg) => arg.startsWith('--megatune.id='));
  const id = idArg ? idArg.slice(idArg.indexOf('=') + 1) : `megatune-${Date.now()}`;
  const outDirArg = process.argv.find((arg) => arg.startsWith('--megatune.outDir='));
  const outDir = outDirArg ? outDirArg.slice(outDirArg.indexOf('=') + 1) : defaultOutDir(id);
  const seed = parseIntFlag('--megatune.seed=', 42);

  const budgets: MegaTuneBudgets = {
    guardrailTrials: parseIntFlag('--megatune.guardrail.trials=', 80),
    stage1MaxCandidates: parseIntFlag('--megatune.stage1.maxCandidates=', 200),
    stage1Trials: parseIntFlag('--megatune.stage1.trials=', 200),
    stage1TopK: parseIntFlag('--megatune.stage1.topK=', 20),
    stage2MaxSemantics: parseIntFlag('--megatune.stage2.maxSemantics=', 40),
    stage2Trials: parseIntFlag('--megatune.stage2.trials=', 150),
    stage2TopK: parseIntFlag('--megatune.stage2.topK=', 10),
    stage3MaxBg: parseIntFlag('--megatune.stage3.maxBgConfigs=', 40),
    stage3Trials: parseIntFlag('--megatune.stage3.trials=', 200),
    stage3TopK: parseIntFlag('--megatune.stage3.topK=', 10),
    stage4Enabled: parseBoolFlag('--megatune.stage4.enabled', false),
    stage4MaxCandidates: parseIntFlag('--megatune.stage4.maxCandidates=', 80),
    stage4Trials: parseIntFlag('--megatune.stage4.trials=', 150),
  };

  const constraints: MegaTuneConstraints = {
    allowImpulse: parseBoolFlag('--megatune.constraints.allowImpulse', false),
    forceLearnEqEval: parseBoolFlag('--megatune.constraints.forceLearnEqEval', true),
    minAcc: parseNumberFlag('--megatune.constraints.minAcc=', 0.2),
    maxCollapseFrac: parseNumberFlag('--megatune.constraints.maxCollapseFrac=', 0.6),
  };

  const guardrail: MegaTuneGuardrailConfig = {
    enabled: parseBoolFlag('--megatune.guardrail.enabled', true),
    minAcc: parseNumberFlag('--megatune.guardrail.minAcc=', 0.1),
    maxCollapseFrac: parseNumberFlag('--megatune.guardrail.maxCollapseFrac=', 0.9),
    continueOnFail: parseBoolFlag('--megatune.guardrail.continueOnFail', true),
  };

  if (parseBoolFlag('--megatune.guardrail.enforce', false)) {
    guardrail.continueOnFail = false;
  }

  const search: MegaTuneSearchSpaces = {
    evalWindows: parseListFlag('--megatune.search.evalWindows=', ['meanNoImpulse', 'tailNoImpulse', 'lateNoImpulse'], (v) =>
      v as TransitionWindowName,
    ),
    settleWindows: parseListFlag('--megatune.search.settleWindows=', ['mean', 'mid', 'late'], (v) => v as TransitionWindowName),
    excludeFirstK: parseListFlag('--megatune.search.excludeFirstK=', [0, 1, 2], (v) => parseInt(v, 10)),
    activityModes: parseListFlag('--megatune.search.activityModes=', ['spike', 'ema_spike'], (v) => v as ActivityMode),
    activityAlphas: parseListFlag('--megatune.search.activityAlphas=', [0.05, 0.1, 0.2], (v) => parseFloat(v)),
    tTrans: parseIntFlag('--megatune.search.tTrans=', 40),
    tailLen: parseIntFlag('--megatune.search.tailLen=', 10),
    bg: {
      epsilons: parseListFlag('--megatune.search.bg.epsilons=', [0.02, 0.05, 0.1], (v) => parseFloat(v)),
      temperatures: parseListFlag('--megatune.search.bg.temperatures=', [1.0, 1.4], (v) => parseFloat(v)),
      waitSteps: parseListFlag('--megatune.search.bg.waitSteps=', [2, 5, 10], (v) => parseInt(v, 10)),
      sampleActions: parseListFlag('--megatune.search.bg.sampleActions=', [true, false], (v) => v === 'true'),
    },
  };

  const topologies = parseTopologies();
  const wAcc = parseNumberFlag('--megatune.wAcc=', 1.0);
  const wSustain = parseNumberFlag('--megatune.wSustain=', 1.0);
  const wCollapse = parseNumberFlag('--megatune.wCollapse=', 1.0);

  return {
    id,
    outDir,
    seed,
    topologies,
    wAcc,
    wSustain,
    wCollapse,
    constraints,
    guardrail,
    budgets,
    search,
  };
}

function generatePhysicsGrid(topology: Topology, tTrans: number, tailLen: number): PhysicsCandidate[] {
  const wAttrRowNorms = [1.5, 2.0, 2.5];

  if (topology === 'ring') {
    const k_inhib = [1, 2];
    const v_th = [0.8, 1.0];
    const alpha = [0.9, 0.95];
    const wInRowNorm = [3.0, 4.0];
    const etaTrans = [0.005, 0.01];

    const grid: PhysicsCandidate[] = [];
    for (const k of k_inhib) {
      for (const v of v_th) {
        for (const a of alpha) {
          for (const wIn of wInRowNorm) {
            for (const wAttr of wAttrRowNorms) {
              for (const eta of etaTrans) {
                grid.push({
                  k_inhib: k,
                  v_th: v,
                  alpha: a,
                  wInRowNorm: wIn,
                  wAttrRowNorm: wAttr,
                  etaTrans: eta,
                  tTrans,
                  tailLen,
                });
              }
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

  const grid: PhysicsCandidate[] = [];
  for (const k of k_inhib) {
    for (const v of v_th) {
      for (const a of alpha) {
        for (const wIn of wInRowNorm) {
          for (const wAttr of wAttrRowNorms) {
            for (const eta of etaTrans) {
              grid.push({
                k_inhib: k,
                v_th: v,
                alpha: a,
                wInRowNorm: wIn,
                wAttrRowNorm: wAttr,
                etaTrans: eta,
                tTrans,
                tailLen,
              });
            }
          }
        }
      }
    }
  }
  return grid;
}

function sampleCandidates<T>(candidates: T[], maxCount: number, rng: MathOps.RNG): T[] {
  if (candidates.length <= maxCount) return candidates;
  const shuffled = [...candidates];
  for (let i = shuffled.length - 1; i > 0; i--) {
    const j = Math.floor(rng() * (i + 1));
    [shuffled[i], shuffled[j]] = [shuffled[j], shuffled[i]];
  }
  return shuffled.slice(0, maxCount);
}

function cloneExperimentConfig(config: ExperimentConfig): ExperimentConfig {
  return {
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
      bg: config.controller.bg ? { ...config.controller.bg, minPhaseDuration: { ...config.controller.bg.minPhaseDuration } } : undefined,
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
  };
}

function applyCandidateConfig(base: ExperimentConfig, candidate: MegaTuneCandidate): ExperimentConfig {
  const config = cloneExperimentConfig(base);
  config.curriculum.phaseBC = { ...config.curriculum.phaseBC, ...candidate.physics };
  config.phaseBCConfig = { ...config.phaseBCConfig };
  config.phaseBCConfig.evalWindow = candidate.semantics.evalWindow;
  config.phaseBCConfig.learnWindow = candidate.semantics.learnWindow;
  config.phaseBCConfig.settleWindow = candidate.semantics.settleWindow;
  config.phaseBCConfig.excludeFirstK = candidate.semantics.excludeFirstK;
  if (candidate.semantics.activityMode) {
    config.curriculum.phaseBC.activityMode = candidate.semantics.activityMode;
  }
  if (candidate.semantics.activityAlpha !== undefined) {
    config.curriculum.phaseBC.activityAlpha = candidate.semantics.activityAlpha;
  }
  if (candidate.semantics.tTrans !== undefined) {
    config.curriculum.phaseBC.tTrans = candidate.semantics.tTrans;
  }
  if (candidate.semantics.tailLen !== undefined) {
    config.curriculum.phaseBC.tailLen = candidate.semantics.tailLen;
  }
  if (candidate.bg) {
    config.controller.mode = 'bg';
    config.controller.bg = {
      ...config.controller.bg!,
      epsilon: candidate.bg.epsilon,
      temperature: candidate.bg.temperature,
      waitSteps: candidate.bg.waitSteps,
      sampleActions: candidate.bg.sampleActions,
    };
  }
  return config;
}

function computeCollapsePenalty(confusion: number[][]): number {
  const totals = confusion.flat().reduce((acc, v) => acc + v, 0);
  if (totals === 0) return 0;
  const preds = confusion[0].map((_, colIdx) => confusion.reduce((sum, row) => sum + row[colIdx], 0));
  const maxFrac = Math.max(...preds.map((p) => p / totals));
  return maxFrac;
}

function evaluateCandidate(
  topology: Topology,
  baseConfig: ExperimentConfig,
  snapshot: ColumnSnapshot,
  candidate: MegaTuneCandidate,
  trials: number,
  weights: { wAcc: number; wSustain: number; wCollapse: number },
  constraints: MegaTuneConstraints,
  seed: number,
): MegaTuneResultEntry {
  const config = applyCandidateConfig(baseConfig, candidate);
  const column = MathColumn.fromSnapshot(snapshot);
  column.setRngSeed(seed);
  column.setPhaseBCConfig(config.phaseBCConfig);
  column.applyPhysicsParams(config.curriculum.phaseBC);
  // eta_trans is determined by applyPhysicsParams (from candidate.physics)
  // Keep readout and attractor weights frozen to isolate transition physics performance
  column.learnParams.eta_out = 0;
  column.learnParams.eta_attr = 0;
  const controller = createController(config.controller, config.phaseBCConfig, MathOps.createRng(seed + 1));
  if (candidate.bg) {
    if (config.controller.mode !== 'bg' || !config.controller.bg) {
      throw new Error(`[MegaTune] Expected BG controller config but got mode=${config.controller.mode}`);
    }
    if (!controller.getStats()) {
      throw new Error(`[MegaTune] Expected BG controller instance but got mode=${config.controller.mode}`);
    }
  }
  controller.resetEpisode();

  const numDigits = column.net.numDigits;
  const confusion = Array.from({ length: numDigits }, () => Array(numDigits).fill(0));
  let evaluated = 0;
  let correct = 0;
  let sustainSum = 0;
  const startDigitRng = MathOps.createRng(seed + 2);

  const windows = resolveTransitionWindows(
    config.phaseBCConfig,
    config.curriculum.phaseBC.tTrans ?? DEFAULT_T_TRANS,
    config.curriculum.phaseBC.tailLen ?? DEFAULT_TAIL_LEN,
  );
  const evalLabel = describeTransitionWindow(windows, config.phaseBCConfig.evalWindow).label;
  const invalidReasons: string[] = [];
  if (!constraints.allowImpulse && !isNoImpulseWindow(config.phaseBCConfig.evalWindow)) {
    invalidReasons.push('impulse window not allowed');
  }
  if (constraints.forceLearnEqEval && config.phaseBCConfig.evalWindow !== config.phaseBCConfig.learnWindow) {
    invalidReasons.push('learn/eval mismatch');
  }
  if (
    !constraints.allowImpulse &&
    config.phaseBCConfig.evalWindow.toLowerCase().includes('noimpulse') &&
    config.phaseBCConfig.excludeFirstK < 1
  ) {
    invalidReasons.push('excludeFirstK too small for no-impulse window');
  }

  for (let i = 0; i < trials; i++) {
    const startDigit = Math.floor(startDigitRng() * numDigits);
    const trial = column.runSuccessorTrial(startDigit, i, undefined, undefined, undefined, controller);
    if (trial.aborted) {
      sustainSum += trial.sustain?.tailSilentFrac ?? 1;
      continue;
    }
    evaluated += 1;
    sustainSum += trial.sustain?.tailSilentFrac ?? 1;
    const pred = trial.predictionsByWindow?.eval ?? trial.predDigit ?? 0;
    if (pred === trial.targetDigit) correct += 1;
    confusion[trial.targetDigit][pred] += 1;
  }

  const denom = evaluated > 0 ? evaluated : 1;
  const acc = correct / denom;
  const sustain = 1 - sustainSum / trials;
  const collapse = computeCollapsePenalty(confusion);
  let score = weights.wAcc * acc + weights.wSustain * sustain - weights.wCollapse * collapse;
  let invalidReason: string | undefined;
  if (invalidReasons.length > 0 || collapse > constraints.maxCollapseFrac) {
    invalidReason = invalidReasons.length ? invalidReasons.join('; ') : 'collapse too high';
    score = -Infinity;
  }
  if (acc < constraints.minAcc) {
    invalidReason = invalidReason
      ? `${invalidReason}; acc below minAcc (${acc.toFixed(3)} < ${constraints.minAcc})`
      : `acc below minAcc (${acc.toFixed(3)} < ${constraints.minAcc})`;
    score = -Infinity;
  }

  console.log(
    `[MegaTune Eval] topology=${topology} score=${score.toFixed(3)} acc=${acc.toFixed(3)} sustain=${sustain.toFixed(3)} ` +
      `collapse=${collapse.toFixed(3)} evalWindow=${evalLabel} seed=${seed}${invalidReason ? ` invalid=${invalidReason}` : ''}`,
  );

  return {
    topology,
    candidate,
    score: { accuracy: acc, sustain, collapse, score, invalidReason },
    trials,
    evaluatedTrials: evaluated,
    seed,
    confusion,
  };
}

function rankResults(results: MegaTuneResultEntry[]): MegaTuneResultEntry[] {
  return [...results].sort((a, b) => b.score.score - a.score.score);
}

function writeJson(file: string, payload: unknown): void {
  ensureDir(path.dirname(file));
  fs.writeFileSync(file, JSON.stringify(payload, null, 2));
}

function summarizeBest(topology: Topology, entry: MegaTuneResultEntry): Record<string, unknown> {
  return {
    topology,
    phaseBC: entry.candidate.physics,
    semantics: entry.candidate.semantics,
    bg: entry.candidate.bg,
    scores: entry.score,
    trials: entry.trials,
    evaluatedTrials: entry.evaluatedTrials,
    seed: entry.seed,
  };
}

function buildBaseExperiment(topology: Topology, seed: number): ExperimentConfig {
  return resolveExperimentConfig({
    topology,
    useCountingPhase: true,
    seed,
    includeAccHistory: false,
  });
}

function loadSnapshot(config: ExperimentConfig): ColumnSnapshot {
  if (!config.phaseASnapshotPath || !fs.existsSync(config.phaseASnapshotPath)) {
    throw new Error(`[MegaTune] Missing phaseA snapshot for ${config.topology} at ${config.phaseASnapshotPath}`);
  }
  const raw = fs.readFileSync(config.phaseASnapshotPath, 'utf-8');
  const parsed = JSON.parse(raw) as ColumnSnapshot;
  return parsed;
}

function stageGuardrail(
  topology: Topology,
  base: ExperimentConfig,
  snapshot: ColumnSnapshot,
  cfg: MegaTuneConfig,
  outDir: string,
): { result?: MegaTuneResultEntry; passed: boolean; reason?: string; skipped: boolean } {
  if (!cfg.guardrail.enabled) {
    console.log(`[MegaTune] Stage 0 guardrail disabled for ${topology}; skipping`);
    return { passed: true, skipped: true };
  }

  console.log(`[MegaTune] Stage 0 guardrail for ${topology}`);
  const candidate: MegaTuneCandidate = {
    physics: base.curriculum.phaseBC,
    semantics: {
      evalWindow: 'meanNoImpulse',
      learnWindow: 'meanNoImpulse',
      settleWindow: 'mean',
      excludeFirstK: 1,
    },
  };
  const res = evaluateCandidate(
    topology,
    base,
    snapshot,
    candidate,
    cfg.budgets.guardrailTrials,
    { wAcc: cfg.wAcc, wSustain: cfg.wSustain, wCollapse: cfg.wCollapse },
    { ...cfg.constraints, maxCollapseFrac: cfg.guardrail.maxCollapseFrac, minAcc: cfg.guardrail.minAcc },
    cfg.seed + 10,
  );
  const passed = res.score.accuracy >= cfg.guardrail.minAcc && res.score.collapse <= cfg.guardrail.maxCollapseFrac;
  const reason = passed
    ? undefined
    : `guardrail thresholds unmet (acc=${res.score.accuracy.toFixed(3)} collapse=${res.score.collapse.toFixed(3)})`;
  const file = path.join(outDir, 'guardrail.json');
  writeJson(file, res);
  return { result: res, passed, reason, skipped: false };
}

function stagePhysics(
  topology: Topology,
  base: ExperimentConfig,
  snapshot: ColumnSnapshot,
  cfg: MegaTuneConfig,
  outDir: string,
): MegaTuneResultEntry[] {
  console.log(`[MegaTune] Stage 1 physics shortlist for ${topology}`);
  const grid = generatePhysicsGrid(topology, cfg.search.tTrans, cfg.search.tailLen);
  const rng = MathOps.createRng(cfg.seed + 100);
  const sampled = sampleCandidates(grid, cfg.budgets.stage1MaxCandidates, rng);
  const defaultMode = cfg.search.activityModes[0] ?? 'spike';
  const defaultAlpha = cfg.search.activityAlphas[0] ?? 0.1;
  const semantics: SemanticsCandidate = {
    evalWindow: 'meanNoImpulse',
    learnWindow: 'meanNoImpulse',
    settleWindow: 'mean',
    excludeFirstK: 1,
    activityMode: defaultMode,
    activityAlpha: defaultAlpha,
  };
  const results = sampled.map((physics, idx) =>
    evaluateCandidate(
      topology,
      base,
      snapshot,
      { physics, semantics },
      cfg.budgets.stage1Trials,
      { wAcc: cfg.wAcc, wSustain: cfg.wSustain, wCollapse: cfg.wCollapse },
      cfg.constraints,
      cfg.seed + 200 + idx,
    ),
  );
  const ranked = rankResults(results).slice(0, cfg.budgets.stage1TopK);
  writeJson(path.join(outDir, 'stage1_physics_topK.json'), ranked);
  if (ranked[0]) writeJson(path.join(outDir, 'stage1_physics_best.json'), ranked[0]);
  return ranked;
}

function buildSemanticsGrid(cfg: MegaTuneConfig): SemanticsCandidate[] {
  const grid: SemanticsCandidate[] = [];
  const evalWindows = cfg.constraints.allowImpulse
    ? cfg.search.evalWindows
    : cfg.search.evalWindows.filter(isNoImpulseWindow);

  for (const evalWindow of evalWindows) {
    const learnChoices = cfg.constraints.forceLearnEqEval ? [evalWindow] : evalWindows;
    for (const learnWindow of learnChoices) {
      for (const settleWindow of cfg.search.settleWindows) {
        for (const excludeFirstK of cfg.search.excludeFirstK) {
          for (const activityMode of cfg.search.activityModes) {
            for (const activityAlpha of cfg.search.activityAlphas) {
              grid.push({
                evalWindow,
                learnWindow,
                settleWindow,
                excludeFirstK,
                activityMode,
                activityAlpha,
              });
            }
          }
        }
      }
    }
  }
  return grid;
}

function stageSemantics(
  topology: Topology,
  base: ExperimentConfig,
  snapshot: ColumnSnapshot,
  physicsTopK: MegaTuneResultEntry[],
  cfg: MegaTuneConfig,
  outDir: string,
): MegaTuneResultEntry[] {
  console.log(`[MegaTune] Stage 2 semantics sweep for ${topology}`);
  const semanticsGrid = sampleCandidates(buildSemanticsGrid(cfg), cfg.budgets.stage2MaxSemantics, MathOps.createRng(cfg.seed + 300));
  const results: MegaTuneResultEntry[] = [];
  physicsTopK.forEach((physicsEntry, physicsIdx) => {
    semanticsGrid.forEach((semantics, semIdx) => {
      results.push(
        evaluateCandidate(
          topology,
          base,
          snapshot,
          { physics: physicsEntry.candidate.physics, semantics },
          cfg.budgets.stage2Trials,
          { wAcc: cfg.wAcc, wSustain: cfg.wSustain, wCollapse: cfg.wCollapse },
          cfg.constraints,
          cfg.seed + 400 + physicsIdx * 100 + semIdx,
        ),
      );
    });
  });
  const ranked = rankResults(results).slice(0, cfg.budgets.stage2TopK);
  writeJson(path.join(outDir, 'stage2_semantics_topK.json'), ranked);
  if (ranked[0]) writeJson(path.join(outDir, 'stage2_semantics_best.json'), ranked[0]);
  return ranked;
}

function buildBgGrid(cfg: MegaTuneConfig): BgCandidate[] {
  const grid: BgCandidate[] = [];
  for (const epsilon of cfg.search.bg.epsilons) {
    for (const temperature of cfg.search.bg.temperatures) {
      for (const waitSteps of cfg.search.bg.waitSteps) {
        for (const sampleActions of cfg.search.bg.sampleActions) {
          grid.push({ epsilon, temperature, waitSteps, sampleActions });
        }
      }
    }
  }
  return grid;
}

function stageBg(
  topology: Topology,
  base: ExperimentConfig,
  snapshot: ColumnSnapshot,
  semanticsTop: MegaTuneResultEntry[],
  cfg: MegaTuneConfig,
  outDir: string,
): MegaTuneResultEntry[] {
  console.log(`[MegaTune] Stage 3 BG sweep for ${topology}`);
  const bgGrid = sampleCandidates(buildBgGrid(cfg), cfg.budgets.stage3MaxBg, MathOps.createRng(cfg.seed + 500));
  const results: MegaTuneResultEntry[] = [];
  const baseCandidate = semanticsTop[0];
  if (!baseCandidate) return results;
  bgGrid.forEach((bg, idx) => {
    console.log(`[MegaTune BG] topology=${topology} ${formatBgConfig(bg)} seed=${cfg.seed + 600 + idx}`);
    results.push(
      evaluateCandidate(
        topology,
        base,
        snapshot,
        { physics: baseCandidate.candidate.physics, semantics: baseCandidate.candidate.semantics, bg },
        cfg.budgets.stage3Trials,
        { wAcc: cfg.wAcc, wSustain: cfg.wSustain, wCollapse: cfg.wCollapse },
        cfg.constraints,
        cfg.seed + 600 + idx,
      ),
    );
  });
  const ranked = rankResults(results).slice(0, cfg.budgets.stage3TopK);
  writeJson(path.join(outDir, 'stage3_bg_topK.json'), ranked);
  if (ranked[0]) writeJson(path.join(outDir, 'stage3_bg_best.json'), ranked[0]);
  return ranked;
}

function stageRefinePhysics(
  topology: Topology,
  base: ExperimentConfig,
  snapshot: ColumnSnapshot,
  bestCandidate: MegaTuneCandidate | undefined,
  cfg: MegaTuneConfig,
  outDir: string,
): MegaTuneResultEntry | undefined {
  if (!cfg.budgets.stage4Enabled) return undefined;
  if (!bestCandidate) return undefined;
  console.log(`[MegaTune] Stage 4 refine physics for ${topology}`);
  const grid = sampleCandidates(
    generatePhysicsGrid(topology, cfg.search.tTrans, cfg.search.tailLen),
    cfg.budgets.stage4MaxCandidates,
    MathOps.createRng(cfg.seed + 700),
  );
  const results = grid.map((physics, idx) =>
    evaluateCandidate(
      topology,
      base,
      snapshot,
      { physics, semantics: bestCandidate.semantics, bg: bestCandidate.bg },
      cfg.budgets.stage4Trials,
      { wAcc: cfg.wAcc, wSustain: cfg.wSustain, wCollapse: cfg.wCollapse },
      cfg.constraints,
      cfg.seed + 800 + idx,
    ),
  );
  const ranked = rankResults(results);
  if (ranked[0]) {
    writeJson(path.join(outDir, 'stage4_refine_physics_best.json'), ranked[0]);
    return ranked[0];
  }
  return undefined;
}

function consolidateBest(
  topology: Topology,
  dir: string,
  base: ExperimentConfig,
  stage2Best: MegaTuneResultEntry | undefined,
  stage3Best: MegaTuneResultEntry | undefined,
  refineBest: MegaTuneResultEntry | undefined,
): MegaTuneResultEntry | undefined {
  const candidates = [stage3Best, stage2Best, refineBest].filter(Boolean) as MegaTuneResultEntry[];
  if (candidates.length === 0) return undefined;
  const ranked = rankResults(candidates);
  const best = ranked[0];
  writeJson(path.join(dir, 'best.json'), best);
  return best;
}

export function runMegaTunePhaseBC(): void {
  const cfg = parseMegaTuneConfig();
  console.log(
    `[MegaTune] BG search space epsilons=${JSON.stringify(cfg.search.bg.epsilons)} ` +
      `temperatures=${JSON.stringify(cfg.search.bg.temperatures)} waitSteps=${JSON.stringify(cfg.search.bg.waitSteps)} ` +
      `sampleActions=${JSON.stringify(cfg.search.bg.sampleActions)} stage3MaxBg=${cfg.budgets.stage3MaxBg} ` +
      `stage3Trials=${cfg.budgets.stage3Trials}`,
  );
  const meta = { id: cfg.id, topologies: cfg.topologies, seed: cfg.seed, timestamp: new Date().toISOString() };
  ensureDir(cfg.outDir);
  writeJson(path.join(cfg.outDir, 'meta.json'), meta);

  const summary: Record<string, unknown> = {};
  const bestFiles: { filename: string; dir: string }[] = [];
  const guardrailBlocked: Topology[] = [];
  const artifactsRoot = artifactsDir();

  cfg.topologies.forEach((topology) => {
    const topoDir = path.join(cfg.outDir, topology);
    ensureDir(topoDir);
    const base = buildBaseExperiment(topology, cfg.seed);
    const snapshot = loadSnapshot(base);
    const guardrail = stageGuardrail(topology, base, snapshot, cfg, topoDir);
    if (!guardrail.passed) {
      guardrailBlocked.push(topology);
      console.warn(
        `[MegaTune] Guardrail failed for ${topology} (${guardrail.reason ?? 'unknown reason'})` +
          (cfg.guardrail.continueOnFail ? ' — continuing with tuning' : ' — stopping topology'),
      );
      summary[topology] = {
        guardrail: guardrail.result?.score,
        guardrailPassed: guardrail.passed,
        guardrailReason: guardrail.reason,
        success: false,
        continuedAfterGuardrail: cfg.guardrail.continueOnFail,
      };
      if (!cfg.guardrail.continueOnFail) return;
    }
    const stage1Top = stagePhysics(topology, base, snapshot, cfg, topoDir);
    const stage2Top = stageSemantics(topology, base, snapshot, stage1Top, cfg, topoDir);
    const stage3Top = stageBg(topology, base, snapshot, stage2Top, cfg, topoDir);
    const refineBest = stageRefinePhysics(
      topology,
      base,
      snapshot,
      stage3Top[0]?.candidate ?? stage2Top[0]?.candidate ?? stage1Top[0]?.candidate,
      cfg,
      topoDir,
    );
    const best = consolidateBest(topology, topoDir, base, stage2Top[0], stage3Top[0], refineBest);
    if (best) {
      const filename = `megatune_best_${topology}.json`;
      const stable = path.join(cfg.outDir, filename);
      const bestSummary = summarizeBest(topology, best);
      writeJson(stable, bestSummary);
      bestFiles.push({ filename, dir: cfg.outDir });
      const rootPath = path.join(artifactsRoot, filename);
      fs.copyFileSync(stable, rootPath);
      summary[topology] = {
        ...bestSummary,
        guardrail: guardrail.result?.score,
        guardrailPassed: guardrail.passed,
        guardrailReason: guardrail.reason,
        success: true,
        bestFile: stable,
      };
    } else {
      summary[topology] = {
        guardrail: guardrail.result?.score,
        guardrailPassed: guardrail.passed,
        guardrailReason: guardrail.reason,
        success: false,
      };
    }
  });

  writeJson(path.join(cfg.outDir, 'summary.json'), summary);
  fs.copyFileSync(path.join(cfg.outDir, 'summary.json'), path.join(artifactsRoot, 'summary.json'));
  bestFiles.forEach((file) => exportToSharedArtifacts(file.filename, file.dir));

  const fatalGuardrail = cfg.guardrail.enabled && !cfg.guardrail.continueOnFail && guardrailBlocked.length === cfg.topologies.length;
  if (fatalGuardrail) {
    const failure = {
      reason: 'guardrail_failed',
      message: 'Guardrail failed for all requested topologies; no tuning executed.',
      topologies: guardrailBlocked,
      guardrail: cfg.guardrail,
    };
    const failurePath = path.join(artifactsRoot, 'megatune_failure.json');
    writeJson(failurePath, failure);
    console.error(`[MegaTune] Fatal guardrail failure for all topologies. Details written to ${failurePath}`);
    process.exitCode = 1;
    return;
  }

  if (bestFiles.length === 0) {
    const guardrailAllFailed = guardrailBlocked.length === cfg.topologies.length;
    const failure = {
      reason: guardrailAllFailed ? 'guardrail_failed' : 'no_winners',
      message: guardrailAllFailed
        ? 'Guardrail failed for all requested topologies; tuning could not produce winners.'
        : 'MegaTune completed without producing any winning candidates.',
      guardrail: cfg.guardrail,
      guardrailBlocked,
    };
    const failurePath = path.join(artifactsRoot, 'megatune_failure.json');
    writeJson(failurePath, failure);
    console.error(`[MegaTune] No winning candidates produced. Failure details written to ${failurePath}`);
    process.exitCode = 1;
  }
}

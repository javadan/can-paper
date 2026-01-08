import fs from 'fs';
import path from 'path';
import { spawnSync } from 'child_process';

import { ActivityMode, PhasePhysicsParams, Topology, TransitionWindowName } from '../types';
import type { ExperimentResult, PhaseSummary } from '../types';
import type { PhaseABestRecord, PhaseATuningRecord } from '../phaseATuner';
import type { ProbeTopKPayload } from '../reporting/reportTypes';

type PlanStep = { id: string; mode: string; args?: string[] };
type PlanSuite = { id: string; description?: string; seed?: number; steps: PlanStep[] };
type OvernightPlan = { suites: PlanSuite[] };

type OvernightOptions = {
  planPath: string;
  outDir: string;
  cleanOutDir: boolean;
  failFast: boolean;
  continueOnFail: boolean;
  topologies: Topology[];
  formats: string[];
  vaultOutDir?: string;
  useHtml: boolean;
  resume: boolean;
};

type StepExportInfo = { copied: string[]; missing: string[] };

type StepMeta = {
  id: string;
  mode: string;
  args: string[];
  startTime: string;
  endTime: string;
  durationMs: number;
  exitCode: number | string | null;
  success: boolean;
  artifactsDir: string;
  skipped?: boolean;
  exports?: StepExportInfo;
  error?: string;
};

type SuiteMeta = {
  id: string;
  description?: string;
  seed?: number;
  success: boolean;
  startTime: string;
  endTime: string;
  durationMs: number;
  steps: StepMeta[];
};

type RunMeta = {
  planPath: string;
  outDir: string;
  options: Omit<OvernightOptions, 'planPath' | 'outDir'>;
  startTime: string;
  endTime?: string;
  durationMs?: number;
  suites: SuiteMeta[];
  success?: boolean;
};

type TraceDefaults = {
  traceOutDir: string;
  args: string[];
};

function parseListFlag(flag: string, defaultValue: string[]): string[] {
  const raw = process.argv.find((arg) => arg.startsWith(flag));
  if (!raw) return defaultValue;
  const [, value] = raw.split('=');
  if (!value) return defaultValue;
  return value
    .split(',')
    .map((v) => v.trim())
    .filter((v) => v.length > 0);
}

function parseBoolFlag(flag: string, defaultValue: boolean): boolean {
  const raw = process.argv.find((arg) => arg.startsWith(flag));
  if (!raw) return defaultValue;
  const idx = raw.indexOf('=');
  const value = idx >= 0 ? raw.slice(idx + 1) : undefined;
  if (value === undefined) return true;
  return !(value === '0' || value.toLowerCase() === 'false');
}

function getStringFlag(prefix: string): string | undefined {
  const raw = process.argv.find((arg) => arg.startsWith(prefix));
  if (!raw) return undefined;
  const idx = raw.indexOf('=');
  return idx >= 0 ? raw.slice(idx + 1) : undefined;
}

function parseTopologies(flag = '--overnight.topologies='): Topology[] {
  const parsed = parseListFlag(flag, ['snake', 'ring']) as Topology[];
  parsed.forEach((t) => {
    if (t !== 'snake' && t !== 'ring') {
      throw new Error(`[Overnight] Invalid topology ${t}; must be snake or ring.`);
    }
  });
  return parsed;
}

function timestampSlug(): string {
  return new Date().toISOString().replace(/[:.]/g, '-');
}

function ensureDir(dir: string): void {
  if (!fs.existsSync(dir)) {
    fs.mkdirSync(dir, { recursive: true });
  }
}

function loadPlan(planPath: string): OvernightPlan {
  if (!fs.existsSync(planPath)) {
    throw new Error(`[Overnight] Plan file not found at ${planPath}`);
  }
  const raw = fs.readFileSync(planPath, 'utf-8');
  const parsed = JSON.parse(raw) as OvernightPlan;
  if (!Array.isArray(parsed.suites)) {
    throw new Error('[Overnight] Plan must contain a suites array');
  }
  return parsed;
}

function writeJson(file: string, payload: unknown): void {
  fs.writeFileSync(file, JSON.stringify(payload, null, 2));
}

function removeDirIfExists(dir: string): void {
  if (fs.existsSync(dir)) {
    fs.rmSync(dir, { recursive: true, force: true });
  }
}

function copyIfExists(src: string, dest: string, missing: string[]): boolean {
  if (!fs.existsSync(src)) {
    missing.push(src);
    return false;
  }
  ensureDir(path.dirname(dest));
  fs.copyFileSync(src, dest);
  return true;
}

function copyReportFolder(src: string, dest: string): void {
  if (fs.existsSync(dest)) {
    removeDirIfExists(dest);
  }
  fs.cpSync(src, dest, { recursive: true });
}

const toRelative = (from: string, to: string): string => path.relative(from, to).replace(/\\/g, '/');

type PhaseABestPayload = {
  timestamp?: string;
  bestPerTopology?: Partial<Record<Topology, PhaseABestRecord | null>>;
  allResults?: PhaseATuningRecord[];
};

function findPhaseAArtifact(filename: string, suiteDir: string, suiteMeta: SuiteMeta): string | undefined {
  const suiteArtifactCandidate = path.join(suiteDir, 'suite_artifacts', filename);
  if (fs.existsSync(suiteArtifactCandidate)) return suiteArtifactCandidate;

  const tuneStep = suiteMeta.steps.find((s) => s.mode === 'tune-phaseA' || s.mode === 'tune:phaseA');
  if (tuneStep) {
    const stepCandidate = path.join(tuneStep.artifactsDir, filename);
    if (fs.existsSync(stepCandidate)) return stepCandidate;
  }

  return undefined;
}

function loadPhaseABest(suiteDir: string, suiteMeta: SuiteMeta): { payload: PhaseABestPayload; path: string } | undefined {
  const bestPath = findPhaseAArtifact('phaseA_best.json', suiteDir, suiteMeta);
  if (!bestPath) return undefined;

  const raw = fs.readFileSync(bestPath, 'utf-8');
  const payload = JSON.parse(raw) as PhaseABestPayload;
  return { payload, path: bestPath };
}

type PhaseAFinetunePayload = {
  timestamp?: string;
  results?: Partial<
    Record<
      Topology,
      | (PhaseABestRecord & {
          accuracyByDigit?: number[];
          meanAcc?: number;
          minDigitAcc?: number;
          stepsTrained?: number;
        })
      | null
    >
  >;
};

type MegatuneScore = {
  accuracy?: number;
  sustain?: number;
  collapse?: number;
  score?: number;
  invalidReason?: string;
};

type MegatuneSemantics = {
  evalWindow?: TransitionWindowName;
  learnWindow?: TransitionWindowName;
  settleWindow?: TransitionWindowName;
  excludeFirstK?: number;
  activityMode?: ActivityMode;
  activityAlpha?: number;
  tTrans?: number;
  tailLen?: number;
};

type MegatuneBg = {
  epsilon?: number;
  temperature?: number;
  waitSteps?: number;
  sampleActions?: boolean;
};

type MegatuneBestPayload = {
  topology?: Topology;
  phaseBC?: PhasePhysicsParams;
  semantics?: MegatuneSemantics;
  bg?: MegatuneBg;
  scores?: MegatuneScore;
  trials?: number;
  evaluatedTrials?: number;
  seed?: number;
};

type MegatuneSummaryEntry = MegatuneBestPayload & {
  guardrail?: MegatuneScore;
  guardrailPassed?: boolean;
  guardrailReason?: string;
  continuedAfterGuardrail?: boolean;
  success?: boolean;
  bestFile?: string;
};

type MegatuneSummaryPayload = Record<string, MegatuneSummaryEntry>;

function loadPhaseAFinetune(suiteDir: string, suiteMeta: SuiteMeta):
  | { payload: PhaseAFinetunePayload; path: string }
  | undefined {
  const bestPath = findPhaseAArtifact('phaseA_finetune_best.json', suiteDir, suiteMeta);
  if (!bestPath) return undefined;

  const raw = fs.readFileSync(bestPath, 'utf-8');
  const payload = JSON.parse(raw) as PhaseAFinetunePayload;
  return { payload, path: bestPath };
}

function findPhaseACompanionArtifacts(suiteDir: string, suiteMeta: SuiteMeta): string[] {
  const filenames = ['phaseA_tuning_progress.json', 'phaseA_state_snake.json', 'phaseA_state_ring.json'];
  const found: string[] = [];

  for (const filename of filenames) {
    const candidate = findPhaseAArtifact(filename, suiteDir, suiteMeta);
    if (candidate) found.push(candidate);
  }

  return found;
}

function loadProbeTopKArtifacts(suiteDir: string): { topology: Topology; payload: ProbeTopKPayload; path: string }[] {
  const artifacts: { topology: Topology; payload: ProbeTopKPayload; path: string }[] = [];
  (['snake', 'ring'] as Topology[]).forEach((topology) => {
    const filename = `probeBC_topK_${topology}.json`;
    const candidate = path.join(suiteDir, 'suite_artifacts', filename);
    if (!fs.existsSync(candidate)) return;
    const payload = JSON.parse(fs.readFileSync(candidate, 'utf-8')) as ProbeTopKPayload;
    artifacts.push({ topology, payload, path: candidate });
  });
  return artifacts;
}

function loadProbeBestArtifacts(
  suiteDir: string,
): { topology: Topology; path: string }[] {
  const artifacts: { topology: Topology; path: string }[] = [];
  (['snake', 'ring'] as Topology[]).forEach((topology) => {
    const filename = `probeBC_best_${topology}.json`;
    const candidate = path.join(suiteDir, 'suite_artifacts', filename);
    if (!fs.existsSync(candidate)) return;
    artifacts.push({ topology, path: candidate });
  });
  return artifacts;
}

function loadMegatuneSummary(
  suiteDir: string,
): { payload: MegatuneSummaryPayload; path: string } | undefined {
  const filename = 'summary.json';
  const candidate = path.join(suiteDir, 'suite_artifacts', filename);
  if (!fs.existsSync(candidate)) return undefined;

  const payload = JSON.parse(fs.readFileSync(candidate, 'utf-8')) as MegatuneSummaryPayload;
  return { payload, path: candidate };
}

function loadMegatuneBestArtifacts(
  suiteDir: string,
): { topology: Topology; payload: MegatuneBestPayload; path: string }[] {
  const artifacts: { topology: Topology; payload: MegatuneBestPayload; path: string }[] = [];
  (['snake', 'ring'] as Topology[]).forEach((topology) => {
    const filename = `megatune_best_${topology}.json`;
    const candidate = path.join(suiteDir, 'suite_artifacts', filename);
    if (!fs.existsSync(candidate)) return;
    const payload = JSON.parse(fs.readFileSync(candidate, 'utf-8')) as MegatuneBestPayload;
    artifacts.push({ topology, payload, path: candidate });
  });
  return artifacts;
}

function loadRunArtifacts(suiteDir: string): { label: string; result: ExperimentResult; path: string }[] {
  const artifacts: { label: string; result: ExperimentResult; path: string }[] = [];
  (['snake', 'ring'] as Topology[]).forEach((topology) => {
    ['countingOn', 'countingOff'].forEach((mode) => {
      const filename = `run_last_${topology}_${mode}.json`;
      const candidate = path.join(suiteDir, 'suite_artifacts', filename);
      if (!fs.existsSync(candidate)) return;
      const payload = JSON.parse(fs.readFileSync(candidate, 'utf-8')) as ExperimentResult;
      artifacts.push({ label: `${topology}_${mode}`, result: payload, path: candidate });
    });
  });
  return artifacts;
}

const formatNum = (value: number | undefined, digits = 3): string =>
  Number.isFinite(value as number) ? (value as number).toFixed(digits) : 'n/a';

function maybeWriteHtml(mdPath: string, enabled: boolean): void {
  if (!enabled || !fs.existsSync(mdPath)) return;
  const content = fs.readFileSync(mdPath, 'utf-8');
  const html = `<!doctype html>\n<html><head><meta charset="utf-8"><title>${path.basename(mdPath)}</title></head><body><pre>${content.replace(
    /</g,
    '&lt;',
  )}</pre></body></html>`;
  fs.writeFileSync(mdPath.replace(/\.md$/, '.html'), html);
}

function exportArtifacts(
  mode: string,
  stepDir: string,
  stepArtifactsDir: string,
  suiteArtifactsDir: string,
  options: { topologies: Topology[]; failFast: boolean; useHtml: boolean },
): StepExportInfo {
  const missing: string[] = [];
  const copied: string[] = [];

  const exportRequiredFile = (filename: string) => {
    const source = path.join(stepArtifactsDir, filename);
    const dest = path.join(suiteArtifactsDir, filename);
    if (copyIfExists(source, dest, missing)) copied.push(filename);
  };

  const exportOptionalFile = (filename: string) => {
    const source = path.join(stepArtifactsDir, filename);
    if (!fs.existsSync(source)) return;
    const dest = path.join(suiteArtifactsDir, filename);
    ensureDir(path.dirname(dest));
    fs.copyFileSync(source, dest);
    copied.push(filename);
  };

  if (mode === 'tune-phaseA' || mode === 'tune:phaseA') {
    exportRequiredFile('phaseA_best.json');
    exportOptionalFile('phaseA_topK.json');
  } else if (mode === 'megatune-phaseBC') {
    options.topologies.forEach((topology) => {
      exportRequiredFile(`megatune_best_${topology}.json`);
    });
    exportOptionalFile('summary.json');
  } else if (mode === 'probe-phaseBC') {
    options.topologies.forEach((topology) => {
      exportRequiredFile(`probeBC_best_${topology}.json`);
      exportRequiredFile(`probeBC_topK_${topology}.json`);
    });
  } else if (mode === 'finetune-phaseA' || mode === 'finetune:phaseA') {
    exportRequiredFile('phaseA_finetune_best.json');
    options.topologies.forEach((topology) => {
      exportRequiredFile(`phaseA_state_${topology}.json`);
    });
    exportOptionalFile('phaseA_finetune_progress.json');
    exportOptionalFile('phaseA_validation.json');
  } else if (mode === 'run') {
    options.topologies.forEach((topology) => {
      exportRequiredFile(`run_last_${topology}_countingOn.json`);
      exportRequiredFile(`run_last_${topology}_countingOff.json`);
    });
  } else if (mode === 'report') {
    const reportDir = path.join(stepDir, 'report');
    const destDir = path.join(suiteArtifactsDir, 'report');
    if (fs.existsSync(reportDir)) {
      const indexPath = path.join(reportDir, 'index.md');
      maybeWriteHtml(indexPath, options.useHtml);
      copyReportFolder(reportDir, destDir);
      copied.push('report/');
    } else {
      missing.push(reportDir);
    }
  }

  return { copied, missing };
}

function sanitizeArgs(args: string[], prefix: string): string[] {
  return args.filter((arg) => !arg.startsWith(prefix));
}

function extractArgValue(args: string[], prefix: string): string | undefined {
  const raw = args.find((arg) => arg.startsWith(prefix));
  if (!raw) return undefined;
  const idx = raw.indexOf('=');
  return idx >= 0 ? raw.slice(idx + 1) : undefined;
}

function ensureTraceDefaults(args: string[], suiteArtifactsDir: string, stepId: string): TraceDefaults {
  const nextArgs = [...args];
  const hasTraceFlag = nextArgs.some((arg) => arg.startsWith('--debug.transitionTrace'));
  if (!hasTraceFlag) {
    nextArgs.push('--debug.transitionTrace=true');
  }

  let traceOutDir = extractArgValue(nextArgs, '--debug.traceOutDir=');
  if (!traceOutDir) {
    traceOutDir = path.join(suiteArtifactsDir, 'transition_traces', stepId);
    nextArgs.push(`--debug.traceOutDir=${traceOutDir}`);
  }

  return { traceOutDir, args: nextArgs };
}

function resolveTraceDir(traceOutDir: string, stepArtifactsDir: string): string {
  const resolvedTraceDir = path.resolve(traceOutDir);
  const resolvedArtifacts = path.resolve(stepArtifactsDir);
  const isTraceInArtifacts =
    resolvedTraceDir === resolvedArtifacts || resolvedTraceDir.startsWith(`${resolvedArtifacts}${path.sep}`);
  if (path.isAbsolute(traceOutDir) || isTraceInArtifacts) {
    return resolvedTraceDir;
  }
  return path.resolve(path.join(resolvedArtifacts, traceOutDir));
}

function runTracePlots(
  traceDir: string,
  stepArtifactsDir: string,
  formats: string[],
): { success: boolean; logs: string } {
  if (!fs.existsSync(traceDir)) {
    return { success: true, logs: `[Overnight] Trace directory not found for plotting: ${traceDir}\n` };
  }

  const traceFiles = fs
    .readdirSync(traceDir)
    .filter((file) => file.startsWith('trace_') && file.endsWith('.json'))
    .map((file) => path.join(traceDir, file));
  if (traceFiles.length === 0) {
    return { success: true, logs: `[Overnight] No trace files found in ${traceDir}\n` };
  }

  const plotDir = path.join(traceDir, 'plots');
  ensureDir(plotDir);

  const logs: string[] = [];
  let success = true;
  const tracePattern = /trace_(snake|ring)_(countingOn|countingOff)_/;
  for (const traceFile of traceFiles) {
    const match = path.basename(traceFile).match(tracePattern);
    const topology = match?.[1] as Topology | undefined;
    if (!topology) {
      logs.push(`[Overnight] Skipping trace file with unknown topology: ${traceFile}`);
      continue;
    }
    const runInputPreferred = path.join(stepArtifactsDir, `run_last_${topology}_countingOn.json`);
    const runInputFallback = path.join(stepArtifactsDir, `run_last_${topology}_countingOff.json`);
    const runInput = fs.existsSync(runInputPreferred)
      ? runInputPreferred
      : fs.existsSync(runInputFallback)
        ? runInputFallback
        : undefined;
    const plotKind = runInput ? 'run' : 'trace';
    if (!runInput) {
      logs.push(
        `[Overnight] Missing run output for trace plotting (${topology}). Expected ${runInputPreferred}. Falling back to trace-only plotting.`,
      );
    }

    const proc = spawnSync(
      'python',
      [
        'scripts/report_plot.py',
        `--kind=${plotKind}`,
        ...(runInput ? [`--input=${runInput}`] : []),
        `--outDir=${plotDir}`,
        `--formats=${formats.join(',')}`,
        `--traceFile=${traceFile}`,
      ],
      { encoding: 'utf-8', maxBuffer: 1024 * 1024 * 10 },
    );
    const combinedLogs = `${proc.stdout ?? ''}${proc.stderr ?? ''}`;
    logs.push(combinedLogs);
    if (proc.error) {
      logs.push(`[Overnight] Trace plot spawn error: ${proc.error.message}`);
      success = false;
      continue;
    }
    if ((proc.status ?? 1) !== 0) {
      logs.push(`[Overnight] Trace plot failed with status=${proc.status ?? 'unknown'}`);
      success = false;
    }
  }

  return { success, logs: logs.join('\n') + (logs.length ? '\n' : '') };
}

function buildStepArgs(
  step: PlanStep,
  stepArtifactsDir: string,
  suiteArtifactsDir: string,
  options: OvernightOptions,
  suiteSeed?: number,
): string[] {
  const args = [...(step.args ?? [])];
  const cleaned = sanitizeArgs(args, '--artifacts.dir=');
  const finalArgs = [...cleaned];

  if (suiteSeed !== undefined) {
    const hasProbeSeed = finalArgs.some((arg) => arg.startsWith('--probe.seed='));
    const hasSeed = finalArgs.some((arg) => arg.startsWith('--seed='));
    if (step.mode === 'probe-phaseBC' && !hasProbeSeed) {
      finalArgs.push(`--probe.seed=${suiteSeed}`);
    }
    if (step.mode === 'run' && !hasSeed) {
      finalArgs.push(`--seed=${suiteSeed}`);
    }
  }

  if (step.mode === 'report') {
    const cleanedReportArgs = sanitizeArgs(sanitizeArgs(finalArgs, '--report.outDir='), '--report.artifactsDir=');
    const withoutTopologies = sanitizeArgs(cleanedReportArgs, '--report.topologies=');
    const withoutFormats = sanitizeArgs(withoutTopologies, '--report.formats=');
    finalArgs.length = 0;
    finalArgs.push(...withoutFormats);
    finalArgs.push(`--report.outDir=${path.join(path.dirname(stepArtifactsDir), 'report')}`);
    finalArgs.push(`--report.artifactsDir=${suiteArtifactsDir}`);
    if (!withoutFormats.some((arg) => arg.startsWith('--report.formats='))) {
      finalArgs.push(`--report.formats=${options.formats.join(',')}`);
    }
    if (!withoutTopologies.some((arg) => arg.startsWith('--report.topologies='))) {
      finalArgs.push(`--report.topologies=${options.topologies.join(',')}`);
    }
  }

  finalArgs.push(`--artifacts.dir=${stepArtifactsDir}`);
  return finalArgs;
}

function executeTraceStep(
  suiteDir: string,
  suiteArtifactsDir: string,
  step: PlanStep,
  idx: number,
  options: OvernightOptions,
  suiteSeed?: number,
): StepMeta {
  const stepLabel = `${String(idx + 1).padStart(2, '0')}_${step.id}`;
  const stepDir = path.join(suiteDir, 'steps', stepLabel);
  const stepArtifactsDir = path.join(stepDir, 'artifacts');
  ensureDir(stepArtifactsDir);
  if (fs.existsSync(suiteArtifactsDir)) {
    fs.cpSync(suiteArtifactsDir, stepArtifactsDir, { recursive: true });
  }

  const timingPath = path.join(stepDir, 'timing.json');
  if (options.resume && fs.existsSync(timingPath)) {
    const existing = JSON.parse(fs.readFileSync(timingPath, 'utf-8')) as StepMeta;
    if (existing.exitCode === 0 && existing.success) {
      return { ...existing, skipped: true };
    }
  }

  const traceDefaults = ensureTraceDefaults(step.args ?? [], suiteArtifactsDir, step.id);
  const runStep: PlanStep = { ...step, mode: 'run', args: traceDefaults.args };
  const args = buildStepArgs(runStep, stepArtifactsDir, suiteArtifactsDir, options, suiteSeed);
  console.log(
    `[Overnight] Executing step ${stepLabel} (trace->run) with args: ${args.join(' ') || '(none)'} | artifacts: ${stepArtifactsDir}`,
  );
  const start = Date.now();
  const spawnOptions = {
    encoding: 'utf-8',
    maxBuffer: 1024 * 1024 * 20,
    shell: process.platform === 'win32',
  } as const;
  const proc = spawnSync('npx', ['ts-node', 'src/main.ts', 'run', ...args], spawnOptions);
  const runEnd = Date.now();
  let combinedLogs = `${proc.stdout ?? ''}${proc.stderr ?? ''}`;

  console.log(
    `[Overnight] Spawn result for ${stepLabel}: status=${proc.status} signal=${proc.signal} error=${proc.error?.message ?? 'none'}`,
  );

  let success = !proc.error && (proc.status ?? 1) === 0;
  let exitCode: number | string | null = proc.status ?? proc.signal ?? (proc.error ? 'spawn-error' : null);
  if (success) {
    const resolvedTraceDir = resolveTraceDir(traceDefaults.traceOutDir, stepArtifactsDir);
    const plotResult = runTracePlots(resolvedTraceDir, stepArtifactsDir, options.formats);
    combinedLogs += plotResult.logs;
    success = plotResult.success;
    if (!success) {
      exitCode = 'plot-failed';
    }
  }
  const end = Date.now();

  ensureDir(stepDir);
  fs.writeFileSync(path.join(stepDir, 'logs.txt'), combinedLogs);
  writeJson(path.join(stepDir, 'args.json'), { command: 'npx ts-node src/main.ts', args });

  const meta: StepMeta = {
    id: step.id,
    mode: step.mode,
    args,
    startTime: new Date(start).toISOString(),
    endTime: new Date(end).toISOString(),
    durationMs: end - start,
    exitCode,
    success,
    artifactsDir: stepArtifactsDir,
    error: proc.error?.message,
  };

  console.log(
    `[Overnight] Step ${stepLabel} (${step.mode}) completed with exitCode=${meta.exitCode} success=${meta.success} duration=${meta.durationMs}ms`,
  );

  writeJson(timingPath, meta);

  if (proc.error) {
    throw new Error(`[Overnight] Step ${stepLabel} failed to spawn: ${proc.error.message}`);
  }
  return meta;
}

function executeStep(
  suiteDir: string,
  suiteArtifactsDir: string,
  step: PlanStep,
  idx: number,
  options: OvernightOptions,
  suiteSeed?: number,
): StepMeta {
  if (step.mode === 'trace') {
    return executeTraceStep(suiteDir, suiteArtifactsDir, step, idx, options, suiteSeed);
  }
  const stepLabel = `${String(idx + 1).padStart(2, '0')}_${step.id}`;
  const stepDir = path.join(suiteDir, 'steps', stepLabel);
  const stepArtifactsDir = path.join(stepDir, 'artifacts');
  ensureDir(stepArtifactsDir);
  if (fs.existsSync(suiteArtifactsDir)) {
    fs.cpSync(suiteArtifactsDir, stepArtifactsDir, { recursive: true });
  }

  const timingPath = path.join(stepDir, 'timing.json');
  if (options.resume && fs.existsSync(timingPath)) {
    const existing = JSON.parse(fs.readFileSync(timingPath, 'utf-8')) as StepMeta;
    if (existing.exitCode === 0 && existing.success) {
      return { ...existing, skipped: true };
    }
  }

  const args = buildStepArgs(step, stepArtifactsDir, suiteArtifactsDir, options, suiteSeed);
  console.log(
    `[Overnight] Executing step ${stepLabel} (${step.mode}) with args: ${args.join(' ') || '(none)'} | artifacts: ${stepArtifactsDir}`,
  );
  const start = Date.now();
  const spawnOptions = {
    encoding: 'utf-8',
    maxBuffer: 1024 * 1024 * 20,
    shell: process.platform === 'win32',
  } as const;
  const proc = spawnSync('npx', ['ts-node', 'src/main.ts', step.mode, ...args], spawnOptions);
  const end = Date.now();
  const combinedLogs = `${proc.stdout ?? ''}${proc.stderr ?? ''}`;

  console.log(
    `[Overnight] Spawn result for ${stepLabel}: status=${proc.status} signal=${proc.signal} error=${proc.error?.message ?? 'none'}`,
  );

  ensureDir(stepDir);
  fs.writeFileSync(path.join(stepDir, 'logs.txt'), combinedLogs);
  writeJson(path.join(stepDir, 'args.json'), { command: 'npx ts-node src/main.ts', args });

  let exports: StepExportInfo | undefined;
  let success = !proc.error && (proc.status ?? 1) === 0;
  if (success) {
    exports = exportArtifacts(step.mode, stepDir, stepArtifactsDir, suiteArtifactsDir, {
      topologies: options.topologies,
      failFast: options.failFast,
      useHtml: options.useHtml,
    });
    if (exports.missing.length > 0 && options.failFast) {
      console.warn(`[Overnight] Missing expected artifacts for ${step.mode}: ${exports.missing.join(', ')}`);
    }
    success = success && exports.missing.length === 0;
  }

  const meta: StepMeta = {
    id: step.id,
    mode: step.mode,
    args,
    startTime: new Date(start).toISOString(),
    endTime: new Date(end).toISOString(),
    durationMs: end - start,
    exitCode: proc.status ?? proc.signal ?? (proc.error ? 'spawn-error' : null),
    success,
    artifactsDir: stepArtifactsDir,
    exports,
    error: proc.error?.message,
  };

  console.log(
    `[Overnight] Step ${stepLabel} (${step.mode}) completed with exitCode=${meta.exitCode} success=${meta.success} duration=${meta.durationMs}ms`,
  );

  writeJson(timingPath, meta);

  if (proc.error) {
    throw new Error(`[Overnight] Step ${stepLabel} failed to spawn: ${proc.error.message}`);
  }
  return meta;
}

function writeCombinedReport(
  suiteDir: string,
  suiteMeta: SuiteMeta,
  options: Pick<OvernightOptions, 'useHtml'>,
): void {
  const combinedDir = path.join(suiteDir, 'combined_report');
  ensureDir(combinedDir);
  const lines: string[] = [];
  lines.push(`# Suite ${suiteMeta.id}`);
  if (suiteMeta.description) {
    lines.push('');
    lines.push(suiteMeta.description);
  }
  lines.push('');
  lines.push(`Status: ${suiteMeta.success ? '✅ success' : '❌ failure'}`);
  lines.push('');
  lines.push('## Steps');
  suiteMeta.steps.forEach((step, stepIdx) => {
    lines.push(`- ${step.id} (${step.mode}): ${step.skipped ? 'skipped' : step.success ? 'ok' : 'failed'}`);
    lines.push(`  - logs: steps/${String(stepIdx + 1).padStart(2, '0')}_${step.id}/logs.txt`);
    if (step.exports?.copied?.length) {
      lines.push(`  - exported: ${step.exports.copied.join(', ')}`);
    }
    if (step.exports?.missing?.length) {
      lines.push(`  - missing: ${step.exports.missing.join(', ')}`);
    }
  });

  const phaseABest = loadPhaseABest(suiteDir, suiteMeta);
  lines.push('');
  lines.push('## Phase A (Digits) — tuned params');
  if (phaseABest) {
    lines.push('');
    lines.push('| topology | finalAccuracy | steps | snapshotWindowAcc | snapshotSteps | params |');
    lines.push('| --- | --- | --- | --- | --- | --- |');

    const warnings: string[] = [];
    (['snake', 'ring'] as Topology[]).forEach((topology) => {
      const record = phaseABest.payload.bestPerTopology?.[topology];
      const finalAccuracy = record?.finalAccuracy ?? NaN;
      const snapshotAcc = record?.snapshot?.finalWindowAcc ?? NaN;
      const snapshotTarget = record?.snapshot?.targetAcc ?? 0.99;
      const params = record?.params;

      const paramText = params
        ? `k_inhib=${params.k_inhib}, v_th=${params.v_th}, alpha=${params.alpha}, ` +
          `wInRowNorm=${params.wInRowNorm}, wAttrRowNorm=${params.wAttrRowNorm}, ` +
          `eta_attr=${params.eta_attr}, eta_out=${params.eta_out}`
        : 'n/a';

      if (record && !Number.isNaN(finalAccuracy) && finalAccuracy < 0.99) {
        warnings.push(`- ${topology}: tuning finalAccuracy ${finalAccuracy.toFixed(3)} < 0.99`);
      }

      if (record?.snapshot && !Number.isNaN(snapshotAcc) && snapshotAcc < snapshotTarget) {
        warnings.push(
          `- ${topology}: snapshot windowAcc ${snapshotAcc.toFixed(3)} < target ${snapshotTarget.toFixed(2)}`,
        );
      }

      lines.push(
        `| ${topology} | ${Number.isFinite(finalAccuracy) ? finalAccuracy.toFixed(3) : 'n/a'} | ${record?.steps ?? 'n/a'} | ` +
          `${Number.isFinite(snapshotAcc) ? snapshotAcc.toFixed(3) : 'n/a'} | ${record?.snapshot?.stepsTrained ?? 'n/a'} | ${paramText} |`,
      );
    });

    const artifactLinks = [phaseABest.path, ...findPhaseACompanionArtifacts(suiteDir, suiteMeta)].map((p) => {
      const rel = toRelative(combinedDir, p);
      return `[${path.basename(p)}](${rel})`;
    });

    if (artifactLinks.length > 0) {
      lines.push('');
      lines.push(`Artifacts: ${artifactLinks.join(', ')}`);
    }

    if (warnings.length > 0) {
      lines.push('');
      lines.push('Warnings:');
      warnings.forEach((w) => lines.push(`- ${w}`));
    }
  } else {
    lines.push('');
    lines.push('- phaseA_best.json not found; Phase A summary unavailable.');
  }

  const phaseAFinetune = loadPhaseAFinetune(suiteDir, suiteMeta);
  lines.push('');
  lines.push('## Phase A finetune summary');
  if (phaseAFinetune?.payload?.results) {
    lines.push('');
    lines.push('| topology | finalAcc | meanAcc | minDigitAcc | worstDigit | stepsTrained |');
    lines.push('| --- | --- | --- | --- | --- | --- |');
    (['snake', 'ring'] as Topology[]).forEach((topology) => {
      const record = phaseAFinetune.payload.results?.[topology];
      const finalAcc = (record as any)?.finalAccuracy ?? (record as any)?.meanAcc;
      const meanAcc = (record as any)?.meanAcc ?? (record as any)?.finalAccuracy;
      const minDigitAcc = (record as any)?.minDigitAcc;
      const accuracyByDigit = (record as any)?.accuracyByDigit as number[] | undefined;
      const stepsTrained = (record as any)?.stepsTrained ?? (record as any)?.steps;

      const worstIdx = accuracyByDigit ? accuracyByDigit.indexOf(Math.min(...accuracyByDigit)) : undefined;
      const worstDigit =
        accuracyByDigit && worstIdx !== undefined && worstIdx >= 0 ? `${worstIdx} (${accuracyByDigit[worstIdx].toFixed(3)})` : 'n/a';

      lines.push(
        `| ${topology} | ${
          Number.isFinite(finalAcc) ? (finalAcc as number).toFixed(3) : 'n/a'
        } | ${
          Number.isFinite(meanAcc) ? (meanAcc as number).toFixed(3) : 'n/a'
        } | ${
          Number.isFinite(minDigitAcc) ? (minDigitAcc as number).toFixed(3) : 'n/a'
        } | ${worstDigit} | ${stepsTrained ?? 'n/a'} |`,
      );
    });

    const finetuneArtifacts = [phaseAFinetune.path, findPhaseAArtifact('phaseA_validation.json', suiteDir, suiteMeta)]
      .filter(Boolean)
      .map((p) => {
        const rel = toRelative(combinedDir, p as string);
        return `[${path.basename(p as string)}](${rel})`;
      });
    if (finetuneArtifacts.length > 0) {
      lines.push('');
      lines.push(`Artifacts: ${finetuneArtifacts.join(', ')}`);
    }

    const perDigitLines = (['snake', 'ring'] as Topology[])
      .map((topology) => {
        const accuracyByDigit = (phaseAFinetune.payload.results?.[topology] as any)?.accuracyByDigit as number[] | undefined;
        if (!accuracyByDigit) return null;
        const formatted = accuracyByDigit.map((acc, idx) => `${idx}:${acc.toFixed(3)}`).join(', ');
        return `- ${topology} per-digit accuracy: ${formatted}`;
      })
      .filter(Boolean) as string[];

    if (perDigitLines.length > 0) {
      lines.push('');
      lines.push('Per-digit accuracy:');
      perDigitLines.forEach((line) => lines.push(line));
    }
  } else {
    lines.push('');
    lines.push('- phaseA_finetune_best.json not found; finetune summary unavailable.');
  }

  const megatuneSummary = loadMegatuneSummary(suiteDir);
  const megatuneBest = loadMegatuneBestArtifacts(suiteDir);
  lines.push('');
  lines.push('## Megatune Phase B/C');

  const megatuneEntries: {
    topology: Topology;
    summary?: MegatuneSummaryEntry;
    best?: { payload: MegatuneBestPayload; path: string };
  }[] = [];

  (['snake', 'ring'] as Topology[]).forEach((topology) => {
    const summaryEntry = megatuneSummary?.payload?.[topology];
    const bestEntry = megatuneBest.find((artifact) => artifact.topology === topology);
    if (summaryEntry || bestEntry) {
      megatuneEntries.push({
        topology,
        summary: summaryEntry,
        best: bestEntry ? { payload: bestEntry.payload, path: bestEntry.path } : undefined,
      });
    }
  });

  if (megatuneEntries.length === 0) {
    lines.push('');
    lines.push('- megatune artifacts not found; Megatune summary unavailable.');
  } else {
    lines.push('');
    lines.push(
      '| topology | success | score | acc | sustain | collapse | guardrailScore | guardrailPass | guardrailReason | trials | evaluated | seed | artifact |',
    );
    lines.push(
      '| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |',
    );

    megatuneEntries.forEach((entry) => {
      const scores = entry.summary?.scores ?? entry.best?.payload.scores;
      const guardrailScore = entry.summary?.guardrail?.score;
      const trials = entry.summary?.trials ?? entry.best?.payload.trials;
      const evaluated = entry.summary?.evaluatedTrials ?? entry.best?.payload.evaluatedTrials;
      const seed = entry.summary?.seed ?? entry.best?.payload.seed;
      const artifactPath = entry.summary?.bestFile ?? entry.best?.path;
      const artifactLink = artifactPath ? `[${path.basename(artifactPath)}](${toRelative(combinedDir, artifactPath)})` : 'n/a';

      lines.push(
        `| ${entry.topology} | ${
          entry.summary?.success === undefined ? 'n/a' : entry.summary.success ? 'yes' : 'no'
        } | ${formatNum(scores?.score)} | ${formatNum(scores?.accuracy)} | ${formatNum(
          scores?.sustain,
        )} | ${formatNum(scores?.collapse)} | ${formatNum(guardrailScore)} | ${
          entry.summary?.guardrailPassed === undefined
            ? 'n/a'
            : entry.summary.guardrailPassed
              ? 'yes'
              : 'no'
        } | ${entry.summary?.guardrailReason ?? 'n/a'} | ${trials ?? 'n/a'} | ${evaluated ?? 'n/a'} | ${
          seed ?? 'n/a'
        } | ${artifactLink} |`,
      );
    });

    lines.push('');
    lines.push(
      '| topology | k_inhib | v_th | alpha | wInRowNorm | wAttrRowNorm | etaTrans | activity | activityAlpha | tTrans | tailLen | eval | learn | settle | excludeFirstK | bg.epsilon | bg.temperature | bg.waitSteps | bg.sampleActions |',
    );
    lines.push(
      '| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |',
    );

    megatuneEntries.forEach((entry) => {
      const phaseBC = entry.summary?.phaseBC ?? entry.best?.payload.phaseBC;
      const semantics = entry.summary?.semantics ?? entry.best?.payload.semantics;
      const bg = entry.summary?.bg ?? entry.best?.payload.bg;
      const activityMode = semantics?.activityMode ?? phaseBC?.activityMode;
      const activityAlpha = semantics?.activityAlpha ?? phaseBC?.activityAlpha;
      const tTrans = semantics?.tTrans ?? phaseBC?.tTrans;
      const tailLen = semantics?.tailLen ?? phaseBC?.tailLen;

      lines.push(
        `| ${entry.topology} | ${formatNum(phaseBC?.k_inhib, 2)} | ${formatNum(phaseBC?.v_th, 2)} | ${formatNum(
          phaseBC?.alpha,
          2,
        )} | ${formatNum(phaseBC?.wInRowNorm, 2)} | ${formatNum(phaseBC?.wAttrRowNorm, 2)} | ${formatNum(
          phaseBC?.etaTrans,
          3,
        )} | ${activityMode ?? 'n/a'} | ${formatNum(activityAlpha, 2)} | ${formatNum(tTrans)} | ${formatNum(
          tailLen,
        )} | ${semantics?.evalWindow ?? 'n/a'} | ${semantics?.learnWindow ?? 'n/a'} | ${
          semantics?.settleWindow ?? 'n/a'
        } | ${semantics?.excludeFirstK ?? 'n/a'} | ${formatNum(bg?.epsilon, 3)} | ${formatNum(
          bg?.temperature,
          3,
        )} | ${bg?.waitSteps ?? 'n/a'} | ${
          bg?.sampleActions === undefined ? 'n/a' : bg.sampleActions ? 'yes' : 'no'
        } |`,
      );
    });

    const megatuneArtifacts = [
      megatuneSummary?.path,
      ...megatuneBest.map((artifact) => artifact.path),
    ]
      .filter(Boolean)
      .map((p) => `[${path.basename(p as string)}](${toRelative(combinedDir, p as string)})`);

    if (megatuneArtifacts.length > 0) {
      lines.push('');
      lines.push(`Artifacts: ${megatuneArtifacts.join(', ')}`);
    }
  }

  const probeTopK = loadProbeTopKArtifacts(suiteDir);
  const probeBest = loadProbeBestArtifacts(suiteDir);
  lines.push('');
  lines.push('## Phase B/C probe summary');
  if (probeTopK.length === 0) {
    lines.push('');
    lines.push('- probeBC_topK artifacts not found; probe summary unavailable.');
  }
  probeTopK.forEach((artifact) => {
    lines.push('');
    lines.push(`### ${artifact.topology}`);
    const probeSettings = artifact.payload.probe;
    const resolved = artifact.payload.resolved;
    lines.push(
      `- probe settings: seed=${probeSettings.seed}, maxCandidates=${probeSettings.maxCandidates}, trialsPerCandidate=${probeSettings.trialsPerCandidate}, rankTop=${probeSettings.rankTop}, weights(sustain=${probeSettings.wSustain}, acc=${probeSettings.wAcc})`,
    );
    lines.push(
      `- resolved windows: eval=${resolved.evalWindow}, learn=${resolved.learnWindow}, settle=${resolved.settleWindow}, tTrans=${resolved.tTrans}, tailLen=${resolved.tailLen}, activity=${resolved.activityMode} (alpha=${resolved.activityAlpha})`,
    );

    lines.push('');
    lines.push(
      '| rank | score | acc | sustain | aborted | tailSilent | tailSpike | k_inhib | v_th | alpha | wIn | wAttr | etaTrans |',
    );
    lines.push('| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |');
    artifact.payload.candidates
      .slice(0, Math.min(artifact.payload.candidates.length, 5))
      .forEach((candidate, idx) => {
        lines.push(
          `| ${idx + 1} | ${formatNum(candidate.metrics.score)} | ${formatNum(candidate.metrics.acc)} | ${formatNum(
            candidate.metrics.sustain,
          )} | ${formatNum(candidate.metrics.abortedFrac)} | ${formatNum(candidate.metrics.tailSilent)} | ${formatNum(
            candidate.metrics.tailSpike,
          )} | ${formatNum(candidate.physics.k_inhib, 2)} | ${formatNum(candidate.physics.v_th, 2)} | ${formatNum(
            candidate.physics.alpha,
            2,
          )} | ${formatNum(candidate.physics.wInRowNorm, 2)} | ${formatNum(candidate.physics.wAttrRowNorm, 2)} | ${formatNum(
            candidate.physics.etaTrans,
            3,
          )} |`,
        );
      });

    const artifactLinks = [artifact.path, probeBest.find((p) => p.topology === artifact.topology)?.path]
      .filter(Boolean)
      .map((p) => `[${path.basename(p as string)}](${toRelative(combinedDir, p as string)})`);
    if (artifactLinks.length > 0) {
      lines.push('');
      lines.push(`Artifacts: ${artifactLinks.join(', ')}`);
    }
  });

  const runArtifacts = loadRunArtifacts(suiteDir);
  lines.push('');
  lines.push('## Experiment run summary');
  if (runArtifacts.length === 0) {
    lines.push('');
    lines.push('- run_last artifacts not found; experiment summary unavailable.');
  }
  if (runArtifacts.length > 0) {
    const summaryRows: string[] = [];
    const phaseDEFRows: string[] = [];
    const detailLines: string[] = [];
    lines.push('');
    lines.push('### Phase B/C summary');
    lines.push('');
    lines.push('| run | phase | finalAcc | abortedFrac | tailSilent | tailSpike | timeToSilence | gateFails | steps |');
    lines.push('| --- | --- | --- | --- | --- | --- | --- | --- | --- |');
    runArtifacts.forEach((artifact) => {
      const { result } = artifact;
      const phaseRows: { label: string; phase?: typeof result.phaseC }[] = [
        { label: 'C', phase: result.phaseC },
      ];
      phaseRows.push({ label: 'B', phase: result.phaseB });

      phaseRows.forEach(({ label, phase }) => {
        if (!phase) return;
        summaryRows.push(
          `| ${artifact.label} | ${label} | ${formatNum(phase.finalAcc)} | ${formatNum(phase.abortedFrac)} | ${formatNum(
            phase.sustain?.tailSilentFracMean,
          )} | ${formatNum(phase.sustain?.tailSpikeMassMean)} | ${formatNum(phase.sustain?.timeToSilenceMean)} | ${formatNum(
            phase.sustain?.gate?.gateFails,
            0,
          )} | ${phase.steps} |`,
        );
      });

      const phaseDEFEntries: {
        label: string;
        finalAcc?: number;
        accPlus?: number;
        accMinus?: number;
        boundaryAccPlus?: number;
        boundaryAccMinus?: number;
        abortedFrac?: number;
        sustain?: PhaseSummary['sustain'];
        steps?: number;
      }[] = [];

      if (result.phaseD) {
        phaseDEFEntries.push({
          label: 'D',
          finalAcc: result.phaseD.finalAcc,
          abortedFrac: result.phaseD.abortedFrac,
          sustain: result.phaseD.sustain,
          steps: result.phaseD.steps,
        });
      }

      if (result.phaseE) {
        phaseDEFEntries.push({
          label: 'E',
          finalAcc: result.phaseE.finalAcc,
          accPlus: result.phaseE.opSplit?.accPlus,
          accMinus: result.phaseE.opSplit?.accMinus,
          boundaryAccPlus: result.phaseE.opSplit?.boundaryAccPlus,
          boundaryAccMinus: result.phaseE.opSplit?.boundaryAccMinus,
          abortedFrac: result.phaseE.abortedFrac,
          sustain: result.phaseE.sustain,
          steps: result.phaseE.steps,
        });
      }

      if (result.phaseF) {
        phaseDEFEntries.push({
          label: 'F',
          finalAcc: result.phaseF.overallAccMean,
          accPlus: result.phaseF.accPlus,
          accMinus: result.phaseF.accMinus,
          boundaryAccPlus: result.phaseF.boundaryAccPlus,
          boundaryAccMinus: result.phaseF.boundaryAccMinus,
        });
      }

      phaseDEFEntries.forEach((entry) => {
        phaseDEFRows.push(
          `| ${artifact.label} | ${entry.label} | ${formatNum(entry.finalAcc)} | ${formatNum(entry.accPlus)} | ${formatNum(
            entry.accMinus,
          )} | ${formatNum(entry.boundaryAccPlus)} | ${formatNum(entry.boundaryAccMinus)} | ${formatNum(
            entry.abortedFrac,
          )} | ${formatNum(entry.sustain?.tailSilentFracMean)} | ${formatNum(
            entry.sustain?.tailSpikeMassMean,
          )} | ${formatNum(entry.sustain?.timeToSilenceMean)} | ${formatNum(
            entry.sustain?.gate?.gateFails,
            0,
          )} | ${entry.steps ?? 'n/a'} |`,
        );
      });

      detailLines.push(
        `- ${artifact.label}: success phaseB=${result.successPhaseB ? 'yes' : 'no'}, phaseC=${result.successPhaseC ? 'yes' : 'no'}, phaseD=${result.phaseD ? 'present' : 'n/a'}, phaseE=${result.phaseE ? 'present' : 'n/a'}, phaseF=${result.phaseF ? 'present' : 'n/a'}, countingPhase=${result.config.useCountingPhase ? 'on' : 'off'}, seed=${result.config.randomSeed}`,
      );

      detailLines.push(
        `  - config: targetAccB=${formatNum(result.config.targetAccPhaseB)}, targetAccC=${formatNum(
          result.config.targetAccPhaseC,
        )}`,
      );

      if (result.phaseE) {
        detailLines.push(`  - phaseE schedule: ${result.config.phaseE?.opSchedule ?? 'n/a'}`);
      }

      detailLines.push(`  - artifact: ${toRelative(combinedDir, artifact.path)}`);
      detailLines.push('');
    });

    lines.push(...summaryRows);

    if (phaseDEFRows.length > 0) {
      lines.push('');
      lines.push('### Phase D/E/F summary');
      lines.push('');
      lines.push(
        '| run | phase | accMean | accPlus | accMinus | boundaryPlus | boundaryMinus | abortedFrac | tailSilent | tailSpike | timeToSilence | gateFails | steps |',
      );
      lines.push(
        '| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |',
      );
      lines.push(...phaseDEFRows);
    }

    lines.push('');
    lines.push(...detailLines);
  }

  const reportStep = suiteMeta.steps.find((s) => s.mode === 'report');
  if (reportStep) {
    lines.push('');
    lines.push('## Report');
    lines.push(`- Step report: steps/${String(suiteMeta.steps.indexOf(reportStep) + 1).padStart(2, '0')}_${reportStep.id}/report/index.md`);
    lines.push('- Suite exports: suite_artifacts/report/');
  }

  const indexPath = path.join(combinedDir, 'index.md');
  fs.writeFileSync(indexPath, lines.join('\n'));
  maybeWriteHtml(indexPath, options.useHtml);
}

function copySuiteToVault(suiteDir: string, vaultOutDir?: string): void {
  if (!vaultOutDir) return;
  const dest = path.join(vaultOutDir, path.basename(suiteDir));
  removeDirIfExists(dest);
  fs.cpSync(suiteDir, dest, { recursive: true });
}

function runSuite(suite: PlanSuite, rootOutDir: string, options: OvernightOptions): SuiteMeta {
  const suiteDir = path.join(rootOutDir, 'suites', suite.id);
  const suiteArtifactsDir = path.join(suiteDir, 'suite_artifacts');
  ensureDir(suiteArtifactsDir);
  const start = Date.now();

  const steps: StepMeta[] = [];
  for (let i = 0; i < suite.steps.length; i += 1) {
    const step = suite.steps[i];
    const meta = executeStep(suiteDir, suiteArtifactsDir, step, i, options, suite.seed);
    steps.push(meta);
    if (!meta.success && options.failFast && !options.continueOnFail) {
      break;
    }
  }

  const success = steps.every((s) => s.success || s.skipped);
  const suiteMeta: SuiteMeta = {
    id: suite.id,
    description: suite.description,
    seed: suite.seed,
    success,
    startTime: new Date(start).toISOString(),
    endTime: new Date().toISOString(),
    durationMs: Date.now() - start,
    steps,
  };
  writeJson(path.join(suiteDir, 'suite_meta.json'), suiteMeta);
  writeCombinedReport(suiteDir, suiteMeta, { useHtml: options.useHtml });
  copySuiteToVault(suiteDir, options.vaultOutDir);
  return suiteMeta;
}

function parseOptions(): OvernightOptions {
  const planPath = getStringFlag('--overnight.plan=');
  if (!planPath) throw new Error('[Overnight] --overnight.plan=<path> is required');
  const outDirArg = getStringFlag('--overnight.outDir=');
  const outDir = outDirArg
    ? path.isAbsolute(outDirArg)
      ? outDirArg
      : path.join(process.cwd(), outDirArg)
    : path.join(process.cwd(), 'artifacts_overnight', timestampSlug());

  return {
    planPath: path.isAbsolute(planPath) ? planPath : path.join(process.cwd(), planPath),
    outDir,
    cleanOutDir: parseBoolFlag('--overnight.cleanOutDir', false),
    failFast: parseBoolFlag('--overnight.failFast', true),
    continueOnFail: parseBoolFlag('--overnight.continueOnFail', false),
    topologies: parseTopologies(),
    formats: parseListFlag('--overnight.formats=', ['png']),
    vaultOutDir: getStringFlag('--overnight.vaultOutDir='),
    useHtml: parseBoolFlag('--overnight.useHtml', false),
    resume: parseBoolFlag('--overnight.resume', true),
  };
}

export function runOvernight(): void {
  console.log('[Overnight] Parsing options...');
  const options = parseOptions();
  console.log('[Overnight] Options parsed:', JSON.stringify(options, null, 2));
  console.log(`[Overnight] Loading plan from ${options.planPath}`);
  const plan = loadPlan(options.planPath);
  const { planPath, outDir, ...optionSnapshot } = options;

  console.log(
    `[Overnight] Loaded plan with ${plan.suites?.length ?? 0} suites. Output dir: ${outDir}. Clean: ${options.cleanOutDir}`,
  );

  if (options.cleanOutDir) removeDirIfExists(outDir);
  ensureDir(outDir);

  const runMeta: RunMeta = {
    planPath,
    outDir,
    options: optionSnapshot,
    startTime: new Date().toISOString(),
    suites: [],
  };

  writeJson(path.join(outDir, 'run_meta.json'), runMeta);

  const suites = plan.suites ?? [];
  suites.forEach((suite) => {
    console.log(`[Overnight] Starting suite ${suite.id} (${suite.steps.length} steps)`);
    const meta = runSuite(suite, outDir, options);
    runMeta.suites.push(meta);
    writeJson(path.join(outDir, 'run_meta.json'), runMeta);
    console.log(
      `[Overnight] Finished suite ${suite.id}: ${meta.success ? 'success' : 'failure'} in ${meta.durationMs}ms`,
    );
    if (!meta.success && options.failFast && !options.continueOnFail) {
      return;
    }
  });

  const end = Date.now();
  runMeta.endTime = new Date(end).toISOString();
  runMeta.durationMs = end - Date.parse(runMeta.startTime);
  runMeta.success = runMeta.suites.every((s) => s.success);
  writeJson(path.join(outDir, 'run_meta.json'), runMeta);
  console.log(`[Overnight] Run completed. Success: ${runMeta.success}. Duration: ${runMeta.durationMs ?? 0}ms.`);
}

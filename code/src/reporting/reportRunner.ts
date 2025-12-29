import fs from 'fs';
import path from 'path';
import { spawnSync } from 'child_process';

import { artifactPath, resolveArtifactPath } from '../artifactPaths';
import { AccHistorySeries, ExperimentResult, PhaseSummary, Topology } from '../types';
import {
  loadProbeTopK,
  loadRunResults,
  resolveReportOutDir,
  summarizeInputs,
  writeIndexMd,
  writeMetaJson,
} from './reportIO';
import { LoadedReportInputs, ProbeTopKPayload, ReportFigure, ReportMeta, RunHistoryFigure } from './reportTypes';

export type ReportOptions = {
  reportId?: string;
  outDir?: string;
  artifactsDir?: string;
  includeProbe: boolean;
  includeRun: boolean;
  probeTopK: number;
  formats: string[];
  topologies: Topology[];
  inputRun?: string;
  inputProbeTopK?: string;
  inputPerturbSummaries?: string[];
  inputPerturbTrials?: string[];
  perturbDirs?: string[];
  failOnMissing: boolean;
};

const SCRIPT_PATH = path.join(process.cwd(), 'scripts', 'report_plot.py');

function resolveGitCommit(): string | undefined {
  try {
    const result = spawnSync('git', ['rev-parse', 'HEAD'], { encoding: 'utf-8' });
    if (result.status === 0) {
      return result.stdout.trim();
    }
  } catch (err) {
    // ignore
  }
  return undefined;
}

function runPythonPlot(kind: 'probe' | 'run' | 'run_acc_history' | 'perturb', args: string[]): void {
  const pythonBin = process.env.REPORT_PYTHON ?? 'python3';
  const finalArgs = [SCRIPT_PATH, '--kind', kind, ...args];
  const proc = spawnSync(pythonBin, finalArgs, { stdio: 'inherit' });
  if (proc.error) {
    throw new Error(`Failed to launch python: ${proc.error.message}`);
  }
  if (proc.status !== 0) {
    throw new Error(`Python plotting failed for ${kind} (exit code ${proc.status}).`);
  }
}

type PerturbArtifactSet = {
  label: string;
  summaryPath?: string;
  trialsPath?: string;
};

type PerturbSummary = {
  config?: {
    recoveryThreshold?: number;
    maxRecoverySteps?: number;
  };
};

function labelFromPath(candidate?: string): string | undefined {
  if (!candidate) return undefined;
  const parent = path.basename(path.dirname(candidate));
  if (parent === 'perturb') {
    return path.basename(path.dirname(path.dirname(candidate)));
  }
  return parent || path.basename(candidate);
}

function ensureUniqueLabels(sets: PerturbArtifactSet[]): PerturbArtifactSet[] {
  const seen = new Map<string, number>();
  return sets.map((set) => {
    const count = seen.get(set.label) ?? 0;
    seen.set(set.label, count + 1);
    if (count === 0) return set;
    return { ...set, label: `${set.label}-${count + 1}` };
  });
}

function findPerturbArtifacts(options: {
  artifactsRoot?: string;
  inputSummaries?: string[];
  inputTrials?: string[];
  perturbDirs?: string[];
}): PerturbArtifactSet[] {
  const { artifactsRoot, inputSummaries, inputTrials, perturbDirs } = options;
  const resolvedSets: PerturbArtifactSet[] = [];
  const resolveInputPath = (candidate: string) =>
    path.isAbsolute(candidate) ? candidate : artifactPath(candidate, artifactsRoot);

  const summaries = (inputSummaries ?? []).filter((s) => s.length > 0).map(resolveInputPath);
  const trials = (inputTrials ?? []).filter((s) => s.length > 0).map(resolveInputPath);
  const explicitCount = Math.max(summaries.length, trials.length);
  for (let i = 0; i < explicitCount; i += 1) {
    const summaryPath = summaries[i];
    const trialsPath = trials[i];
    if (!summaryPath && !trialsPath) continue;
    const label = labelFromPath(summaryPath) ?? labelFromPath(trialsPath) ?? `perturb-set-${i + 1}`;
    resolvedSets.push({ label, summaryPath, trialsPath });
  }

  (perturbDirs ?? []).forEach((dir) => {
    const resolvedDir = resolveInputPath(dir);
    const directSummary = path.join(resolvedDir, 'perturb_summary.json');
    const directTrials = path.join(resolvedDir, 'perturb_trials.csv');
    const nestedSummary = path.join(resolvedDir, 'perturb', 'perturb_summary.json');
    const nestedTrials = path.join(resolvedDir, 'perturb', 'perturb_trials.csv');
    const hasDirect = fs.existsSync(directSummary) || fs.existsSync(directTrials);
    const hasNested = fs.existsSync(nestedSummary) || fs.existsSync(nestedTrials);
    if (!hasDirect && !hasNested) return;
    const summaryPath = hasDirect ? (fs.existsSync(directSummary) ? directSummary : undefined) : (fs.existsSync(nestedSummary) ? nestedSummary : undefined);
    const trialsPath = hasDirect ? (fs.existsSync(directTrials) ? directTrials : undefined) : (fs.existsSync(nestedTrials) ? nestedTrials : undefined);
    const label = labelFromPath(summaryPath) ?? labelFromPath(trialsPath) ?? path.basename(resolvedDir);
    resolvedSets.push({ label, summaryPath, trialsPath });
  });

  if (resolvedSets.length > 0) {
    return ensureUniqueLabels(resolvedSets);
  }

  const traceRoot = artifactPath('transition_traces', artifactsRoot);
  if (!fs.existsSync(traceRoot) || !fs.statSync(traceRoot).isDirectory()) {
    return [];
  }

  const directPerturbDir = path.join(traceRoot, 'perturb');
  if (fs.existsSync(directPerturbDir) && fs.statSync(directPerturbDir).isDirectory()) {
    const summaryPath = path.join(directPerturbDir, 'perturb_summary.json');
    const trialsPath = path.join(directPerturbDir, 'perturb_trials.csv');
    if (fs.existsSync(summaryPath) || fs.existsSync(trialsPath)) {
      const label = 'perturb';
      resolvedSets.push({
        label,
        summaryPath: fs.existsSync(summaryPath) ? summaryPath : undefined,
        trialsPath: fs.existsSync(trialsPath) ? trialsPath : undefined,
      });
    }
  }

  const entries = fs
    .readdirSync(traceRoot, { withFileTypes: true })
    .filter((entry) => entry.isDirectory())
    .filter((entry) => entry.name !== 'perturb')
    .map((entry) => ({ name: entry.name, dir: path.join(traceRoot, entry.name) }));

  const candidates: PerturbArtifactSet[] = [];
  entries.forEach(({ name, dir }) => {
    const perturbDir = path.join(dir, 'perturb');
    const summaryPath = path.join(perturbDir, 'perturb_summary.json');
    const trialsPath = path.join(perturbDir, 'perturb_trials.csv');
    if (!fs.existsSync(summaryPath) && !fs.existsSync(trialsPath)) {
      return;
    }
    const summary = fs.existsSync(summaryPath) ? summaryPath : undefined;
    const trials = fs.existsSync(trialsPath) ? trialsPath : undefined;
    const candidate: PerturbArtifactSet = { label: name };
    if (summary) {
      candidate.summaryPath = summary;
    }
    if (trials) {
      candidate.trialsPath = trials;
    }
    candidates.push(candidate);
  });

  return ensureUniqueLabels([...resolvedSets, ...candidates]);
}

function buildProbeTable(payload: ProbeTopKPayload, limit: number): string {
  const header = ['Rank', 'Score', 'Acc', 'Sustain', 'Aborted', 'k_inhib', 'v_th', 'alpha', 'wIn', 'wAttr', 'etaTrans'];
  const lines = [`| ${header.join(' | ')} |`, `| ${header.map(() => '---').join(' | ')} |`];
  payload.candidates.slice(0, limit).forEach((c, idx) => {
    const row = [
      idx + 1,
      c.metrics.score.toFixed(3),
      c.metrics.acc.toFixed(3),
      c.metrics.sustain.toFixed(3),
      c.metrics.abortedFrac.toFixed(3),
      c.physics.k_inhib.toFixed(2),
      c.physics.v_th.toFixed(2),
      c.physics.alpha.toFixed(2),
      c.physics.wInRowNorm.toFixed(2),
      c.physics.wAttrRowNorm.toFixed(2),
      c.physics.etaTrans.toFixed(3),
    ];
    lines.push(`| ${row.join(' | ')} |`);
  });
  return lines.join('\n');
}

function buildRunTable(result: ExperimentResult): string {
  const phaseC = result.phaseC;
  const aborted = phaseC.abortedFrac ?? 0;
  const tailSilent = phaseC.sustain?.tailSilentFracMean ?? 0;
  const tailSpike = phaseC.sustain?.tailSpikeMassMean ?? 0;
  const timeToSilence = phaseC.sustain?.timeToSilenceMean ?? 0;
  const gates = phaseC.sustain?.gate?.gateFails ?? 0;
  const headers = ['Final Acc (C)', 'Aborted', 'Tail Silent', 'Tail Spike', 'Time-to-silence', 'Gate Fails'];
  const values = [
    phaseC.finalAcc.toFixed(3),
    aborted.toFixed(3),
    tailSilent.toFixed(3),
    tailSpike.toFixed(3),
    timeToSilence.toFixed(3),
    gates,
  ];
  return [`| ${headers.join(' | ')} |`, `| ${headers.map(() => '---').join(' | ')} |`, `| ${values.join(' | ')} |`].join('\n');
}

function collectProbeFigures(topology: Topology, format: string): ReportFigure[] {
  return [
    { filename: `probe_score_scatter_${topology}.${format}`, caption: 'Probe score vs accuracy' },
    { filename: `probe_heat_k_inhib_vs_v_th_${topology}.${format}`, caption: 'k_inhib vs v_th (accuracy heat)' },
    { filename: `probe_accuracy_tailSilent_${topology}.${format}`, caption: 'Accuracy vs tail silence' },
  ];
}

function collectRunFigures(result: ExperimentResult, format: string): ReportFigure[] {
  const suffix = `${result.config.topology}_${result.config.useCountingPhase ? 'countingOn' : 'countingOff'}`;
  const figs: ReportFigure[] = [
    { filename: `confusion_phaseC_readout_${suffix}.${format}`, caption: 'Phase C confusion (readout)' },
    { filename: `confusion_phaseC_proto_${suffix}.${format}`, caption: 'Phase C confusion (prototype)' },
    { filename: `metrics_phaseC_aborts_${suffix}.${format}`, caption: 'Phase C aborts and accuracy' },
  ];
  if (result.phaseB?.confusion) {
    figs.push({ filename: `confusion_phaseB_readout_${suffix}.${format}`, caption: 'Phase B confusion (readout)' });
    figs.push({ filename: `confusion_phaseB_proto_${suffix}.${format}`, caption: 'Phase B confusion (prototype)' });
  }
  return figs;
}

function normalizeAccHistory(history: AccHistorySeries | undefined):
  | { series: { readout?: number[]; proto?: number[] }; note?: string }
  | undefined {
  if (!history) return undefined;
  if (Array.isArray(history)) {
    return { series: { readout: history } };
  }
  if (typeof history === 'object') {
    const maybeHistory = history as Record<string, unknown>;
    const readoutCandidate = maybeHistory['readout'];
    const protoCandidate = maybeHistory['proto'];
    const readout = Array.isArray(readoutCandidate) ? readoutCandidate : undefined;
    const proto = Array.isArray(protoCandidate) ? protoCandidate : undefined;
    if (readout || proto) {
      return { series: { readout, proto } };
    }
    const samplesCandidate = maybeHistory['samples'];
    const samples = Array.isArray(samplesCandidate) ? samplesCandidate : undefined;
    if (samples) return { series: { readout: samples } };
    return { series: {}, note: `Unknown accHistory keys: ${Object.keys(maybeHistory).join(', ')}` };
  }
  return { series: {}, note: 'Unrecognized accHistory format' };
}

function buildRunHistory(
  phase: 'B' | 'C',
  summary: PhaseSummary | undefined,
  configIncludesHistory: boolean | undefined,
  format: string,
  suffix: string,
): RunHistoryFigure {
  const finalAcc = summary?.finalAcc;
  const abortedFrac = summary?.abortedFrac;
  const includesHistory = configIncludesHistory ?? true;
  if (includesHistory === false) {
    return {
      phase,
      available: false,
      note: 'Accuracy history not available; run without --no-acc-history',
      finalAcc,
      abortedFrac,
    };
  }
  if (!summary) {
    return { phase, available: false, note: 'Phase not present in results', finalAcc, abortedFrac };
  }

  const normalized = normalizeAccHistory(summary.accHistory);
  if (!normalized || !normalized.series || Object.keys(normalized.series).length === 0) {
    const note = normalized?.note ?? 'Accuracy history missing from artifact';
    return { phase, available: false, note, finalAcc, abortedFrac };
  }

  const filename = `acc_history_phase${phase}_${suffix}.${format}`;
  return {
    phase,
    filename,
    caption: `Phase ${phase} accuracy history`,
    available: true,
    note: 'available',
    finalAcc,
    abortedFrac,
  };
}

export function runReportEntry(options: ReportOptions): void {
  const { reportId, outDir } = resolveReportOutDir(options.reportId, options.outDir, options.artifactsDir);
  const embedFormat = options.formats[0] ?? 'png';
  const artifactsRoot = options.artifactsDir;
  const resolveInput = (
    explicit: string | undefined,
    fallback: string,
    topology?: Topology,
    mode?: string,
  ) => {
    if (explicit) {
      return path.isAbsolute(explicit) ? explicit : artifactPath(explicit, artifactsRoot);
    }
    return (
      resolveArtifactPath({ key: fallback, topology, mode, artifactsDirOverride: artifactsRoot }) ??
      artifactPath(fallback, artifactsRoot)
    );
  };

  const inputs: LoadedReportInputs = {};
  if (options.includeProbe) {
    inputs.probeTopK = {};
    options.topologies.forEach((topology) => {
      const probe = loadProbeTopK(topology, options.inputProbeTopK, options.failOnMissing, artifactsRoot);
      if (probe) inputs.probeTopK![topology] = probe;
    });
  }
  if (options.includeRun) {
    inputs.runResults = loadRunResults(options.topologies, options.inputRun, options.failOnMissing, artifactsRoot);
  }

  const meta: ReportMeta = {
    reportId,
    createdAt: new Date().toISOString(),
    gitCommit: resolveGitCommit(),
    inputs: summarizeInputs(inputs),
    cliArgs: process.argv.slice(2),
  };
  writeMetaJson(meta, outDir);

  const probeSections: { topology: Topology; figures: ReportFigure[]; table?: string }[] = [];
  if (inputs.probeTopK) {
    Object.entries(inputs.probeTopK).forEach(([topology, payload]) => {
      if (!payload) return;
      const figures = collectProbeFigures(topology as Topology, embedFormat);
      const table = buildProbeTable(payload, options.probeTopK);
      const args = [
        '--input',
        resolveInput(options.inputProbeTopK, 'probeBC_topK', topology as Topology),
        '--outDir',
        path.join(outDir, 'figures'),
        '--formats',
        options.formats.join(','),
        '--topology',
        topology,
        '--probeTopK',
        options.probeTopK.toString(),
      ];
      runPythonPlot('probe', args);
      probeSections.push({ topology: topology as Topology, figures, table });
    });
  }

  const runSections: { label: string; figures: ReportFigure[]; history: RunHistoryFigure[]; table?: string }[] = [];
  inputs.runResults?.forEach((result) => {
    const suffix = `${result.config.topology}_${result.config.useCountingPhase ? 'countingOn' : 'countingOff'}`;
    const figures = collectRunFigures(result, embedFormat);
    const includeAccHistory = result.config.includeAccHistory ?? true;
    const args = [
      '--input',
      resolveInput(
        options.inputRun,
        'run_last',
        result.config.topology,
        result.config.useCountingPhase ? 'countingOn' : 'countingOff',
      ),
      '--outDir',
      path.join(outDir, 'figures'),
      '--formats',
      options.formats.join(','),
      '--topology',
      result.config.topology,
    ];
    runPythonPlot('run', args);
    const history: RunHistoryFigure[] = [
      buildRunHistory('C', result.phaseC, includeAccHistory, embedFormat, suffix),
    ];
    if (result.phaseB) {
      history.push(buildRunHistory('B', result.phaseB, includeAccHistory, embedFormat, suffix));
    }
    const hasHistoryToPlot = history.some((h) => h.available);
    if (hasHistoryToPlot) {
      const historyArgs = [
        '--input',
        resolveInput(
          options.inputRun,
          'run_last',
          result.config.topology,
          result.config.useCountingPhase ? 'countingOn' : 'countingOff',
        ),
        '--outDir',
        path.join(outDir, 'figures'),
        '--formats',
        options.formats.join(','),
        '--topology',
        result.config.topology,
      ];
      runPythonPlot('run_acc_history', historyArgs);
    }
    runSections.push({
      label: `${result.config.topology.toUpperCase()} | counting ${result.config.useCountingPhase ? 'ON' : 'OFF'}`,
      figures,
      history,
      table: buildRunTable(result),
    });
  });

  const perturbSets = findPerturbArtifacts({
    artifactsRoot,
    inputSummaries: options.inputPerturbSummaries,
    inputTrials: options.inputPerturbTrials,
    perturbDirs: options.perturbDirs,
  });
  const perturbFigures: ReportFigure[] = [];
  let perturbLabels: string[] | undefined;
  let perturbRecoveryDefinition: string | undefined;
  if (perturbSets.length > 0) {
    const perturbArgs = ['--outDir', path.join(outDir, 'figures'), '--formats', options.formats.join(',')];
    const labels = perturbSets.map((set) => set.label);
    perturbLabels = labels;
    const baseDefinition =
      'Recovery is defined as cosine similarity to the pre-perturb baseline state exceeding the configured threshold within maxRecoverySteps.';
    const recoveryDetails: Array<{ label: string; recoveryThreshold?: number; maxRecoverySteps?: number }> = [];
    perturbSets.forEach((set) => {
      if (!set.summaryPath) return;
      const raw = fs.readFileSync(set.summaryPath, 'utf-8');
      const parsed = JSON.parse(raw) as PerturbSummary;
      const config = parsed?.config ?? {};
      const recoveryThreshold = typeof config.recoveryThreshold === 'number' ? config.recoveryThreshold : undefined;
      const maxRecoverySteps = typeof config.maxRecoverySteps === 'number' ? config.maxRecoverySteps : undefined;
      if (recoveryThreshold === undefined && maxRecoverySteps === undefined) return;
      recoveryDetails.push({ label: set.label, recoveryThreshold, maxRecoverySteps });
    });
    if (recoveryDetails.length > 0) {
      const detailString = recoveryDetails
        .map((detail) => {
          const thresholdLabel =
            detail.recoveryThreshold === undefined ? 'n/a' : detail.recoveryThreshold.toFixed(3);
          const maxLabel = detail.maxRecoverySteps === undefined ? 'n/a' : detail.maxRecoverySteps.toString();
          return `${detail.label} (threshold=${thresholdLabel}, maxRecoverySteps=${maxLabel})`;
        })
        .join('; ');
      perturbRecoveryDefinition = `${baseDefinition} Thresholds: ${detailString}.`;
    } else {
      perturbRecoveryDefinition = baseDefinition;
    }
    perturbSets.forEach((set) => {
      if (set.summaryPath) {
        perturbArgs.push('--perturbSummary', set.summaryPath);
      }
      if (set.trialsPath) {
        perturbArgs.push('--perturbTrials', set.trialsPath);
      }
      perturbArgs.push('--perturbLabel', set.label);
    });
    const labelSuffix = labels.length > 0 ? ` (conditions: ${labels.join(', ')})` : '';
    const hasSummary = perturbSets.some((set) => set.summaryPath);
    const hasTrials = perturbSets.some((set) => set.trialsPath);
    if (hasSummary) {
      perturbFigures.push({
        filename: `perturb_similarity_mean.${embedFormat}`,
        caption: `Mean similarity vs time (aligned to perturb step)${labelSuffix}`,
      });
      perturbFigures.push({
        filename: `perturb_recovery_rate.${embedFormat}`,
        caption: `Recovery rate${labelSuffix}`,
      });
    }
    if (hasTrials) {
      perturbFigures.push({
        filename: `perturb_recovery_steps_hist.${embedFormat}`,
        caption: `Histogram of recovery steps${labelSuffix}`,
      });
    }
    runPythonPlot('perturb', perturbArgs);
  }

  writeIndexMd({
    outDir,
    reportId,
    meta,
    probeSections,
    runSections,
    perturbFigures,
    perturbLabels,
    perturbRecoveryDefinition,
  });
  console.log(`[Report] Written to ${outDir}`);
}

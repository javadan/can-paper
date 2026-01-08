import fs from 'fs';
import path from 'path';

import { artifactPath, artifactsDir, resolveArtifactPath } from '../artifactPaths';
import { ExperimentResult, Topology } from '../types';
import { LoadedReportInputs, ProbeTopKPayload, ReportMeta, ReportFigure, RunHistoryFigure } from './reportTypes';

function resolvePath(candidate: string, artifactsRoot?: string): string {
  if (path.isAbsolute(candidate)) return candidate;
  return artifactPath(candidate, artifactsRoot);
}

export function resolveReportOutDir(
  reportId?: string,
  customOutDir?: string,
  artifactsRoot?: string,
): { reportId: string; outDir: string } {
  const timestamp = new Date().toISOString().replace(/[:.]/g, '-');
  const id = reportId ?? `report-${timestamp}`;
  const baseDir = artifactsDir(artifactsRoot);
  const resolved = customOutDir
    ? path.isAbsolute(customOutDir)
      ? customOutDir
      : path.join(baseDir, customOutDir)
    : path.join(baseDir, 'reports', id);
  fs.mkdirSync(resolved, { recursive: true });
  fs.mkdirSync(path.join(resolved, 'figures'), { recursive: true });
  return { reportId: id, outDir: resolved };
}

export function loadProbeTopK(
  topology: Topology,
  explicitPath?: string,
  failOnMissing = true,
  artifactsRoot?: string,
): ProbeTopKPayload | undefined {
  const candidatePath = explicitPath
    ? resolvePath(explicitPath, artifactsRoot)
    : resolveArtifactPath({ key: 'probeBC_topK', topology, artifactsDirOverride: artifactsRoot });
  if (!candidatePath || !fs.existsSync(candidatePath)) {
    if (failOnMissing) {
      throw new Error(`[Report] Missing probe artifact for ${topology} at ${candidatePath}`);
    }
    return undefined;
  }
  const raw = fs.readFileSync(candidatePath, 'utf-8');
  return JSON.parse(raw) as ProbeTopKPayload;
}

export function loadRunResults(
  topologies: Topology[],
  explicitPath?: string,
  failOnMissing = true,
  artifactsRoot?: string,
): ExperimentResult[] {
  const results: ExperimentResult[] = [];
  if (explicitPath) {
    const resolved = resolvePath(explicitPath, artifactsRoot);
    if (!fs.existsSync(resolved)) {
      if (failOnMissing) {
        throw new Error(`[Report] Missing run artifact at ${resolved}`);
      }
    } else {
      const raw = fs.readFileSync(resolved, 'utf-8');
      results.push(JSON.parse(raw) as ExperimentResult);
    }
    return results;
  }

  topologies.forEach((topology) => {
    ['countingOn', 'countingOff'].forEach((mode) => {
      const candidatePath = resolveArtifactPath({
        key: 'run_last',
        topology,
        mode,
        artifactsDirOverride: artifactsRoot,
      });
      if (!candidatePath || !fs.existsSync(candidatePath)) {
        if (failOnMissing) {
          throw new Error(`[Report] Missing run artifact for ${topology}/${mode} at ${candidatePath}`);
        }
        return;
      }
      const raw = fs.readFileSync(candidatePath, 'utf-8');
      results.push(JSON.parse(raw) as ExperimentResult);
    });
  });
  return results;
}

export function writeMetaJson(meta: ReportMeta, outDir: string): void {
  const metaPath = path.join(outDir, 'meta.json');
  fs.writeFileSync(metaPath, JSON.stringify(meta, null, 2));
}

export function writeIndexMd(args: {
  outDir: string;
  reportId: string;
  meta: ReportMeta;
  probeSections: { topology: Topology; figures: ReportFigure[]; table?: string }[];
  runSections: { label: string; figures: ReportFigure[]; history: RunHistoryFigure[]; table?: string }[];
  perturbFigures?: ReportFigure[];
  perturbLabels?: string[];
  perturbRecoveryDefinition?: string;
}): void {
  const lines: string[] = [];
  lines.push(`# Report ${args.reportId}`);
  lines.push('');
  lines.push(`Generated at ${args.meta.createdAt}`);
  lines.push('');
  lines.push('## Inputs');
  const probeEntries = args.meta.inputs.probes ?? {};
  Object.entries(probeEntries).forEach(([topo, val]) => {
    lines.push(`- Probe (${topo}): ${val ?? 'n/a'}`);
  });
  const runEntries = args.meta.inputs.runs ?? {};
  Object.entries(runEntries).forEach(([label, val]) => {
    lines.push(`- Run (${label}): ${val ?? 'n/a'}`);
  });
  lines.push('');

  args.probeSections.forEach((section) => {
    lines.push(`## Probe (${section.topology})`);
    lines.push('');
    if (section.table) {
      lines.push(section.table);
      lines.push('');
    }
    section.figures.forEach((fig) => {
      lines.push(`![${fig.caption ?? fig.filename}](figures/${fig.filename})`);
      if (fig.caption) lines.push(`*${fig.caption}*`);
      lines.push('');
    });
  });

  if (args.runSections.length > 0) {
    lines.push('## Run Results');
    lines.push('');
  }

  args.runSections.forEach((section) => {
    lines.push(`### ${section.label}`);
    lines.push('');
    if (section.table) {
      lines.push(section.table);
      lines.push('');
    }

    lines.push('#### Accuracy over time');
    lines.push('');
    const sortedHistory = [...section.history].sort((a, b) =>
      a.phase === b.phase ? 0 : a.phase === 'C' ? -1 : 1,
    );
    sortedHistory.forEach((hist) => {
      const phaseLabel = hist.phase === 'C' ? 'Phase C' : 'Phase B';
      lines.push(`##### ${phaseLabel}`);
      lines.push('');
      const accLabel = hist.finalAcc !== undefined ? hist.finalAcc.toFixed(3) : 'n/a';
      const abortedLabel = hist.abortedFrac !== undefined ? hist.abortedFrac.toFixed(3) : 'n/a';
      lines.push(`- Final accuracy: ${accLabel}`);
      lines.push(`- Aborted fraction: ${abortedLabel}`);
      lines.push(`- History: ${hist.note}`);
      if (hist.available && hist.filename) {
        lines.push('');
        lines.push(`![${hist.caption ?? hist.filename}](figures/${hist.filename})`);
        if (hist.caption) lines.push(`*${hist.caption}*`);
      }
      lines.push('');
    });

    section.figures.forEach((fig) => {
      lines.push(`![${fig.caption ?? fig.filename}](figures/${fig.filename})`);
      if (fig.caption) lines.push(`*${fig.caption}*`);
      lines.push('');
    });
  });

  if (args.perturbFigures && args.perturbFigures.length > 0) {
    lines.push('## Perturbation Recovery');
    lines.push('');
    lines.push(
      args.perturbRecoveryDefinition ??
        'Recovery is defined as cosine similarity to the pre-perturb baseline state exceeding the configured threshold within maxRecoverySteps.',
    );
    lines.push('');
    if (args.perturbLabels && args.perturbLabels.length > 0) {
      lines.push(`Conditions: ${args.perturbLabels.join(', ')}`);
      lines.push('');
    }
    args.perturbFigures.forEach((fig) => {
      lines.push(`![${fig.caption ?? fig.filename}](figures/${fig.filename})`);
      if (fig.caption) lines.push(`*${fig.caption}*`);
      lines.push('');
    });
  }

  const indexPath = path.join(args.outDir, 'index.md');
  fs.writeFileSync(indexPath, lines.join('\n'));
}

export function summarizeInputs(inputs: LoadedReportInputs): { probes: Record<Topology, string | undefined>; runs: Record<string, string | undefined> } {
  const probes: Record<Topology, string | undefined> = { snake: undefined, ring: undefined };
  (['snake', 'ring'] as Topology[]).forEach((t) => {
    if (inputs.probeTopK?.[t]) probes[t] = `probeBC_topK_${t}.json`;
  });

  const runs: Record<string, string | undefined> = {};
  inputs.runResults?.forEach((res) => {
    const countingLabel = res.config.useCountingPhase ? 'countingOn' : 'countingOff';
    runs[`${res.config.topology}_${countingLabel}`] = `run_last_${res.config.topology}_${countingLabel}.json`;
  });
  return { probes, runs };
}

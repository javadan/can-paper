import fs from 'fs';
import path from 'path';

import { Topology } from './types';

function ensureArtifactsDirExists(dirPath: string): void {
  if (!fs.existsSync(dirPath)) {
    fs.mkdirSync(dirPath, { recursive: true });
  }
}

function resolveDir(rawDir: string): string {
  const resolved = path.isAbsolute(rawDir) ? rawDir : path.join(process.cwd(), rawDir);
  ensureArtifactsDirExists(resolved);
  return resolved;
}

let cachedDir: string | undefined;

function findCliOverride(): string | undefined {
  const override = process.argv.find((arg) => arg.startsWith('--artifacts.dir='));
  if (!override) return undefined;
  const [, value] = override.split('=');
  return value || undefined;
}

export function artifactsDir(override?: string): string {
  if (override) return resolveDir(override);
  if (!cachedDir) {
    const cliOverride = findCliOverride();
    const envOverride = process.env.ARTIFACTS_DIR;
    const rawDir = cliOverride ?? envOverride ?? path.join(process.cwd(), 'artifacts');
    cachedDir = resolveDir(rawDir);
  }
  return cachedDir;
}

export function artifactPath(filename: string, override?: string): string {
  const dir = artifactsDir(override);
  return path.join(dir, filename);
}

const SHARED_DIR_NAMES = ['suite_artifacts', 'shared_artifacts'];
const SHARED_CANONICAL_NAME = SHARED_DIR_NAMES[0];

type ResolutionTier = 'local' | 'shared' | 'legacy';

export type ArtifactKey =
  | 'phaseA_best'
  | 'phaseA_finetune_best'
  | 'phaseA_finetune_progress'
  | 'phaseA_validation'
  | 'phaseA_state'
  | 'phaseA_topK'
  | 'phaseB_best'
  | 'phaseB_tuning_progress'
  | 'phaseB_finetune_best'
  | 'phaseB_finetune_progress'
  | 'probeBC_best'
  | 'probeBC_topK'
  | 'run_last';

const KNOWN_ARTIFACT_KEYS: ArtifactKey[] = [
  'phaseA_best',
  'phaseA_finetune_best',
  'phaseA_finetune_progress',
  'phaseA_validation',
  'phaseA_state',
  'phaseA_topK',
  'phaseB_best',
  'phaseB_tuning_progress',
  'phaseB_finetune_best',
  'phaseB_finetune_progress',
  'probeBC_best',
  'probeBC_topK',
  'run_last',
];

function artifactFilename(key: ArtifactKey, options?: { topology?: Topology; mode?: string }): string {
  switch (key) {
    case 'phaseA_best':
      return 'phaseA_best.json';
    case 'phaseA_finetune_best':
      return 'phaseA_finetune_best.json';
    case 'phaseA_finetune_progress':
      return 'phaseA_finetune_progress.json';
    case 'phaseA_validation':
      return 'phaseA_validation.json';
    case 'phaseA_state':
      if (!options?.topology) throw new Error('[Artifacts] topology is required for phaseA_state');
      return `phaseA_state_${options.topology}.json`;
    case 'phaseA_topK':
      return 'phaseA_topK.json';
    case 'phaseB_best':
      return 'phaseB_best.json';
    case 'phaseB_tuning_progress':
      return 'phaseB_tuning_progress.json';
    case 'phaseB_finetune_best':
      return 'phaseB_finetune_best.json';
    case 'phaseB_finetune_progress':
      return 'phaseB_finetune_progress.json';
    case 'probeBC_best':
      if (!options?.topology) throw new Error('[Artifacts] topology is required for probeBC_best');
      return `probeBC_best_${options.topology}.json`;
    case 'probeBC_topK':
      if (!options?.topology) throw new Error('[Artifacts] topology is required for probeBC_topK');
      return `probeBC_topK_${options.topology}.json`;
    case 'run_last':
      if (!options?.topology || !options?.mode) {
        throw new Error('[Artifacts] topology and mode are required for run_last');
      }
      return `run_last_${options.topology}_${options.mode}.json`;
  }
}

function describeKey(logicalArtifactName: ArtifactKey | string, opts: { topology?: Topology; mode?: string }): string {
  const parts = [`${logicalArtifactName}`];
  if (opts.topology) parts.push(`topology=${opts.topology}`);
  if (opts.mode) parts.push(`mode=${opts.mode}`);
  return parts.join(' ');
}

function findArtifactsRoot(startDir: string): { root: string; sharedDir: string; marker: string } | undefined {
  let current = path.resolve(startDir);
  while (true) {
    const base = path.basename(current);
    if (SHARED_DIR_NAMES.includes(base)) {
      return { root: path.dirname(current), sharedDir: current, marker: `self:${base}` };
    }

    const foundShared = SHARED_DIR_NAMES.find((name) => fs.existsSync(path.join(current, name)));
    if (foundShared) {
      return { root: current, sharedDir: path.join(current, foundShared), marker: foundShared };
    }

    const foundMeta = ['suite_meta.json', 'run_meta.json'].find((name) => fs.existsSync(path.join(current, name)));
    if (foundMeta) {
      const candidate = path.join(current, SHARED_CANONICAL_NAME);
      return { root: current, sharedDir: candidate, marker: foundMeta };
    }

    const parent = path.dirname(current);
    if (parent === current) break;
    current = parent;
  }
  return undefined;
}

type ArtifactResolutionOptions = {
  key: ArtifactKey | string;
  topology?: Topology;
  mode?: string;
  artifactsDirOverride?: string;
  required?: boolean;
  logger?: (message: string) => void;
};

export function resolveArtifactPath(options: ArtifactResolutionOptions): string | undefined {
  const { key, topology, mode, artifactsDirOverride, required = false } = options;
  const resolvedFilename = ((): string => {
    const artifactKey = key as ArtifactKey;
    if ((KNOWN_ARTIFACT_KEYS as string[]).includes(key as string)) {
      return artifactFilename(artifactKey, { topology, mode });
    }
    return key as string;
  })();
  const logger = options.logger ?? ((msg: string) => console.log(msg));
  const tried: { tier: ResolutionTier; path: string }[] = [];

  const localDir = artifactsDir(artifactsDirOverride);
  const localPath = path.join(localDir, resolvedFilename);
  tried.push({ tier: 'local', path: localPath });
  if (fs.existsSync(localPath)) {
    logger(`[Artifacts] ${describeKey(key, { topology, mode })} -> ${localPath} (tier=local)`);
    return localPath;
  }

  const discovery = findArtifactsRoot(localDir);
  if (discovery) {
    const sharedPath = path.join(discovery.sharedDir, resolvedFilename);
    tried.push({ tier: 'shared', path: sharedPath });
    if (fs.existsSync(sharedPath)) {
      logger(
        `[Artifacts] ${describeKey(key, { topology, mode })} -> ${sharedPath} (tier=shared root=${discovery.root} marker=${discovery.marker})`,
      );
      return sharedPath;
    }
  }

  const legacyDir = path.join(process.cwd(), 'artifacts');
  const legacyPath = path.join(legacyDir, resolvedFilename);
  if (legacyDir !== localDir) {
    tried.push({ tier: 'legacy', path: legacyPath });
    if (fs.existsSync(legacyPath)) {
      logger(`[Artifacts] ${describeKey(key, { topology, mode })} -> ${legacyPath} (tier=legacy)`);
      return legacyPath;
    }
  }

  if (required) {
    const attempted = tried.map((t) => `- ${t.tier}: ${t.path}`).join('\n');
    const rootMsg = discovery
      ? `discoveredRoot=${discovery.root} sharedCandidate=${discovery.sharedDir}`
      : 'discoveredRoot=none';
    throw new Error(
      `[Artifacts] Missing required artifact for ${describeKey(key, { topology, mode })}. Tried:\n${attempted}\n${rootMsg}`,
    );
  }

  logger(
    `[Artifacts] ${describeKey(key, { topology, mode })} not found. Tried local/shared/legacy paths and giving up (required=${required}).`,
  );
  return undefined;
}

export function exportToSharedArtifacts(filename: string, artifactsDirOverride?: string): string | undefined {
  const localDir = artifactsDir(artifactsDirOverride);
  const source = path.join(localDir, filename);
  if (!fs.existsSync(source)) return undefined;

  const discovery = findArtifactsRoot(localDir);
  if (!discovery) return undefined;

  ensureArtifactsDirExists(discovery.sharedDir);
  const dest = path.join(discovery.sharedDir, filename);
  fs.copyFileSync(source, dest);
  console.log(`[Artifacts] Exported ${filename} to shared dir at ${dest} (root=${discovery.root})`);
  return dest;
}

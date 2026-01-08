import { PhaseBCConfig, TransitionWindowName } from './types';

export type TransitionWindowRange = {
  name: TransitionWindowName;
  start: number;
  end: number;
  length: number;
};

export function describeTransitionWindow(
  windows: TransitionWindowRange[],
  name: TransitionWindowName,
): { label: string; range: string } {
  const resolved = windows.find((w) => w.name === name || w.name.startsWith(`${name}(k=`));
  if (!resolved) return { label: name, range: 'n/a' };
  return { label: resolved.name, range: `[${resolved.start},${resolved.end})` };
}

export function normalizeExcludeFirstValues(excludeFirst: number[]): number[] {
  const values = excludeFirst.slice();
  if (!values.includes(1)) values.push(1);
  const filtered = values.filter((k) => Number.isFinite(k) && k >= 0 && Number.isInteger(k));
  const unique = Array.from(new Set(filtered));
  unique.sort((a, b) => a - b);
  return unique;
}

export function resolveTransitionWindows(
  config: Pick<PhaseBCConfig, 'evalWindow' | 'learnWindow' | 'settleWindow' | 'excludeFirstK'>,
  tTrans: number,
  tailLen: number,
): TransitionWindowRange[] {
  if (tTrans <= 0) {
    throw new Error(`[Transition windows] Expected tTrans > 0, got ${tTrans}`);
  }

  const windowNames = new Set<TransitionWindowName>(['mean', 'tail', 'late', 'impulseOnly']);
  windowNames.add(config.evalWindow as TransitionWindowName);
  windowNames.add(config.learnWindow as TransitionWindowName);
  windowNames.add(config.settleWindow as TransitionWindowName);

  return buildTransitionWindows(Array.from(windowNames), tTrans, tailLen, config.excludeFirstK, true);
}

export function buildTransitionWindows(
  names: TransitionWindowName[],
  tTrans: number,
  tailLen: number,
  excludeFirst: number,
  includeKInName = true,
  warnings?: string[],
): TransitionWindowRange[] {
  if (tTrans <= 0) {
    throw new Error(`[Transition windows] Cannot build windows: tTrans must be > 0 (got ${tTrans})`);
  }

  const ranges: TransitionWindowRange[] = [];
  const seen = new Set<string>();
  const localWarnings: string[] = warnings ?? [];
  const emitWarning = (msg: string): void => {
    localWarnings.push(msg);
  };
  for (const name of names) {
    const range = buildSingleTransitionWindow(name, tTrans, tailLen, excludeFirst, includeKInName, emitWarning);
    if (!range) continue;
    if (seen.has(range.name)) continue;
    seen.add(range.name);
    ranges.push(range);
  }
  if (!warnings && localWarnings.length > 0) {
    for (const warning of localWarnings) console.warn(warning);
  }
  return ranges;
}

export function buildNoImpulseTransitionWindows(
  excludeFirst: number[],
  tTrans: number,
  tailLen: number,
  warnings?: string[],
): Record<number, TransitionWindowRange[]> {
  const windowsByK: Record<number, TransitionWindowRange[]> = {};
  const windowNames: TransitionWindowName[] = ['meanNoImpulse', 'tailNoImpulse', 'lateNoImpulse'];
  for (const k of excludeFirst) {
    windowsByK[k] = buildTransitionWindows(windowNames, tTrans, tailLen, k, false, warnings);
  }
  return windowsByK;
}

export function createWindowSums(
  windows: TransitionWindowRange[],
  dim: number,
): Record<string, Float32Array> {
  const sums: Record<string, Float32Array> = {};
  for (const window of windows) {
    sums[window.name] = new Float32Array(dim);
  }
  return sums;
}

export function accumulateWindowSums(
  spikes: Float32Array,
  windows: TransitionWindowRange[],
  windowSums: Record<string, Float32Array>,
  t: number,
): void {
  for (const window of windows) {
    if (t >= window.start && t < window.end) {
      const sumVec = windowSums[window.name];
      for (let i = 0; i < spikes.length; i++) {
        sumVec[i] += spikes[i];
      }
    }
  }
}

export function finalizeWindowMeans(
  windows: TransitionWindowRange[],
  windowSums: Record<string, Float32Array>,
): Record<string, Float32Array> {
  const means: Record<string, Float32Array> = {};
  for (const window of windows) {
    const sumVec = windowSums[window.name];
    const meanVec = new Float32Array(sumVec.length);
    const invLen = window.length > 0 ? 1 / window.length : 0;
    for (let i = 0; i < sumVec.length; i++) {
      meanVec[i] = sumVec[i] * invLen;
    }
    means[window.name] = meanVec;
  }

  // Ensure callers using base window names (e.g., meanNoImpulse) can still access
  // results even when the configured window names include the k-suffix
  // (meanNoImpulse(k=1), etc.). This avoids missing-window errors when
  // excludeFirstK > 0.
  for (const [name, meanVec] of Object.entries(means)) {
    const match = name.match(/^(.*)\(k=\d+\)$/);
    if (match) {
      const baseName = match[1];
      if (!means[baseName]) {
        means[baseName] = meanVec;
      }
    }
  }

  return means;
}

function buildSingleTransitionWindow(
  name: TransitionWindowName,
  tTrans: number,
  tailLen: number,
  excludeFirst: number,
  includeKInName: boolean,
  onCollapse?: (msg: string) => void,
): TransitionWindowRange | null {
  let effectiveName: TransitionWindowName = name;
  let effectiveExclude = excludeFirst;
  const match = (name as string).match(/(lateNoImpulse|tailNoImpulse|meanNoImpulse)\(k=(\d+)\)/);
  if (match) {
    effectiveName = match[1] as TransitionWindowName;
    effectiveExclude = Math.max(excludeFirst, parseInt(match[2], 10));
  }

  let start = 0;
  let end = tTrans;
  switch (effectiveName) {
    case 'early':
      start = 0;
      end = Math.min(6, tTrans);
      break;
    case 'mid': {
      const center = Math.floor(tTrans / 2);
      start = Math.max(0, center - 3);
      end = Math.min(tTrans, center + 3);
      break;
    }
    case 'late':
      start = Math.max(0, tTrans - 6);
      end = tTrans;
      break;
    case 'tail':
      start = Math.max(0, tTrans - tailLen);
      end = tTrans;
      break;
    case 'mean':
      start = 0;
      end = tTrans;
      break;
    case 'tailNoImpulse':
      start = Math.max(0, Math.max(tTrans - tailLen, effectiveExclude));
      end = tTrans;
      break;
    case 'lateNoImpulse':
      start = Math.max(Math.max(0, tTrans - 6), effectiveExclude);
      end = tTrans;
      break;
    case 'meanNoImpulse':
      start = Math.min(tTrans, Math.max(0, effectiveExclude));
      end = tTrans;
      break;
    case 'impulseOnly':
      start = 0;
      end = Math.min(1, tTrans);
      break;
    default:
      break;
  }

  if (end <= start) {
    const collapseReasons: string[] = [];
    if (tTrans <= 0) collapseReasons.push(`tTrans <= 0`);
    if (tailLen <= 0 && ['tail', 'tailNoImpulse'].includes(effectiveName)) {
      collapseReasons.push(`tailLen=${tailLen}`);
    }
    if (
      effectiveExclude >= tTrans &&
      ['meanNoImpulse', 'tailNoImpulse', 'lateNoImpulse'].includes(effectiveName)
    ) {
      collapseReasons.push(`excludeFirstK (${effectiveExclude}) >= tTrans (${tTrans})`);
    }
    const collapseReason = collapseReasons.length > 0 ? collapseReasons.join('; ') : 'start >= end';
    onCollapse?.(
      `[Transition windows] Collapsed window ${effectiveName} (k=${effectiveExclude}) ` +
        `start=${start} end=${end} tTrans=${tTrans} tailLen=${tailLen}; reason=${collapseReason}`,
    );
  }
  if (end <= start) end = Math.min(tTrans, start + 1);
  if (end <= start) return null;

  const label =
    includeKInName && ['tailNoImpulse', 'lateNoImpulse', 'meanNoImpulse'].includes(effectiveName)
      ? (`${effectiveName}(k=${effectiveExclude})` as TransitionWindowName)
      : effectiveName;

  return { name: label, start, end, length: end - start };
}


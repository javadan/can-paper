import { ActivityMode, PhasePhysicsParams } from './types';

const ACTIVITY_MODES: ActivityMode[] = ['spike', 'ema_spike', 'ema_voltage'];

export function parseActivityOverrides(
  argv: string[],
): { phaseA?: Partial<PhasePhysicsParams>; phaseBC?: Partial<PhasePhysicsParams> } | undefined {
  const getString = (prefix: string): string | undefined => {
    const raw = argv.find((arg) => arg.startsWith(prefix));
    if (!raw) return undefined;
    const idx = raw.indexOf('=');
    return idx >= 0 ? raw.slice(idx + 1) : undefined;
  };

  const parseMode = (prefix: string): ActivityMode | undefined => {
    const value = getString(prefix);
    if (!value) return undefined;
    if (!ACTIVITY_MODES.includes(value as ActivityMode)) {
      throw new Error(`[CLI] ${prefix.slice(2, -1)} must be one of ${ACTIVITY_MODES.join(', ')}. Received: ${value}`);
    }
    return value as ActivityMode;
  };

  const parseAlpha = (prefix: string): number | undefined => {
    const value = getString(prefix);
    if (value === undefined) return undefined;
    const parsed = parseFloat(value);
    if (!Number.isFinite(parsed)) {
      throw new Error(`[CLI] ${prefix.slice(2, -1)} must be numeric. Received: ${value}`);
    }
    return parsed;
  };

  const columnMode = parseMode('--column.activityMode=');
  const columnAlpha = parseAlpha('--column.activityAlpha=');
  const phaseBCMode = parseMode('--phaseBC.activityMode=');
  const phaseBCAlpha = parseAlpha('--phaseBC.activityAlpha=');

  if (!columnMode && columnAlpha === undefined && !phaseBCMode && phaseBCAlpha === undefined) return undefined;

  const phaseA: Partial<PhasePhysicsParams> = {};
  const phaseBC: Partial<PhasePhysicsParams> = {};

  if (columnMode) {
    phaseA.activityMode = columnMode;
    phaseBC.activityMode = columnMode;
  }
  if (columnAlpha !== undefined) {
    phaseA.activityAlpha = columnAlpha;
    phaseBC.activityAlpha = columnAlpha;
  }
  if (phaseBCMode) phaseBC.activityMode = phaseBCMode;
  if (phaseBCAlpha !== undefined) phaseBC.activityAlpha = phaseBCAlpha;

  return {
    phaseA: Object.keys(phaseA).length ? phaseA : undefined,
    phaseBC: Object.keys(phaseBC).length ? phaseBC : undefined,
  };
}

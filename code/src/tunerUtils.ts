import { MathColumn } from './mathColumn';
import * as MathOps from './mathHelpers';
import { PhasePhysicsParams, Topology } from './types';

export interface PhaseAEvalResult {
  finalAccuracy: number;
  steps: number;
}

export interface PhaseATunePoint {
  k_inhib: number;
  v_th: number;
  alpha: number;
  wInRowNorm: number;
  wAttrRowNorm: number;
  eta_attr: number;
  eta_out: number;
}

export function trainPhaseAUntilTarget(
  topology: Topology,
  physics: PhasePhysicsParams,
  eta_attr: number,
  eta_out: number,
  baseSeed: number,
  targetAcc: number,
  windowSize: number,
  minWindowFill: number,
  maxSteps: number,
): { col: MathColumn; steps: number; finalWindowAcc: number } {
  const col = new MathColumn(topology, baseSeed);
  col.learnParams.eta_attr = eta_attr;
  col.learnParams.eta_out = eta_out;
  col.applyPhysicsParams(physics);
  col.resetState();

  const rng = MathOps.createRng(baseSeed + 1);
  const numDigits = col.net.numDigits;

  const history: number[] = [];
  let sum = 0;

  for (let step = 0; step < maxSteps; step++) {
    const digit = Math.floor(rng() * numDigits);
    const { correct } = col.runPhaseA(digit);
    const val = correct ? 1 : 0;

    history.push(val);
    sum += val;
    if (history.length > windowSize) {
      sum -= history.shift() ?? 0;
    }

    const windowAcc = sum / history.length;

    if (history.length >= minWindowFill && windowAcc >= targetAcc) {
      return { col, steps: step + 1, finalWindowAcc: windowAcc };
    }
  }

  const finalWindowAcc = history.length ? sum / history.length : 0;
  return { col, steps: maxSteps, finalWindowAcc };
}

export function evaluatePhaseACandidate(
  topology: Topology,
  physics: PhasePhysicsParams,
  eta_attr: number,
  eta_out: number,
  baseSeed: number,
  maxSteps: number,
): PhaseAEvalResult {
  const col = new MathColumn(topology, baseSeed);
  col.learnParams.eta_attr = eta_attr;
  col.learnParams.eta_out = eta_out;
  col.applyPhysicsParams(physics);
  col.resetState();

  const windowSize = 200;
  const history: number[] = [];
  let sum = 0;
  const rng = MathOps.createRng(baseSeed + 1);
  const numDigits = col.net.numDigits;

  for (let step = 0; step < maxSteps; step++) {
    const digit = Math.floor(rng() * numDigits);
    const { correct } = col.runPhaseA(digit);
    const val = correct ? 1 : 0;
    history.push(val);
    sum += val;
    if (history.length > windowSize) {
      sum -= history.shift() ?? 0;
    }

    const windowAcc = sum / history.length;
    if (history.length >= windowSize / 2 && windowAcc >= 0.99) {
      return { finalAccuracy: windowAcc, steps: step + 1 };
    }
  }

  const finalAcc = history.length > 0 ? sum / history.length : 0;
  return { finalAccuracy: finalAcc, steps: maxSteps };
}

export function generatePhaseAGrid(): PhaseATunePoint[] {
  const k_inhib = [6, 8];
  const v_th = [1.6, 1.8];
  const alpha = [0.95, 0.98];
  const wInRowNorm = [2.0, 3.0];
  const wAttrRowNorm = [1.0, 1.5, 2.0];
  const eta_attr = [0.01];
  const eta_out = [0.05];

  const grid: PhaseATunePoint[] = [];
  for (const k of k_inhib) {
    for (const v of v_th) {
      for (const a of alpha) {
        for (const win of wInRowNorm) {
          for (const wa of wAttrRowNorm) {
            for (const etaAttr of eta_attr) {
              for (const etaOut of eta_out) {
                grid.push({
                  k_inhib: k,
                  v_th: v,
                  alpha: a,
                  wInRowNorm: win,
                  wAttrRowNorm: wa,
                  eta_attr: etaAttr,
                  eta_out: etaOut,
                });
              }
            }
          }
        }
      }
    }
  }

  return grid;
}

import {
  BGControllerAction,
  BGControllerConfig,
  ControllerConfig,
  ControllerMode,
  ControllerPhase,
  PhaseBCConfig,
  TransitionSustainStats,
} from './types';
import type { RNG } from './mathHelpers';

export type BeforeTransitionContext = {
  phase: ControllerPhase;
  trialIdx: number;
  features?: number[];
};

export type ControllerDecision = {
  applyUpdates?: boolean;
  allowStopEpisode?: boolean;
  suppressStopEpisode?: boolean;
  settleSteps?: number;
  action?: BGControllerAction;
  abortEpisode?: boolean;
};

export type ControllerOutcome = {
  phase: ControllerPhase;
  trialIdx: number;
  correct: boolean | null;
  aborted?: boolean;
  sustain: TransitionSustainStats;
  tailSpikeMass: number;
  transSpikeMass: number;
  gateFailed: boolean;
  updatesSkipped: boolean;
  stopEpisode: boolean;
};

export type ControllerStats = {
  actions?: Partial<Record<BGControllerAction, number>>;
  totalReward?: number;
  trials?: number;
};

export interface Controller {
  resetEpisode(): void;
  beforeTransition(context: BeforeTransitionContext): ControllerDecision;
  afterTransition(outcome: ControllerOutcome): void;
  shouldAbortEpisode(): boolean;
  getStats(): ControllerStats | undefined;
}

class StandardController implements Controller {
  constructor(private config: ControllerConfig, private phaseBCConfig: PhaseBCConfig) {}

  resetEpisode(): void {}

  beforeTransition(context: BeforeTransitionContext): ControllerDecision {
    const abortAfter = this.phaseBCConfig.sustainGate.abortAfterTrials ?? 0;
    const suppressStopEpisode = abortAfter > 0 && abortAfter > context.trialIdx;
    return {
      suppressStopEpisode,
    };
  }

  afterTransition(): void {}

  shouldAbortEpisode(): boolean {
    return false;
  }

  getStats(): ControllerStats | undefined {
    return undefined;
  }
}

class BGController implements Controller {
  private weights!: Record<BGControllerAction, number[]>;
  private lastAction!: BGControllerAction | null;
  private lastFeatures!: number[];
  private rewardSum!: number;
  private trials!: number;
  private actionCounts!: Partial<Record<BGControllerAction, number>>;
  private dwellCount!: number;
  private phaseTrials!: Partial<Record<ControllerPhase, number>>;
  private lastPhase!: ControllerPhase | null;

  constructor(private config: BGControllerConfig, private rng: RNG) {
    if (this.config.temperature <= 0) {
      throw new Error('BGController temperature must be greater than zero.');
    }
    this.config.sampleActions ??= true;
    this.config.minPhaseDuration ??= {};
    this.resetEpisode();
  }

  resetEpisode(): void {
    this.weights = this.buildZeroWeights();
    this.lastAction = null;
    this.lastFeatures = Array(5).fill(0);
    this.rewardSum = 0;
    this.trials = 0;
    this.actionCounts = {};
    this.dwellCount = 0;
    this.phaseTrials = {};
    this.lastPhase = null;
  }

  beforeTransition(context: BeforeTransitionContext): ControllerDecision {
    const phaseChanged = this.lastPhase !== context.phase;
    this.incrementPhaseTrials(context.phase);
    const features = context.features ?? this.lastFeatures;
    const { action: sampledAction, logits } = this.sampleAction(features);
    const previousAction = phaseChanged ? null : this.lastAction;
    const previousDwell = phaseChanged ? 0 : this.dwellCount;
    const action = this.applyConstraints(sampledAction, logits, context, previousAction, previousDwell);
    this.lastAction = action;
    this.dwellCount = action === previousAction ? previousDwell + 1 : 1;
    this.lastFeatures = features;
    this.actionCounts[action] = (this.actionCounts[action] ?? 0) + 1;

    return this.buildDecision(action);
  }

  afterTransition(outcome: ControllerOutcome): void {
    if (!this.lastAction) return;
    const reward = this.computeReward(outcome);
    this.rewardSum += reward;
    this.trials += 1;
    const features = outcome.aborted ? this.lastFeatures : this.extractFeatures(outcome);
    this.lastFeatures = features;
    this.updateWeights(this.lastAction, features, reward);
  }

  shouldAbortEpisode(): boolean {
    return false;
  }

  getStats(): ControllerStats {
    return {
      actions: this.actionCounts,
      totalReward: this.rewardSum,
      trials: this.trials,
    };
  }

  private sampleAction(features: number[]): { action: BGControllerAction; logits: number[] } {
    const available = this.config.actions;
    if (available.length === 0) return { action: 'GO', logits: [] };
    const logits = available.map((a) => this.dot(this.weights[a], features) / this.config.temperature);
    if (!this.config.sampleActions) {
      return { action: this.argmaxAction(available, logits), logits };
    }
    if (this.rng() < this.config.epsilon) {
      return { action: available[Math.floor(this.rng() * available.length)], logits };
    }
    const maxLogit = Math.max(...logits);
    const exps = logits.map((l) => Math.exp(l - maxLogit));
    const sumExp = exps.reduce((acc, v) => acc + v, 0);
    const probs = exps.map((v) => v / (sumExp || 1));
    const r = this.rng();
    let acc = 0;
    for (let i = 0; i < probs.length; i++) {
      acc += probs[i];
      if (r <= acc) return { action: available[i], logits };
    }
    return { action: available[available.length - 1], logits };
  }

  private argmaxAction(available: BGControllerAction[], logits: number[]): BGControllerAction {
    let bestIdx = 0;
    let bestLogit = logits[0];
    for (let i = 1; i < logits.length; i++) {
      if (logits[i] > bestLogit) {
        bestLogit = logits[i];
        bestIdx = i;
      }
    }
    return available[bestIdx];
  }

  private computeReward(outcome: ControllerOutcome): number {
    if (outcome.aborted) return this.config.reward.abort;
    if (outcome.correct === null) return 0;
    const base = outcome.correct ? this.config.reward.correct : this.config.reward.wrong;
    const penalty = outcome.gateFailed ? 0.1 : 0;
    return base - penalty;
  }

  private buildDecision(action: BGControllerAction): ControllerDecision {
    if (action === 'ABORT') {
      return { action, abortEpisode: true };
    }

    if (action === 'WAIT') {
      return { action, settleSteps: this.config.waitSteps, applyUpdates: true, allowStopEpisode: false };
    }

    if (action === 'GO_NO_LEARN') {
      return { action, applyUpdates: false, allowStopEpisode: false };
    }

    return { action, applyUpdates: true, allowStopEpisode: false };
  }

  private extractFeatures(outcome: ControllerOutcome): number[] {
    return [
      outcome.tailSpikeMass,
      outcome.transSpikeMass,
      outcome.sustain.tailSilentFrac,
      outcome.sustain.lateSilentFrac,
      outcome.sustain.timeToSilence,
    ];
  }

  private incrementPhaseTrials(phase: ControllerPhase): void {
    if (this.lastPhase !== phase) {
      this.lastPhase = phase;
      this.dwellCount = 0;
      this.phaseTrials[phase] = 0;
    }
    this.phaseTrials[phase] = (this.phaseTrials[phase] ?? 0) + 1;
  }

  private applyConstraints(
    desired: BGControllerAction,
    logits: number[],
    context: BeforeTransitionContext,
    previousAction: BGControllerAction | null,
    previousDwell: number,
  ): BGControllerAction {
    if (!previousAction) return desired;

    const phaseTrials = this.phaseTrials[context.phase] ?? 0;
    const minDwell = Math.max(0, this.config.minDwell ?? 0);
    const minPhaseDuration = Math.max(0, this.config.minPhaseDuration?.[context.phase] ?? 0);
    const hysteresis = Math.max(0, this.config.hysteresis ?? 0);

    const orderBlocked =
      this.config.enforceOrder && this.getActionIndex(desired) < this.getActionIndex(previousAction);
    const dwellBlocked = desired !== previousAction && previousDwell < minDwell;
    const phaseBlocked = desired !== previousAction && phaseTrials <= minPhaseDuration;
    const hysteresisBlocked =
      desired !== previousAction && hysteresis > 0 && this.logitGap(desired, previousAction, logits) < hysteresis;

    if (orderBlocked || dwellBlocked || phaseBlocked || hysteresisBlocked) {
      return previousAction;
    }

    return desired;
  }

  private getActionIndex(action: BGControllerAction): number {
    return this.config.actions.indexOf(action);
  }

  private logitGap(
    nextAction: BGControllerAction,
    prevAction: BGControllerAction,
    logits: number[],
  ): number {
    const nextIdx = this.getActionIndex(nextAction);
    const prevIdx = this.getActionIndex(prevAction);
    const nextLogit = nextIdx >= 0 ? logits[nextIdx] : Number.NEGATIVE_INFINITY;
    const prevLogit = prevIdx >= 0 ? logits[prevIdx] : Number.NEGATIVE_INFINITY;
    return nextLogit - prevLogit;
  }

  private updateWeights(action: BGControllerAction, features: number[], reward: number): void {
    const weights = this.weights[action];
    for (let i = 0; i < weights.length; i++) {
      weights[i] += this.config.eta * reward * features[i];
    }
  }

  private buildZeroWeights(): Record<BGControllerAction, number[]> {
    return Object.fromEntries(
      this.config.actions.map((action) => [action, Array(5).fill(0)]),
    ) as Record<BGControllerAction, number[]>;
  }

  private dot(a: number[], b: number[]): number {
    let sum = 0;
    for (let i = 0; i < Math.min(a.length, b.length); i++) sum += a[i] * b[i];
    return sum;
  }
}

export function createController(
  config: ControllerConfig,
  phaseBCConfig: PhaseBCConfig,
  rng: RNG = Math.random,
): Controller {
  if (config.mode === 'bg' && config.bg) {
    return new BGController(config.bg, rng);
  }
  return new StandardController(config, phaseBCConfig);
}

export function controllerMode(config: ControllerConfig): ControllerMode {
  return config.mode ?? 'standard';
}

import { ExperimentResult, PhasePhysicsParams, Topology } from '../types';

export type ProbeCandidateMetrics = {
  score: number;
  sustain: number;
  acc: number;
  tailSilent: number;
  tailSpike: number;
  transSpike: number;
  abortedFrac: number;
  trials: number;
  trialsEvaluated: number;
  trialsAborted: number;
};

export type ProbeCandidate = {
  physics: PhasePhysicsParams;
  metrics: ProbeCandidateMetrics;
  controller?: { mode: string; stats?: unknown };
  seeds?: { column?: number; controller?: number };
  trialStarts?: { seed?: number; digits?: number[] };
};

export type ProbeSampledCandidate = {
  sampleOrder: number;
  physics: PhasePhysicsParams;
  controller?: { mode: string; stats?: unknown };
  seeds?: { column?: number; controller?: number };
  trialStarts?: { seed?: number; digits?: number[] };
};

export type ProbeResolvedFields = {
  evalWindow: string;
  learnWindow: string;
  settleWindow: string;
  tTrans: number;
  tailLen: number;
  activityMode: string;
  activityAlpha: number;
};

export type ProbeTopKPayload = {
  version: number;
  topology: Topology;
  probe: {
    seed: number;
    maxCandidates: number;
    trialsPerCandidate: number;
    rankTop: number;
    wSustain: number;
    wAcc: number;
  };
  resolved: ProbeResolvedFields;
  candidates: ProbeCandidate[];
  sampledCandidates: ProbeSampledCandidate[];
};

export type ReportMeta = {
  reportId: string;
  createdAt: string;
  gitCommit?: string;
  inputs: {
    probes?: Record<Topology, string | undefined>;
    runs?: Record<string, string | undefined>;
  };
  cliArgs: string[];
};

export type LoadedReportInputs = {
  probeTopK?: Partial<Record<Topology, ProbeTopKPayload>>;
  runResults?: ExperimentResult[];
};

export type ReportFigure = {
  filename: string;
  caption?: string;
};

export type RunHistoryFigure = {
  phase: 'B' | 'C';
  filename?: string;
  caption?: string;
  available: boolean;
  note: string;
  finalAcc?: number;
  abortedFrac?: number;
};

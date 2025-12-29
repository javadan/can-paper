import argparse
import csv
import json
import os
from typing import Dict, List, Mapping, MutableMapping, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np

plt.switch_backend('Agg')


def save_figure(base: str, out_dir: str, formats: Sequence[str], dpi: int = 250):
  for fmt in formats:
    path = os.path.join(out_dir, f"{base}.{fmt}")
    plt.savefig(path, bbox_inches='tight', dpi=dpi)
  plt.close()


def plot_confusion(confusion: dict, title: str, base: str, out_dir: str, formats: Sequence[str]):
  data = confusion.get('normalized') or confusion.get('counts')
  if not data:
    return
  arr = np.array(data)
  fig, ax = plt.subplots(figsize=(4, 3))
  im = ax.imshow(arr, cmap='Blues')
  ax.set_title(title)
  ax.set_xlabel('Predicted')
  ax.set_ylabel('Actual')
  for i in range(arr.shape[0]):
    for j in range(arr.shape[1]):
      ax.text(j, i, f"{arr[i, j]:.2f}", ha='center', va='center', color='black', fontsize=8)
  fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
  save_figure(base, out_dir, formats)


def plot_run(payload: dict, out_dir: str, formats: Sequence[str], topology: str):
  phase_c = payload.get('phaseC') or {}
  suffix = f"{payload['config']['topology']}_{'countingOn' if payload['config'].get('useCountingPhase') else 'countingOff'}"
  confusion = phase_c.get('confusion')
  if confusion:
    plot_confusion(confusion.get('readout', {}), f"Phase C readout ({topology})", f"confusion_phaseC_readout_{suffix}", out_dir, formats)
    plot_confusion(confusion.get('proto', {}), f"Phase C proto ({topology})", f"confusion_phaseC_proto_{suffix}", out_dir, formats)
  phase_b = payload.get('phaseB') or {}
  confusion_b = phase_b.get('confusion')
  if confusion_b:
    plot_confusion(confusion_b.get('readout', {}), f"Phase B readout ({topology})", f"confusion_phaseB_readout_{suffix}", out_dir, formats)
    plot_confusion(confusion_b.get('proto', {}), f"Phase B proto ({topology})", f"confusion_phaseB_proto_{suffix}", out_dir, formats)

  fig, ax = plt.subplots(figsize=(4, 3))
  aborted = phase_c.get('abortedFrac') or 0
  acc = phase_c.get('finalAcc') or 0
  bars = ax.bar(['FinalAcc', 'Aborted'], [acc, aborted], color=['#4C78A8', '#F58518'])
  ax.bar_label(bars, fmt='%.3f', padding=3)
  ax.set_ylim(0, max(1.0, acc, aborted) * 1.1)
  ax.set_title(f"Phase C metrics ({topology})")
  save_figure(f"metrics_phaseC_aborts_{suffix}", out_dir, formats)


def _normalize_acc_history(acc_history: object) -> Tuple[Optional[Dict[str, List[float]]], Optional[str]]:
  if acc_history is None:
    return None, 'missing'

  if isinstance(acc_history, list):
    return {'readout': [float(v) for v in acc_history]}, None

  if isinstance(acc_history, MutableMapping):
    series: Dict[str, List[float]] = {}
    for key in ['readout', 'proto']:
      values = acc_history.get(key)
      if isinstance(values, list):
        series[key] = [float(v) for v in values]
    samples = acc_history.get('samples')
    if not series and isinstance(samples, list):
      series['readout'] = [float(v) for v in samples]
    if series:
      return series, None
    return None, f"unknown keys: {', '.join(acc_history.keys())}"

  return None, 'unsupported format'


def _smooth(values: List[float], window: int = 25) -> List[float]:
  if len(values) < 3 or window <= 1:
    return values
  arr = np.array(values, dtype=float)
  kernel = np.ones(window) / window
  smoothed = np.convolve(arr, kernel, mode='same')
  return smoothed.tolist()


def _build_title(topology: str, config: Mapping[str, object], phase_label: str) -> str:
  controller = (config.get('controller') or {}) if isinstance(config, Mapping) else {}
  mode = controller.get('mode', 'standard')
  phase_bc_cfg = (config.get('phaseBCConfig') or {}) if isinstance(config, Mapping) else {}
  sustain = (phase_bc_cfg.get('sustainGate') or {}).get('enabled', False)
  curriculum = (config.get('curriculum') or {}) if isinstance(config, Mapping) else {}
  phase_bc = curriculum.get('phaseBC', {}) if isinstance(curriculum, Mapping) else {}
  t_trans = phase_bc.get('tTrans')
  tail_len = phase_bc.get('tailLen')
  eval_window = phase_bc_cfg.get('evalWindow')
  learn_window = phase_bc_cfg.get('learnWindow')
  parts = [
    f"{topology} | ctrl={mode}",
    f"sustain={'on' if sustain else 'off'}",
    f"tTrans={t_trans}" if t_trans is not None else None,
    f"tailLen={tail_len}" if tail_len is not None else None,
    f"eval={eval_window}" if eval_window else None,
    f"learn={learn_window}" if learn_window else None,
    f"{phase_label}",
  ]
  return ' | '.join([p for p in parts if p])


def _plot_history_series(
  series: Mapping[str, List[float]],
  title: str,
  base: str,
  out_dir: str,
  formats: Sequence[str],
):
  fig, ax = plt.subplots(figsize=(5, 3.2))
  for label, values in series.items():
    if not values:
      continue
    xs = list(range(len(values)))
    ax.plot(xs, values, label=label)
    if len(values) >= 10:
      smoothed = _smooth(values, window=min(25, max(3, len(values) // 8)))
      ax.plot(xs, smoothed, linestyle='--', alpha=0.7, label=f"{label} (smoothed)")
  ax.set_xlabel('Trial index')
  ax.set_ylabel('Accuracy')
  ax.set_ylim(0, 1)
  ax.set_title(title)
  ax.grid(True, linestyle='--', alpha=0.3)
  if len(series.keys()) > 1:
    ax.legend()
  save_figure(base, out_dir, formats)


def plot_run_history(payload: dict, out_dir: str, formats: Sequence[str], topology: str):
  config = payload.get('config', {}) if isinstance(payload, Mapping) else {}
  suffix = f"{config.get('topology', topology)}_{'countingOn' if config.get('useCountingPhase') else 'countingOff'}"
  phase_c = payload.get('phaseC') if isinstance(payload, Mapping) else None
  phase_b = payload.get('phaseB') if isinstance(payload, Mapping) else None

  acc_c, note_c = _normalize_acc_history(phase_c.get('accHistory') if isinstance(phase_c, Mapping) else None)
  if acc_c:
    title_c = _build_title(topology, config, 'Phase C')
    _plot_history_series(acc_c, title_c, f"acc_history_phaseC_{suffix}", out_dir, formats)
  elif note_c:
    print(f"[report_plot] Phase C history skipped ({note_c})")

  acc_b, note_b = _normalize_acc_history(phase_b.get('accHistory') if isinstance(phase_b, Mapping) else None)
  if acc_b:
    title_b = _build_title(topology, config, 'Phase B')
    _plot_history_series(acc_b, title_b, f"acc_history_phaseB_{suffix}", out_dir, formats)
  elif note_b and phase_b is not None:
    print(f"[report_plot] Phase B history skipped ({note_b})")


def binned_stat(x: np.ndarray, y: np.ndarray, values: np.ndarray, bins: int = 6):
  x_edges = np.linspace(x.min(), x.max(), bins + 1)
  y_edges = np.linspace(y.min(), y.max(), bins + 1)
  grid = np.zeros((bins, bins))
  counts = np.zeros((bins, bins))
  for xv, yv, v in zip(x, y, values):
    xi = np.searchsorted(x_edges, xv, side='right') - 1
    yi = np.searchsorted(y_edges, yv, side='right') - 1
    xi = np.clip(xi, 0, bins - 1)
    yi = np.clip(yi, 0, bins - 1)
    grid[yi, xi] += v
    counts[yi, xi] += 1
  counts[counts == 0] = 1
  return grid / counts, x_edges, y_edges


def plot_probe(payload: dict, out_dir: str, formats: Sequence[str], topology: str, probe_top_k: int):
  candidates = payload.get('candidates', [])[:probe_top_k]
  if not candidates:
    return
  scores = np.array([c['metrics']['score'] for c in candidates])
  acc = np.array([c['metrics']['acc'] for c in candidates])
  sustain = np.array([c['metrics']['sustain'] for c in candidates])
  aborted = np.array([c['metrics'].get('abortedFrac', 0) for c in candidates])
  tail_silent = np.array([c['metrics'].get('tailSilent', 0) for c in candidates])
  k_inhib = np.array([c['physics']['k_inhib'] for c in candidates])
  v_th = np.array([c['physics']['v_th'] for c in candidates])

  fig, ax = plt.subplots(figsize=(4.5, 3.5))
  sc = ax.scatter(acc, scores, c=aborted, cmap='magma', s=60, edgecolor='k', alpha=0.8)
  ax.set_xlabel('Accuracy')
  ax.set_ylabel('Score')
  ax.set_title(f"Probe scores ({topology})")
  cbar = fig.colorbar(sc, ax=ax)
  cbar.set_label('Aborted frac')
  save_figure(f"probe_score_scatter_{topology}", out_dir, formats)

  fig, ax = plt.subplots(figsize=(4.5, 3.5))
  sc2 = ax.scatter(acc, tail_silent, c=sustain, cmap='viridis', s=60, edgecolor='k', alpha=0.8)
  ax.set_xlabel('Accuracy')
  ax.set_ylabel('Tail silent frac')
  ax.set_title(f"Silence vs accuracy ({topology})")
  cbar2 = fig.colorbar(sc2, ax=ax)
  cbar2.set_label('Sustain score')
  save_figure(f"probe_accuracy_tailSilent_{topology}", out_dir, formats)

  heat, x_edges, y_edges = binned_stat(k_inhib, v_th, acc)
  fig, ax = plt.subplots(figsize=(4.5, 3.5))
  im = ax.imshow(heat, origin='lower', aspect='auto',
                 extent=[x_edges[0], x_edges[-1], y_edges[0], y_edges[-1]], cmap='plasma')
  ax.set_xlabel('k_inhib')
  ax.set_ylabel('v_th')
  ax.set_title(f"Accuracy heatmap ({topology})")
  fig.colorbar(im, ax=ax)
  save_figure(f"probe_heat_k_inhib_vs_v_th_{topology}", out_dir, formats)


def plot_traces(json_path: str, out_dir: str, formats: Sequence[str]):
  if not os.path.exists(json_path):
    print(f"Skipping traces, file not found: {json_path}")
    return

  with open(json_path, 'r', encoding='utf-8') as f:
    data = json.load(f)

  traces = data.get('traces', [])
  if not traces:
    return

  sims = []
  for trace in traces:
    values = trace.get('targetProtoSim')
    if isinstance(values, list):
      sims.append(values)

  if not sims:
    return

  arr = np.array(sims)
  mean_trace = np.mean(arr, axis=0)
  std_trace = np.std(arr, axis=0)
  x = np.arange(len(mean_trace))

  fig, ax = plt.subplots(figsize=(6, 4))
  for i in range(min(10, len(sims))):
    ax.plot(x, sims[i], color='gray', alpha=0.2, linewidth=0.5)

  ax.plot(x, mean_trace, color='#4C78A8', linewidth=2, label='Mean Similarity')
  ax.fill_between(x, mean_trace - std_trace, mean_trace + std_trace, color='#4C78A8', alpha=0.3)

  ax.set_ylim(-0.1, 1.1)
  ax.set_xlabel('Time Step ($t$)')
  ax.set_ylabel('Similarity to Target')
  meta = data.get('meta', {})
  topo = meta.get('topology', 'unknown')
  t_trans = meta.get('tTrans', 0)
  ax.set_title(f"Attractor Stability ({topo}, t={t_trans})")
  ax.axhline(1.0, linestyle='--', color='k', alpha=0.5)

  base_name = f"trace_dynamics_{topo}"
  save_figure(base_name, out_dir, formats)
  print(f"Generated trace plot: {base_name}")


def _load_perturb_summary(summary_path: str) -> Optional[dict]:
  if not summary_path or not os.path.exists(summary_path):
    return None
  with open(summary_path, 'r', encoding='utf-8') as f:
    return json.load(f)


def _load_perturb_trials(trials_path: str) -> List[dict]:
  if not trials_path or not os.path.exists(trials_path):
    return []
  with open(trials_path, 'r', encoding='utf-8') as f:
    reader = csv.DictReader(f)
    return [row for row in reader]


def _derive_label(path: Optional[str], fallback: str) -> str:
  if not path:
    return fallback
  parent = os.path.basename(os.path.dirname(path))
  if parent == 'perturb':
    return os.path.basename(os.path.dirname(os.path.dirname(path))) or fallback
  return parent or fallback


def plot_perturbation_sets(
  sets: List[Tuple[str, Optional[str], Optional[str]]],
  out_dir: str,
  formats: Sequence[str],
):
  summaries: List[Tuple[str, dict]] = []
  trials_sets: List[Tuple[str, List[dict]]] = []
  for label, summary_path, trials_path in sets:
    summary = _load_perturb_summary(summary_path) if summary_path else None
    if summary:
      summaries.append((label, summary))
    trials = _load_perturb_trials(trials_path) if trials_path else []
    if trials:
      trials_sets.append((label, trials))

  if summaries:
    fig, ax = plt.subplots(figsize=(6.5, 3.8))
    title_parts = []
    for label, summary in summaries:
      similarity = summary.get('similarityByOffset', {}).get('mean', [])
      if not similarity:
        continue
      values = np.array([np.nan if v is None else float(v) for v in similarity])
      xs = np.arange(len(values))
      ax.plot(xs, values, linewidth=2, label=label)
      config = summary.get('config', {})
      kind = config.get('kind', 'unknown')
      at_step = config.get('atStep', 0)
      title_parts.append(f"{label}: {kind} @ t={at_step}")

    if len(summaries) == 1:
      config = summaries[0][1].get('config', {})
      duration = config.get('durationSteps', 1)
      ax.axvspan(0, max(0.5, duration), color='#F58518', alpha=0.2, label='Perturb window')

    ax.set_xlabel('Steps since perturb')
    ax.set_ylabel('Similarity to baseline')
    ax.set_ylim(-0.1, 1.1)
    ax.axhline(1.0, linestyle='--', color='k', alpha=0.4)
    ax.set_title('Perturbation recovery (mean similarity)')
    ax.grid(True, linestyle='--', alpha=0.3)
    ax.legend()
    if title_parts:
      fig.text(0.5, -0.02, '; '.join(title_parts), ha='center', fontsize=8)
    save_figure('perturb_similarity_mean', out_dir, formats)

    rates = []
    labels = []
    for label, summary in summaries:
      recovery_rate = summary.get('metrics', {}).get('recoveryRate')
      if recovery_rate is None:
        continue
      labels.append(label)
      rates.append(float(recovery_rate))
    if rates:
      fig, ax = plt.subplots(figsize=(4.8, 3))
      bars = ax.bar(labels, rates, color='#54A24B')
      ax.bar_label(bars, fmt='%.2f', padding=3)
      ax.set_ylim(0, 1)
      ax.set_title('Recovery rate by condition')
      ax.set_ylabel('Recovery rate')
      save_figure('perturb_recovery_rate', out_dir, formats)

  if trials_sets:
    fig, ax = plt.subplots(figsize=(6, 3.4))
    for label, trials in trials_sets:
      recovery_steps = []
      for row in trials:
        raw = row.get('recoverySteps')
        if raw is None or raw == '':
          continue
        try:
          recovery_steps.append(int(raw))
        except ValueError:
          continue
      if not recovery_steps:
        continue
      bins = min(20, max(5, len(set(recovery_steps))))
      ax.hist(recovery_steps, bins=bins, alpha=0.45, label=label)
    ax.set_xlabel('Recovery steps')
    ax.set_ylabel('Trials')
    ax.set_title('Recovery steps distribution by condition')
    ax.grid(True, linestyle='--', alpha=0.3)
    ax.legend()
    save_figure('perturb_recovery_steps_hist', out_dir, formats)


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument('--kind', choices=['probe', 'run', 'run_acc_history', 'perturb'], required=True)
  parser.add_argument('--input')
  parser.add_argument('--outDir', required=True)
  parser.add_argument('--formats', default='png')
  parser.add_argument('--topology', default='')
  parser.add_argument('--probeTopK', type=int, default=20)
  parser.add_argument('--traceFile', help='Path to transition_debug_*.json')
  parser.add_argument('--perturbSummary', action='append', help='Path to perturb_summary.json')
  parser.add_argument('--perturbTrials', action='append', help='Path to perturb_trials.csv')
  parser.add_argument('--perturbLabel', action='append', help='Label for perturb condition')
  args = parser.parse_args()

  out_dir = args.outDir
  os.makedirs(out_dir, exist_ok=True)
  formats: List[str] = [fmt.strip() for fmt in args.formats.split(',') if fmt.strip()]

  payload = None
  if args.kind != 'perturb':
    if not args.input:
      raise ValueError('Missing --input for run/probe plotting')
    with open(args.input, 'r', encoding='utf-8') as f:
      payload = json.load(f)

    if args.kind == 'run':
      plot_run(payload, out_dir, formats, args.topology)
    elif args.kind == 'run_acc_history':
      plot_run_history(payload, out_dir, formats, args.topology)
    else:
      plot_probe(payload, out_dir, formats, args.topology, args.probeTopK)
  else:
    summaries = []
    trials = []
    labels = []
    if args.perturbSummary:
      for entry in args.perturbSummary:
        summaries.extend([part for part in entry.split(',') if part])
    if args.perturbTrials:
      for entry in args.perturbTrials:
        trials.extend([part for part in entry.split(',') if part])
    if args.perturbLabel:
      for entry in args.perturbLabel:
        labels.extend([part for part in entry.split(',') if part])

    max_len = max(len(summaries), len(trials), len(labels), 1)
    sets: List[Tuple[str, Optional[str], Optional[str]]] = []
    for idx in range(max_len):
      summary_path = summaries[idx] if idx < len(summaries) else None
      trials_path = trials[idx] if idx < len(trials) else None
      label = labels[idx] if idx < len(labels) else _derive_label(summary_path or trials_path, f"set-{idx + 1}")
      if summary_path is None and trials_path is None:
        continue
      sets.append((label, summary_path, trials_path))

    if sets:
      plot_perturbation_sets(sets, out_dir, formats)

  if args.traceFile:
    plot_traces(args.traceFile, out_dir, formats)


if __name__ == '__main__':
  main()

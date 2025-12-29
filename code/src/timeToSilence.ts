export function computeTimeToSilence(spikeMasses: number[], silentSpikeThreshold: number, tTrans: number): number {
  const maxT = Math.min(tTrans, spikeMasses.length);
  for (let t = 0; t < maxT; t++) {
    if (spikeMasses[t] <= silentSpikeThreshold) {
      return t;
    }
  }
  return tTrans;
}

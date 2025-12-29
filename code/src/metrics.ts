export function makeConfusionMatrix(numDigits: number): number[][] {
  const matrix: number[][] = [];
  for (let i = 0; i < numDigits; i++) {
    matrix.push(new Array(numDigits).fill(0));
  }
  return matrix;
}

export function addConfusion(conf: number[][], target: number, pred: number): void {
  if (target < 0 || target >= conf.length) return;
  if (pred < 0 || pred >= conf[target].length) return;
  conf[target][pred] += 1;
}

export function normalizeConfusion(conf: number[][]): number[][] {
  return conf.map((row) => {
    const rowSum = row.reduce((sum, val) => sum + val, 0);
    if (rowSum === 0) return row.map(() => 0);
    return row.map((val) => val / rowSum);
  });
}

export function formatConfusion(conf: number[][]): string {
  const header = [' tgt\\pred', ...conf[0].map((_, idx) => idx.toString().padStart(3, ' '))].join(' ');
  const rows = conf
    .map((row, idx) => {
      const values = row.map((val) => val.toFixed(2).padStart(5, ' ')).join(' ');
      return `${idx.toString().padStart(3, ' ')} | ${values}`;
    })
    .join('\n');
  return `${header}\n${rows}`;
}

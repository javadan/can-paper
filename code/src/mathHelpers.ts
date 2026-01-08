export type RNG = () => number;

export function createRng(seed: number): RNG {
  let state = seed >>> 0;
  return () => {
    state = (1664525 * state + 1013904223) >>> 0;
    return state / 0xffffffff;
  };
}

export function randNormal(mean: number, std: number, rng: RNG): number {
  let u = 0;
  let v = 0;
  while (u === 0) u = rng();
  while (v === 0) v = rng();
  const mag = Math.sqrt(-2.0 * Math.log(u));
  const z0 = mag * Math.cos(2.0 * Math.PI * v);
  return mean + std * z0;
}

export function matVecMul(
  mat: Float32Array,
  vec: Float32Array,
  rows: number,
  cols: number,
  out: Float32Array,
): void {
  for (let r = 0; r < rows; r++) {
    let acc = 0;
    const base = r * cols;
    for (let c = 0; c < cols; c++) {
      acc += mat[base + c] * vec[c];
    }
    out[r] = acc;
  }
}

export function outerUpdate(
  mat: Float32Array,
  post: Float32Array,
  pre: Float32Array,
  rows: number,
  cols: number,
  eta: number,
): void {
  for (let r = 0; r < rows; r++) {
    const base = r * cols;
    const postVal = post[r];
    if (postVal === 0) continue;
    for (let c = 0; c < cols; c++) {
      mat[base + c] += eta * postVal * pre[c];
    }
  }
}

export type RowNormType = 'l1' | 'l2';

export function normalizeRows(
  mat: Float32Array,
  rows: number,
  cols: number,
  rowNorm: number,
  normType: RowNormType = 'l1',
): void {
  if (rowNorm <= 0) return;
  const eps = 1e-4;
  for (let r = 0; r < rows; r++) {
    const base = r * cols;
    let norm = 0;
    if (normType === 'l1') {
      for (let c = 0; c < cols; c++) {
        norm += Math.abs(mat[base + c]);
      }
    } else {
      let norm2 = 0;
      for (let c = 0; c < cols; c++) {
        const v = mat[base + c];
        norm2 += v * v;
      }
      norm = Math.sqrt(norm2);
    }
    if (norm <= eps) continue;
    const scale = rowNorm / norm;
    for (let c = 0; c < cols; c++) {
      mat[base + c] *= scale;
    }
  }
}

export function argmax(vec: Float32Array): number {
  let maxIdx = 0;
  let maxVal = vec[0];
  for (let i = 1; i < vec.length; i++) {
    if (vec[i] > maxVal) {
      maxVal = vec[i];
      maxIdx = i;
    }
  }
  return maxIdx;
}

export function oneHot(idx: number, dim: number): Float32Array {
  const v = new Float32Array(dim);
  if (idx >= 0 && idx < dim) {
    v[idx] = 1;
  }
  return v;
}

import { Topology } from './types';

export class SDREncoder {
  dim: number;
  numDigits: number;
  topology: Topology;
  prototypes: Map<number, Float32Array>;
  gridWidth: number;
  gridHeight: number;
  barHeight = 3;
  step = 1;

  constructor(dim: number, numDigits: number, topology: Topology) {
    this.dim = dim;
    this.numDigits = numDigits;
    this.topology = topology;
    this.prototypes = new Map();
    this.gridWidth = Math.round(Math.sqrt(dim));
    this.gridHeight = Math.ceil(dim / this.gridWidth);
    this.buildPrototypes();
  }

  private buildPrototypes(): void {
    for (let d = 0; d < this.numDigits; d++) {
      const proto = this.topology === 'ring'
        ? this.buildRingPrototype(d)
        : this.buildSnakePrototype(d);
      this.prototypes.set(d, proto);
    }
  }

  private buildSnakePrototype(digit: number): Float32Array {
    const proto = new Float32Array(this.dim);
    const baseRow = Math.min(this.gridHeight - this.barHeight, digit);
    for (let r = 0; r < this.barHeight; r++) {
      const row = Math.min(this.gridHeight - 1, baseRow + r);
      for (let c = 0; c < this.gridWidth; c++) {
        const idx = row * this.gridWidth + c;
        if (idx < this.dim) {
          proto[idx] = 1;
        }
      }
    }
    return proto;
  }

  private buildRingPrototype(digit: number): Float32Array {
    const proto = new Float32Array(this.dim);
    const activeBits = Math.min(this.dim, this.barHeight * this.gridWidth);

    const cx = (this.gridWidth - 1) / 2;
    const cy = (this.gridHeight - 1) / 2;
    const radius = Math.max(1, Math.min(this.gridWidth, this.gridHeight) / 2 - 2);
    const theta = (2 * Math.PI * digit) / this.numDigits;
    const px = cx + radius * Math.cos(theta);
    const py = cy + radius * Math.sin(theta);

    const distances: { idx: number; dist: number }[] = [];
    for (let idx = 0; idx < this.dim; idx++) {
      const x = idx % this.gridWidth;
      const y = Math.floor(idx / this.gridWidth);
      const dx = x - px;
      const dy = y - py;
      const dist = dx * dx + dy * dy;
      distances.push({ idx, dist });
    }

    distances.sort((a, b) => a.dist - b.dist);
    for (let i = 0; i < activeBits; i++) {
      const idx = distances[i].idx;
      proto[idx] = 1;
    }

    return proto;
  }

  encode(digit: number): Float32Array {
    const proto = this.prototypes.get(digit % this.numDigits);
    if (!proto) throw new Error(`No prototype for digit ${digit}`);
    return new Float32Array(proto);
  }

  overlap(d1: number, d2: number): number {
    const a = this.prototypes.get(d1 % this.numDigits);
    const b = this.prototypes.get(d2 % this.numDigits);
    if (!a || !b) return 0;
    let acc = 0;
    for (let i = 0; i < this.dim; i++) {
      acc += a[i] * b[i];
    }
    return acc;
  }
}

export function printOverlapExamples(dim = 256, numDigits = 10): void {
  const ring = new SDREncoder(dim, numDigits, 'ring');
  const snake = new SDREncoder(dim, numDigits, 'snake');

  const ringNeighbors = ring.overlap(0, 1);
  const ringWrapNeighbors = ring.overlap(0, numDigits - 1);
  const ringOpposite = ring.overlap(0, Math.floor(numDigits / 2));

  const snakeNeighbors = snake.overlap(0, 1);
  const snakeWrapNeighbors = snake.overlap(0, numDigits - 1);
  const snakeOpposite = snake.overlap(0, Math.floor(numDigits / 2));

  // These logs act as lightweight runtime checks to verify the expected ordering.
  /* eslint-disable no-console */
  console.log('Ring overlaps:', {
    '0~1': ringNeighbors,
    '0~9': ringWrapNeighbors,
    '0~5': ringOpposite,
  });
  console.log('Snake overlaps:', {
    '0~1': snakeNeighbors,
    '0~9': snakeWrapNeighbors,
    '0~5': snakeOpposite,
  });
  /* eslint-enable no-console */
}

if (process.env.SDR_ENCODER_DEBUG === '1') {
  printOverlapExamples();
}

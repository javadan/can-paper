import { Operator } from './types';

export function computeTargetDigit(startDigit: number, op: Operator, numDigits: number): number {
  if (op === 'plus') {
    return (startDigit + 1) % numDigits;
  }
  return (startDigit - 1 + numDigits) % numDigits;
}

export function isBoundaryStartDigit(startDigit: number, op: Operator, numDigits: number): boolean {
  if (op === 'plus') {
    return startDigit === numDigits - 1;
  }
  return startDigit === 0;
}

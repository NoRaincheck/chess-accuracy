// Log-likelihood / score surface utilities for anchor-grid ELO localization.
// Mirrors the Python pipeline in estimate_elo.py (_make_grid, bicubic
// upsample, marginal mode finding) — JS uses bilinear interpolation, which is
// sufficient for localization and never overshoots on noisy score surfaces.

import { MIN_ELO, MAX_ELO } from './scoring.js';

// Uniform ELO grid from lo to hi (inclusive, hi snapped into the grid).
// Mirrors python estimate_elo.py:_make_grid.
function makeGrid(lo, hi, step) {
  const n = Math.ceil((hi - lo) / step);
  const vals = [];
  for (let k = 0; k <= n; k++) vals.push(Math.min(hi, lo + k * step));
  return vals;
}

// Fine grid centered on `center` with half-width `margin`, clamped to bounds.
function windowGrid(center, margin, nPoints, lo, hi) {
  const start = Math.max(lo, center - margin);
  const end = Math.min(hi, center + margin);
  if (nPoints <= 1) return [start];
  const vals = [];
  for (let k = 0; k < nPoints; k++) vals.push(start + ((end - start) * k) / (nPoints - 1));
  return vals;
}

// Bilinear upsample of a row-major (srcW x srcB) surface onto a denser
// (dstW x dstB) grid spanning the same range (corners aligned).
function upsampleBilinear(src, srcW, srcB, dstW, dstB) {
  const out = new Float64Array(dstW * dstB);
  const xMax = Math.max(srcW - 2, 0);
  const yMax = Math.max(srcB - 2, 0);
  for (let i = 0; i < dstW; i++) {
    const x = dstW === 1 ? 0 : (i * (srcW - 1)) / Math.max(dstW - 1, 1);
    const x0 = Math.max(0, Math.min(Math.floor(x), xMax));
    const x1 = Math.min(x0 + 1, srcW - 1);
    const fx = x - x0;
    for (let j = 0; j < dstB; j++) {
      const y = dstB === 1 ? 0 : (j * (srcB - 1)) / Math.max(dstB - 1, 1);
      const y0 = Math.max(0, Math.min(Math.floor(y), yMax));
      const y1 = Math.min(y0 + 1, srcB - 1);
      const fy = y - y0;

      const v00 = src[x0 * srcB + y0];
      const v01 = src[x0 * srcB + y1];
      const v10 = src[x1 * srcB + y0];
      const v11 = src[x1 * srcB + y1];

      out[i * dstB + j] =
        v00 * (1 - fx) * (1 - fy) +
        v01 * (1 - fx) * fy +
        v10 * fx * (1 - fy) +
        v11 * fx * fy;
    }
  }
  return out;
}

// Per-color modes from an interpolated surface: argmax of each axis marginal.
function marginalModes(surface, nW, nB) {
  const rowSum = new Float64Array(nW);
  const colSum = new Float64Array(nB);
  for (let i = 0; i < nW; i++) {
    for (let j = 0; j < nB; j++) {
      rowSum[i] += surface[i * nB + j];
      colSum[j] += surface[i * nB + j];
    }
  }
  let wi = 0;
  let bi = 0;
  for (let i = 1; i < nW; i++) if (rowSum[i] > rowSum[wi]) wi = i;
  for (let j = 1; j < nB; j++) if (colSum[j] > colSum[bi]) bi = j;
  return { wi, bi };
}

export { makeGrid, windowGrid, upsampleBilinear, marginalModes, MIN_ELO, MAX_ELO };

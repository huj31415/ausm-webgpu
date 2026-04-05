/**
 * Airfoil DAT file parser and Catmull-Rom interpolator.
 *
 * Supports two common formats, auto-detected from the data:
 *
 *   Selig   — one continuous list starting at x≈1 (upper TE), wrapping around
 *             the leading edge, ending at x≈1 (lower TE).  The boundary between
 *             surfaces is the x-minimum (leading edge), not a blank line.
 *
 *   Lednicer — two separate blocks each starting at x≈0 (leading edge) and
 *              running to x≈1 (trailing edge), separated by a blank line.
 *              A "N. N." point-count header may precede the data.
 *
 * Regardless of input format, `parseDat` normalises the output so that:
 *   upper runs TE → LE   (upper[0] is near x=1, upper[last] is near x=0)
 *   lower runs LE → TE   (lower[0] is near x=0, lower[last] is near x=1)
 */

// ---------------------------------------------------------------------------
// 1. Parser
// ---------------------------------------------------------------------------

/**
 * Extract raw numeric lines from the file, grouped by blank-line-separated
 * blocks.  Header lines and point-count lines are dropped.
 * @param {string} text
 * @returns {[number,number][][]}  Array of blocks, each block an array of [x,y].
 */
function extractBlocks(text) {
  const blocks = [];
  let current = [];

  for (const line of text.split('\n')) {
    const trimmed = line.trim();

    if (!trimmed) {
      // Blank line → flush current block
      if (current.length) { blocks.push(current); current = []; }
      continue;
    }

    const tokens = trimmed.split(/\s+/);
    if (tokens.length < 2) continue;

    const x = parseFloat(tokens[0]);
    const y = parseFloat(tokens[1]);
    if (isNaN(x) || isNaN(y)) continue;

    // Drop "103. 103." style point-count headers (integers > 1 in both columns)
    if (Number.isInteger(x) && Number.isInteger(y) && x > 1 && y > 1) continue;

    current.push([x, y]);
  }
  if (current.length) blocks.push(current);

  return blocks;
}

/**
 * Detect format and return normalised { upper, lower } where:
 *   upper = TE → LE
 *   lower = LE → TE
 *
 * @param {string} text - Raw file contents.
 * @returns {{ upper: [number,number][], lower: [number,number][], format: string }}
 */
function parseDat(text) {
  const blocks = extractBlocks(text);
  if (!blocks.length) throw new Error('No numeric data found in DAT file.');

  // ── Format detection ──────────────────────────────────────────────────────
  // Lednicer: 2+ blocks, each starting near x=0.
  // Selig:    1 block (or 2 blocks that together form a continuous wrap-around)
  //           whose first point is near x=1.

  const firstX = blocks[0][0][0];
  const isLednicer = blocks.length >= 2 && firstX < 0.1;

  if (isLednicer) {
    // ── Lednicer ─────────────────────────────────────────────────────────────
    // Block 0: upper surface, LE→TE  →  reverse to get TE→LE
    // Block 1: lower surface, LE→TE  →  keep as-is
    const upper = [...blocks[0]].reverse();
    const lower = blocks[1];
    return { upper, lower, format: 'lednicer' };

  } else {
    // ── Selig ─────────────────────────────────────────────────────────────────
    // One continuous list.  Split at the leading edge = point with minimum x.
    const pts = blocks.flat();
    if (pts.length < 4) throw new Error('Too few coordinate pairs found in DAT file.');

    // Find the index of the leading-edge point (minimum x)
    let leIndex = 0;
    for (let i = 1; i < pts.length; i++) {
      if (pts[i][0] < pts[leIndex][0]) leIndex = i;
    }

    // upper: index 0 (TE) … leIndex (LE)
    // lower: leIndex (LE) … last (TE)
    const upper = pts.slice(0, leIndex + 1);
    const lower = pts.slice(leIndex);
    return { upper, lower, format: 'selig' };
  }
}

// ---------------------------------------------------------------------------
// 2. Arc-length parameterization
// ---------------------------------------------------------------------------

/**
 * Build a cumulative arc-length parameter array for a sequence of points.
 * @param {[number,number][]} pts
 * @returns {number[]} t values in [0, totalLength]
 */
function arcLengthParams(pts) {
  const t = [0];
  for (let i = 1; i < pts.length; i++) {
    const dx = pts[i][0] - pts[i - 1][0];
    const dy = pts[i][1] - pts[i - 1][1];
    t.push(t[i - 1] + Math.sqrt(dx * dx + dy * dy));
  }
  return t;
}

// ---------------------------------------------------------------------------
// 3. Catmull-Rom spline evaluation
// ---------------------------------------------------------------------------

/**
 * Evaluate a Catmull-Rom spline at local parameter u ∈ [0,1] given four
 * control points p0, p1, p2, p3 (p1→p2 is the active segment).
 * Uses the centripetal parameterization (alpha=0.5) to avoid cusps on
 * unevenly spaced data.
 *
 * @param {[number,number]} p0
 * @param {[number,number]} p1
 * @param {[number,number]} p2
 * @param {[number,number]} p3
 * @param {number} u  - parameter in [0, 1] along the p1→p2 segment
 * @returns {[number, number]}
 */
function catmullRom(p0, p1, p2, p3, u) {
  // Centripetal Catmull-Rom: compute knot intervals with alpha = 0.5
  const alpha = 0.5;
  const knot = (a, b) => {
    const dx = b[0] - a[0], dy = b[1] - a[1];
    return Math.pow(dx * dx + dy * dy, alpha / 2);
  };

  const t0 = 0;
  const t1 = t0 + knot(p0, p1);
  const t2 = t1 + knot(p1, p2);
  const t3 = t2 + knot(p2, p3);

  // Remap u ∈ [0,1] → t ∈ [t1, t2]
  const t = t1 + u * (t2 - t1);

  // Barry–Goldman recursive evaluation
  function lerp2(a, b, ta, tb, tv) {
    if (Math.abs(tb - ta) < 1e-12) return [(a[0] + b[0]) / 2, (a[1] + b[1]) / 2];
    const f = (tv - ta) / (tb - ta);
    return [a[0] + f * (b[0] - a[0]), a[1] + f * (b[1] - a[1])];
  }

  const A1 = lerp2(p0, p1, t0, t1, t);
  const A2 = lerp2(p1, p2, t1, t2, t);
  const A3 = lerp2(p2, p3, t2, t3, t);
  const B1 = lerp2(A1, A2, t0, t2, t);
  const B2 = lerp2(A2, A3, t1, t3, t);
  return lerp2(B1, B2, t1, t2, t);
}

// ---------------------------------------------------------------------------
// 4. Spline builder — wraps the raw control points
// ---------------------------------------------------------------------------

/**
 * Build a spline sampler from an ordered array of control points.
 * Phantom endpoints are added at each end to provide tangent direction
 * without distorting the curve at the leading/trailing edge.
 *
 * @param {[number,number][]} pts - Control points in traversal order.
 * @returns {(t: number) => [number, number]}  Sampler for t ∈ [0, 1].
 */
function buildSpline(pts) {
  const n = pts.length;

  // Phantom endpoints: reflect neighbours across the boundary point
  const phantom0 = [
    2 * pts[0][0] - pts[1][0],
    2 * pts[0][1] - pts[1][1],
  ];
  const phantomN = [
    2 * pts[n - 1][0] - pts[n - 2][0],
    2 * pts[n - 1][1] - pts[n - 2][1],
  ];

  // Extended control-point array: [phantom0, ...pts, phantomN]
  const cp = [phantom0, ...pts, phantomN];

  // Arc-length params for the original points only (indices 1..n in cp)
  const alParams = arcLengthParams(pts);
  const totalLen = alParams[alParams.length - 1];
  // Normalize to [0,1]
  const tNorm = alParams.map(v => v / totalLen);

  /**
   * Given a global t ∈ [0, 1], return the interpolated [x, y].
   */
  return function sample(t) {
    // Clamp to valid range
    t = Math.max(0, Math.min(1, t));

    // Binary search for the segment straddling t in tNorm
    let lo = 0, hi = tNorm.length - 2;
    while (lo < hi) {
      const mid = (lo + hi) >> 1;
      if (tNorm[mid + 1] < t) lo = mid + 1;
      else hi = mid;
    }
    const seg = lo; // segment index in [0, n-2]

    // Local parameter within this segment
    const t0s = tNorm[seg], t1s = tNorm[seg + 1];
    const u = t1s > t0s ? (t - t0s) / (t1s - t0s) : 0;

    // Map segment index → extended cp indices (offset by 1 because of phantom0)
    const i1 = seg + 1;   // p1 in cp
    const i0 = i1 - 1;   // p0
    const i2 = i1 + 1;   // p2
    const i3 = i1 + 2;   // p3

    return catmullRom(cp[i0], cp[i1], cp[i2], cp[i3], u);
  };
}

// ---------------------------------------------------------------------------
// 5. Main public API
// ---------------------------------------------------------------------------

/**
 * Parse a DAT file string and return an interpolation function.
 * Auto-detects Selig and Lednicer formats.
 *
 * The contour runs:  upper trailing edge → leading edge → lower trailing edge.
 * t = 0   → upper trailing edge
 * t ≈ 0.5 → leading edge (exact value depends on surface arc lengths)
 * t → 1   → lower trailing edge  (t=1 is excluded; use t < 1)
 *
 * @param {string} datText - Raw DAT file contents (Selig or Lednicer format).
 * @returns {(t: number) => [number, number]}
 */
function parseAirfoil(datText) {
  const { upper, lower } = parseDat(datText);

  // upper[-1] and lower[0] are both the leading-edge point; drop the duplicate.
  const lastUpper = upper[upper.length - 1];
  const firstLower = lower[0];
  const leIsDuplicated =
    Math.abs(lastUpper[0] - firstLower[0]) < 1e-9 &&
    Math.abs(lastUpper[1] - firstLower[1]) < 1e-9;

  const fullContour = leIsDuplicated
    ? [...upper, ...lower.slice(1)]
    : [...upper, ...lower];

  return buildSpline(fullContour);
}

/**
 * Sample the airfoil at n evenly-spaced t values.
 *
 * @param {string} datText - Raw DAT file contents.
 * @param {number} n       - Number of output points.
 * @param {number} scale   - Optional scaling factor for the output coordinates.
 * @param {number} xOffset - Optional horizontal offset to apply to all points.
 * @returns {[number, number][]}  Array of n [x, y] pairs centered at the origin.
 */
function sampleAirfoil(datText, n, scale = 1.0, xOffset = 0.0) {
  scale *= 2.0; // scale from [0,1] to [-1,1]
  const sample = parseAirfoil(datText);
  const points = [];
  for (let k = 0; k < n; k++) {
    let point = sample(k / n);
    points.push([(point[0] - 0.5) * scale + xOffset, point[1] * scale]);
  }
  return points;
}

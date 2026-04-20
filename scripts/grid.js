
// let prepareGrid = true;
// let runPoisson = true;
// will be set in main.js
let prepareGrid = () => {};

// let poissonIterationsPerFrame = 10000;
let poissonIterations = 0;
let maxPoissonIterations = 1e4;
// let gridFinalized = false;

const simulationDomain = [512 - 2, 384 - 3]; // circumferential * radial for o grid, keep the wg dispatch size constant by setting the largest texture size to a mulitple of wg size
const gridVertexCount = [(simulationDomain[0]), (simulationDomain[1] + 1)];
const xFluxTexSize = [(simulationDomain[0] + 1), (simulationDomain[1])];
const yFluxTexSize = [(simulationDomain[0]), (simulationDomain[1] + 1)];
const totalCellCount = [simulationDomain[0], simulationDomain[1] + 3]; // for boundary condition ghost cells - 2 for outer boundary, 1 for object boundary

const gridDisplayProperties = Object.freeze({
  full: [0, (gridVertexCount[0] + 1) * (gridVertexCount[1] - 1) *2],
  mesh: [1, gridVertexCount[0] * (gridVertexCount[1] - 1) * 4],
  vertices: [2, gridVertexCount[0] * gridVertexCount[1]],
});
let [gridDisplayMode, numVertices] = gridDisplayProperties.full;
uni.values.gridDisplayMode.set([gridDisplayMode]);

gui.io.gridResX(totalCellCount[0]);
gui.io.gridResY(totalCellCount[1]);
gui.addDropdown("gridDisplayMode", "Grid display mode", ["full", "mesh", "vertices"], "grid", null, (value) => {
  [gridDisplayMode, numVertices] = gridDisplayProperties[value];
  uni.values.gridDisplayMode.set([gridDisplayMode]);
});

const gridVtxData = new Float32Array(gridVertexCount[0] * gridVertexCount[1] * 2);
const gridBoundaryData = new Int16Array(gridVertexCount[0] * gridVertexCount[1] * 2);
const cellIdx = (x, y) => (y * simulationDomain[0] + x);
const vtxIdx = (x, y) => (y * gridVertexCount[0] + x) * 2;

function generateNACA4Boundary(t, naca4, size=1.0, xOffset=0, pointDistExponent = 7/5) {
  t /= gridVertexCount[0] / 2;
  size *= 2;
  const thickness = naca4 % 100 / 100;
  const m = Math.floor(naca4 / 1000) / 100; // maximum camber
  const p = Math.floor(naca4 / 100) % 10 / 10; // location of maximum camber
  const thicknessFunction = (t) => 5 * thickness * (0.2969 * Math.sqrt(t) - 0.126 * t - 0.3516 * t**2 + 0.2843 * t**3 - 0.1036 * t**4);
  const camberFunction = (t) => t < p ? (m / p**2) * (2*p*t - t**2) : (m / (1-p)**2) * ((1 - 2*p) + 2*p*t - t**2);
  const camberDerivativeFunction = (t) => (t < p ? (m / p**2) : (m / (1-p)**2)) * (2*p - 2*t);
  const symmetricAirfoil = (t) => [thicknessFunction(t) * -Math.sin(Math.atan(camberDerivativeFunction(t))), thicknessFunction(t) / Math.sqrt(1 + camberDerivativeFunction(t)**2)];
  const airfoilUpper = (t) => [(t - 0.5 + symmetricAirfoil(t)[0]) * size + xOffset, (camberFunction(t) + symmetricAirfoil(t)[1]) * size];
  const airfoilLower = (t) => [(t - 0.5 - symmetricAirfoil(t)[0]) * size + xOffset, (camberFunction(t) - symmetricAirfoil(t)[1]) * size];
  // return points in counterclockwise order starting from trailing edge
  return (t <= 1) ? airfoilUpper((1 - t) ** pointDistExponent) : airfoilLower((t - 1) ** pointDistExponent);
}
function generateSearsHaackBoundary(t, V = 0.2, L = 10, xOffset = -0.7) {
  t /= (gridVertexCount[0] / 2);
  const r = (x) => 8 / Math.PI * Math.sqrt(2 * V / (3 * L)) * (x * (1 - x))**(3/4);
  if (t < 1) {
    return [L * (-t) + xOffset, r(t)];
  }
  t -= 1;
  return [L * (t-1) + xOffset, -r(t)];
}

/**
 * Generates a haack nosecone projectile boundary
 * @param {Number} t parameter coordinate, [0,1)
 * @param {Number} F Nosecone fineness ratio
 * @param {Number} R Max radius
 * @param {Number} L Body length (fraction of nosecone length)
 * @param {Number} Rb Boat tail radius fraction
 * @param {Number} Lb Boat tail length fraction
 * @param {Number} C Haack cone parameter
 * @param {Number} xOffset X offset from origin relative to nosecone base, negative -> upstream
 * @returns [x, y] at t, starts at middle trailing edge, CCW around object, ends at middle trailing edge
 */
function generateHaackProjectileBoundary(t, F = 6, R = 0.03, L = 0.5, Rb = 0.5, Lb = 0.2, C = 0, xOffset = -0.7) {
  t /= (gridVertexCount[0] / 2);
  const r = (T) => R * Math.sqrt((T - Math.sin(2 * T) * 0.5 + C / 3 * (Math.sin(T) ** 3)) / Math.PI);
  // approximate as triangle hypot
  const coneLength = R * Math.hypot(1, 2 * F);
  const FR = F * R;
  const bodyLength = FR * L * (1 - Lb);
  const boatTailXLength = L * FR * Lb;
  const boatTailLength = Math.hypot(boatTailXLength, (1 - Rb) * R);
  const boatTailSlope = -(1 - Rb) * R / (boatTailXLength);
  const tailLength = R * Rb;
  // uniformly distribute points along surface
  const totalLengthInv = 1 / (coneLength + bodyLength + boatTailLength + tailLength);
  const tailFrac = tailLength * totalLengthInv;
  const boatTailFrac = boatTailLength * totalLengthInv + tailFrac;
  const bodyFrac = bodyLength * totalLengthInv + boatTailFrac;
  if (t < 1) {
    if (t < tailFrac) {
      t /= tailFrac;
      return [FR * L + xOffset, R * Rb * t];
    } else if (t < boatTailFrac) {
      t = (t - tailFrac) / (boatTailFrac - tailFrac);
      const x = FR * L - boatTailXLength * t;
      return [x + xOffset, R + boatTailSlope * (x - bodyLength)];
    } else if (t < bodyFrac) {
      t = (t - boatTailFrac) / (bodyFrac - boatTailFrac);
      return [bodyLength * (1 - t) + xOffset, R];
    }
    t = (t - bodyFrac) / (1 - bodyFrac);
    const T = Math.acos(2 * t - 1);
    return [2 * F * R * (-t) + xOffset, r(T)];
  }
  t -= 1;
  if (t < 1 - bodyFrac) {
    t = t / (1 - bodyFrac);
    const T = Math.acos(1 - 2 * t);
    return [2 * F * R * (t - 1) + xOffset, -r(T)];
  } else if (t < 1 - boatTailFrac) {
    t = (t - (1 - bodyFrac)) / ((1 - boatTailFrac) - (1 - bodyFrac));
    return [bodyLength * t + xOffset, -R];
  } else if (t < 1 - tailFrac) {
    t = (t - (1 - boatTailFrac)) / ((1 - tailFrac) - (1 - boatTailFrac));
    const x = FR * L - boatTailXLength * (1 - t);
    return [x + xOffset, -R - boatTailSlope * (x - bodyLength)];
  }
  t = (t - (1 - tailFrac)) / tailFrac;
  return [FR * L + xOffset, -R * Rb * (1 - t)];
}
/**
 * Generates a Haack nosecone projectile boundary.
 * @param {number} t    Raw parameter coordinate, remapped internally to [0, 2)
 * @param {number} F    Nosecone fineness ratio
 * @param {number} R    Max radius
 * @param {number} L    Body length as a fraction of nosecone length
 * @param {number} Rb   Boat-tail radius fraction
 * @param {number} Lb   Boat-tail length fraction
 * @param {number} C    Haack series parameter
 * @param {number} xOffset  X offset from origin relative to nosecone base (negative = upstream)
 * @returns {[number, number]} [x, y] at t.
 *   Traversal starts at the middle trailing edge, goes CCW, ends at the middle trailing edge.
 */
function generateHaackProjectileBoundary2(
  t,
  F = 6, R = 0.03, L = 0.5, Rb = 0.5, Lb = 0.2, C = 2,
  xOffset = -0.7,
) {
  // Remap raw vertex index → [0, 2)
  t /= (gridVertexCount[0] / 2);

  // ── Geometry ────────────────────────────────────────────────────────────────
  const FR             = F * R;
  const coneLength     = R * Math.hypot(1, 2 * F);           // triangle hypotenuse approx
  const bodyLength     = FR * L * (1 - Lb);
  const boatTailXLen   = FR * L * Lb;
  const boatTailSlope  = -(1 - Rb) * R / boatTailXLen;
  const boatTailLen    = Math.hypot(boatTailXLen, (1 - Rb) * R);
  const tailLen        = R * Rb;

  // Arc-length fractions (cumulative) so points are uniformly distributed
  const totalLen    = coneLength + bodyLength + boatTailLen + tailLen;
  const tailFrac     = tailLen      / totalLen;
  const boatTailFrac = boatTailLen  / totalLen + tailFrac;
  const bodyFrac     = bodyLength   / totalLen + boatTailFrac;
  // coneFrac = 1 (remainder)

  // ── Haack profile ───────────────────────────────────────────────────────────
  /** Haack radius at angle T ∈ [0, π] */
  const haackR = (T) =>
    R * Math.sqrt((T - 0.5 * Math.sin(2 * T) + (C / 3) * Math.sin(T) ** 3) / Math.PI);

  /** Linearly remap t from [lo, hi] → [0, 1] */
  const remap = (t, lo, hi) => (t - lo) / (hi - lo);

  // ── Section resolvers ───────────────────────────────────────────────────────
  // Each returns [x, y] for a normalised s ∈ [0, 1].
  // sign = +1 for upper half, −1 for lower half.
  const sections = [
    // Tail cap (vertical segment at the base)
    {
      hi: tailFrac,
      resolve: (s, sign) => [FR * L + xOffset, sign * R * Rb * s],
    },
    // Boat tail (angled taper)
    {
      hi: boatTailFrac,
      resolve: (s, sign) => {
        const x = FR * L - boatTailXLen * s;
        return [x + xOffset, sign * (R + boatTailSlope * (x - bodyLength))];
      },
    },
    // Cylindrical body
    {
      hi: bodyFrac,
      resolve: (s, sign) => [bodyLength * (1 - s) + xOffset, sign * R],
    },
    // Haack nosecone (s goes tip → base, so x goes 0 → −FR·L·2)
    {
      hi: 1,
      resolve: (s, sign) => {
        const T = Math.acos(sign < 0 ? 2 * s - 1 : 1 - 2 * s);
        return [2 * F * R * (s - 1) + xOffset, sign * haackR(T)];
      },
    },
  ];

  // ── Traverse: upper half [0,1), then lower half [1,2) ──────────────────────
  const sign = t < 1 ? 1 : -1;
  if (sign < 0) t -= 1;

  // For the lower half the section order reverses, so flip t
  const u = sign > 0 ? t : 1 - t;

  let lo = 0;
  for (const { hi, resolve } of sections) {
    if (u <= hi) {
      const s = remap(u, lo, hi);
      return resolve(s, sign);
    }
    lo = hi;
  }
}


// loop through boundaries
// O grid: n points fixed by domain x size
// C grid: n points less than domain x size, remaining points joined together
function generatePolygonBoundary(t, R = 0.2, N = 4, F = 0.1, a = 0, xOffset = -0.7) {
  // normalize to [0, 1]
  t /= gridVertexCount[0];
  t *= 2 * Math.PI;

  const b = 2 * Math.PI / N;
  const r = R / Math.cos(((t + a + b) % b) - Math.PI / N);
  const x = r * Math.cos(t) + xOffset;
  const y = r * F * Math.sin(t);
  return [x, y];
}
function generateCircularOuterBoundary(t, radius = 1) {
  t /= gridVertexCount[0];
  const x = radius * Math.cos(t * Math.PI * 2);
  const y = radius * Math.sin(t * Math.PI * 2);
  return [x, y];
}
function generateRectangularOuterBoundary(t, lengthRatio) {
  t /= gridVertexCount[0];
  t = (t + 0.125) % 1; // rotate to align with object boundary
  if (t < 0.25) {
    if (t > 1.0) t -= 1.0;
    t *= 4;
    return [lengthRatio, t*2-1];
  } else if (t < 0.5) {
    t = (t - 0.25) * 4;
    return [((1 - t)*2-1) * lengthRatio, 1];
  } else if (t < 0.75) {
    t = (t - 0.5) * 4;
    return [-lengthRatio, (1 - t)*2-1];
  } else {
    t = (t - 0.75) * 4;
    return [(t*2-1) * lengthRatio, -1];
  }
}

const objectCoords = {
  "polygon": new Array(gridVertexCount[0]).fill(0).map((_, t) => generatePolygonBoundary(t, 0.1, 4, .1, 0*Math.PI/4, -0.7)),
  "circle": new Array(gridVertexCount[0]).fill(0).map((_, t) => generatePolygonBoundary(t, 0.1, 400, 1, 0, -0.7)),
  "naca-4": new Array(gridVertexCount[0]).fill(0).map((_, t) => generateNACA4Boundary(t, 4412, 0.2, -0.7)),
  "airfoil-dat": sampleAirfoil(whitcombIntegral, gridVertexCount[0], 0.3, -0.7),
  "sears-haack": new Array(gridVertexCount[0]).fill(0).map((_, t) => generateSearsHaackBoundary(t, 0.0002, 0.4, -0.7)),
  "haack-projectile": new Array(gridVertexCount[0]).fill(0).map((_, t) => generateHaackProjectileBoundary(t)),
};
const boundaryCoords = new Array(gridVertexCount[0]).fill(0).map((_, t) => generateRectangularOuterBoundary(t, 1.5));

function updateGridBoundaries(objCoords = objectCoords["polygon"], boundCoords = boundaryCoords) {
  for (let x = 0; x < gridVertexCount[0]; x++) {
    const i = vtxIdx(x, 0);
    gridVtxData[i] = objCoords[x][0];
    gridVtxData[i + 1] = objCoords[x][1];

    const objI = vtxIdx(x, gridVertexCount[1] - 1);
    gridVtxData[objI] = boundCoords[x][0];
    gridVtxData[objI + 1] = boundCoords[x][1];

    // if (x < simulationDomain[0]) {
    //   const i = cellIdx(x, 0);
    //   const objI = cellIdx(x, simulationDomain[1] - 1);
    //   gridBoundaryData[i] = -1; // outer wall
    //   gridBoundaryData[objI] = -2; // object
    // }
  }

  // add O grid connections to boundary texture
  for (let y = 0; y < gridVertexCount[1]; y++) {
    const leftBoundary = vtxIdx(0, y);
    const rightBoundary = vtxIdx(gridVertexCount[0] - 1, y);
    gridBoundaryData[leftBoundary] = gridVertexCount[0] - 1;
    gridBoundaryData[leftBoundary + 1] = y;
    gridBoundaryData[rightBoundary] = 0;
    gridBoundaryData[rightBoundary + 1] = y;
  }
  // need solution for corners - negative for solid boundary?
}
updateGridBoundaries();


gui.addDropdown("objectType", "Object type", ["polygon", "circle", "naca-4", "airfoil-dat", "sears-haack", "haack-projectile"], "grid", {
  "polygon": [],
  "circle": [],
  "naca-4": ["naca4"],
  "airfoil-dat": ["airfoilDatFile"],
  "sears-haack": [],
  "haack-projectile": [],
});
gui.addFileInput("airfoilDatFile", "Airfoil .dat file", "grid", (file) => {
  const reader = new FileReader();
  reader.onload = (event) => {
    const text = event.target.result;
    const coords = sampleAirfoil(text, gridVertexCount[0], 0.3, -0.7);
    objectCoords["airfoil-dat"] = coords;
  }
  reader.readAsText(file);
});
gui.addNumericInput("naca4", false, "NACA 4-digit", { min: 1, max: 9999, step: 1, val: 4412, float: 0 }, "grid", (value) => {
  objectCoords["naca-4"] = new Array(gridVertexCount[0]).fill(0).map((_, t) => generateNACA4Boundary(t, value, 0.3, -0.7)); // should do this only when loading new grid
});
gui.addButton("updateGrid", "Update grid", true, "grid", () => {
  updateGridBoundaries(objectCoords[gui.io.objectType.value]);
  prepareGrid();
});
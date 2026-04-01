
let prepareGrid = true;
let runPoisson = true;
let poissonIterationsPerFrame = 10000;
let poissonIterations = 0;
let maxPoissonIterations = 10000;
let gridFinalized = false;

const simulationDomain = [512, 256]; // circumferential * radial for o grid
const gridVertexCount = [(simulationDomain[0]), (simulationDomain[1] + 1)];
const xFluxTexSize = [(simulationDomain[0] + 1), (simulationDomain[1])];
const yFluxTexSize = [(simulationDomain[0]), (simulationDomain[1] + 1)];
const totalCellCount = [simulationDomain[0], simulationDomain[1] + 3]; // for boundary condition ghost cells - 2 for outer boundary, 1 for object boundary

const gridDisplayProperties = Object.freeze({
  full: [0, (gridVertexCount[0] + 1) * (gridVertexCount[1] - 1) *2],
  mesh: [1, gridVertexCount[0] * (gridVertexCount[1] - 1) * 4],//[1, (gridVertexCount[0] - 1.5) * gridVertexCount[1] * 4 + 4],
  vertices: [2, gridVertexCount[0] * gridVertexCount[1]],
});
let [gridDisplayMode, numVertices] = gridDisplayProperties.full;
uni.values.gridDisplayMode.set([gridDisplayMode]);

gui.io.gridResX(simulationDomain[0]);
gui.io.gridResY(simulationDomain[1]);
gui.addDropdown("gridDisplayMode", "Grid display mode", ["full", "mesh", "vertices"], "grid", null, (value) => {
  [gridDisplayMode, numVertices] = gridDisplayProperties[value];
  uni.values.gridDisplayMode.set([gridDisplayMode]);
});

const gridVtxData = new Float32Array(gridVertexCount[0] * gridVertexCount[1] * 2);
const gridBoundaryData = new Int16Array(gridVertexCount[0] * gridVertexCount[1] * 2);
const cellIdx = (x, y) => (y * simulationDomain[0] + x);
const vtxIdx = (x, y) => (y * gridVertexCount[0] + x) * 2;

function generateNACA4Boundary(t, naca4, size=1.0) {
  t /= gridVertexCount[0] / 2;
  size *= 2;
  const thickness = naca4 % 100 / 100;
  const m = Math.floor(naca4 / 1000) / 100; // maximum camber
  const p = Math.floor(naca4 / 100) % 10 / 10; // location of maximum camber
  const thicknessFunction = (t) => 5 * thickness * (0.2969 * Math.sqrt(t) - 0.126 * t - 0.3516 * t**2 + 0.2843 * t**3 - 0.1015 * t**4);
  const camberFunction = (t) => t <= p ? (m / p**2) * (2*p*t - t**2) : (m / (1-p)**2) * ((1 - 2*p) + 2*p*t - t**2);
  const camberDerivativeFunction = (t) => (t <= p ? (m / p**2) : (m / (1-p)**2)) * (2*p - 2*t);
  const symmetricAirfoil = (t) => [thicknessFunction(t) * -Math.sin(Math.atan(camberDerivativeFunction(t))), thicknessFunction(t) / Math.sqrt(1 + camberDerivativeFunction(t)**2)];
  const airfoilUpper = (t) => [(t - 0.5 + symmetricAirfoil(t)[0]) * size, (camberFunction(t) + symmetricAirfoil(t)[1]) * size];
  const airfoilLower = (t) => [(t - 0.5 - symmetricAirfoil(t)[0]) * size, (camberFunction(t) - symmetricAirfoil(t)[1]) * size];
  // return points in counterclockwise order starting from trailing edge
  return (t <= 1) ? airfoilUpper(1 - t) : airfoilLower(t - 1);
}

// loop through boundaries
// O grid: n points fixed by domain x size
// C grid: n points less than domain x size, remaining points joined together
function generateObjectBoundary(t) {
  // normalize to [0, 1]
  t /= gridVertexCount[0];
  t *= 2 * Math.PI;
  const x = 0.1 * Math.cos(t)-0.3;
  const y = 0.1 * Math.sin(t);
  // const x = (0.1 + 0.1 * Math.abs(Math.cos(t))) * Math.cos(t);
  // const y = (0.1 + 0.1 * Math.abs(Math.cos(t))) * Math.sin(t);
  
  // const N = 6;
  // const F = 0.1;
  // const r = 0.2 / Math.cos((t % (2 * Math.PI / N)) - Math.PI / N);
  // const x = r * Math.cos(t);
  // const y = r * F * Math.sin(t);
  return [x, y];
}
function generateCircularOuterBoundary(t) {
  t /= gridVertexCount[0];
  const x = Math.cos(t * Math.PI * 2);
  const y = Math.sin(t * Math.PI * 2);
  return [x, y];
}
function generateSquareOuterBoundary(t) {
  t /= gridVertexCount[0];
  t = (t + 0.125) % 1; // rotate to align with object boundary
  if (t < 0.25) {
    if (t > 1.0) t -= 1.0;
    t *= 4;
    return [1, t*2-1];
  } else if (t < 0.5) {
    t = (t - 0.25) * 4;
    return [(1 - t)*2-1, 1];
  } else if (t < 0.75) {
    t = (t - 0.5) * 4;
    return [-1, (1 - t)*2-1];
  } else {
    t = (t - 0.75) * 4;
    return [t*2-1, -1];
  }
}
const objectCoords = new Array(gridVertexCount[0]).fill(0).map((_, t) => generateObjectBoundary(t));
// const objectCoords = new Array(gridVertexCount[0]).fill(0).map((_, t) => generateNACA4Boundary(t, 4415, 0.2));
const boundaryCoords = new Array(gridVertexCount[0]).fill(0).map((_, t) => generateCircularOuterBoundary(t));
for (let x = 0; x < gridVertexCount[0]; x++) {
  const i = vtxIdx(x, 0);
  gridVtxData[i] = objectCoords[x][0];
  gridVtxData[i + 1] = objectCoords[x][1];

  const objI = vtxIdx(x, gridVertexCount[1] - 1);
  gridVtxData[objI] = boundaryCoords[x][0];
  gridVtxData[objI + 1] = boundaryCoords[x][1];

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
// need solution for corners - 
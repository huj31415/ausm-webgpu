const uni = new Uniforms();
uni.addUniform("simDomain", "vec2f");
uni.addUniform("resRatio", "vec2f");

uni.addUniform("objectCenter", "vec2f");
uni.addUniform("pan", "vec2f");

uni.addUniform("inflowV", "vec2f");
uni.addUniform("zoom", "f32");
uni.addUniform("dt", "f32");

uni.addUniform("inPressure", "f32");
uni.addUniform("inRho", "f32");
uni.addUniform("K_p", "f32");
uni.addUniform("K_u", "f32");

uni.addUniform("inState", "vec4f");

uni.addUniform("gamma", "f32");
uni.addUniform("gridDisplayMode", "f32");
uni.addUniform("simDisplayMode", "f32");
uni.addUniform("cflFactor", "f32");


uni.finalize();



const Int16Max = 32767;

const storage = {
  // conformal structured grid 
  gridPoints0: null,      // rg32float (M+1)x(N+1)  xy points in clip space, derive normals and lengths from this
  gridPoints1: null,      //                        same as above, used for ping-ponging during iterative Poisson solve
  gridBoundaries: null,   // r32sint   (M+1)x(N+1)  boundary conditions and indices of connections between nonadjacent cells for each vertex
                          //                        positive: index of connected cell, otherwise: type of boundary (-1: object, -2: domain outer boundary (derive inflow and outflow from normal direction))//-2: inlet, -4: outlet)
                          // replace with linear indices for 4 sides if no boundaries inside domain?

  // rgbafloat32 (M)x(N+3) rho, rho*u, rho*v, rho*E
  state0: null,
  state1: null,
  state2: null,

  // fluxes at cell faces
  fluxX: null, // (M+1)x(N)
  fluxY: null, // (M)x(N+1)

  // residuals
  residual: null, // (M)x(N)
}

let deltaTime = lastFrameTime = fps = jsTime = renderTime = postprocessingTime = cflTime = 0;
let poissonTime = 0;
let dt = 1e-4;
let oldDt;
let stepsPerFrame = 70;

let inflowVel = 3.8;
let actualInflowVel = 0;
let velRampUpStrength = 128;
let AoA = 0;
let xyAoA = [1, 0];
let gamma = 1.4;
let inPressure = 1.0 / gamma;
let inRho = 1.0;
let K_p = 0;//.25;
let K_u = 0.75;

const simDisplayModes = Object.freeze({
  schlieren: 0,
  vorticity: 1,
  density: 2,
  temperature: 3,
  pressure: 4,
  mach: 5,
  velocity: 6,
  entropy: 7,
  pressureLoss: 8,
});

let displayMode = simDisplayModes.schlieren;
uni.values.simDisplayMode.set([displayMode]);


const canvas = document.getElementById("canvas");
let pixelRatio = window.devicePixelRatio || 1;

const gui = new GUI("AUSM+-up compressible fluid sim", canvas);

gui.addGroup("perf", "Performance");
gui.addStringOutput("res", "Resolution", "", "perf");
gui.addHalfWidthGroups("perfL", "perfR", "perf");

gui.addNumericOutput("fps", "FPS", "", 1, "perfL");
gui.addNumericOutput("frameTime", "Frame", "ms", 2, "perfL");
gui.addNumericOutput("jsTime", "JS", "ms", 2, "perfL");
// gui.addNumericOutput("computeTime", "Compute", "ms", 2, "perfL");
gui.addNumericOutput("postTime", "Postprocess", "ms", 2, "perfL");
gui.addNumericOutput("cflTime", "CFL", "ms", 2, "perfL");
gui.addNumericOutput("renderTime", "Render", "ms", 2, "perfL");

gui.addGroup("grid", "Grid");
gui.addNDimensionalOutput(["gridResX", "gridResY"], "Grid res", "", ", ", 0, "grid");
gui.addNumericOutput("poissonIterations", "Poisson iterations", "", 0, "grid");

gui.addGroup("sim", "Simulation");
gui.addDropdown("simDisplayMode", "Visualization mode", ["schlieren", "density", "temperature", "pressure", "mach", "velocity", "vorticity", "entropy", "pressureLoss"], "sim", null, (value) => {
  displayMode = simDisplayModes[value];
  uni.values.simDisplayMode.set([displayMode]);
});
gui.addNumericInput("cflFactor", true, "CFL factor", { min: 0.1, max: 3, step: 0.1, val: 2.0, float: 1 }, "sim", (value) => {
  uni.values.cflFactor.set([value]);
});
gui.addNumericOutput("mach", "Mach number", "", 2, "sim");
gui.addNumericInput("inflowVel", true, "Inflow velocity", { min: 0, max: 40, step: 0.01, val: 3.8, float: 2 }, "sim", (value) => {
  inflowVel = value;
});
gui.addNumericInput("rampFactor", true, "V smoothing", { min: 0, max: 10, step: 0.1, val: 7, float: 1 }, "sim", (value) => {
  velRampUpStrength = Math.pow(2, value);
}, "How fast the velocity changes when ramping up or down");
gui.addNumericInput("AoA", true, "Angle of attack", { min: -90, max: 90, step: 1, val: 0, float: 1 }, "sim", (value) => {
  AoA = value;
  let AoARad = AoA * Math.PI / 180;
  xyAoA[0] = Math.cos(AoARad);
  xyAoA[1] = Math.sin(AoARad);
});
gui.addNumericInput("inPressure", true, "Inflow pressure", { min: 0.01, max: 2, step: 0.01, val: inPressure, float: 3 }, "sim", (value) => {
  inPressure = value;
  uni.values.inPressure.set([inPressure]);
});
gui.addNumericInput("inRho", true, "Inflow density", { min: 0.01, max: 2, step: 0.01, val: inRho, float: 3 }, "sim", (value) => {
  inRho = value;
  uni.values.inRho.set([inRho]);
});
gui.addNumericInput("K_p", true, "K_p", { min: 0, max: 1, step: 0.01, val: K_p, float: 2 }, "sim", (value) => {
  uni.values.K_p.set([value]);
});
gui.addNumericInput("K_u", true, "K_u", { min: 0, max: 1, step: 0.01, val: K_u, float: 2 }, "sim", (value) => {
  uni.values.K_u.set([value]);
});
gui.addNumericInput("dtPerFrame", true, "dt/frame", { min: 10, max: 500, step: 10, val: stepsPerFrame, float: 0 }, "sim", (value) => {
  stepsPerFrame = value;
});
gui.addButton("toggleSim", "Play / Pause", true, "sim", () => {
  if (oldDt) {
    dt = oldDt;
    oldDt = null;
  } else {
    oldDt = dt;
    dt = 0;
  }
  uni.values.dt.set([dt]);
});

// handle resizing
window.onresize = window.onload = () => {
  pixelRatio = window.devicePixelRatio || 1;
  // canvas.style.zoom = 1 / pixelRatio;
  canvas.width = window.innerWidth * pixelRatio;
  canvas.height = window.innerHeight * pixelRatio;
  // uni.values.resolution.set([canvas.width, canvas.height]);
  const invMinRes = 1 / Math.min(canvas.width, canvas.height);
  uni.values.resRatio.set([canvas.width * invMinRes, canvas.height * invMinRes]);
  gui.io.res([window.innerWidth, window.innerHeight]);
};

// handle panning and zooming
let isPanning = false;
let lastMousePos = [0, 0];
let currentZoom = 1.0;
canvas.onmousedown = (e) => {
  isPanning = true;
}
canvas.onmouseup = canvas.onmouseleave = () => {
  isPanning = false;
}
canvas.onmousemove = (e) => {
  if (isPanning) {
    let size = Math.min(canvas.width, canvas.height);
    uni.values.pan[0] += e.movementX / size * 2 * pixelRatio;
    uni.values.pan[1] -= e.movementY / size * 2 * pixelRatio;
  }
}
canvas.onwheel = (e) => {
  const zoomAmount = 1.1;
  let size = Math.min(canvas.width, canvas.height);
  const mousePos = [
    (e.clientX / window.innerWidth * 2 - 1) * pixelRatio,
    (1 - e.clientY / window.innerHeight * 2) * pixelRatio
  ]
  const worldMousePos = [
    (mousePos[0] - uni.values.pan[0]) / currentZoom,
    (mousePos[1] - uni.values.pan[1]) / currentZoom
  ];
  if (e.deltaY < 0) {
    currentZoom *= zoomAmount;
    uni.values.zoom[0] = currentZoom;
  } else {
    currentZoom /= zoomAmount;
    uni.values.zoom[0] = currentZoom;
  }
  const panOffset = [
    mousePos[0] - worldMousePos[0] * currentZoom,
    mousePos[1] - worldMousePos[1] * currentZoom
  ];
  uni.values.pan[0] = panOffset[0];
  uni.values.pan[1] = panOffset[1];
}
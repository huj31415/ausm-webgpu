const uni = new Uniforms();
uni.addUniform("simDomain", "vec2f");
uni.addUniform("resolution", "vec2f");

uni.addUniform("resRatio", "vec2f");
uni.addUniform("objectCenter", "vec2f");

uni.addUniform("pan", "vec2f");
uni.addUniform("zoom", "f32");
uni.addUniform("dt", "f32");

uni.addUniform("inVel", "vec2f");
uni.addUniform("inPressure", "f32");
uni.addUniform("inRho", "f32");

uni.addUniform("gamma", "f32");
// uni.addUniform("inState", "vec4f");

uni.addUniform("gridDisplayMode", "f32");

uni.finalize();



const Int16Max = 32767;

const storage = {
  // conformal structured grid 
  gridPoints0: null,      // rg32float (M+1)x(N+1)  xy points in clip space, derive normals and lengths from this
  gridPoints1: null,      //                        same as above, used for ping-ponging during iterative Poisson solve
  gridArea: null,         // r32float  (M)x(N)      area of each cell, Jacobian of transformation from world to grid space
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

let deltaTime = lastFrameTime = fps = jsTime = renderTime = 0;
let poissonTime = 0;
let dt = 1e-4;
let oldDt;
let stepsPerFrame = 100;


const canvas = document.getElementById("canvas");

const gui = new GUI("fluid sim", canvas);

gui.addGroup("perf", "Performance");
gui.addStringOutput("res", "Resolution", "", "perf");
gui.addHalfWidthGroups("perfL", "perfR", "perf");

gui.addNumericOutput("fps", "FPS", "", 1, "perfL");
gui.addNumericOutput("frameTime", "Frame", "ms", 2, "perfL");
gui.addNumericOutput("jsTime", "JS", "ms", 2, "perfL");
gui.addNumericOutput("computeTime", "Compute", "ms", 2, "perfL");
gui.addNumericOutput("renderTime", "Render", "ms", 2, "perfL");

gui.addGroup("grid", "Grid");
gui.addNDimensionalOutput(["gridResX", "gridResY"], "Grid res", "", ", ", 0, "grid");
gui.addNumericOutput("poissonIterations", "Poisson iterations", "", 0, "grid");
gui.addButton("toggleSim", "Play / Pause", true, "grid", () => {
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
  canvas.width = window.innerWidth;
  canvas.height = window.innerHeight;
  uni.values.resolution.set([canvas.width, canvas.height]);
  const invMinRes = 1 / Math.min(...uni.values.resolution);
  uni.values.resRatio.set([canvas.width * invMinRes, canvas.height * invMinRes]);
  gui.io.res([window.innerWidth, window.innerHeight]);
};

// handle panning and zooming
let isPanning = false;
let lastMousePos = [0, 0];
let currentZoom = 1.0;
canvas.onmousedown = (e) => {
  isPanning = true;
  lastMousePos = [e.clientX, e.clientY];
}
canvas.onmouseup = canvas.onmouseleave = () => {
  isPanning = false;
}
canvas.onmousemove = (e) => {
  if (isPanning) {
    const deltaX = e.clientX - lastMousePos[0];
    const deltaY = e.clientY - lastMousePos[1];
    uni.values.pan[0] += deltaX / canvas.width * 2;
    uni.values.pan[1] -= deltaY / canvas.height * 2;
    lastMousePos = [e.clientX, e.clientY];
  }
}
canvas.onwheel = (e) => {
  const zoomAmount = 1.1;
  const mousePos = [
    e.clientX / canvas.width * 2 - 1,
    1 - e.clientY / canvas.height * 2
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
// prepare initial grid guess using interpolation
// O grid - x coordinate of compute space goes around object, y coordinate connects object boundary to outer boundary
// run for each vertex
const gridInterpolationShaderCode = /* wgsl */`
${uni.uniformStruct}

@group(0) @binding(0) var<uniform> uni: Uniforms;
@group(0) @binding(1) var gridPoints0: texture_2d<f32>; // rg32float
@group(0) @binding(2) var gridPoints1: texture_storage_2d<rg32float, write>;
@group(0) @binding(3) var gridBoundaries: texture_2d<i32>; // rg16sint

override WG_X: u32;
override WG_Y: u32;

@compute @workgroup_size(WG_X, WG_Y)
fn main(
  @builtin(global_invocation_id) gid: vec3u
) {
  let boundary = textureLoad(gridBoundaries, gid.xy, 0);
  if (boundary.x < 0) { return; } // only compute for interior points, skip boundaries

  // get boundaries at this x point, interpolate between them based on y coordinate
  let objectBoundary = textureLoad(gridPoints0, vec2u(gid.x, 0), 0).xy;
  let outerBoundary = textureLoad(gridPoints0, vec2u(gid.x, u32(uni.simDomain.y)), 0).xy;
  let t = f32(gid.y) / (uni.simDomain.y);
  let interpolatedPoint = mix(objectBoundary, outerBoundary, t);
  textureStore(gridPoints1, gid.xy, vec4f(interpolatedPoint, 0.0, 0.0));
}
`;

// solve elliptic Poisson equation for grid orthogonality
// run for each vertex
const gridEllipticPoissonShaderCode = /* wgsl */`
${uni.uniformStruct}

@group(0) @binding(0) var<uniform> uni: Uniforms;
@group(0) @binding(1) var gridPoints0: texture_2d<f32>; // rg32float
@group(0) @binding(2) var gridPoints1: texture_storage_2d<rg32float, write>;
@group(0) @binding(3) var gridBoundaries: texture_2d<i32>; // rg16sint

override WG_X: u32;
override WG_Y: u32;

@compute @workgroup_size(WG_X, WG_Y)
fn main(
  @builtin(global_invocation_id) gid: vec3u
) {
  let boundary = textureLoad(gridBoundaries, gid.xy, 0).xy;
  if (gid.y == 0 || gid.y == u32(uni.simDomain.y)) { return; }
  // if (boundary.x < 0) { return; } // only compute for interior points, skip boundaries

  // wrap indices, only for o-grid
  // var leftIdx = (gid.x + u32(uni.simDomain.x) - 1) % u32(uni.simDomain.x);
  // var rightIdx = (gid.x + 1) % u32(uni.simDomain.x);
  let upIdx = clamp(gid.y + 1, 0u, u32(uni.simDomain.y));
  let downIdx = clamp(gid.y - 1, 0u, u32(uni.simDomain.y));

  var leftIdx = gid.x - 1;
  var rightIdx = gid.x + 1;
  if (any(boundary > vec2i(-1, 0))) {
    if (gid.x == 0) {
      leftIdx = u32(boundary.x);
    } else if (gid.x == u32(uni.simDomain.x - 1)) {
      rightIdx = u32(boundary.x);
    }
  }

  // move to shared memory
  let center = textureLoad(gridPoints0, gid.xy, 0).xy;
  let left = textureLoad(gridPoints0, vec2u(leftIdx, gid.y), 0).xy;
  let right = textureLoad(gridPoints0, vec2u(rightIdx, gid.y), 0).xy;
  let down = textureLoad(gridPoints0, vec2u(gid.x, downIdx), 0).xy;
  let up = textureLoad(gridPoints0, vec2u(gid.x, upIdx), 0).xy;
  let downleft = textureLoad(gridPoints0, vec2u(leftIdx, downIdx), 0).xy;
  let downright = textureLoad(gridPoints0, vec2u(rightIdx, downIdx), 0).xy;
  let upleft = textureLoad(gridPoints0, vec2u(leftIdx, upIdx), 0).xy;
  let upright = textureLoad(gridPoints0, vec2u(rightIdx, upIdx), 0).xy;

  let xy_xi = 0.5 * (right - left);
  let xy_eta = 0.5 * (up - down);
  let xy_xieta = 0.25 * (upright + downleft - upleft - downright);

  let alpha = dot(xy_eta, xy_eta);
  let gamma = dot(xy_xi, xy_xi);
  let beta = dot(xy_xi, xy_eta);

  let ds_xi  = length(right - left);
  let ds_eta = length(up - down);

  // adjust Q to push cells towards more square shape
  // let aspectRatio = ds_xi / (ds_eta + 1e-6);
  // let Q = 0.1 * (1 - 1/aspectRatio);
  
  // adjust Q based on length difference and position in domain
  let lengthDiff = (ds_xi - ds_eta) / (max(ds_xi, ds_eta) + 1e-6);
  // let Q = lengthDiff * smoothstep(1, -2, f32(gid.y) / uni.simDomain.y);// * exp(-f32(gid.y));
  let Q = 0.0;
  // adjust P based on skewness
  // let P = -.03 * beta / sqrt(alpha * gamma);
  let P = 0.0;

  let newPos = (alpha * (left + right + P * xy_xi) + gamma * (down + up + Q * xy_eta) - 2 * beta * xy_xieta) / (2.0 * (alpha + gamma));
  let jac = xy_xi.x * xy_eta.y - xy_xi.y * xy_eta.x;

  // adjust relaxation
  let omega = 1.0;
  // only update if not inverted
  let newPoint = select(center, mix(center, newPos, omega), jac < 0.0);
  textureStore(gridPoints1, gid.xy, vec4f(newPoint, 0.0, 0.0));
}
`;

// finalize grid by cell areas from grid points using jacobian and initialize state variables
// run for each cell center
const stateInitShaderCode = /* wgsl */`
${uni.uniformStruct}

@group(0) @binding(0) var<uniform> uni: Uniforms;
@group(0) @binding(1) var state: texture_storage_2d<rgba32float, write>;

override WG_X: u32;
override WG_Y: u32;

@compute @workgroup_size(WG_X, WG_Y)
fn main(
  @builtin(global_invocation_id) gid: vec3u
) {
  // initialize interior state to air at rest, leave rest to boundary shader
  let rhoE = uni.inPressure / (uni.gamma - 1.0) + 0.5 * uni.inRho * 0.0;
  textureStore(state, gid.xy + vec2u(0, 1), vec4f(uni.inRho, 0.0, 0.0, rhoE));
}
`;

// handle boundary "ghost cells", run for all cells (ghost + domain)
// object boundary - 1 ghost cell, reflect normal velocity
// outer boundary - 2 ghost cells for 2nd order muscl
const boundaryShaderCode = /* wgsl */`
${uni.uniformStruct}
@group(0) @binding(0) var<uniform> uni: Uniforms;
@group(0) @binding(1) var gridPoints: texture_2d<f32>; // rg32float
@group(0) @binding(2) var gridBoundaries: texture_2d<i32>; // rg16sint
@group(0) @binding(3) var stateIn: texture_2d<f32>;   // rgba32float
@group(0) @binding(4) var stateOut: texture_storage_2d<rgba32float, write>;

override WG_X: u32;
override WG_Y: u32;

@compute @workgroup_size(WG_X, WG_Y)
fn main(
  @builtin(global_invocation_id) gid: vec3u
) {

  let simDomain = vec2u(uni.simDomain);
  let rightIdx = (gid.x + 1) % simDomain.x;

  let boundary = textureLoad(gridBoundaries, gid.xy, 0).x;
  
  let vtxYCoord = clamp(gid.y, 0, simDomain.y);
  let vtx1 = textureLoad(gridPoints, vec2u(gid.x, vtxYCoord), 0).xy;
  let vtx2 = textureLoad(gridPoints, vec2u(rightIdx, vtxYCoord), 0).xy;
  
  // outward pointing normal
  let tangent = normalize(vtx1 - vtx2);
  let normal = vec2f(-tangent.y, tangent.x);

  var ghostState = textureLoad(stateIn, gid.xy, 0); // default to copy of interior state, may be modified by boundary conditions below

  if (gid.y == 0) {
    // handle object boundary - reflect velocity for slip condition
    let interiorState = textureLoad(stateIn, vec2u(gid.x, 1), 0);
    let rho_int = interiorState.x;
    let u_int = interiorState.yz / rho_int;
    let ghost_U = reflect(u_int, normal); // reflect velocity across normal
    let rhoE = (interiorState.w - 0.5 * dot(u_int, u_int) * rho_int) + 0.5 * rho_int * dot(ghost_U, ghost_U);
    ghostState = vec4f(rho_int, ghost_U * rho_int, rhoE); // reflect velocity, keep other states, apply new rhoE
  } else if (gid.y > simDomain.y) {
    // handle outer boundary
    let interiorState = textureLoad(stateIn, vec2u(gid.x, simDomain.y), 0);
    let rho_int = interiorState.x;
    let u_int = interiorState.yz / rho_int;
    
    let boundaryInOrOut = dot(normal, uni.inflowV); // negative for inflow, positive for outflow
    if(boundaryInOrOut < 0.0) {
      // inflow
      // rho * E = p / (uni.gamma - 1) + 0.5 * rho * (u^2 + v^2)
      let rhoE = uni.inPressure / ((uni.gamma - 1.0) * uni.inRho) + 0.5 * dot(uni.inflowV, uni.inflowV);
      ghostState = uni.inRho * vec4f(1, uni.inflowV, rhoE); // rho, rho*u, rho*v, rho*E
    } else {
      // outflow
      let P_int = (interiorState.w - 0.5 * dot(u_int, u_int) * rho_int) * (uni.gamma - 1.0);
      let a = sqrt(uni.gamma * P_int / rho_int);
      let M_int = length(u_int) / a;
      let ambientRhoE = uni.inPressure / (uni.gamma - 1.0) + 0.5 * rho_int * dot(u_int, u_int); //dot(uni.inflowV, uni.inflowV)
      ghostState = select(vec4f(interiorState.xyz, ambientRhoE), interiorState, M_int >= 1.0); // copy for M > 1.0, fix pressure to ambient for M < 1.0
      if (gid.y == simDomain.y + 2) {
        ghostState = 2.0 * ghostState - interiorState; // for 2nd order extrapolation in muscl
      }
    }
  }
  textureStore(stateOut, gid.xy, ghostState);
}
`;

// calculate fluxes at cell faces based on sim state and grid geometry
// run for each face, Mx(N+1) for vertical faces, (M+1)xN for horizontal faces
// vertical flux calculated for face between (x, y) and (x, y+1), horizontal flux calculated for face between (x, y) and (x+1, y)
// vertical - face at a given index is below cell at that index (using shifted index for ghost cells)
// "A sequel to AUSM, Part II: AUSM+-up for all speeds", Liou, sec. 3.3
const commonFluxShaderCode = (vertical) => /* wgsl */`
${uni.uniformStruct}

@group(0) @binding(0) var<uniform> uni: Uniforms;
@group(0) @binding(1) var gridPoints: texture_2d<f32>; // rg32float
@group(0) @binding(2) var gridBoundaries: texture_2d<i32>; // rg16sint
@group(0) @binding(3) var state: texture_2d<f32>; // rgba32float
@group(0) @binding(4) var flux: texture_storage_2d<rgba32float, write>;

override WG_X: u32;
override WG_Y: u32;

// shift for ghost cells and account for o-grid wrapping
fn loadState(idx: vec2i) -> vec4f {
  let xDim = i32(uni.simDomain.x);
  // shift up by 1 to account for ghost cells and apply x wrapping, clamp object ghosts for zero gradient
  let index = vec2i((idx.x + xDim) % xDim, max(idx.y + 1, 0));
  return textureLoad(state, vec2u(index), 0);
}

fn toPrimitive(Q: vec4f) -> vec4f {
  let rho = Q.x;
  let vel = Q.yz / rho;
  let p   = max(1e-7, (Q.w - 0.5 * rho * dot(vel, vel)) * (uni.gamma - 1.0));
  return vec4f(rho, vel, p);
}

fn toConservative(W: vec4f) -> vec4f {
  let rho = W.x;
  let vel = W.yz;
  let rhoE = W.w / (uni.gamma - 1.0) + 0.5 * rho * dot(vel, vel);
  return vec4f(rho, rho * vel, rhoE);
}

fn M_1 (M: f32, plusOrMinus: i32) -> f32 {
  return 0.5 * (M + f32(plusOrMinus) * abs(M));
}

fn M_2 (M: f32, plusOrMinus: i32) -> f32 {
  return f32(plusOrMinus) * 0.25 * (M + f32(plusOrMinus)) * (M + f32(plusOrMinus));
}

fn M_4(M: f32, beta: f32, plusOrMinus: i32) -> f32 {
  return select(
    M_1(M, plusOrMinus),
    M_2(M, plusOrMinus) * (1 - f32(plusOrMinus) * 16 * beta * M_2(M, -plusOrMinus)),
    abs(M) < 1.0
  );
}

fn P_5(M: f32, alpha: f32, plusOrMinus: i32) -> f32 {
  return select(
    M_1(M, plusOrMinus) / M,
    M_2(M, plusOrMinus) * ((f32(plusOrMinus) * 2 - M) - f32(plusOrMinus) * 16 * alpha * M * M_2(M, -plusOrMinus)),
    abs(M) < 1.0
  );
}

@compute @workgroup_size(WG_X, WG_Y)
fn main(
  @builtin(global_invocation_id) gid: vec3u
) {
  let cellIdx = vec2i(gid.xy);
  let boundary = textureLoad(gridBoundaries, gid.xy, 0).xy;

  let sigma = 1.0;

  let M_inf2 = dot(uni.inflowV, uni.inflowV) / (uni.gamma * uni.inPressure / uni.inRho);

  // load current face
  let vtx1 = textureLoad(gridPoints, gid.xy, 0).xy;
  let vtx2 = textureLoad(gridPoints, vec2u(${vertical ? "(gid.x + 1) % u32(uni.simDomain.x), gid.y" : "gid.x, gid.y + 1"}), 0).xy;

  // +Y/+X pointing normal
  let tangent = normalize(${vertical ? "vtx1 - vtx2" : "vtx2 - vtx1"});
  let normal = vec2f(-tangent.y, tangent.x);

  // y = 0: cell above of object face
  // y = (simDomain.y - 1): cell below outer boundary face
  // state above the face relative to object, or outer boundary ghost(s)
  let Q_R = loadState(cellIdx);
  let Q_R2 = loadState(cellIdx + vec2i(${vertical ? "0, 1" : "1, 0"}));
  // state below the face, or object boundary ghost
  let Q_L = loadState(cellIdx - vec2i(${vertical ? "0, 1" : "1, 0"}));
  let Q_L2 = loadState(cellIdx - vec2i(${vertical ? "0, 2" : "2, 0"})); // will be clamped to ghost state for object boundary

  // MUSCL
  // let faceDiff = (Q_R - Q_L);
  // let r_L = ((Q_L - Q_L2) * faceDiff + 1e-6) / (faceDiff * faceDiff + 1e-6);
  // let r_R = ((Q_R2 - Q_R) * faceDiff + 1e-6) / (faceDiff * faceDiff + 1e-6);
  // // van leer limiter
  // let vanLeer_L = (r_L + abs(r_L)) / (1.0 + abs(r_L));
  // let vanLeer_R = (r_R + abs(r_R)) / (1.0 + abs(r_R));
  // let Q_LMUSCL = Q_L;// + 0.5 * vanLeer_L * (Q_R - Q_L); //(Q_L - Q_L2);
  // let Q_RMUSCL = Q_R;// - 0.5 * vanLeer_R * (Q_R2 - Q_R); //(Q_R2 - Q_R);

  let Q_Lprimitive = toPrimitive(Q_L); // rho, u, v, p
  let Q_L2primitive = toPrimitive(Q_L2);
  let Q_Rprimitive = toPrimitive(Q_R);
  let Q_R2primitive = toPrimitive(Q_R2);

  // MUSCL
  let faceDiff = (Q_Rprimitive - Q_Lprimitive);
  let r_L = ((Q_Lprimitive - Q_L2primitive) * faceDiff + 1e-6) / (faceDiff * faceDiff + 1e-6);
  let r_R = ((Q_Rprimitive - Q_R2primitive) * faceDiff + 1e-6) / (faceDiff * faceDiff + 1e-6);
  // van leer limiter
  let vanLeer_L = (r_L + abs(r_L)) / (1.0 + abs(r_L));
  let vanLeer_R = (r_R + abs(r_R)) / (1.0 + abs(r_R));
  let Q_LMUSCLprimitive = Q_Lprimitive;// + 0.5 * vanLeer_L * (Q_Lprimitive - Q_L2primitive);
  let Q_RMUSCLprimitive = Q_Rprimitive;// - 0.5 * vanLeer_R * (Q_R2primitive - Q_Rprimitive);

  let Q_LMUSCL = toConservative(Q_LMUSCLprimitive);
  let Q_RMUSCL = toConservative(Q_RMUSCLprimitive);

  // interface states, with u = velocity normal to face
  // left state
  let rho_L = Q_LMUSCL.x;
  let vel_L = Q_LMUSCL.yz / rho_L;
  let u_L = dot(vel_L, normal);

  let p_L = (Q_LMUSCL.w - 0.5 * dot(vel_L, vel_L) * rho_L) * (uni.gamma - 1.0);
  let a_L = sqrt(uni.gamma * p_L / rho_L);
  let h_L = (Q_LMUSCL.w + p_L) / rho_L;

  // right state
  let rho_R = Q_RMUSCL.x;
  let vel_R = Q_RMUSCL.yz / rho_R;
  let u_R = dot(vel_R, normal);

  let p_R = (Q_RMUSCL.w - 0.5 * dot(vel_R, vel_R) * rho_R) * (uni.gamma - 1.0);
  let a_R = sqrt(uni.gamma * p_R / rho_R);
  let h_R = (Q_RMUSCL.w + p_R) / rho_R;

  // more accurate
  let h_t_interface = 0.5 * (h_L + h_R); // total enthalpy
  let aStar = sqrt(max(1e-6, 2 * (uni.gamma - 1.0) / (uni.gamma + 1) * h_t_interface)); // eq 29
  let aHat_L = aStar * aStar / max(u_L, aStar); // eq 30
  let aHat_R = aStar * aStar / max(-u_R, aStar);
  let a_interface = min(aHat_L, aHat_R); // eq 28

  // let a_interface = (a_L + a_R) / 2.0;
  let M_L = u_L / a_interface; // eq 69
  let M_R = u_R / a_interface;
  let a_interface2 = a_interface * a_interface;
  
  let Mbar2 = (u_L * u_L + u_R * u_R) / (a_interface2 * 2); // eq 70
  let M_o2 = min(1, max(Mbar2, M_inf2)); // eq 71
  let f_a = M_o2 * (2 - M_o2); // eq 72
  let rho_interface = 0.5 * (rho_L + rho_R);

  let alpha = 3.0 / 16.0 * (-4 + 5 * f_a * f_a); // eq 76
  let beta = 0.125; // 1/8, eq 76

  let M_interface = M_4(M_L, beta, 1) + M_4(M_R, beta, -1) - uni.K_p / f_a * max(1 - sigma * Mbar2, 0) * (p_R - p_L) / (rho_interface * a_interface2); // eq 73

  let mdot = a_interface * M_interface * select(rho_R, rho_L, M_interface > 0); // eq 74

  let P_5plus = P_5(M_L, alpha, 1);
  let P_5minus = P_5(M_R, alpha, -1);
  let P_interface = P_5plus * p_L + P_5minus * p_R - uni.K_u * P_5plus * P_5minus * (rho_L + rho_R) * f_a * a_interface * (u_R - u_L); // eq 75

  let phi_R = vec4f(1, vel_R, h_R); // eq 3
  let phi_L = vec4f(1, vel_L, h_L);
  let convectiveFlux = select(phi_R, phi_L, mdot > 0.0); // eq 6
  let pressureFlux = vec4f(0.0, normal * P_interface, 0.0);

  let totalFlux = convectiveFlux * mdot + pressureFlux;
  textureStore(flux, gid.xy, totalFlux);
}
`;

const verticalFluxShaderCode = commonFluxShaderCode(true);
const horizontalFluxShaderCode = commonFluxShaderCode(false);

// compute residuals for each cell based on fluxes at faces, run for each cell inside domain
// residual dQ/dt = -1/area * (F_right - F_left + G_up - G_down)
const residualShaderCode = /* wgsl */`
${uni.uniformStruct}

@group(0) @binding(0) var<uniform> uni: Uniforms;
@group(0) @binding(1) var xFlux: texture_2d<f32>; // rgba32float
@group(0) @binding(2) var yFlux: texture_2d<f32>; // rgba32float
@group(0) @binding(3) var residual: texture_storage_2d<rgba32float, write>; // rgba32float
@group(0) @binding(4) var gridPoints: texture_2d<f32>; // rg32float

override WG_X: u32;
override WG_Y: u32;

@compute @workgroup_size(WG_X, WG_Y)
fn main(
  @builtin(global_invocation_id) gid: vec3u
) {
  if (gid.y >= u32(uni.simDomain.y)) { return; } // only update interior cells, skip ghost cells
  let rightIdx = (gid.x + 1) % u32(uni.simDomain.x);
  let xF_left = textureLoad(xFlux, gid.xy, 0);
  let xF_right = textureLoad(xFlux, vec2u(rightIdx, gid.y), 0);
  let yF_up = textureLoad(yFlux, gid.xy + vec2u(0, 1), 0);
  let yF_down = textureLoad(yFlux, gid.xy, 0);

  let vtx1 = textureLoad(gridPoints, gid.xy, 0).xy;
  let vtx2 = textureLoad(gridPoints, vec2u(rightIdx, gid.y), 0).xy;
  let vtx3 = textureLoad(gridPoints, vec2u(gid.x, gid.y + 1), 0).xy;
  let vtx4 = textureLoad(gridPoints, vec2u(rightIdx, gid.y + 1), 0).xy;
  let leftFace = length(vtx1 - vtx3);
  let rightFace = length(vtx2 - vtx4);
  let upFace = length(vtx3 - vtx4);
  let downFace = length(vtx1 - vtx2);

  let area = 0.5 * abs((vtx4.x - vtx1.x) * (vtx3.y - vtx2.y) - (vtx3.x - vtx2.x) * (vtx4.y - vtx1.y));

  let dQdt = -(xF_right * rightFace - xF_left * leftFace + yF_up * upFace - yF_down * downFace) / area;
  textureStore(residual, gid.xy, dQdt);
}
`;

// TVD RK3 for time integration, run for each cell inside domain
// part 1: Q1 = Qn + dt * L(Qn)
const integrationStage1ShaderCode = /* wgsl */`
${uni.uniformStruct}

@group(0) @binding(0) var<uniform> uni: Uniforms;
@group(0) @binding(1) var residual: texture_2d<f32>;  // rgba32float
@group(0) @binding(2) var stateIn: texture_2d<f32>; // rgba32float
@group(0) @binding(3) var stateOut: texture_storage_2d<rgba32float, write>; // rgba32float

override WG_X: u32;
override WG_Y: u32;

@compute @workgroup_size(WG_X, WG_Y)
fn main(
  @builtin(global_invocation_id) gid: vec3u
) {
  if (gid.y >= u32(uni.simDomain.y)) { return; } // only update interior cells, skip ghost cells
  let cellIdx = gid.xy + vec2u(0, 1); // shift up by 1 to account for ghost cells
  let res = textureLoad(residual, gid.xy, 0);
  let Q = textureLoad(stateIn, cellIdx, 0);

  let newQ = Q + uni.dt * res;
  textureStore(stateOut, cellIdx, newQ);
}
`;

// TVD RK3 part 2
// Q2 = 3/4 Qn + 1/4 Q1 + 1/4 dt * L(Q1)
const integrationStage2ShaderCode = /* wgsl */`
${uni.uniformStruct}

@group(0) @binding(0) var<uniform> uni: Uniforms;
@group(0) @binding(1) var residual: texture_2d<f32>;  // rgba32float
@group(0) @binding(2) var stateIn: texture_2d<f32>; // rgba32float
@group(0) @binding(3) var stateIn1: texture_2d<f32>; // rgba32float
@group(0) @binding(4) var stateOut: texture_storage_2d<rgba32float, write>; // rgba32float

override WG_X: u32;
override WG_Y: u32;

@compute @workgroup_size(WG_X, WG_Y)
fn main(
  @builtin(global_invocation_id) gid: vec3u
) {
  if (gid.y >= u32(uni.simDomain.y)) { return; } // only update interior cells, skip ghost cells
  let cellIdx = gid.xy + vec2u(0, 1); // shift up by 1 to account for ghost cells
  let res = textureLoad(residual, gid.xy, 0);
  let Q = textureLoad(stateIn, cellIdx, 0);
  let Q1 = textureLoad(stateIn1, cellIdx, 0);

  let newQ = 0.75 * Q + 0.25 * (Q1 + uni.dt * res);
  textureStore(stateOut, cellIdx, newQ);
}
`;

// TVD RK3 part 3
// Q^(n+1) = 1/3 Qn + 2/3 Q2 + 2/3 dt * L(Q2)
const integrationStage3ShaderCode = /* wgsl */`
${uni.uniformStruct}

@group(0) @binding(0) var<uniform> uni: Uniforms;
@group(0) @binding(1) var residual: texture_2d<f32>;  // rgba32float
@group(0) @binding(2) var stateIn: texture_2d<f32>; // rgba32float
@group(0) @binding(3) var stateIn2: texture_2d<f32>; // rgba32float
@group(0) @binding(4) var stateOut: texture_storage_2d<rgba32float, write>; // rgba32float

override WG_X: u32;
override WG_Y: u32;

@compute @workgroup_size(WG_X, WG_Y)
fn main(
  @builtin(global_invocation_id) gid: vec3u
) {
  if (gid.y >= u32(uni.simDomain.y)) { return; } // only update interior cells, skip ghost cells
  let cellIdx = gid.xy + vec2u(0, 1); // shift up by 1 to account for ghost cells
  let res = textureLoad(residual, gid.xy, 0);
  let Q = textureLoad(stateIn, cellIdx, 0);
  let Q2 = textureLoad(stateIn2, cellIdx, 0);

  let newQ = (Q + 2.0 * (Q2 + uni.dt * res)) / 3.0;

  // let dx = 
  // let rho = newQ.x;
  // let vel = length(Q2.yz / rho);
  // let p = (newQ.w - 0.5 * dot(vel, vel) * rho) * (uni.gamma - 1.0);
  // let a = sqrt(uni.gamma * max(p, 1e-7) / max(rho, 1e-7));
  // let cfl = dx / (vel + a);
  textureStore(stateOut, cellIdx, newQ);
}
`;

// find minimum CFL value across grid for stable time step
// Full reduction in 1 dispatch
// https://developer.download.nvidia.com/assets/cuda/files/reduction.pdf
// need to clear each frame: device.queue.writeBuffer(cflOutBuffer, 0, new Uint32Array([0x7F7FFFFF]));
// read as let dt = bitcast<f32>(cfl) * uni.cflFactor;
const CFLReductionShaderCode = /* wgsl */`
${uni.uniformStruct}

@group(0) @binding(0) var<uniform> uni: Uniforms;
@group(0) @binding(1) var<storage, read> cflValues: array<f32>;
@group(0) @binding(2) var<storage, read_write> cflOut: atomic<u32>;

const blockSize = 64u;
var<workgroup> shared: array<f32, blockSize>;

@compute @workgroup_size(blockSize)
fn main(
  @builtin(global_invocation_id) gid: vec3u,
  @builtin(local_invocation_id) lid: vec3u,
  // @builtin(workgroup_id) wid: vec3u,
  @builtin(num_workgroups) nwg: vec3u
) {
  let lidx = lid.x;
  var i = gid.x;
  var acc = 1e38f;
  let arraySize = u32(uni.simDomain.x) * u32(uni.simDomain.y);
  while (i < arraySize) {
    acc = min(acc, cflValues[i]);
    i += blockSize * nwg.x;
  }
  shared[lidx] = acc;

  workgroupBarrier();

  // parallel reduction to find min CFL value
  for (var stride = blockSize / 2u; stride > 0u; stride >>= 1u) {
    if (lidx < stride) {
      shared[lidx] = min(shared[lidx], shared[lidx + stride]);
    }
    workgroupBarrier();
  }
  if (lidx == 0u) {
    atomicMin(&cflOut, bitcast<u32>(shared[0]));
  }
}
`;

const renderShaderCode = /* wgsl */`
${uni.uniformStruct}

@group(0) @binding(0) var<uniform> uni: Uniforms;
@group(0) @binding(1) var gridPoints: texture_2d<f32>; // rg32float
@group(0) @binding(2) var state: texture_2d<f32>;   // rgba32float
@group(0) @binding(3) var gridSampler: sampler;

struct VertexOut {
  @builtin(position) position: vec4f,
  @location(0) fragCoord: vec2f,
  // @location(1) @interpolate(flat) normal: vec2f,
};

fn vIdxToGridPosX(vIdx: u32) -> vec2u {
  let xRes = u32(uni.simDomain.x);
  let x = vIdx % xRes;
  let y = vIdx / xRes;
  return vec2u(x, y);
}

fn vIdxToGridPosY(vIdx: u32) -> vec2u {
  let yRes = u32(uni.simDomain.y);
  let x = vIdx / yRes;
  let y = vIdx % yRes;
  return vec2u(x, y);
}

fn vIdxToTriangleStrip(vIdx: u32) -> vec2u {
  let res = vec2u(uni.simDomain) + vec2u(0, 1);
  let vtxPerRow = res.x * 2 + 2;
  let strip = vIdx / vtxPerRow;
  let idxInRow = vIdx % vtxPerRow;
  let x = idxInRow / 2;
  let y = strip + (idxInRow % 2);
  return vec2u(x, y);
}

fn vIdxToLineStrip(vIdx: u32) -> vec2u {
  let res = vec2u(uni.simDomain) + vec2u(0, 1);
  let vtxPerRow = res.x * 4;
  let strip = vIdx / vtxPerRow;
  let idxInRow = vIdx % vtxPerRow;
  let x = (idxInRow + 1) / 2;
  let y = strip + u32(step(2.0, f32(idxInRow % 4)));
  return vec2u(x, y);
}

@vertex
fn vs(@builtin(vertex_index) vIdx: u32) -> VertexOut {
  var out: VertexOut;
  var gridPos: vec2u;
  switch (u32(uni.gridDisplayMode)) {
    case 0: { gridPos = vIdxToTriangleStrip(vIdx); }
    case 1: { gridPos = vIdxToLineStrip(vIdx); }
    case 2: { gridPos = vIdxToGridPosX(vIdx); }
    default: { gridPos = vIdxToGridPosX(vIdx); }
  }
  let vtxIdx = vec2u(gridPos.x % (u32(uni.simDomain.x)), gridPos.y);
  let vtx = textureLoad(gridPoints, vtxIdx, 0).xy;
  let vtxPos = (vtx * uni.zoom + uni.pan) / uni.resRatio;
  out.position = vec4f(vtxPos, 0.0, 1.0);
  out.fragCoord = vec2f(gridPos) / vec2f(uni.simDomain);
  
  // let adjIdx = vec2u((gridPos.x + 1) % (u32(uni.simDomain.x)), gridPos.y);
  // let tangent = normalize(vtx - textureLoad(gridPoints, adjIdx, 0).xy);
  // let adjIdx = vec2u(gridPos.x, (gridPos.y + 1));
  // let tangent = normalize(textureLoad(gridPoints, adjIdx, 0).xy - vtx);
  // out.normal = vec2f(-tangent.y, tangent.x);
  return out;
}

@fragment
fn fs(vtx: VertexOut) -> @location(0) vec4f {
  // sample sim state
  let state = textureSample(state, gridSampler, vtx.fragCoord);
  // return vec4f(vtx.fragCoord, 1.0 - vtx.fragCoord.y, 1.0);
  // return vec4f(abs(vtx.normal), dot(vtx.normal, vec2f(1.0, 0.0)) * 0.5 + 0.5, 1.0);
  // return vec4f(abs(state.yz) / (state.x * 5.0), f32(any(state != state)), 0);
  return abs(state / 10.0);
  // return vec4f(state.x / 2.0);// - f32((state.x * 100.0 % 5.0 <= (0.1)));

  
  // let rho = state.x;
  // let vel = length(state.yz / rho);
  // let p = (state.w - 0.5 * vel * vel * rho) * (uni.gamma - 1.0);
  // let a = sqrt(uni.gamma * max(p, 1e-7) / max(rho, 1e-7));
  // return vec4f(vel / a, f32(any(state != state)), 0.0, 1.0);
}
`;
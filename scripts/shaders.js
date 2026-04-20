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
  let Q = 0.0; //-0.01;
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

// finalize grid, compute face lengths, cell areas, todo: calculate face normals
// run for each cell center
const gridFinalizeShaderCode = /* wgsl */`
${uni.uniformStruct}

@group(0) @binding(0) var<uniform> uni: Uniforms;
@group(0) @binding(1) var gridPoints: texture_2d<f32>; // rg32float
@group(0) @binding(2) var gridBoundaries: texture_2d<i32>; // rg16sint
@group(0) @binding(3) var gridArea: texture_storage_2d<r32float, write>;
@group(0) @binding(4) var faceLengths: texture_storage_2d<rgba32float, write>;
@group(0) @binding(5) var cellDistances: texture_storage_2d<rgba32float, write>;

override WG_X: u32;
override WG_Y: u32;

@compute @workgroup_size(WG_X, WG_Y)
fn main(
  @builtin(global_invocation_id) gid: vec3u
) {
  let boundary = textureLoad(gridBoundaries, gid.xy, 0).x;

  let rightIdx = (gid.x + 1) % u32(uni.simDomain.x);
  let a = textureLoad(gridPoints, gid.xy, 0).xy;
  let b = textureLoad(gridPoints, vec2u(rightIdx, gid.y), 0).xy;
  let c = textureLoad(gridPoints, vec2u(gid.x, gid.y + 1), 0).xy;
  let d = textureLoad(gridPoints, vec2u(rightIdx, gid.y + 1), 0).xy;

  let area = 0.5 * abs((d.x - a.x) * (c.y - b.y) - (c.x - b.x) * (d.y - a.y));
  textureStore(gridArea, gid.xy, vec4f(area, 0.0, 0.0, 0.0));

  let leftFace = length(a - c);
  let rightFace = length(b - d);
  let upFace = length(c - d);
  let downFace = length(a - b);
  textureStore(faceLengths, gid.xy, vec4f(leftFace, rightFace, upFace, downFace));

  let cellCenter = 0.25 * (a + b + c + d);
  let rightIdx2 = (gid.x + 2) % u32(uni.simDomain.x);
  let leftIdx = (gid.x + u32(uni.simDomain.x) - 1) % u32(uni.simDomain.x);
  let up1 = textureLoad(gridPoints, vec2u(gid.x, gid.y + 2), 0).xy;
  let up2 = textureLoad(gridPoints, vec2u(rightIdx, gid.y + 2), 0).xy;
  let right1 = textureLoad(gridPoints, vec2u(rightIdx2, gid.y), 0).xy;
  let right2 = textureLoad(gridPoints, vec2u(rightIdx2, gid.y + 1), 0).xy;
  let left1 = textureLoad(gridPoints, vec2u(leftIdx, gid.y), 0).xy;
  let left2 = textureLoad(gridPoints, vec2u(leftIdx, gid.y + 1), 0).xy;
  let down1 = textureLoad(gridPoints, vec2u(gid.x, max(gid.y, 0)), 0).xy;
  let down2 = textureLoad(gridPoints, vec2u(rightIdx, max(gid.y, 0)), 0).xy;
  let leftCenter = 0.25 * (left1 + left2 + a + c);
  let rightCenter = 0.25 * (right1 + right2 + b + d);
  let upCenter = 0.25 * (up1 + up2 + c + d);
  let downCenter = 0.25 * (down1 + down2 + a + b);
  let leftDist = length(cellCenter - leftCenter);
  let rightDist = length(cellCenter - rightCenter);
  let upDist = length(cellCenter - upCenter);
  let downDist = length(cellCenter - downCenter);

  textureStore(cellDistances, gid.xy, vec4f(leftDist, rightDist, upDist, downDist));
}
`;

const prepareStateShaderCode = /* wgsl */`
${uni.uniformStruct}

@group(0) @binding(0) var<uniform> uni: Uniforms;
@group(0) @binding(1) var state: texture_storage_2d<rgba32float, write>;

override WG_X: u32;
override WG_Y: u32;

@compute @workgroup_size(WG_X, WG_Y)
fn main(
  @builtin(global_invocation_id) gid: vec3u
) {
  // initialize interior state to air at rest
  let rhoE = uni.inPressure / (uni.gamma - 1.0) + 0.5 * uni.inRho * 0.0;
  textureStore(state, gid.xy, vec4f(uni.inRho, 0.0, 0.0, rhoE));
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
  let boundary = textureLoad(gridBoundaries, gid.xy, 0).x;
  let isObjectBoundary = gid.y == 0;
  let isOuterBoundary = gid.y > simDomain.y;

  // if (!isObjectBoundary && !isOuterBoundary) { return; } // only compute for ghost cells, skip interior

  let rightIdx = (gid.x + 1) % simDomain.x;
  
  let vtxYCoord = clamp(gid.y, 0, simDomain.y);
  let vtx1 = textureLoad(gridPoints, vec2u(gid.x, vtxYCoord), 0).xy;
  let vtx2 = textureLoad(gridPoints, vec2u(rightIdx, vtxYCoord), 0).xy;
  
  // outward pointing normal
  let tangent = normalize(vtx1 - vtx2);
  let normal = vec2f(-tangent.y, tangent.x);

  var ghostState = textureLoad(stateIn, gid.xy, 0); // pass through by default for non-boundary cells, will be overwritten for ghost cells

  if (isObjectBoundary) {
    // handle object boundary - reflect velocity for slip condition
    let interiorState = textureLoad(stateIn, vec2u(gid.x, 1), 0);
    let rho_int = interiorState.x;
    let u_int = interiorState.yz / rho_int;
    let ghost_U = reflect(u_int, normal); // reflect velocity across normal
    // let rhoE = (interiorState.w - 0.5 * dot(u_int, u_int) * rho_int) + 0.5 * rho_int * dot(ghost_U, ghost_U);
    ghostState = vec4f(rho_int, ghost_U * rho_int, interiorState.w); // reflect velocity, keep other states, KE doesn't change
  } else if (isOuterBoundary) {
    // handle outer boundary
    let interiorState = textureLoad(stateIn, vec2u(gid.x, simDomain.y), 0);
    let rho_int = interiorState.x;
    let u_int = interiorState.yz / rho_int;
    
    let boundaryInOrOut = dot(normal, uni.inflowV); // negative for inflow, positive for outflow
    if(boundaryInOrOut < 0.0) {
      // inflow
      ghostState = uni.inState;
    } else {
      // outflow
      let gammaMinus1Inv = 1.0 / (uni.gamma - 1.0);
      let u_int2 = dot(u_int, u_int);
      let P_int = (interiorState.w - 0.5 * u_int2 * rho_int) * (uni.gamma - 1.0);
      let a2 = (uni.gamma * P_int) / rho_int;
      let a2inf = (uni.gamma * uni.inPressure / uni.inRho);
      let M_int = u_int2 / a2;
      let M_inf2 = dot(uni.inflowV, uni.inflowV) / a2inf;
      if (M_int < 1.0 && M_inf2 < 1.0) {
        // subsonic outflow using Riemann invariant to extrapolate ghost state
        let u_normal_int = dot(u_int, normal);
        let u_normal_inf = boundaryInOrOut;
        let a_int = sqrt(a2);
        let a_inf = sqrt(a2inf);
        
        // Riemann invariants
        let J_int = u_normal_int + 2 * a_int * gammaMinus1Inv;
        let J_ext = u_normal_inf - 2 * a_inf * gammaMinus1Inv;
        
        // velocity in normal direction from Riemann invariant, tangential velocity from interior
        let u_normal_ghost = 0.5 * (J_int + J_ext);
        let a_ghost = max(a_int * 0.1, 0.25 * (J_int - J_ext) * (uni.gamma - 1.0));
        let X = max(1e-6, a_ghost / a_int);
        let X2 = X * X;
        let rho_ghost = max(1e-6, rho_int * X2 * X2 * X); // for gamma = 1.4

        // combine tangential velocity with normal velocity from Riemann
        let vel_ghost = (u_normal_ghost - u_normal_int) * normal + u_int;
        let rhoE_ghost = max(1e-6, (rho_ghost * a_ghost * a_ghost) / uni.gamma) * gammaMinus1Inv + 0.5 * rho_ghost * dot(vel_ghost, vel_ghost);

        ghostState = vec4f(rho_ghost, vel_ghost * rho_ghost, rhoE_ghost); // Riemann invariant for normal velocity, copy tangential momentum from interior
      } else if (M_inf2 > 1.0) {
        let ambientRhoE = uni.inPressure / (uni.gamma - 1.0) + 0.5 * rho_int * u_int2; //dot(uni.inflowV, uni.inflowV)
        ghostState = vec4f(interiorState.xyz, ambientRhoE); // fix pressure to ambient for M < 1.0
      } else {
        // supersonic outflow, copy interior state
        ghostState = interiorState;
      }
      if (gid.y == simDomain.y + 2) {
        ghostState = 2.0 * ghostState - interiorState; // for 2nd order extrapolation in muscl
      }
    }
  }
  textureStore(stateOut, gid.xy, ghostState);
}
`;

// common to all flux calculations
// run for each face, Mx(N+1) for vertical faces, (M+1)xN for horizontal faces
// vertical flux calculated for face between (x, y) and (x, y+1), horizontal flux calculated for face between (x, y) and (x+1, y)
// vertical - face at a given index is below cell at that index (using shifted index for ghost cells)
// calculate face normal and left/right rho, u, u_normal, p, h
const fluxInterfaceStateCode = (vertical) => /* wgsl */`
${uni.uniformStruct}

@group(0) @binding(0) var<uniform> uni: Uniforms;
@group(0) @binding(1) var gridPoints: texture_2d<f32>; // rg32float
@group(0) @binding(2) var gridBoundaries: texture_2d<i32>; // rg16sint
@group(0) @binding(3) var state: texture_2d<f32>; // rgba32float
@group(0) @binding(4) var flux: texture_storage_2d<rgba32float, write>;

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

fn vanLeerLimiter(r: vec4f) -> vec4f {
  let absR = abs(r);
  return (r + absR) / (1.0 + absR);
}

fn minmod(a: f32, b: f32) -> f32 {
  return 0.5 * (sign(a) + sign(b)) * min(abs(a), abs(b));
}

fn vanAlbadaLimiter(r: vec4f) -> vec4f {
  let r2 = r * r;
  return (r2 + r) / (r2 + 1.0);
}

override WG_X: u32;
override WG_Y: u32;

var<workgroup> localStates: array<vec4f, WG_X * (WG_Y + 3)>;

@compute @workgroup_size(WG_X, WG_Y)
fn main(
  @builtin(global_invocation_id) gid: vec3u,
  @builtin(local_invocation_id) lid: vec3u
) {
  let gammaMinus1 = (uni.gamma - 1.0);

  let cellIdx = vec2i(gid.xy);
  let boundary = textureLoad(gridBoundaries, gid.xy, 0).xy;

  let epsilon = 1e-6;

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

  let Q_Lprimitive = toPrimitive(Q_L); // rho, u, v, p
  let Q_L2primitive = toPrimitive(Q_L2);
  let Q_Rprimitive = toPrimitive(Q_R);
  let Q_R2primitive = toPrimitive(Q_R2);

  // relative to current cell:
  // y-2: Q_L2
  // y-1: Q_L
  // y+0: Q_R
  // y+1: Q_R2
  // let localIdx = (lid.y + 2) * WG_X + lid.x;

  // let Q_Rprimitive = toPrimitive(loadState(cellIdx));
  // localStates[localIdx] = Q_Rprimitive;

  // if (lid.y == WG_Y - 1) {
  //   localStates[localIdx + WG_X] = toPrimitive(loadState(cellIdx + vec2i(${vertical ? "0, 1" : "1, 0"}))); // Q_R2
  // }
  // // first 2 rows of workgroup each load 1 extra row for Q_L and Q_L2, with clamping for object boundary handled in loadState
  // if (lid.y <= 1) {
  //   // y=1 -> Q_L halo
  //   // y=0 -> Q_L2 halo
  //   localStates[localIdx - 2 * WG_X] = toPrimitive(loadState(cellIdx - vec2i(${vertical ? "0, 2" : "2, 0"})));
  // }
  
  // workgroupBarrier();

  // let Q_Lprimitive = localStates[localIdx - WG_X];
  // let Q_L2primitive = localStates[localIdx - 2 * WG_X];
  // let Q_R2primitive = localStates[localIdx + WG_X];

  var Q_LMUSCLprimitive = Q_Lprimitive;
  var Q_RMUSCLprimitive = Q_Rprimitive;
  if (uni.muscl > 0) {
    // MUSCL
    let delta_L = Q_Lprimitive - Q_L2primitive;
    let delta_Face = Q_Rprimitive - Q_Lprimitive;
    let delta_R = Q_R2primitive - Q_Rprimitive;
    let r_L = (delta_L * delta_Face + epsilon) / (delta_Face * delta_Face + epsilon);
    let r_R = (delta_Face * delta_R + epsilon) / (delta_Face * delta_Face + epsilon); // / delta_R^2
    Q_LMUSCLprimitive += 0.5 * vanAlbadaLimiter(r_L) * (delta_Face);
    Q_RMUSCLprimitive -= 0.5 * vanAlbadaLimiter(r_R) * (delta_Face); // * delta_R
  }

  // interface states, with u = velocity normal to face
  // left state
  let rho_L = Q_LMUSCLprimitive.x;
  let vel_L = Q_LMUSCLprimitive.yz;
  let u_L = dot(vel_L, normal);

  let p_L = Q_LMUSCLprimitive.w;
  let h_L = p_L / (gammaMinus1 * rho_L) + 0.5 * dot(vel_L, vel_L) + p_L / rho_L;

  // right state
  let rho_R = Q_RMUSCLprimitive.x;
  let vel_R = Q_RMUSCLprimitive.yz;
  let u_R = dot(vel_R, normal);

  let p_R = Q_RMUSCLprimitive.w;
  let h_R = p_R / (gammaMinus1 * rho_R) + 0.5 * dot(vel_R, vel_R) + p_R / rho_R;
`;

// "A sequel to AUSM, Part II: AUSM+-up for all speeds", Liou 2006, sec. 3.3
// https://www.sciencedirect.com/science/article/pii/S0021999105004274
const AUSMupFluxShaderCode = (vertical) => /* wgsl */`
${fluxInterfaceStateCode(vertical)}
  let M_inf2 = dot(uni.inflowV, uni.inflowV) / (uni.gamma * uni.inPressure / rho_L);
  let sigma = 1.0;

  // more accurate
  let h_t_interface = 0.5 * (h_L + h_R); // total enthalpy
  let aStar = sqrt(max(1e-6, 2 * gammaMinus1 / (uni.gamma + 1) * h_t_interface)); // eq 29
  let aHat_L = aStar * aStar / max(u_L, aStar); // eq 30
  let aHat_R = aStar * aStar / max(-u_R, aStar);
  let a_interface = min(aHat_L, aHat_R); // eq 28

  // let a_interface = (a_L + a_R) * 0.5;
  let M_L = u_L / a_interface; // eq 69
  let M_R = u_R / a_interface;
  let a_interface2 = a_interface * a_interface;
  
  let Mbar2 = (u_L * u_L + u_R * u_R) / (a_interface2 * 2); // eq 70
  let M_o2 = min(1, max(Mbar2, M_inf2)); // eq 71
  let f_a = M_o2 * (2 - M_o2); // eq 72
  let rho_interface = 0.5 * (rho_L + rho_R);

  let M_LR = vec2f(M_L, M_R);

  // precompute the M2 components for both L and R
  let M_plus_1  = M_LR + 1.0;
  let M_minus_1 = M_LR - 1.0;
  let M2_plus  =  0.25 * M_plus_1  * M_plus_1;
  let M2_minus = -0.25 * M_minus_1 * M_minus_1;

  // vectorized M4 splitting
  let M4_sub = vec2f(
    M2_plus.x  * (1.0 + 2.0 * M2_minus.x),
    M2_minus.y * (1.0 - 2.0 * M2_plus.y)
  );

  // supersonic check for both sides independently
  let is_super = step(vec2f(1.0), abs(M_LR)); 
  let M4_super = vec2f(max(0.0, M_L), min(0.0, M_R));
  let M4_final = mix(M4_sub, M4_super, is_super);


  let M_interface = M4_final.x + M4_final.y - uni.K_p / f_a * max(1 - sigma * Mbar2, 0) * (p_R - p_L) / (rho_interface * a_interface2); // eq 73

  let alpha4 = 0.75 * (-4 + 5 * f_a * f_a); // eq 76 * 4
  
  // vectorized P5 splitting
  let P5_sub = vec2f(
    M2_plus.x  * (( 2.0 - M_L) + alpha4 * M_L * M2_minus.x),
    M2_minus.y * ((-2.0 - M_R) - alpha4 * M_R * M2_plus.y)
  );

  let P5_super = vec2f(step(0.0, M_L), step(0.0, -M_R));
  let P5_final = mix(P5_sub, P5_super, is_super);
  let P_interface = dot(P5_final, vec2f(p_L, p_R)) - uni.K_u * P5_final.x * P5_final.y * (rho_L + rho_R) * f_a * a_interface * (u_R - u_L); // eq 75
  
  let mdot = a_interface * M_interface * select(rho_R, rho_L, M_interface > 0); // eq 74

  let phi_R = vec4f(1, vel_R, h_R); // eq 3
  let phi_L = vec4f(1, vel_L, h_L);
  let convectiveFlux = select(phi_R, phi_L, mdot > 0.0); // eq 6
  let pressureFlux = vec4f(0.0, normal * P_interface, 0.0);

  let totalFlux = convectiveFlux * mdot + pressureFlux;
  textureStore(flux, gid.xy, totalFlux);
}
`;
const AUSMup_verticalFluxShaderCode   = AUSMupFluxShaderCode(true);
const AUSMup_horizontalFluxShaderCode = AUSMupFluxShaderCode(false);

// Parameter-Free Simple Low-Dissipation AUSM-Family Scheme for All Speeds, Kitamura and Shima 2011
// https://arc.aiaa.org/doi/pdf/10.2514/1.J050905
// Towards shock-stable and accurate hypersonic heating computations: A new pressure flux for AUSM-family schemes, Kitamura and Shima 2013
// https://www.sciencedirect.com/science/article/pii/S0021999113001769
const SLAUFluxShaderCode = (vertical, version=2) => /* wgsl */`
${fluxInterfaceStateCode(vertical)}

  let a_L = sqrt(uni.gamma * max(p_L, 1e-10) / max(rho_L, 1e-10));
  let a_R = sqrt(uni.gamma * max(p_R, 1e-10) / max(rho_R, 1e-10));
  let a_interface = (a_L + a_R) * 0.5;
  let M_L = u_L / a_interface;
  let M_R = u_R / a_interface;
  
  let Uarc_unclamped = sqrt(0.5 * (dot(vel_L, vel_L) + dot(vel_R, vel_R)));
  let Marc = min(1.0, Uarc_unclamped / a_interface); // eq 2.3e
  let X = (1 - Marc) * (1 - Marc); // eq 2.3d

  let M_LR = vec2f(M_L, M_R);

  // precompute the M2 components for both L and R
  let M_LRplusminus1 = vec2f(M_L + 1.0, M_R - 1.0);
  let M_LR1sq = 0.5 * M_LRplusminus1 * M_LRplusminus1;
  let plusOrMinus = vec2f(1.0, -1.0);
  let is_super = step(vec2f(1.0), abs(M_LR)); 

  let f_p_plusminus = saturate(0.5 * mix(
    M_LR1sq * (2 - plusOrMinus * M_LR),
    1 + plusOrMinus * sign(M_LR),
    is_super
  ));

  let f_p_plus = f_p_plusminus.x;
  let f_p_minus = f_p_plusminus.y;
  let P_interface = 0.5 * ((f_p_plus - f_p_minus) * (p_L - p_R) + ${version == 2
    ? "(p_L + p_R) + (Uarc_unclamped * (f_p_plus + f_p_minus - 1) * (rho_L + rho_R) * a_interface)); // eq 3.4 SLAU2"
    : "(1.0 + (1 - X) * (f_p_plus + f_p_minus - 1)) * (p_L + p_R)); // eq 2.3c SLAU original"
  }

  let delta_p = p_R - p_L;
  let ubar_n = (rho_L * abs(u_L) + rho_R * abs(u_R)) / (rho_L + rho_R); // eq 2.3k
  let g = saturate(-M_L) * saturate(M_R); // eq 2.3l
  let ubar_n_plus = mix(ubar_n, abs(u_L), g);
  let ubar_n_minus = mix(ubar_n, abs(u_R), g);
  let mdot = 0.5 * ((rho_L * (u_L + ubar_n_plus) + rho_R * (u_R - ubar_n_minus)) * (1 - g) - X / a_interface * delta_p); // eq 2.3i

  let phi_R = vec4f(1, vel_R, h_R);
  let phi_L = vec4f(1, vel_L, h_L);
  let convectiveFlux = select(phi_R, phi_L, mdot > 0.0);
  let pressureFlux = vec4f(0.0, normal * P_interface, 0.0);

  let totalFlux = convectiveFlux * mdot + pressureFlux;
  textureStore(flux, gid.xy, totalFlux);
}
`;
const SLAU_verticalFluxShaderCode   = SLAUFluxShaderCode(true, 1);
const SLAU_horizontalFluxShaderCode = SLAUFluxShaderCode(false, 1);
const SLAU2_verticalFluxShaderCode   = SLAUFluxShaderCode(true, 2);
const SLAU2_horizontalFluxShaderCode = SLAUFluxShaderCode(false, 2);

// compute residuals for each cell based on fluxes at faces, run for each cell inside domain
// residual dQ/dt = -1/area * (F_right - F_left + G_up - G_down)
const residualShaderCode = /* wgsl */`
${uni.uniformStruct}

@group(0) @binding(0) var<uniform> uni: Uniforms;
@group(0) @binding(1) var xFlux: texture_2d<f32>; // rgba32float
@group(0) @binding(2) var yFlux: texture_2d<f32>; // rgba32float
@group(0) @binding(3) var residual: texture_storage_2d<rgba32float, write>; // rgba32float
@group(0) @binding(4) var faceLengths: texture_2d<f32>; // rgba32float
@group(0) @binding(5) var gridArea: texture_2d<f32>; // r32float

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

  let faceLengths = textureLoad(faceLengths, gid.xy, 0);
  let leftFace = faceLengths.x;
  let rightFace = faceLengths.y;
  let upFace = faceLengths.z;
  let downFace = faceLengths.w;

  let area = textureLoad(gridArea, gid.xy, 0).x;

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
@group(0) @binding(4) var<storage, read> maxWaveSpeed: u32;

override WG_X: u32;
override WG_Y: u32;

@compute @workgroup_size(WG_X, WG_Y)
fn main(
  @builtin(global_invocation_id) gid: vec3u
) {
  let dt = min(uni.maxdt, uni.cflFactor / bitcast<f32>(maxWaveSpeed));
  if (gid.y >= u32(uni.simDomain.y)) { return; } // only update interior cells, skip ghost cells
  let cellIdx = gid.xy + vec2u(0, 1); // shift up by 1 to account for ghost cells
  let res = textureLoad(residual, gid.xy, 0);
  let Q = textureLoad(stateIn, cellIdx, 0);

  let newQ = Q + dt * res;
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
@group(0) @binding(5) var<storage, read> maxWaveSpeed: u32;

override WG_X: u32;
override WG_Y: u32;

@compute @workgroup_size(WG_X, WG_Y)
fn main(
  @builtin(global_invocation_id) gid: vec3u
) {
  let dt = min(uni.maxdt, uni.cflFactor / bitcast<f32>(maxWaveSpeed));
  if (gid.y >= u32(uni.simDomain.y)) { return; } // only update interior cells, skip ghost cells
  let cellIdx = gid.xy + vec2u(0, 1); // shift up by 1 to account for ghost cells
  let res = textureLoad(residual, gid.xy, 0);
  let Q = textureLoad(stateIn, cellIdx, 0);
  let Q1 = textureLoad(stateIn1, cellIdx, 0);

  let newQ = 0.75 * Q + 0.25 * (Q1 + dt * res);
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
@group(0) @binding(5) var<storage, read> maxWaveSpeed: u32;

override WG_X: u32;
override WG_Y: u32;

@compute @workgroup_size(WG_X, WG_Y)
fn main(
  @builtin(global_invocation_id) gid: vec3u
) {
  let dt = min(uni.maxdt, uni.cflFactor / bitcast<f32>(maxWaveSpeed));
  if (gid.y >= u32(uni.simDomain.y)) { return; } // only update interior cells, skip ghost cells
  let cellIdx = gid.xy + vec2u(0, 1); // shift up by 1 to account for ghost cells
  let res = textureLoad(residual, gid.xy, 0);
  let Q = textureLoad(stateIn, cellIdx, 0);
  let Q2 = textureLoad(stateIn2, cellIdx, 0);

  let newQ = (Q + 2.0 * (Q2 + dt * res)) / 3.0;
  textureStore(stateOut, cellIdx, newQ);
}
`;

// find minimum CFL value across grid for stable time step
// Full reduction in 1 dispatch
// https://developer.download.nvidia.com/assets/cuda/files/reduction.pdf
// need to clear each frame: device.queue.writeBuffer(cflOutBuffer, 0, new Uint32Array([0]));
// read as let dt = uni.cflFactor / bitcast<f32>(maxWaveSpeed);
const cflReductionShaderCode = /* wgsl */`
${uni.uniformStruct}

@group(0) @binding(0) var<uniform> uni: Uniforms;
@group(0) @binding(1) var<storage, read> cflValues: array<f32>;
@group(0) @binding(2) var<storage, read_write> maxWaveSpeed: atomic<u32>;

override WG_X: u32;
override WG_Y: u32;

const blockSize = 256u;
var<workgroup> wgShared: array<f32, blockSize>;

@compute @workgroup_size(blockSize)
fn main(
  @builtin(global_invocation_id) gid: vec3u,
  @builtin(local_invocation_id) lid: vec3u,
  // @builtin(workgroup_id) wid: vec3u,
  @builtin(num_workgroups) nwg: vec3u
) {
  if (all(gid == vec3u(0))) {
    atomicStore(&maxWaveSpeed, 0u); // reset max wave speed for this frame, will be updated by reduction
  }
  let lidx = lid.x;
  var i = gid.x;
  var acc = 0.0;
  let arraySize = u32(uni.simDomain.x) * u32(uni.simDomain.y);
  let stride = blockSize * nwg.x;
  while (i < arraySize) {
    acc = max(acc, cflValues[i]);
    i += stride;
  }
  wgShared[lidx] = acc;

  workgroupBarrier();

  // parallel reduction to find max CFL value
  for (var s = blockSize / 2u; s > 0u; s >>= 1u) {
    if (lidx < s) {
      wgShared[lidx] = max(wgShared[lidx], wgShared[lidx + s]);
    }
    workgroupBarrier();
  }
  if (lidx == 0u) {
    atomicMax(&maxWaveSpeed, bitcast<u32>(wgShared[0]));
  }
}
`;

// visualization shader, run for each cell, read state and other variables to calculate color based on display mode
// also calculates wave speeds for CFL condition and stores in buffer for reduction
// run once per frame after state update
const visualizationShaderCode = /* wgsl */`
${uni.uniformStruct}

@group(0) @binding(0) var<uniform> uni: Uniforms;
@group(0) @binding(1) var state: texture_2d<f32>;   // rgba32float
@group(0) @binding(2) var faceLengths: texture_2d<f32>;   // rgba32float
@group(0) @binding(3) var area: texture_2d<f32>;   // r32float
@group(0) @binding(4) var vis: texture_storage_2d<rgba8unorm, write>;
@group(0) @binding(5) var<storage, read_write> waveSpeeds: array<f32>;

override WG_X: u32;
override WG_Y: u32;

fn colorMapBRYW(value: f32) -> vec4f {
  return vec4f(value, value - 1.0, saturate(1.0 - value) + saturate(2.0 * (value - 1.0) / value - 1.0), value / (value + uni.contourCompression)); // atan(value) / 1.5708);//
  // return vec4f(saturate(value), saturate(abs(value) - 1.0), mix(saturate(-value), saturate(2.0 * (value - 1.0) / value - 1.0), f32(value > 0.0)), value / (value + uni.contourCompression)); // atan(value) / 1.5708);//
  // return vec4f(
  //   smoothstep(0.0, 1.0, value),
  //   smoothstep(1.0, 2.0, value),
  //   smoothstep(1.0, 0.0, value) + smoothstep(0.0, 1.0, 2.0 * (value - 1.0) / value - 1.0),
  //   value / (value + uni.contourCompression)
  // );
}

@compute @workgroup_size(WG_X, WG_Y)
fn main(
  @builtin(global_invocation_id) gid: vec3u
) {
  if (any(gid.xy >= vec2u(uni.simDomain))) { return; }

  let state0 = textureLoad(state, gid.xy + vec2u(0, 1), 0); // shift up by 1 to account for ghost cells
  let rho = state0.x;
  let inVelMag = length(uni.inflowV);
  let velocity = state0.yz / rho;
  let pressure = (state0.w - 0.5 * rho * dot(velocity, velocity)) * (uni.gamma - 1.0);
  let temperature = pressure / rho;
  let inTemp = uni.inPressure / uni.inRho;
  let a = sqrt(uni.gamma * temperature);
  let speed = length(velocity);

  // calculate wave speed for CFL condition, store in buffer for reduction
  let faceLengths = textureLoad(faceLengths, gid.xy, 0);
  let area = textureLoad(area, gid.xy, 0).x;
  waveSpeeds[gid.x + gid.y * u32(uni.simDomain.x)] = (speed + a) * dot(faceLengths, vec4f(1.0)) / (2.0 * area);


  var color: vec4f;

  if (uni.simDisplayMode <= 1) {
    // schlieren and vorticity

    // gradient of density for numerical schlieren
    let leftIdx = (gid.x + u32(uni.simDomain.x) - 1) % u32(uni.simDomain.x);
    let rightIdx = (gid.x + 1) % u32(uni.simDomain.x);

    let stateLeft = textureLoad(state, vec2u(leftIdx, gid.y + 1), 0);
    let stateRight = textureLoad(state, vec2u(rightIdx, gid.y + 1), 0);
    let stateDown = textureLoad(state, vec2u(gid.x, max(gid.y, 1)), 0);
    let stateUp = textureLoad(state, vec2u(gid.x, min(gid.y + 2, u32(uni.simDomain.y) - 1)), 0);

    if (uni.simDisplayMode == 0) {
      // numerical schlieren
      let gradX = (stateRight.x - stateLeft.x); // / (distances.x + distances.y);
      let gradY = (stateUp.x - stateDown.x); // / (distances.z + distances.w);
      let gradMag = length(vec2f(gradX, gradY));
      color = vec4f(exp(-gradMag * 2.0));
    } else {
      // vorticity
      let vorticity = (stateRight.z / stateRight.x - stateLeft.z / stateLeft.x) - (stateUp.y / stateUp.x - stateDown.y / stateDown.x); // dv/dx - du/dy, central difference with ghost cells
      color = vec4f(colorMapBRYW(vorticity * 0.5 + 0.5));
      // let divergence = (stateRight.y / stateRight.x - stateLeft.y / stateLeft.x) + (stateUp.z / stateUp.x - stateDown.z / stateDown.x); // du/dx + dv/dy
      // color = vec4f(colorMapBRYW(divergence * 0.5 + 0.5));
    }
    textureStore(vis, gid.xy, color);
    return;
  } else if (uni.simDisplayMode == 7) {
    // let velND = abs(velocity) - inVelMag; // non-dimensionalize velocity by inflow velocity for visualization
    // textureStore(vis, gid.xy, vec4f(velND.x, velND.y, 0.0, speed / (speed + uni.contourCompression)));
    // relative velocity
    let vRelAbs = abs(velocity - uni.inflowV) * uni.visMultiplier;
    let adjSpeed = speed * uni.visMultiplier;
    let velocityRGB = vec3f(vRelAbs, velocity.y) / (vec3f(vRelAbs, abs(velocity.y)) + 1);
    textureStore(vis, gid.xy, vec4f(velocityRGB, adjSpeed / (adjSpeed + uni.contourCompression)));
    return;
  }
  var localValue: f32;
  var freeStreamValue: f32;
  switch (u32(uni.simDisplayMode)) {
    case 2: {
      localValue = rho;
      freeStreamValue = uni.inRho;
    }
    case 3: {
      localValue = temperature;
      freeStreamValue = inTemp;
    }
    case 4: {
      localValue = pressure;
      freeStreamValue = uni.inPressure;
    }
    case 5,6: {
      localValue = speed / a;
      freeStreamValue = select(0.5, inVelMag / sqrt(uni.gamma * inTemp), uni.simDisplayMode == 6);
    }
    case 8: {
      localValue = log(pressure / pow(rho, uni.gamma));
      freeStreamValue = log(uni.inPressure / pow(uni.inRho, uni.gamma));
    }
    case 9: {
      let mach = speed / a;
      let inMach = inVelMag / sqrt(uni.gamma * inTemp);
      let totalPressure = pressure * pow(1.0 + 0.5 * (uni.gamma - 1.0) * mach * mach, uni.gamma / (uni.gamma - 1.0));
      let totalPressureInf = uni.inPressure * pow(1.0 + 0.5 * (uni.gamma - 1.0) * inMach * inMach, uni.gamma / (uni.gamma - 1.0));
      localValue = totalPressure;
      freeStreamValue = totalPressureInf;
    }
    default: {
      localValue = rho;
      freeStreamValue = uni.inRho;
    }
  }
  textureStore(vis, gid.xy, colorMapBRYW(((localValue - freeStreamValue) * uni.visMultiplier) / (freeStreamValue * 2) + 0.5));
  // textureStore(vis, gid.xy, colorMapBRYW((localValue - freeStreamValue) * uni.visMultiplier));
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
  
  // adjust for ghost cells in visualization
  // let gridPos_f = vec2f(gridPos);
  // out.fragCoord = vec2f(gridPos_f.x, gridPos_f.y * uni.simDomain.y / (uni.simDomain.y + 3) + 1) / uni.simDomain;

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
  if (uni.contourLevels > 0) {
    // https://observablehq.com/@rreusser/locally-scaled-domain-coloring-part-1-contour-plots#contours
    // let plotValue = tan(state.a * 1.5708) * uni.contourLevels;
    let plotValue = uni.contourCompression * state.a / (1.0 - state.a) * uni.contourLevels;
    // let screenSpaceGradient = fwidthFine(plotValue);
    let screenSpaceGradient = length(vec2f(dpdxFine(plotValue), dpdyFine(plotValue)));
    let contourLineWidth = 1.0;
    let contour = step(contourLineWidth, (0.5 - abs(fract(plotValue) - 0.5)) / (screenSpaceGradient + 1e-6));
    return mix(state, vec4f(0.0), 1 - contour);
  }
  return state;
}
`;
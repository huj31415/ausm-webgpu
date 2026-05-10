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
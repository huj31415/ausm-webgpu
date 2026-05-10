
// calculate forces on aerofoil by integrating pressure along surface, run for the object boundary once per frame or at graph update rate
// calculate pressure * area for each boundary segment and store in buffer for reduction
const forceCalculationShaderCode = /* wgsl */`
${uni.uniformStruct}

@group(0) @binding(0) var<uniform> uni: Uniforms;
@group(0) @binding(1) var state: texture_2d<f32>;   // rgba32float
@group(0) @binding(2) var faceLengths: texture_2d<f32>; // rgba32float
@group(0) @binding(3) var<storage, read_write> forceValues: array<vec2f>;

override WG_X: u32;
override WG_Y: u32;

@compute @workgroup_size(WG_X, WG_Y)
fn main(
  @builtin(global_invocation_id) gid: vec3u,
) {
}
`;

// integrate aero force vectors to calculate total force vector and store in output buffer for force vector plot
// calculate magnitude, C_l and C_d and store each in ring buffers for graphing
const forceIntegrationShaderCode = /* wgsl */`
${uni.uniformStruct}

@group(0) @binding(0) var<uniform> uni: Uniforms;
@group(0) @binding(1) var<storage, read> forceValues: array<vec2f>;
@group(0) @binding(2) var<storage, read_write> totalForce: atomic<u32>;

override WG_X: u32;
override WG_Y: u32;

const blockSize = 256u;
var<workgroup> wgShared: array<f32, blockSize>;

@compute @workgroup_size(blockSize)
fn main(
  @builtin(global_invocation_id) gid: vec3u,
  @builtin(local_invocation_id) lid: vec3u,
  @builtin(num_workgroups) nwg: vec3u
) {
  if (all(gid == vec3u(0))) {
    atomicStore(&totalForce, 0u); // reset for this frame, will be updated by reduction
  }
  let lidx = lid.x;
  var i = gid.x;
  var acc = 0.0;
  let arraySize = u32(uni.simDomain.x) * u32(uni.simDomain.y);
  let stride = blockSize * nwg.x;
  while (i < arraySize) {
    acc += forceValues[i];
    i += stride;
  }
  wgShared[lidx] = acc;

  workgroupBarrier();

  // parallel reduction to integrate force
  for (var s = blockSize / 2u; s > 0u; s >>= 1u) {
    if (lidx < s) {
      wgShared[lidx] += wgShared[lidx + s];
    }
    workgroupBarrier();
  }
  if (lidx == 0u) {
    atomicAdd(&totalForce, bitcast<u32>(wgShared[0]));
  }
}
`;

const lineGraphRenderShaderCode = /* wgsl */`
${uni.uniformStruct}

// @group(0) @binding(0) var<uniform> uni: Uniforms;
@group(0) @binding(1) var<storage, read> data: array<f32>;

struct VertexOut {
  @builtin(position) position: vec4f,
  @location(0) fragCoord: vec2f,
};

@vertex
fn vs(@builtin(vertex_index) vIdx: u32) -> VertexOut {
  // use data to draw triangle strip from bottom of the graph to data point height
  // store the relative data point height for coloring
  // vertex order for area graph: (0,0) -> (0, data[0]) -> (1,0) -> (1, data[1]) -> ...
  // for thick line graph: (0, data[0] - thickness) -> (0, data[0]) -> (1, data[1] - thickness) -> ...
  // need to correct for normal to avoid line thickness decreasing at high slopes
  let idx = vIdx / 2u;
  let dataValue = data[idx];
  let thickness = 0.02;
  let isOdd = (vIdx & 1u) == 1;
  let pos = vec2f(f32(idx) / 255.0, select(dataValue, 0, isOdd));
  var out: VertexOut;
  out.position = vec4f(pos * 2.0 - 1.0, 0.0, 1.0);
  out.fragCoord = pos;
  return out;
}

@fragment
fn fs(vtx: VertexOut) -> @location(0) vec4f {
  return vec4f((vtx.fragCoord.y + 1.0) * 0.5, 0.0, 0.0, 1.0);
}

`;
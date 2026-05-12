
// calculate forces on aerofoil by integrating pressure along surface, run for the object boundary once per frame or at graph update rate
// calculate pressure * area for each boundary segment and store in buffer for reduction
// currently uses cell center pressures, todo: use pressure splitting to get boundary face pressures
const forceCalcShaderCode = /* wgsl */`
${uni.uniformStruct}

@group(0) @binding(0) var<uniform> uni: Uniforms;
@group(0) @binding(1) var state: texture_2d<f32>;   // rgba32float
@group(0) @binding(2) var gridPoints: texture_2d<f32>; // rg32float
@group(0) @binding(3) var<storage, read_write> forceValues: array<vec2f>;

override WG_X: u32;
override WG_Y: u32;

@compute @workgroup_size(16, 1)
fn main(
  @builtin(global_invocation_id) gid: vec3u,
) {
  let idx = gid.x;
  let simDomain = vec2u(uni.simDomain);
  if (idx >= simDomain.x) {
    return;
  }
  
  let state0 = textureLoad(state, vec2u(idx, 1), 0); // get row 1 to account for ghost cells
  let rho = state0.x;
  let velocity = state0.yz / rho;
  let pressure = (state0.w - 0.5 * rho * dot(velocity, velocity)) * (uni.gamma - 1.0);

  let rightIdx = (idx + 1) % simDomain.x;
  
  let vtx1 = textureLoad(gridPoints, vec2u(idx, 0), 0).xy;
  let vtx2 = textureLoad(gridPoints, vec2u(rightIdx, 0), 0).xy;
  let faceLength = length(vtx2 - vtx1);
  let normal = vec2f(vtx2.y - vtx1.y, vtx1.x - vtx2.x) / faceLength;

  // inward normal points towards object, force is pressure * area * normal
  forceValues[idx] = pressure * faceLength * -normal;
}
`;

// integrate aero force vectors to calculate total force vector and store in output buffer for force vector plot
// dispatch 1 workgroup
// calculate magnitude, C_l and C_d and store each in ring buffers for graphing, can also be done in graphing shader
const forceIntegrationShaderCode = /* wgsl */`
${uni.uniformStruct}
${graphUni.uniformStruct}

@group(0) @binding(0) var<uniform> uni: Uniforms;
@group(0) @binding(1) var<uniform> graphUni: GraphUniforms;
@group(0) @binding(2) var<storage, read> forceValues: array<vec2f>;
@group(0) @binding(3) var<storage, read_write> forceRing: array<vec2f>;

override WG_X: u32;
override WG_Y: u32;

const blockSize = 256u;
var<workgroup> wgShared: array<vec2f, blockSize>;

@compute @workgroup_size(blockSize)
fn main(
  @builtin(local_invocation_id) lid: vec3u
) {
  let writeIdx = u32(graphUni.writeHead);

  let lidx = lid.x;
  var i = lidx;
  var acc = vec2f(0.0);
  let arraySize = u32(uni.simDomain.x);
  while (i < arraySize) {
    acc += forceValues[i];
    i += blockSize;
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
    forceRing[writeIdx] = wgShared[0];
  }
}
`;

const lineGraphRenderShaderCode = /* wgsl */`
${graphUni.uniformStruct}

@group(0) @binding(0) var<uniform> graphUni: GraphUniforms;
@group(0) @binding(1) var<storage, read> data: array<vec2f>;

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
  let dataValue = length(data[(idx + u32(graphUni.writeHead)) % u32(graphUni.nPoints)]);
  // let dataValue = (data[(idx + u32(graphUni.writeHead)) % u32(graphUni.nPoints)]).x;
  let thickness = 0.01;
  let isOdd = (vIdx & 1u) == 1;
  let pos = vec2f(f32(idx) / 255.0, select(dataValue, dataValue + thickness, isOdd));
  var out: VertexOut;
  out.position = vec4f(pos * 2.0 - 1.0, 0.0, 1.0);
  out.fragCoord = pos;
  return out;
}

@fragment
fn fs(vtx: VertexOut) -> @location(0) vec4f {
  return (vtx.fragCoord.y + 1.0) * 0.5 * vec4f(1.0);
}
`;

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
  
  // adjust for ghost cells in visualization
  // let gridPos_f = vec2f(gridPos);
  // out.fragCoord = vec2f(gridPos_f.x, gridPos_f.y * uni.simDomain.y / (uni.simDomain.y + 3) + 1) / uni.simDomain;

  // let adjIdx = vec2u((gridPos.x + 1) % (u32(uni.simDomain.x)), gridPos.y);
  // let tangent = normalize(vtx - textureLoad(gridPoints, adjIdx, 0).xy);

  // let adjIdx = vec2u(gridPos.x, (gridPos.y + 1));
  // let tangent = normalize(textureLoad(gridPoints, adjIdx, 0).xy - vtx);
  // out.normal = vec2f(-tangent.y, tangent.x);

  return VertexOut(
    vec4f(vtxPos, 0.0, 1.0),
    vec2f(gridPos) / vec2f(uni.simDomain)
  );
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
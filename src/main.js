
let adapter, device;
let gpuInfo = false;

async function main() {

  if (device) device.destroy();

  adapter = await navigator.gpu?.requestAdapter();

  const maxComputeInvocationsPerWorkgroup = adapter.limits.maxComputeInvocationsPerWorkgroup;
  const maxBufferSize = adapter.limits.maxBufferSize;
  const maxStorageBufferBindingSize = adapter.limits.maxStorageBufferBindingSize;
  const f32filterable = adapter.features.has("float32-filterable");
  // const shaderf16 = adapter.features.has("shader-f16");
  // const subgroups = adapter.features.has("subgroups");

//   const floatPrecision = shaderf16 ? 16 : 32;
//   const f16header = shaderf16 ? `
// enable f16;
// // alias vec4h = vec4<f${floatPrecision}>;
// // alias vec3h = vec3<f${floatPrecision}>;
// // alias vec2h = vec2<f${floatPrecision}>;
// ` : "";

  const textureTier1 = adapter.features.has("texture-formats-tier1");
  if (!textureTier1 && !gpuInfo) alert("texture-formats-tier1 feature required");
  // const textureTier2 = adapter.features.has("texture-formats-tier2");
  // if (!textureTier2 && !gpuInfo) alert("texture-formats-tier2 unsupported, may reduce performance");

  // compute workgroup size 32^2 = 1024 threads if maxComputeInvocationsPerWorkgroup >= 1024, otherwise 16^2 = 256 threads
  // const largeWg = maxComputeInvocationsPerWorkgroup >= 1024;
  // seems like smaller workgroups are faster
  const [wg_x, wg_y] = [16, 16]; //largeWg ? [32, 32] : [16, 16];

  if (!gpuInfo) {
    gui.addGroup("deviceInfo", "Device info", `
<pre>maxComputeInvocationsPerWorkgroup: ${maxComputeInvocationsPerWorkgroup}
workgroup: [${wg_x}, ${wg_y}]
maxBufferSize: ${maxBufferSize}
maxStorageBufferBindingSize: ${maxStorageBufferBindingSize}
f32filterable: ${f32filterable}
texture-formats-tier1: ${textureTier1}
</pre>
    `);
    // <span ${!textureTier2 ? "class='warn'" : ""}>texture-formats-tier2: ${textureTier2}</span>
    // shader-f16: ${shaderf16}
    // subgroups: ${subgroups}
    gpuInfo = true;
  }


  device = await adapter?.requestDevice({
    requiredFeatures: [
      ...(adapter.features.has("timestamp-query") ? ["timestamp-query"] : []),
      ...(f32filterable ? ["float32-filterable"] : []),
      ...(textureTier1 ? ["texture-formats-tier1"] : []),
      // ...(textureTier2 ? ["texture-formats-tier2"] : []),
      // ...(shaderf16 ? ["shader-f16"] : []),
      // ...(subgroups ? ["subgroups"] : []),
    ],
    requiredLimits: {
      maxComputeInvocationsPerWorkgroup: maxComputeInvocationsPerWorkgroup,
      maxBufferSize: maxBufferSize,
      maxStorageBufferBindingSize: maxStorageBufferBindingSize,
    }
  });
  device.addEventListener('uncapturederror', event => {
    const msg = event.error.message;
    if (msg.includes("max buffer size limit"))
      alert(`Max buffer size exceeded. Reduce the simulation domain size to decrease buffer size`);
    else {
      // alert(msg);
    }
    cancelAnimationFrame(rafId);
    return;
  });

  // restart if device crashes
  device.lost.then((info) => {
    if (info.reason != "destroyed") {
      // hardReset();
      console.warn("WebGPU device lost, reinitializing.");
    }
  });

  // }
  if (!device) {
    alert("Browser does not support WebGPU");
    document.body.textContent = "WebGPU is not supported in this browser.";
    return;
  }
  const context = canvas.getContext("webgpu");
  const swapChainFormat = navigator.gpu.getPreferredCanvasFormat();
  context.configure({
    device: device,
    format: swapChainFormat,
  });

  const colorbarCtx = colorbar.getContext("2d");

  const new2dTexture = (name, size = simulationDomain, format = `r${floatPrecision}float`, copy = false, storage = true) => device.createTexture({
    size: size,
    dimension: "2d",
    format: format,
    usage: GPUTextureUsage.TEXTURE_BINDING | (copy ? GPUTextureUsage.COPY_DST | GPUTextureUsage.COPY_SRC : 0) | (storage ? GPUTextureUsage.STORAGE_BINDING : 0),
    label: `${name} texture`
  });

  storage.gridPoints0 = new2dTexture("gridPoints0", gridVertexCount, `rg32float`, true);
  storage.gridPoints1 = new2dTexture("gridPoints1", gridVertexCount, `rg32float`, true);
  storage.gridBoundaries = new2dTexture("gridBoundaries", gridVertexCount, `rg16sint`, true);
  storage.gridArea = new2dTexture("gridArea", simulationDomain, `r32float`);
  storage.faceLengths = new2dTexture("faceLengths", simulationDomain, `rgba32float`);
  storage.cellDistances = new2dTexture("cellDistances", simulationDomain, `rgba32float`);

  storage.state0 = new2dTexture("state0", totalCellCount, `rgba32float`, true);
  storage.state1 = new2dTexture("state1", totalCellCount, `rgba32float`, true);
  storage.state2 = new2dTexture("state2", totalCellCount, `rgba32float`, true);

  storage.fluxX = new2dTexture("fluxX", xFluxTexSize, `rgba32float`);
  storage.fluxY = new2dTexture("fluxY", yFluxTexSize, `rgba32float`);
  storage.residual = new2dTexture("residual", simulationDomain, `rgba32float`);
  
  storage.vis = new2dTexture("visualization", simulationDomain, `rgba8unorm`);
  storage.waveSpeeds = device.createBuffer({
    size: simulationDomain[0] * simulationDomain[1] * 4,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    label: "waveSpeeds buffer"
  });
  storage.maxWaveSpeed = device.createBuffer({
    size: 4,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.UNIFORM, // | GPUBufferUsage.COPY_DST
    label: "maxWaveSpeed buffer"
  });
  // device.queue.writeBuffer(storage.maxWaveSpeed, 0, new Float32Array([23000]));

  storage.forceValues = device.createBuffer({
    size: simulationDomain[0] * 4 * 2, // vec2f per boundary segment
    usage: GPUBufferUsage.STORAGE,
    label: "forceValues buffer"
  });
  storage.ringBuffer = device.createBuffer({
    size: graphPoints * 4 * 4, // vec4f
    usage: GPUBufferUsage.STORAGE,
    label: "forceRingBuffer buffer"
  });
  storage.forceVector = device.createBuffer({
    size: 8, // single vec2f
    usage: GPUBufferUsage.STORAGE,
    label: "forceVector buffer"
  });
  // // test data for graph
  // let forceValues = new Float32Array(graphPoints * 2);
  // for (let i = 0; i < graphPoints; i++) {
  //   forceValues[i * 2] = Math.random();
  //   forceValues[i * 2 + 1] = Math.random();
  // }
  // device.queue.writeBuffer(storage.forceRingBuffer, 0, forceValues);

  // const stagingBuffer = device.createBuffer({
  //   size: 4,
  //   usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
  // });

  const textureViews = Object.fromEntries(
    Object.entries(storage).filter(([key, value]) => value instanceof GPUTexture).map(([key, texture]) => [key, texture.createView()])
  );

  const uniformBuffer = uni.createBuffer(device);
  const graphUniformBuffer = graphUni.createBuffer(device);

  const newComputePipeline = (shaderCode, name, layout = "auto") =>
    device.createComputePipeline({
      layout: layout,
      compute: {
        module: device.createShaderModule({
          code: shaderCode, // f16header +
          label: `${name} compute module`
        }),
        constants: {
          WG_X: wg_x,
          WG_Y: wg_y
        },
        entryPoint: 'main'
      },
      label: `${name} compute pipeline`
    });

  const gridInterpolationComputePipeline = newComputePipeline(gridInterpolationShaderCode, "grid interpolation");
  const gridInterpolationBindGroup = device.createBindGroup({
    layout: gridInterpolationComputePipeline.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: { buffer: uniformBuffer } },
      { binding: 1, resource: textureViews.gridPoints0 },
      { binding: 2, resource: textureViews.gridPoints1 },
      { binding: 3, resource: textureViews.gridBoundaries },
    ],
    label: "grid interpolation compute bind group"
  });

  const gridEllipticPoissonComputePipeline = newComputePipeline(gridEllipticPoissonShaderCode, "grid elliptic poisson");
  const gridEllipticPoissonBindGroup = (texInView, texOutView) => device.createBindGroup({
    layout: gridEllipticPoissonComputePipeline.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: { buffer: uniformBuffer } },
      { binding: 1, resource: texInView },
      { binding: 2, resource: texOutView },
      { binding: 3, resource: textureViews.gridBoundaries },
    ],
    label: "grid elliptic poisson compute bind group"
  });
  const gridEllipticPoissonBindGroups = [
    gridEllipticPoissonBindGroup(textureViews.gridPoints1, textureViews.gridPoints0),
    gridEllipticPoissonBindGroup(textureViews.gridPoints0, textureViews.gridPoints1),
  ];

  const gridFinalizeComputePipeline = newComputePipeline(gridFinalizeShaderCode, "grid finalize");
  const gridFinalizeBindGroup = (pointTexView) => device.createBindGroup({
    layout: gridFinalizeComputePipeline.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: { buffer: uniformBuffer } },
      { binding: 1, resource: pointTexView },
      { binding: 2, resource: textureViews.gridBoundaries },
      { binding: 3, resource: textureViews.gridArea },
      { binding: 4, resource: textureViews.faceLengths },
      { binding: 5, resource: textureViews.cellDistances },
    ],
    label: "grid finalize compute bind group"
  });
  const gridFinalizeBindGroups = [
    gridFinalizeBindGroup(textureViews.gridPoints1),
    gridFinalizeBindGroup(textureViews.gridPoints0),
  ];
  const prepareStateComputePipeline = newComputePipeline(prepareStateShaderCode, "prepare state");
  const prepareStateBindGroup = device.createBindGroup({
    layout: prepareStateComputePipeline.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: { buffer: uniformBuffer } },
      { binding: 1, resource: textureViews.state0 },
    ],
    label: "prepare state compute bind group"
  });

  const boundaryComputePipeline = newComputePipeline(boundaryShaderCode, "boundary condition");
  const boundaryBindGroup = (stateInView, stateOutView) => device.createBindGroup({
    layout: boundaryComputePipeline.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: { buffer: uniformBuffer } },
      { binding: 1, resource: textureViews.gridPoints0 },
      { binding: 2, resource: textureViews.gridBoundaries },
      { binding: 3, resource: stateInView },
      { binding: 4, resource: stateOutView },
    ],
    label: "boundary compute bind group"
  });
  const boundaryBindGroups = [
    boundaryBindGroup(textureViews.state2, textureViews.state0),
    boundaryBindGroup(textureViews.state1, textureViews.state2),
    boundaryBindGroup(textureViews.state2, textureViews.state1),
  ];

  const fluxBindGroupLayout = device.createBindGroupLayout({
    entries: [
      { // uniform
        binding: 0,
        visibility: GPUShaderStage.COMPUTE,
        buffer: {
          type: "uniform",
          hasDynamicOffset: false,
        },
      },
      { // grid points
        binding: 1,
        visibility: GPUShaderStage.COMPUTE,
        texture: {
          sampleType: "float",
          viewDimension: "2d",
          multisampled: false,
        },
      },
      { // grid boundaries
        binding: 2,
        visibility: GPUShaderStage.COMPUTE,
        texture: {
          sampleType: "sint",
          viewDimension: "2d",
          multisampled: false,
        },
      },
      { // state
        binding: 3,
        visibility: GPUShaderStage.COMPUTE,
        texture: {
          sampleType: "float",
          viewDimension: "2d",
          multisampled: false,
        },
      },
      { // flux output
        binding: 4,
        visibility: GPUShaderStage.COMPUTE,
        storageTexture: {
          access: "write-only",
          viewDimension: "2d",
          format: "rgba32float",
        },
      },
    ],
  });
  const fluxPipelineLayout = device.createPipelineLayout({
    bindGroupLayouts: [ fluxBindGroupLayout ],
  });
  const fluxPipelines = Object.freeze({
    "SLAU2": [
      newComputePipeline(SLAU2_verticalFluxShaderCode, "SLAU2 vertical flux", fluxPipelineLayout),
      newComputePipeline(SLAU2_horizontalFluxShaderCode, "SLAU2 horizontal flux", fluxPipelineLayout)
    ],
    "SLAU": [
      newComputePipeline(SLAU_verticalFluxShaderCode, "SLAU vertical flux", fluxPipelineLayout),
      newComputePipeline(SLAU_horizontalFluxShaderCode, "SLAU horizontal flux", fluxPipelineLayout)
    ],
    "AUSM+-up": [
      newComputePipeline(AUSMup_verticalFluxShaderCode, "AUSM+-up vertical flux", fluxPipelineLayout),
      newComputePipeline(AUSMup_horizontalFluxShaderCode, "AUSM+-up horizontal flux", fluxPipelineLayout)
    ],
  });
  let [verticalFluxComputePipeline, horizontalFluxComputePipeline] = fluxPipelines.SLAU2;
  const verticalFluxBindGroup = (stateView) => device.createBindGroup({
    layout: fluxBindGroupLayout,
    entries: [
      { binding: 0, resource: { buffer: uniformBuffer } },
      { binding: 1, resource: textureViews.gridPoints0 },
      { binding: 2, resource: textureViews.gridBoundaries },
      { binding: 3, resource: stateView },
      { binding: 4, resource: textureViews.fluxY },
    ],
    label: "vertical flux compute bind group"
  });
  const verticalFluxBindGroups = [
    verticalFluxBindGroup(textureViews.state0),
    verticalFluxBindGroup(textureViews.state2),
    verticalFluxBindGroup(textureViews.state1),
  ];

  const horizontalFluxBindGroup = (stateView) => device.createBindGroup({
    layout: fluxBindGroupLayout,
    entries: [
      { binding: 0, resource: { buffer: uniformBuffer } },
      { binding: 1, resource: textureViews.gridPoints0 },
      { binding: 2, resource: textureViews.gridBoundaries },
      { binding: 3, resource: stateView },
      { binding: 4, resource: textureViews.fluxX },
    ],
    label: "horizontal flux compute bind group"
  });
  const horizontalFluxBindGroups = [
    horizontalFluxBindGroup(textureViews.state0),
    horizontalFluxBindGroup(textureViews.state2),
    horizontalFluxBindGroup(textureViews.state1),
  ];

  const residualComputePipeline = newComputePipeline(residualShaderCode, "residual compute");
  const residualBindGroup = device.createBindGroup({
    layout: residualComputePipeline.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: { buffer: uniformBuffer } },
      { binding: 1, resource: textureViews.fluxX },
      { binding: 2, resource: textureViews.fluxY },
      { binding: 3, resource: textureViews.residual },
      { binding: 4, resource: textureViews.faceLengths },
      { binding: 5, resource: textureViews.gridArea },
    ],
    label: "residual compute bind group"
  });

  const integrationStage1ComputePipeline = newComputePipeline(integrationStage1ShaderCode, "integration stage 1");
  const integrationStage1BindGroup = device.createBindGroup({
    layout: integrationStage1ComputePipeline.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: { buffer: uniformBuffer } },
      { binding: 1, resource: textureViews.residual },
      { binding: 2, resource: textureViews.state0 },
      { binding: 3, resource: textureViews.state1 },
      { binding: 4, resource: { buffer: storage.maxWaveSpeed } },
    ],
    label: "integration stage 1 compute bind group"
  });

  const integrationStage2ComputePipeline = newComputePipeline(integrationStage2ShaderCode, "integration stage 2");
  const integrationStage2BindGroup = device.createBindGroup({
    layout: integrationStage2ComputePipeline.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: { buffer: uniformBuffer } },
      { binding: 1, resource: textureViews.residual },
      { binding: 2, resource: textureViews.state0 },
      { binding: 3, resource: textureViews.state1 },
      { binding: 4, resource: textureViews.state2 },
      { binding: 5, resource: { buffer: storage.maxWaveSpeed } },
    ],
    label: "integration stage 2 compute bind group"
  });

  const integrationStage3ComputePipeline = newComputePipeline(integrationStage3ShaderCode, "integration stage 3");
  const integrationStage3BindGroup = device.createBindGroup({
    layout: integrationStage3ComputePipeline.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: { buffer: uniformBuffer } },
      { binding: 1, resource: textureViews.residual },
      { binding: 2, resource: textureViews.state0 },
      { binding: 3, resource: textureViews.state1 },
      { binding: 4, resource: textureViews.state2 },
      { binding: 5, resource: { buffer: storage.maxWaveSpeed } },
    ],
    label: "integration stage 3 compute bind group"
  });

  const visualizationComputePipeline = newComputePipeline(visualizationShaderCode, "visualization");
  const visualizationBindGroup = device.createBindGroup({
    layout: visualizationComputePipeline.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: { buffer: uniformBuffer } },
      { binding: 1, resource: textureViews.state2 },
      { binding: 2, resource: textureViews.faceLengths },
      { binding: 3, resource: textureViews.gridArea },
      { binding: 4, resource: textureViews.vis },
      { binding: 5, resource: { buffer: storage.waveSpeeds } },
    ],
    label: "visualization compute bind group"
  });

  const cflReductionComputePipeline = newComputePipeline(cflReductionShaderCode, "CFL reduction");
  const cflReductionBindGroup = device.createBindGroup({
    layout: cflReductionComputePipeline.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: { buffer: uniformBuffer } },
      { binding: 1, resource: { buffer: storage.waveSpeeds } },
      { binding: 2, resource: { buffer: storage.maxWaveSpeed } },
    ],
    label: "CFL reduction compute bind group"
  });

  const filter = f32filterable ? "linear" : "nearest";
  const f32sampler = device.createSampler({
    magFilter: filter,
    minFilter: filter,
    addressModeU: "repeat",
    addressModeV: "clamp-to-edge",
  });
  const gridSampler = device.createSampler({
    magFilter: "linear",//"nearest",
    minFilter: "linear",//"nearest",
    addressModeU: "repeat",
    addressModeV: "clamp-to-edge",
  })

  const renderModule = device.createShaderModule({
    code: renderShaderCode,
    label: "render module"
  });

  const renderBindGroupLayout = device.createBindGroupLayout({
    entries: [
      {
        binding: 0,
        visibility: GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT,
        buffer: {
          type: "uniform",
          hasDynamicOffset: false,
          // minBindingSize: 80,
        },
      },
      {
        binding: 1,
        visibility: GPUShaderStage.VERTEX,
        texture: {
          sampleType: "float",
          viewDimension: "2d",
          multisampled: false,
        },
      },
      {
        binding: 2,
        visibility: GPUShaderStage.FRAGMENT,
        texture: {
          sampleType: "float",
          viewDimension: "2d",
          multisampled: false,
        },
      },
      {
        binding: 3,
        visibility: GPUShaderStage.FRAGMENT,
        sampler: {
          type: "filtering",
        }
      },
    ],
    label: "render bind group layout"
  });
  const renderPipelineLayout = device.createPipelineLayout({
    bindGroupLayouts: [ renderBindGroupLayout ],
  });
  
  const newRenderPipeline = (topology, name = "main", module = renderModule, layout = renderPipelineLayout, consts = {}) =>
    device.createRenderPipeline({
      label: `${topology} ${name} rendering pipeline`,
      layout: layout,
      vertex: { module: module },
      fragment: {
        module: module,
        targets: [{ format: swapChainFormat }],
        constants: consts
      },
      primitive: { topology: topology },
    });
  const renderPipelines = [
    newRenderPipeline("triangle-strip"),
    newRenderPipeline("line-strip"),
    newRenderPipeline("point-list"),
  ]

  const renderBindGroup = device.createBindGroup({
    layout: renderBindGroupLayout,
    entries: [
      { binding: 0, resource: { buffer: uniformBuffer } },
      { binding: 1, resource: textureViews.gridPoints0 },
      { binding: 2, resource: textureViews.vis },
      { binding: 3, resource: gridSampler },
    ],
  });

  const renderPassDescriptor = {
    label: 'render pass',
    colorAttachments: [{
      clearValue: [0.1, 0.1, 0.1, 1],
      loadOp: 'clear',
      storeOp: 'store',
    }]
  };

  // force calculation and graphing
  const forceCalcComputePipeline = newComputePipeline(forceCalcShaderCode, "Force calculation");
  const forceCalcBindGroup = device.createBindGroup({
    layout: forceCalcComputePipeline.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: { buffer: uniformBuffer } },
      { binding: 1, resource: textureViews.state2 }, // latest state after integration
      { binding: 2, resource: textureViews.gridPoints0 },
      { binding: 3, resource: { buffer: storage.forceValues } },
    ],
    label: "Force calculation compute bind group"
  });
  const forceIntegrationComputePipeline = newComputePipeline(forceIntegrationShaderCode, "Force integration");
  const forceIntegrationBindGroup = device.createBindGroup({
    layout: forceIntegrationComputePipeline.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: { buffer: uniformBuffer } },
      { binding: 1, resource: { buffer: graphUniformBuffer } },
      { binding: 2, resource: { buffer: storage.forceValues } },
      { binding: 3, resource: { buffer: storage.ringBuffer } },
      { binding: 4, resource: { buffer: storage.forceVector } },
    ],
    label: "Force integration compute bind group"
  });


  // todo: disable graph update when hidden
  lineGraphCtx.configure({
    device: device,
    format: swapChainFormat,
  });
  const lineGraphRenderModule = device.createShaderModule({
    code: lineGraphRenderShaderCode,
    label: "line graph render module"
  });
  const lineGraphRenderPipeline = newRenderPipeline("triangle-strip", "line graph", lineGraphRenderModule, "auto");
  const lineGraphRenderBindGroup = device.createBindGroup({
    layout: lineGraphRenderPipeline.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: { buffer: graphUniformBuffer } },
      { binding: 1, resource: { buffer: storage.ringBuffer } },
    ],
    label: "line graph render bind group"
  });
  const lineGraphRenderPassDescriptor = {
    label: 'line graph render pass',
    colorAttachments: [{
      clearValue: [0, 0, 0, 0],
      loadOp: 'load', // clear
      storeOp: 'store',
    }]
  };

  const filterStrength = 20;

  const computeTimingHelper = new TimingHelper(device);
  const renderTimingHelper = new TimingHelper(device);
  const postprocessingTimingHelper = new TimingHelper(device);
  const gridTimingHelper = new TimingHelper(device);
  const forceCalcTimingHelper = new TimingHelper(device);
  const graphTimingHelper = new TimingHelper(device);

  const wgDispatchSize = (texSize) => [
    Math.ceil(texSize[0] / wg_x),
    Math.ceil(texSize[1] / wg_y)
  ];
  const gridDispatchSize = wgDispatchSize(gridVertexCount);
  const totalDispatchSize = wgDispatchSize(totalCellCount);
  const yFluxDispatchSize = wgDispatchSize(yFluxTexSize);
  const xFluxDispatchSize = wgDispatchSize(xFluxTexSize);
  const simulationDomainDispatchSize = wgDispatchSize(simulationDomain);

  function createComputePass(pass, pipeline, bindGroup, dispatchSize = wgDispatchSize(simulationDomain)) {
    pass.setPipeline(pipeline);
    pass.setBindGroup(0, bindGroup);
    pass.dispatchWorkgroups(...dispatchSize);
    // pass.end();
  }

  // get target frame time by measuring the time of two consecutive frames
  targetFrameTime = 0;
  requestAnimationFrame((frameStartTime) => {
    requestAnimationFrame((timestamp) => {
      targetFrameTime = timestamp - frameStartTime - 1; // subtract 1ms for overhead
    });
  });

  updateSolver = (solver) => [verticalFluxComputePipeline, horizontalFluxComputePipeline] = fluxPipelines[solver];

  prepareState = () => {
    uni.update(device.queue);
    const encoder = device.createCommandEncoder();

    const statePrepPass = encoder.beginComputePass();
    createComputePass(statePrepPass, prepareStateComputePipeline, prepareStateBindGroup, totalDispatchSize);
    statePrepPass.end();
    encoder.copyTextureToTexture(
      { texture: storage.state0 },
      { texture: storage.state1 },
      totalCellCount
    );
    encoder.copyTextureToTexture(
      { texture: storage.state0 },
      { texture: storage.state2 },
      totalCellCount
    );
    // update visualization for initial state and run CFL calculation before sim runs
    const postPass = encoder.beginComputePass();
    createComputePass(postPass, visualizationComputePipeline, visualizationBindGroup, wgDispatchSize(simulationDomain));
    createComputePass(postPass, cflReductionComputePipeline, cflReductionBindGroup, [Math.ceil(simulationDomain[0] * simulationDomain[1] / 256)]);
    postPass.end();
    device.queue.submit([encoder.finish()]);
    actualInflowVel = 0;
  }

  // todo: check residuals after every n iterations and stop if converged
  prepareGrid = () => {
    poissonIterations = 0;
    let pingPongIndex = 0;
    
    uni.update(device.queue);
    
    // write boundaries
    device.queue.writeTexture(
      { texture: storage.gridPoints0 },
      gridVtxData,
      { offset: 0, bytesPerRow: gridVertexCount[0] * 8, rowsPerImage: gridVertexCount[1] },
      { width: gridVertexCount[0], height: gridVertexCount[1] },
    );
    device.queue.writeTexture(
      { texture: storage.gridBoundaries },
      gridBoundaryData,
      { offset: 0, bytesPerRow: gridVertexCount[0] * 4, rowsPerImage: gridVertexCount[1] },
      { width: gridVertexCount[0], height: gridVertexCount[1] },
    );

    const encoder = device.createCommandEncoder();

    // create initial guess using linear interpolation
    const initGuessPass = encoder.beginComputePass();
    createComputePass(initGuessPass, gridInterpolationComputePipeline, gridInterpolationBindGroup, gridDispatchSize);
    initGuessPass.end();
    encoder.copyTextureToTexture(
      { texture: storage.gridPoints1 },
      { texture: storage.gridPoints0 },
      gridVertexCount
    );
    // iteratively solve elliptic Poisson equation to smooth the grid
    const gridSolveAndFinalizePass = gridTimingHelper.beginComputePass(encoder);
    for (let i = 0; i < maxPoissonIterations; i++) {
      createComputePass(gridSolveAndFinalizePass, gridEllipticPoissonComputePipeline, gridEllipticPoissonBindGroups[pingPongIndex], gridDispatchSize);
      pingPongIndex = 1 - pingPongIndex;
    }
    poissonIterations = maxPoissonIterations;

    // finalize grid by computing cell areas, face lengths, and cell distances
    createComputePass(gridSolveAndFinalizePass, gridFinalizeComputePipeline, gridFinalizeBindGroups[pingPongIndex], simulationDomainDispatchSize);
    gridSolveAndFinalizePass.end();
    // pingPongIndex = 1 - pingPongIndex;

    device.queue.submit([encoder.finish()]);
    setTimeout(() => {
      gridTimingHelper.getResult().then(gpuTime => gui.io.gridTime(gpuTime / 1e6));
    }, 100);
    prepareState();
  };

  prepareGrid();

  let [rawComputeTime, rawRenderTime, rawPostprocessingTime, rawForceTime, rawGraphTime, avgTimePerStep] = [0, 0, 0, 0, 0, 1];
  const cflReader = new AsyncBufferReader(device, 4, 5);

  function render() {
    // update performance info
    const startTime = performance.now();
    deltaTime += Math.min(startTime - lastFrameTime - deltaTime, 1e4) / filterStrength;
    fps += (1e3 / deltaTime - fps) / filterStrength;

    const run = maxdt > 0;
    // adaptive time stepping: adjust stepsPerFrame based on how long the last frame took compared to the target frame time
    const totalTime = (rawComputeTime + rawRenderTime + rawPostprocessingTime + rawForceTime + rawGraphTime) / 1e6;
    if (autoStepsPerFrame && totalTime > 0 && run) {
      const timeDifference = (startTime - lastFrameTime) - targetFrameTime;
      // const timeDifference = totalTime - targetFrameTime;
      // avgTimePerStep = totalTime / stepsPerFrame * 0.1 + avgTimePerStep * 0.9;
      // const stepDifference = timeDifference / avgTimePerStep;
      if (totalTime < targetFrameTime - 2) stepsPerFrame += 1;
      stepsPerFrame = Math.max(1, Math.round(stepsPerFrame - Math.min(100, timeDifference) * 0.05));
      // stepsPerFrame -= stepDifference * 0.1; //Math.max(1, Math.round(stepsPerFrame - Math.min(100, timeDifference) * 0.05));
    }
    
    lastFrameTime = startTime;

    uni.update(device.queue);

    graphUni.update(device.queue);

    const encoder = device.createCommandEncoder();

    // simulate
    if (run) {
      const computePass = computeTimingHelper.beginComputePass(encoder);
      for (let i = 0; i < stepsPerFrame; i++) {
        // state2 -> state0 (Qn)
        createComputePass(computePass, boundaryComputePipeline, boundaryBindGroups[0], totalDispatchSize);
        // state0 -> fluxY
        createComputePass(computePass, verticalFluxComputePipeline, verticalFluxBindGroups[0], yFluxDispatchSize);
        // state0 -> fluxX
        createComputePass(computePass, horizontalFluxComputePipeline, horizontalFluxBindGroups[0], xFluxDispatchSize);
        // fluxX, fluxY -> residual
        createComputePass(computePass, residualComputePipeline, residualBindGroup, simulationDomainDispatchSize);
        // state0 (Qn) + residual -> state1 (Q1)
        createComputePass(computePass, integrationStage1ComputePipeline, integrationStage1BindGroup, simulationDomainDispatchSize);

        // state1 -> state2 (Q1)
        createComputePass(computePass, boundaryComputePipeline, boundaryBindGroups[1], totalDispatchSize);
        // state2 -> fluxY
        createComputePass(computePass, verticalFluxComputePipeline, verticalFluxBindGroups[1], yFluxDispatchSize);
        // state2 -> fluxX
        createComputePass(computePass, horizontalFluxComputePipeline, horizontalFluxBindGroups[1], xFluxDispatchSize);
        // fluxX, fluxY -> residual
        createComputePass(computePass, residualComputePipeline, residualBindGroup, simulationDomainDispatchSize);
        // state0 (Qn), state1 (Q1) + residual -> state2 (Q2)
        createComputePass(computePass, integrationStage2ComputePipeline, integrationStage2BindGroup, simulationDomainDispatchSize);

        // state2 -> state1 (Q2)
        createComputePass(computePass, boundaryComputePipeline, boundaryBindGroups[2], totalDispatchSize);
        // state1 -> fluxY
        createComputePass(computePass, verticalFluxComputePipeline, verticalFluxBindGroups[2], yFluxDispatchSize);
        // state1 -> fluxX
        createComputePass(computePass, horizontalFluxComputePipeline, horizontalFluxBindGroups[2], xFluxDispatchSize);
        // fluxX, fluxY -> residual
        createComputePass(computePass, residualComputePipeline, residualBindGroup, simulationDomainDispatchSize);
        // state0 (Qn), state1 (Q2) + residual -> state2 (Qn+1)
        createComputePass(computePass, integrationStage3ComputePipeline, integrationStage3BindGroup, simulationDomainDispatchSize);
      }
      // update inflow velocity, will be 1 frame behind
      actualInflowVel += (inflowVel - actualInflowVel) / (velRampUpStrength / stepsPerFrame * 50);
      const inflowFinal = [actualInflowVel * xyAoA[0], actualInflowVel * xyAoA[1]];
      uni.values.inflowV.set(inflowFinal);
      // compute inflow state
      const rhoE = inPressure / (gamma - 1.0) + 0.5 * (actualInflowVel * actualInflowVel) * inRho;
      uni.values.inState.set([inRho, inflowFinal[0] * inRho, inflowFinal[1] * inRho, rhoE]);

      computePass.end();
      cflReader.recordCopy(encoder, storage.maxWaveSpeed);

      const postPass = postprocessingTimingHelper.beginComputePass(encoder);
      createComputePass(postPass, visualizationComputePipeline, visualizationBindGroup, wgDispatchSize(simulationDomain));
      createComputePass(postPass, cflReductionComputePipeline, cflReductionBindGroup, [Math.ceil(simulationDomain[0] * simulationDomain[1] / 256)]);
      postPass.end();
    } // else { computeTime = 0; }

    renderPassDescriptor.colorAttachments[0].view = context.getCurrentTexture().createView();
    const renderPass = renderTimingHelper.beginRenderPass(encoder, renderPassDescriptor);
    renderPass.setPipeline(renderPipelines[gridDisplayMode]);
    renderPass.setBindGroup(0, renderBindGroup);
    renderPass.draw(numVertices);
    renderPass.end();

    // graphing, todo: only update when graph is visible, adjust update rate to dt/frame
    if (run) {
      const forceGraphPass = forceCalcTimingHelper.beginComputePass(encoder);
      createComputePass(forceGraphPass, forceCalcComputePipeline, forceCalcBindGroup, [Math.ceil(simulationDomain[0] / 16)]);
      createComputePass(forceGraphPass, forceIntegrationComputePipeline, forceIntegrationBindGroup, [1]);
      forceGraphPass.end();

      lineGraphRenderPassDescriptor.colorAttachments[0].view = lineGraphCtx.getCurrentTexture().createView();
      const lineGraphPass = graphTimingHelper.beginRenderPass(encoder, lineGraphRenderPassDescriptor);
      lineGraphPass.setPipeline(lineGraphRenderPipeline);
      lineGraphPass.setBindGroup(0, lineGraphRenderBindGroup);
      lineGraphPass.draw(graphPoints * 2, 4); // graphPoints vertices per line, 4 lines (force, Cl, Cd, Cl/Cd)
      lineGraphPass.end();
      // shift graph data in ring buffer
      graphUni.values.writeHead.set([(graphUni.values.writeHead[0] + 1) % graphPoints]);
    }

    device.queue.submit([encoder.finish()]);

    if (run) {
      computeTimingHelper.getResult().then(gpuTime => rawComputeTime = gpuTime);
      postprocessingTimingHelper.getResult().then(gpuTime => rawPostprocessingTime = gpuTime);
      forceCalcTimingHelper.getResult().then(gpuTime => rawForceTime = gpuTime);
      graphTimingHelper.getResult().then(gpuTime => rawGraphTime = gpuTime);
      computeTime += (rawComputeTime - computeTime) / filterStrength;
      postprocessingTime += (rawPostprocessingTime - postprocessingTime) / filterStrength;
      forceTime += (rawForceTime - forceTime) / filterStrength;
      graphTime += (rawGraphTime - graphTime) / filterStrength;
      gui.io.mach(actualInflowVel / Math.sqrt(gamma * inPressure / inRho));
    } else {
      computeTime = postprocessingTime = forceTime = graphTime = 0;
    }
    renderTimingHelper.getResult().then(gpuTime => rawRenderTime = gpuTime);
    renderTime += (rawRenderTime - renderTime) / filterStrength;


    jsTime += (performance.now() - startTime - jsTime) / filterStrength;

    rafId = requestAnimationFrame(render);
    // rafId = setTimeout(render, 1000);
  }

  perfIntId = setInterval(async () => {
    gui.io.fps(fps);
    gui.io.jsTime(jsTime);
    gui.io.frameTime(deltaTime);
    gui.io.computeTime(computeTime / 1e6);
    gui.io.postTime(postprocessingTime / 1e3);
    gui.io.renderTime(renderTime / 1e6);
    gui.io.poissonIterations(poissonIterations);
    gui.io.stepsPerFrame(stepsPerFrame);
    gui.io.forceTime(forceTime / 1e3);
    gui.io.graphTime(graphTime / 1e3);
    // read CFL value from GPU and cast to float
    const max = await cflReader.readLatest();
    const buffer = new ArrayBuffer(4);
    const intView = new Uint32Array(buffer);
    const floatView = new Float32Array(buffer);
    intView[0] = max;
    if (max !== undefined) {
      gui.io.dt(cflFactor * 1e6 / floatView[0]);
    }
  }, 100);


  rafId = requestAnimationFrame(render);
}

uni.values.zoom.set([1.0]);
uni.values.pan.set([0.0, 0.0]);
uni.values.simDomain.set(simulationDomain);
uni.values.maxdt.set([maxdt]);
uni.values.inflowV.set([0, 0]);
uni.values.gamma.set([gamma]);
uni.values.K_p.set([K_p]);
uni.values.K_u.set([K_u]);
uni.values.inPressure.set([inPressure]);
uni.values.inRho.set([inRho]);
uni.values.cflFactor.set([1.5]);
uni.values.muscl.set([1.0]);
uni.values.contourCompression.set([1.001]);
uni.values.visMultiplier.set([1.0]);

gui.updateAllVisibility();

main()
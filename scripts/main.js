
let adapter, device;
let gpuInfo = false;

async function main() {

  if (device) device.destroy();

  adapter = await navigator.gpu?.requestAdapter();

  const maxComputeInvocationsPerWorkgroup = adapter.limits.maxComputeInvocationsPerWorkgroup;
  const maxBufferSize = adapter.limits.maxBufferSize;
  const maxStorageBufferBindingSize = adapter.limits.maxStorageBufferBindingSize;
  const f32filterable = adapter.features.has("float32-filterable");
  const shaderf16 = adapter.features.has("shader-f16");
  const subgroups = adapter.features.has("subgroups");
  
  if (!shaderf16 && !gpuInfo) {
  }

  const floatPrecision = shaderf16 ? 16 : 32;
  const f16header = shaderf16 ? `
enable f16;
// alias vec4h = vec4<f${floatPrecision}>;
// alias vec3h = vec3<f${floatPrecision}>;
// alias vec2h = vec2<f${floatPrecision}>;
` : "";

  const textureTier1 = adapter.features.has("texture-formats-tier1");
  if (!textureTier1 && !gpuInfo) alert("texture-formats-tier1 feature required");
  const textureTier2 = adapter.features.has("texture-formats-tier2");
  if (!textureTier2 && !gpuInfo) alert("texture-formats-tier2 unsupported, may reduce performance");

  // compute workgroup size 32^2 = 1024 threads if maxComputeInvocationsPerWorkgroup >= 1024, otherwise 16^2 = 256 threads
  const largeWg = maxComputeInvocationsPerWorkgroup >= 1024;
  const [wg_x, wg_y] = largeWg ? [32, 32] : [16, 16];

  if (!gpuInfo) {
    gui.addGroup("deviceInfo", "Device info", `
<pre><span ${!largeWg ? "class='warn'" : ""}>maxComputeInvocationsPerWorkgroup: ${maxComputeInvocationsPerWorkgroup}
workgroup: [${wg_x}, ${wg_y}]</span>
maxBufferSize: ${maxBufferSize}
maxStorageBufferBindingSize: ${maxStorageBufferBindingSize}
f32filterable: ${f32filterable}
shader-f16: ${shaderf16}
subgroups: ${subgroups}
texture-formats-tier1: ${textureTier1}
<span ${!textureTier2 ? "class='warn'" : ""}>texture-formats-tier2: ${textureTier2}</span>
</pre>
    `);
    gpuInfo = true;
  }


  device = await adapter?.requestDevice({
    requiredFeatures: [
      ...(adapter.features.has("timestamp-query") ? ["timestamp-query"] : []),
      ...(f32filterable ? ["float32-filterable"] : []),
      ...(textureTier1 ? ["texture-formats-tier1"] : []),
      ...(textureTier2 ? ["texture-formats-tier2"] : []),
      ...(shaderf16 ? ["shader-f16"] : []),
      ...(subgroups ? ["subgroups"] : []),
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
      hardReset();
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
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC,
    label: "maxWaveSpeed buffer"
  });
  device.queue.writeBuffer(storage.maxWaveSpeed, 0, new Float32Array([23000]));

  // const stagingBuffer = device.createBuffer({
  //   size: 4,
  //   usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
  // });


  const uniformBuffer = uni.createBuffer(device);

  const newComputePipeline = (shaderCode, name, layout = "auto") =>
    device.createComputePipeline({
      layout: layout,
      compute: {
        module: device.createShaderModule({
          code: f16header + shaderCode, //(floatPrecision, textureTier2 ? floatPrecision : 32),
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
      { binding: 1, resource: storage.gridPoints0.createView() },
      { binding: 2, resource: storage.gridPoints1.createView() },
      { binding: 3, resource: storage.gridBoundaries.createView() },
    ],
    label: "grid interpolation compute bind group"
  });

  const gridEllipticPoissonComputePipeline = newComputePipeline(gridEllipticPoissonShaderCode, "grid elliptic poisson");
  const gridEllipticPoissonBindGroup = (texIn, texOut) => device.createBindGroup({
    layout: gridEllipticPoissonComputePipeline.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: { buffer: uniformBuffer } },
      { binding: 1, resource: texIn.createView() },
      { binding: 2, resource: texOut.createView() },
      { binding: 3, resource: storage.gridBoundaries.createView() },
    ],
    label: "grid elliptic poisson compute bind group"
  });
  const gridEllipticPoissonBindGroups = [
    gridEllipticPoissonBindGroup(storage.gridPoints1, storage.gridPoints0),
    gridEllipticPoissonBindGroup(storage.gridPoints0, storage.gridPoints1),
  ];

  const gridFinalizeComputePipeline = newComputePipeline(gridFinalizeShaderCode, "grid finalize");
  const gridFinalizeBindGroup = (tex) => device.createBindGroup({
    layout: gridFinalizeComputePipeline.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: { buffer: uniformBuffer } },
      { binding: 1, resource: tex.createView() },
      { binding: 2, resource: storage.gridBoundaries.createView() },
      { binding: 3, resource: storage.gridArea.createView() },
      { binding: 4, resource: storage.faceLengths.createView() },
      { binding: 5, resource: storage.cellDistances.createView() },
    ],
    label: "grid finalize compute bind group"
  });
  const gridFinalizeBindGroups = [
    gridFinalizeBindGroup(storage.gridPoints1),
    gridFinalizeBindGroup(storage.gridPoints0),
  ];
  const prepareStateComputePipeline = newComputePipeline(prepareStateShaderCode, "prepare state");
  const prepareStateBindGroup = device.createBindGroup({
    layout: prepareStateComputePipeline.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: { buffer: uniformBuffer } },
      { binding: 1, resource: storage.state0.createView() },
    ],
    label: "prepare state compute bind group"
  });

  const boundaryComputePipeline = newComputePipeline(boundaryShaderCode, "boundary condition");
  const boundaryBindGroup = (stateIn, stateOut) => device.createBindGroup({
    layout: boundaryComputePipeline.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: { buffer: uniformBuffer } },
      { binding: 1, resource: storage.gridPoints0.createView() },
      { binding: 2, resource: storage.gridBoundaries.createView() },
      { binding: 3, resource: stateIn.createView() },
      { binding: 4, resource: stateOut.createView() },
    ],
    label: "boundary compute bind group"
  });
  const boundaryBindGroups = [
    boundaryBindGroup(storage.state2, storage.state0),
    boundaryBindGroup(storage.state1, storage.state2),
    boundaryBindGroup(storage.state2, storage.state1),
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
  const verticalFluxBindGroup = (state) => device.createBindGroup({
    layout: fluxBindGroupLayout,
    entries: [
      { binding: 0, resource: { buffer: uniformBuffer } },
      { binding: 1, resource: storage.gridPoints0.createView() },
      { binding: 2, resource: storage.gridBoundaries.createView() },
      { binding: 3, resource: state.createView() },
      { binding: 4, resource: storage.fluxY.createView() },
    ],
    label: "vertical flux compute bind group"
  });
  const verticalFluxBindGroups = [
    verticalFluxBindGroup(storage.state0),
    verticalFluxBindGroup(storage.state2),
    verticalFluxBindGroup(storage.state1),
  ];

  const horizontalFluxBindGroup = (state) => device.createBindGroup({
    layout: fluxBindGroupLayout,
    entries: [
      { binding: 0, resource: { buffer: uniformBuffer } },
      { binding: 1, resource: storage.gridPoints0.createView() },
      { binding: 2, resource: storage.gridBoundaries.createView() },
      { binding: 3, resource: state.createView() },
      { binding: 4, resource: storage.fluxX.createView() },
    ],
    label: "horizontal flux compute bind group"
  });
  const horizontalFluxBindGroups = [
    horizontalFluxBindGroup(storage.state0),
    horizontalFluxBindGroup(storage.state2),
    horizontalFluxBindGroup(storage.state1),
  ];

  const residualComputePipeline = newComputePipeline(residualShaderCode, "residual compute");
  const residualBindGroup = device.createBindGroup({
    layout: residualComputePipeline.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: { buffer: uniformBuffer } },
      { binding: 1, resource: storage.fluxX.createView() },
      { binding: 2, resource: storage.fluxY.createView() },
      { binding: 3, resource: storage.residual.createView() },
      { binding: 4, resource: storage.faceLengths.createView() },
      { binding: 5, resource: storage.gridArea.createView() },
    ],
    label: "residual compute bind group"
  });

  const integrationStage1ComputePipeline = newComputePipeline(integrationStage1ShaderCode, "integration stage 1");
  const integrationStage1BindGroup = device.createBindGroup({
    layout: integrationStage1ComputePipeline.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: { buffer: uniformBuffer } },
      { binding: 1, resource: storage.residual.createView()},
      { binding: 2, resource: storage.state0.createView() },
      { binding: 3, resource: storage.state1.createView() },
      { binding: 4, resource: { buffer: storage.maxWaveSpeed } },
    ],
    label: "integration stage 1 compute bind group"
  });

  const integrationStage2ComputePipeline = newComputePipeline(integrationStage2ShaderCode, "integration stage 2");
  const integrationStage2BindGroup = device.createBindGroup({
    layout: integrationStage2ComputePipeline.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: { buffer: uniformBuffer } },
      { binding: 1, resource: storage.residual.createView()},
      { binding: 2, resource: storage.state0.createView() },
      { binding: 3, resource: storage.state1.createView() },
      { binding: 4, resource: storage.state2.createView() },
      { binding: 5, resource: { buffer: storage.maxWaveSpeed } },
    ],
    label: "integration stage 2 compute bind group"
  });

  const integrationStage3ComputePipeline = newComputePipeline(integrationStage3ShaderCode, "integration stage 3");
  const integrationStage3BindGroup = device.createBindGroup({
    layout: integrationStage3ComputePipeline.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: { buffer: uniformBuffer } },
      { binding: 1, resource: storage.residual.createView()},
      { binding: 2, resource: storage.state0.createView() },
      { binding: 3, resource: storage.state1.createView() },
      { binding: 4, resource: storage.state2.createView() },
      { binding: 5, resource: { buffer: storage.maxWaveSpeed } },
    ],
    label: "integration stage 3 compute bind group"
  });

  const visualizationComputePipeline = newComputePipeline(visualizationShaderCode, "visualization");
  const visualizationBindGroup = device.createBindGroup({
    layout: visualizationComputePipeline.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: { buffer: uniformBuffer } },
      { binding: 1, resource: storage.state2.createView() },
      { binding: 2, resource: storage.faceLengths.createView() },
      { binding: 3, resource: storage.gridArea.createView() },
      { binding: 4, resource: storage.vis.createView() },
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
  });
  
  const newRenderPipeline = (topology) => device.createRenderPipeline({
    label: `${topology} rendering pipeline`,
    layout: device.createPipelineLayout({
      bindGroupLayouts: [ renderBindGroupLayout ],
    }),
    vertex: { module: renderModule },
    fragment: {
      module: renderModule,
      targets: [{ format: swapChainFormat }],
      constants: {}
    },
    primitive: {
      topology: topology,
    },
  });
  const renderPipelines = [
    newRenderPipeline("triangle-strip"),
    newRenderPipeline("line-strip"),
    newRenderPipeline("point-list"),
  ]

  const renderBindGroup = device.createBindGroup({
    layout: renderBindGroupLayout, //renderPipeline.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: { buffer: uniformBuffer } },
      { binding: 1, resource: storage.gridPoints0.createView() },
      { binding: 2, resource: storage.vis.createView() },
      { binding: 3, resource: gridSampler },
    ],
  });

  const renderPassDescriptor = {
    label: 'render pass',
    colorAttachments: [
      {
        clearValue: [0.1, 0.1, 0.1, 1],
        loadOp: 'clear',
        storeOp: 'store',
      },
    ]
  };
  const filterStrength = 50;

  const renderTimingHelper = new TimingHelper(device);
  const postprocessingTimingHelper = new TimingHelper(device);
  const reductionTimingHelper = new TimingHelper(device);

  const wgDispatchSize = (texSize) => [
    Math.ceil(texSize[0] / wg_x),
    Math.ceil(texSize[1] / wg_y)
  ];


  function createComputePass(pass, pipeline, bindGroup, dispatchSize = wgDispatchSize(simulationDomain)) {
    pass.setPipeline(pipeline);
    pass.setBindGroup(0, bindGroup);
    pass.dispatchWorkgroups(...dispatchSize);
    pass.end();
  }

  // get target frame time by measuring the time of two consecutive frames
  let targetFrameTime = 0;
  requestAnimationFrame((timestamp) => {
    let frameStartTime = timestamp;
    requestAnimationFrame((timestamp) => {
      targetFrameTime = timestamp - frameStartTime - 2; // subtract 2ms for overhead
    });
  });

  updateSolver = (solver) => {
    [verticalFluxComputePipeline, horizontalFluxComputePipeline] = fluxPipelines[solver];
  }

  prepareState = () => {
    uni.update(device.queue);
    const encoder = device.createCommandEncoder();

    createComputePass(encoder.beginComputePass(), prepareStateComputePipeline, prepareStateBindGroup, wgDispatchSize(simulationDomain));
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
    createComputePass(encoder.beginComputePass(), gridInterpolationComputePipeline, gridInterpolationBindGroup, wgDispatchSize(gridVertexCount));
    encoder.copyTextureToTexture(
      { texture: storage.gridPoints1 },
      { texture: storage.gridPoints0 },
      gridVertexCount
    );
    // iteratively solve elliptic Poisson equation to smooth the grid
    for (let i = 0; i < maxPoissonIterations; i++) {
      createComputePass(encoder.beginComputePass(), gridEllipticPoissonComputePipeline, gridEllipticPoissonBindGroups[pingPongIndex], wgDispatchSize(gridVertexCount));
      pingPongIndex = 1 - pingPongIndex;
    }
    poissonIterations = maxPoissonIterations;

    // finalize grid by computing cell areas, face lengths, and cell distances
    createComputePass(encoder.beginComputePass(), gridFinalizeComputePipeline, gridFinalizeBindGroups[pingPongIndex], wgDispatchSize(simulationDomain));
    // pingPongIndex = 1 - pingPongIndex;

    device.queue.submit([encoder.finish()]);
    prepareState();
  }

  prepareGrid();


  function render() {
    // update performance info
    const startTime = performance.now();
    deltaTime += Math.min(startTime - lastFrameTime - deltaTime, 1e4) / filterStrength;
    fps += (1e3 / deltaTime - fps) / filterStrength;

    // adaptive time stepping: adjust stepsPerFrame based on how long the last frame took compared to the target frame time
    const timeDifference = (startTime - lastFrameTime) - targetFrameTime;
    stepsPerFrame = Math.max(1, Math.round(stepsPerFrame - Math.min(100, timeDifference) * 0.05));
    if (timeDifference < -1) stepsPerFrame += 1;
    lastFrameTime = startTime;

    const canvasTexture = context.getCurrentTexture();
    renderPassDescriptor.colorAttachments[0].view = canvasTexture.createView();

    const encoder = device.createCommandEncoder();

    uni.update(device.queue);

    // simulate
    if (dt > 0) {
      for (let step = 0; step < stepsPerFrame; step++) {
        // state2 -> state0 (Qn)
        encoder.copyTextureToTexture(
          { texture: storage.state2 },
          { texture: storage.state0 },
          totalCellCount
        );
        createComputePass(encoder.beginComputePass(), boundaryComputePipeline, boundaryBindGroups[0], wgDispatchSize(totalCellCount));
        // state0 -> fluxY
        createComputePass(encoder.beginComputePass(), verticalFluxComputePipeline, verticalFluxBindGroups[0], wgDispatchSize(yFluxTexSize));
        // state0 -> fluxX
        createComputePass(encoder.beginComputePass(), horizontalFluxComputePipeline, horizontalFluxBindGroups[0], wgDispatchSize(xFluxTexSize));
        // fluxX, fluxY -> residual
        createComputePass(encoder.beginComputePass(), residualComputePipeline, residualBindGroup, wgDispatchSize(simulationDomain));
        // state0 (Qn) + residual -> state1 (Q1)
        createComputePass(encoder.beginComputePass(), integrationStage1ComputePipeline, integrationStage1BindGroup, wgDispatchSize(simulationDomain));

        // state1 -> state2 (Q1)
        encoder.copyTextureToTexture(
          { texture: storage.state1 },
          { texture: storage.state2 },
          totalCellCount
        );
        createComputePass(encoder.beginComputePass(), boundaryComputePipeline, boundaryBindGroups[1], wgDispatchSize(totalCellCount));
        // state2 -> fluxY
        createComputePass(encoder.beginComputePass(), verticalFluxComputePipeline, verticalFluxBindGroups[1], wgDispatchSize(yFluxTexSize));
        // state2 -> fluxX
        createComputePass(encoder.beginComputePass(), horizontalFluxComputePipeline, horizontalFluxBindGroups[1], wgDispatchSize(xFluxTexSize));
        // fluxX, fluxY -> residual
        createComputePass(encoder.beginComputePass(), residualComputePipeline, residualBindGroup, wgDispatchSize(simulationDomain));
        // state0 (Qn), state1 (Q1) + residual -> state2 (Q2)
        createComputePass(encoder.beginComputePass(), integrationStage2ComputePipeline, integrationStage2BindGroup, wgDispatchSize(simulationDomain));

        // state2 -> state1 (Q2)
        encoder.copyTextureToTexture(
          { texture: storage.state2 },
          { texture: storage.state1 },
          totalCellCount
        );
        createComputePass(encoder.beginComputePass(), boundaryComputePipeline, boundaryBindGroups[2], wgDispatchSize(totalCellCount));
        // state1 -> fluxY
        createComputePass(encoder.beginComputePass(), verticalFluxComputePipeline, verticalFluxBindGroups[2], wgDispatchSize(yFluxTexSize));
        // state1 -> fluxX
        createComputePass(encoder.beginComputePass(), horizontalFluxComputePipeline, horizontalFluxBindGroups[2], wgDispatchSize(xFluxTexSize));
        // fluxX, fluxY -> residual
        createComputePass(encoder.beginComputePass(), residualComputePipeline, residualBindGroup, wgDispatchSize(simulationDomain));
        // state0 (Qn), state1 (Q2) + residual -> state2 (Qn+1)
        createComputePass(encoder.beginComputePass(), integrationStage3ComputePipeline, integrationStage3BindGroup, wgDispatchSize(simulationDomain));
      }
      
      // update inflow velocity, will be 1 frame behind
      actualInflowVel += (inflowVel - actualInflowVel) / (velRampUpStrength * stepsPerFrame / 50);
      const inflowFinal = [actualInflowVel * xyAoA[0], actualInflowVel * xyAoA[1]];
      uni.values.inflowV.set(inflowFinal);
      // compute inflow state
      const rhoE = inPressure / (gamma - 1.0) + 0.5 * (actualInflowVel * actualInflowVel) * inRho;
      uni.values.inState.set([inRho, inflowFinal[0] * inRho, inflowFinal[1] * inRho, rhoE]);
    }

    createComputePass(postprocessingTimingHelper.beginComputePass(encoder), visualizationComputePipeline, visualizationBindGroup, wgDispatchSize(simulationDomain));
    createComputePass(reductionTimingHelper.beginComputePass(encoder), cflReductionComputePipeline, cflReductionBindGroup, [Math.ceil(simulationDomain[0] * simulationDomain[1] / 256), 1]);
    // encoder.copyBufferToBuffer(
    //   storage.maxWaveSpeed,          // Source buffer
    //   0,                  // Source offset
    //   stagingBuffer,      // Destination buffer
    //   0,                  // Destination offset
    //   4          // Size to copy
    // );

    const renderPass = renderTimingHelper.beginRenderPass(encoder, renderPassDescriptor);
    renderPass.setPipeline(renderPipelines[gridDisplayMode]);
    renderPass.setBindGroup(0, renderBindGroup);
    renderPass.draw(numVertices);
    renderPass.end();

    const commandBuffer = encoder.finish();
    device.queue.submit([commandBuffer]);

    // await stagingBuffer.mapAsync(GPUMapMode.READ);
    // const copyArrayBuffer = stagingBuffer.getMappedRange();
    // const data = copyArrayBuffer.slice(0, 4);
    // stagingBuffer.unmap();
    // const resultValues = new Float32Array(data);
    // console.log(resultValues);

    // if (run) {
    //   computeTimingHelper.getResult().then(gpuTime => computeTime += (gpuTime / 1e6 - computeTime) / filterStrength);
    // } else {
    //   computeTime = 0;
    // }
    renderTimingHelper.getResult().then(gpuTime => renderTime += (gpuTime - renderTime) / filterStrength);
    postprocessingTimingHelper.getResult().then(gpuTime => postprocessingTime += (gpuTime - postprocessingTime) / filterStrength);
    reductionTimingHelper.getResult().then(gpuTime => cflTime += (gpuTime - cflTime) / filterStrength);

    gui.io.mach(actualInflowVel / Math.sqrt(gamma * inPressure / inRho));

    jsTime += (performance.now() - startTime - jsTime) / filterStrength;

    rafId = requestAnimationFrame(render);
    // rafId = setTimeout(render, 1000);
  }

  perfIntId = setInterval(() => {
    gui.io.fps(fps);
    gui.io.jsTime(jsTime);
    gui.io.frameTime(deltaTime);
    // gui.io.computeTime();
    gui.io.postTime(postprocessingTime / 1e6);
    gui.io.cflTime(cflTime / 1e6);
    gui.io.renderTime(renderTime / 1e6);
    gui.io.poissonIterations(poissonIterations);
    gui.io.stepsPerFrame(stepsPerFrame);
  }, 100);


  rafId = requestAnimationFrame(render);
}

uni.values.zoom.set([1.0]);
uni.values.pan.set([0.0, 0.0]);
uni.values.simDomain.set(simulationDomain);
uni.values.dt.set([dt]);
uni.values.inflowV.set([0, 0]);
uni.values.gamma.set([gamma]);
uni.values.K_p.set([K_p]);
uni.values.K_u.set([K_u]);
uni.values.inPressure.set([inPressure]);
uni.values.inRho.set([inRho]);
uni.values.cflFactor.set([2.0]);
uni.values.muscl.set([1.0]);
uni.values.contourCompression.set([1.001]);
uni.values.visMultiplier.set([1.0]);


main()
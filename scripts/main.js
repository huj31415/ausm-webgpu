
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

  storage.state0 = new2dTexture("state0", totalCellCount, `rgba32float`, true);
  storage.state1 = new2dTexture("state1", totalCellCount, `rgba32float`, true);
  storage.state2 = new2dTexture("state2", totalCellCount, `rgba32float`, true);

  storage.fluxX = new2dTexture("fluxX", xFluxTexSize, `rgba32float`);
  storage.fluxY = new2dTexture("fluxY", yFluxTexSize, `rgba32float`);
  storage.residual = new2dTexture("residual", simulationDomain, `rgba32float`);

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


  const uniformBuffer = uni.createBuffer(device);

  const newComputePipeline = (shaderCode, name) =>
    device.createComputePipeline({
      layout: 'auto',
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

  const stateInitComputePipeline = newComputePipeline(stateInitShaderCode, "state init");
  const stateInitBindGroup = device.createBindGroup({
    layout: stateInitComputePipeline.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: { buffer: uniformBuffer } },
      { binding: 1, resource: storage.state0.createView() },
    ],
    label: "state init compute bind group"
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

  const verticalFluxComputePipeline = newComputePipeline(verticalFluxShaderCode, "vertical flux");
  const verticalFluxBindGroup = (state) => device.createBindGroup({
    layout: verticalFluxComputePipeline.getBindGroupLayout(0),
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

  const horizontalFluxComputePipeline = newComputePipeline(horizontalFluxShaderCode, "horizontal flux");
  const horizontalFluxBindGroup = (state) => device.createBindGroup({
    layout: horizontalFluxComputePipeline.getBindGroupLayout(0),
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
      { binding: 4, resource: storage.gridPoints0.createView() },
    ],
    label: "residual compute bind group"
  });

  const integrationStage1ComputePipeline = newComputePipeline(integrationStage1ShaderCode, "integration stage 1");
  const integrationStage1BindGroup = (stateIn, stateOut) => device.createBindGroup({
    layout: integrationStage1ComputePipeline.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: { buffer: uniformBuffer } },
      { binding: 1, resource: storage.residual.createView()},
      { binding: 2, resource: stateIn.createView() },
      { binding: 3, resource: stateOut.createView() },
    ],
    label: "integration stage 1 compute bind group"
  });
  const integrationStage1BindGroups = [
    integrationStage1BindGroup(storage.state0, storage.state1), // state1 = Qn, state0 = Q1
  ];

  const integrationStage2ComputePipeline = newComputePipeline(integrationStage2ShaderCode, "integration stage 2");
  const integrationStage2BindGroup = (stateIn, stateIn1, stateOut) => device.createBindGroup({
    layout: integrationStage2ComputePipeline.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: { buffer: uniformBuffer } },
      { binding: 1, resource: storage.residual.createView()},
      { binding: 2, resource: stateIn.createView() },
      { binding: 3, resource: stateIn1.createView() },
      { binding: 4, resource: stateOut.createView() },
    ],
    label: "integration stage 2 compute bind group"
  });
  const integrationStage2BindGroups = [
    integrationStage2BindGroup(storage.state0, storage.state1, storage.state2), // state0 = Qn, state1 = Q1, state2 = Q2
  ];

  const integrationStage3ComputePipeline = newComputePipeline(integrationStage3ShaderCode, "integration stage 3");
  const integrationStage3BindGroup = (stateIn, stateIn2, stateOut) => device.createBindGroup({
    layout: integrationStage3ComputePipeline.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: { buffer: uniformBuffer } },
      { binding: 1, resource: storage.residual.createView()},
      { binding: 2, resource: stateIn.createView() },
      { binding: 3, resource: stateIn2.createView() },
      { binding: 4, resource: stateOut.createView() },
    ],
    label: "integration stage 3 compute bind group"
  });
  const integrationStage3BindGroups = [
    integrationStage3BindGroup(storage.state0, storage.state1, storage.state2), // state0 = Qn, state1 = Q2, state2 = Qn+1
    // integrationStage3BindGroup(storage.state0, storage.state2, storage.state1),
  ];

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
        visibility: GPUShaderStage.VERTEX,
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

  const renderBindGroup = (tex) => device.createBindGroup({
    layout: renderBindGroupLayout, //renderPipeline.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: { buffer: uniformBuffer } },
      { binding: 1, resource: tex.createView() },
      { binding: 2, resource: storage.state2.createView() },
      // { binding: 2, resource: storage.fluxX.createView() },
      { binding: 3, resource: gridSampler },
    ],
  });

  const renderBindGroups = [
    renderBindGroup(storage.gridPoints0),
    renderBindGroup(storage.gridPoints1),
  ];

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

  const wgDispatchSize = (texSize) => [
    Math.ceil(texSize[0] / wg_x),
    Math.ceil(texSize[1] / wg_y)
  ];

  let pingPongIndex = 0;

  function createComputePass(pass, pipeline, bindGroup, dispatchSize = wgDispatchSize(simulationDomain)) {
    pass.setPipeline(pipeline);
    pass.setBindGroup(0, bindGroup);
    pass.dispatchWorkgroups(...dispatchSize);
    pass.end();
  }


  function render() {
    const startTime = performance.now();
    deltaTime += Math.min(startTime - lastFrameTime - deltaTime, 1e4) / filterStrength;
    fps += (1e3 / deltaTime - fps) / filterStrength;
    lastFrameTime = startTime;

    const canvasTexture = context.getCurrentTexture();
    renderPassDescriptor.colorAttachments[0].view = canvasTexture.createView();

    actualInflowVel += (inflowVel - actualInflowVel) / velRampUpStrength;

    uni.values.inflowV.set([actualInflowVel * xyAoA[0], actualInflowVel * xyAoA[1]]);

    uni.update(device.queue);

    const encoder = device.createCommandEncoder();

    const run = dt > 0;

    // move out of render loop
    if (prepareGrid) {
      createComputePass(encoder.beginComputePass(), gridInterpolationComputePipeline, gridInterpolationBindGroup, wgDispatchSize(gridVertexCount));
      encoder.copyTextureToTexture(
        { texture: storage.gridPoints1 },
        { texture: storage.gridPoints0 },
        gridVertexCount
      );
      prepareGrid = false;
    }
    if (runPoisson && poissonIterations < maxPoissonIterations) {
      for (let i = 0; i < poissonIterationsPerFrame; i++) {
        createComputePass(encoder.beginComputePass(), gridEllipticPoissonComputePipeline, gridEllipticPoissonBindGroups[pingPongIndex], wgDispatchSize(gridVertexCount));
        pingPongIndex = 1 - pingPongIndex;
      }
      poissonIterations += poissonIterationsPerFrame;
    }
    if (poissonIterations >= maxPoissonIterations && !gridFinalized) {
      gridFinalized = true;
      createComputePass(encoder.beginComputePass(), stateInitComputePipeline, stateInitBindGroup, wgDispatchSize(simulationDomain));
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
      pingPongIndex = 1 - pingPongIndex;
    }
    if (gridFinalized && run) {
      for (let step = 0; step < stepsPerFrame; step++) {
        // state2 -> state0 (Qn)
        createComputePass(encoder.beginComputePass(), boundaryComputePipeline, boundaryBindGroups[0], wgDispatchSize(totalCellCount));
        // state0 -> fluxY
        createComputePass(encoder.beginComputePass(), verticalFluxComputePipeline, verticalFluxBindGroups[0], wgDispatchSize(yFluxTexSize));
        // state0 -> fluxX
        createComputePass(encoder.beginComputePass(), horizontalFluxComputePipeline, horizontalFluxBindGroups[0], wgDispatchSize(xFluxTexSize));
        // fluxX, fluxY -> residual
        createComputePass(encoder.beginComputePass(), residualComputePipeline, residualBindGroup, wgDispatchSize(simulationDomain));
        // state0 (Qn) + residual -> state1 (Q1)
        createComputePass(encoder.beginComputePass(), integrationStage1ComputePipeline, integrationStage1BindGroups[0], wgDispatchSize(simulationDomain));

        // state1 -> state2 (Q1)
        createComputePass(encoder.beginComputePass(), boundaryComputePipeline, boundaryBindGroups[1], wgDispatchSize(totalCellCount));
        // state2 -> fluxY
        createComputePass(encoder.beginComputePass(), verticalFluxComputePipeline, verticalFluxBindGroups[1], wgDispatchSize(yFluxTexSize));
        // state2 -> fluxX
        createComputePass(encoder.beginComputePass(), horizontalFluxComputePipeline, horizontalFluxBindGroups[1], wgDispatchSize(xFluxTexSize));
        // fluxX, fluxY -> residual
        createComputePass(encoder.beginComputePass(), residualComputePipeline, residualBindGroup, wgDispatchSize(simulationDomain));
        // state0 (Qn), state1 (Q1) + residual -> state2 (Q2)
        createComputePass(encoder.beginComputePass(), integrationStage2ComputePipeline, integrationStage2BindGroups[0], wgDispatchSize(simulationDomain));

        // state2 -> state1 (Q2)
        createComputePass(encoder.beginComputePass(), boundaryComputePipeline, boundaryBindGroups[2], wgDispatchSize(totalCellCount));
        // state1 -> fluxY
        createComputePass(encoder.beginComputePass(), verticalFluxComputePipeline, verticalFluxBindGroups[2], wgDispatchSize(yFluxTexSize));
        // state1 -> fluxX
        createComputePass(encoder.beginComputePass(), horizontalFluxComputePipeline, horizontalFluxBindGroups[2], wgDispatchSize(xFluxTexSize));
        // fluxX, fluxY -> residual
        createComputePass(encoder.beginComputePass(), residualComputePipeline, residualBindGroup, wgDispatchSize(simulationDomain));
        // state0 (Qn), state1 (Q2) + residual -> state2 (Qn+1)
        createComputePass(encoder.beginComputePass(), integrationStage3ComputePipeline, integrationStage3BindGroups[0], wgDispatchSize(simulationDomain));
      }
    }

    const renderPass = renderTimingHelper.beginRenderPass(encoder, renderPassDescriptor);
    renderPass.setPipeline(renderPipelines[gridDisplayMode]);
    renderPass.setBindGroup(0, renderBindGroups[0]);
    renderPass.draw(numVertices);
    renderPass.end();

    const commandBuffer = encoder.finish();
    device.queue.submit([commandBuffer]);

    // if (run) {
    //   computeTimingHelper.getResult().then(gpuTime => computeTime += (gpuTime / 1e6 - computeTime) / filterStrength);
    // } else {
    //   computeTime = 0;
    // }
    renderTimingHelper.getResult().then(gpuTime => renderTime += (gpuTime / 1e6 - renderTime) / filterStrength);

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
    gui.io.renderTime(renderTime);
    gui.io.poissonIterations(poissonIterations);
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


main()
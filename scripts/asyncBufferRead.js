// thanks to claude
class AsyncBufferReader {
  constructor(device, bufferByteSize, poolSize = 3) {
    this.device = device;
    this.bufferByteSize = bufferByteSize;

    // Ring buffer of staging buffers
    this.stagingBuffers = Array.from({ length: poolSize }, () =>
      device.createBuffer({
        size: bufferByteSize,
        usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
      })
    );

    this.writeIndex = 0;   // next staging buffer to copy into
    this.readIndex = 0;    // next staging buffer to map & read
    this.pendingCopies = 0;// how many buffers have been copied but not yet read

    this.onValue = null;   // callback: (value: number) => void
  }

  // Call once per frame (or every N frames) from your render loop,
  // passing the commandEncoder you're already building this frame.
  recordCopy(commandEncoder, sourceBuffer) {
    const poolSize = this.stagingBuffers.length;
    if (this.pendingCopies >= poolSize) return; // pool exhausted, skip this frame

    const staging = this.stagingBuffers[this.writeIndex];
    commandEncoder.copyBufferToBuffer(sourceBuffer, 0, staging, 0, this.bufferByteSize);

    this.writeIndex = (this.writeIndex + 1) % poolSize;
    this.pendingCopies++;
  }

  // Call from a setInterval
  // Safe to call even if no copies are pending.
  async readLatest() {
    if (this.pendingCopies === 0) return;

    const staging = this.stagingBuffers[this.readIndex];

    // Non-blocking from the GPU's perspective - the GPU has already
    // finished writing to this buffer by the time we ask (it's the oldest).
    try { await staging.mapAsync(GPUMapMode.READ); } catch (e) {}
    const view = new Uint32Array(staging.getMappedRange());
    const value = view[0];
    staging.unmap();

    this.readIndex = (this.readIndex + 1) % this.stagingBuffers.length;
    this.pendingCopies--;

    return value;
  }
}
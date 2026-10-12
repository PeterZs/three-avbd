// Compare an optimization with an unchanged checkout on the same device, sequentially.
// node --experimental-transform-types scripts/compare3d-gpu.ts <baseline checkout>
import assert from 'node:assert/strict';
import { resolve, dirname } from 'node:path';
import { pathToFileURL, fileURLToPath } from 'node:url';
import { device, skip } from '../tests-gpu/device.ts';

if (!device) throw new Error(`GPU comparison unavailable: ${skip}`);
if (!process.argv[2]) throw new Error('Pass the baseline checkout directory');
const root = resolve(dirname(fileURLToPath(import.meta.url)), '..');
const load = async (dir: string) => {
  const module = (name: string) => import(pathToFileURL(resolve(dir, `src/avbd3d/${name}.ts`)).href);
  const [{ Solver }, { Rigid }, { GpuSolver3D }] = await Promise.all([module('ref/solver'), module('ref/body'), module('gpu/solver')]);
  return { Solver, Rigid, GpuSolver3D };
};
const baseline = await load(resolve(process.argv[2])), candidate = await load(root);
const median = (v: number[]) => [...v].sort((a, b) => a - b)[Math.floor(v.length / 2)]!;

async function measure(api: Awaited<ReturnType<typeof load>>, scene: string) {
  const ref = new api.Solver();
  const body = (size: number[], density: number, position: number[]) => new api.Rigid(ref, size, density, 0.5, position);
  body([60, 60, 1], 0, [0, 0, -0.5]);
  if (scene === 'mixed-sizes') {
    for (let i = 0; i < 160; i++) body([4, 4, 0.5], 0, [200 + (i % 16) * 5, Math.floor(i / 16) * 5, 0]);
    for (let i = 0; i < 4000; i++) body([0.45, 0.45, 0.45], 0, [(i % 80) * 0.5 - 20, Math.floor(i / 80) * 0.5 - 12.5, 0.225]);
    for (let i = 0; i < 512; i++) body([0.4, 0.4, 0.4], 1, [(i % 32) * 0.5 - 8, Math.floor(i / 32) * 0.5 - 4, 0.65]);
  } else if (scene === 'fixed-heavy') {
    for (let i = 0; i < 10000; i++) body([0.9, 0.9, 0.9], 0, [(i % 100) - 50, Math.floor(i / 100) - 50, 0.45]);
    for (let i = 0; i < 512; i++) body([0.8, 0.8, 0.8], 1, [(i % 32) - 16, Math.floor(i / 32) - 8, 1.3]);
  } else if (scene === 'pile') {
    for (let i = 0; i < 125; i++) body([0.95, 0.95, 0.95], 1, [(i % 5) - 2, Math.floor(i / 5) % 5 - 2, Math.floor(i / 25) + 0.5]);
  } else if (scene === 'dense-tail') {
    for (let i = 0; i < 1000; i++) body([0.1, 0.1, 0.1], 1, [(i % 40) - 20, Math.floor(i / 40) - 12, 3]);
  } else {
    body([1, 1, 1], 1, [0, 0, 8]);
    for (let i = 0; i < 32; i++) {
      const a = 2 * Math.PI * i / 32;
      body([0.15, 0.15, 0.15], 1, [2 * Math.cos(a), 2 * Math.sin(a), 8]);
    }
  }
  const gpu = new api.GpuSolver3D(device, ref, { spatialSort: false });
  if (scene === 'hub' || scene === 'sparse-hub') gpu.appendJoints(Array.from({ length: 32 }, (_, i) => {
    const a = 2 * Math.PI * i / 32;
    return { a: 1, b: i + 2, rA: [2 * Math.cos(a), 2 * Math.sin(a), 0], rB: [0, 0, 0] };
  }));
  if (scene === 'sparse-hub') gpu.fixedColors = Uint32Array.from({ length: 34 }, (_, i) => i === 0 ? 255 : i + 30);
  if (scene === 'dense-tail') gpu.fixedColors = Uint32Array.from({ length: 1001 }, (_, i) => i === 0 ? 255 : 63);
  try {
    for (let i = 0; i < 60; i++) {
      // Exercise uniform reuse through iteration, alpha and cap growth/shrink changes.
      gpu.params.iterations = [0, 4, 14][Math.floor(i / 10) % 3]!;
      gpu.params.alpha = i < 30 ? 0.95 : 0.8;
      gpu.colorCap = i < 20 ? 12 : i < 40 ? 64 : 16;
      gpu.step();
    }
    gpu.params.iterations = 14;
    gpu.params.alpha = 0.95;
    gpu.colorCap = 64;
    await device!.queue.onSubmittedWorkDone();
    const states = new Float32Array(await gpu.readBodies());
    gpu.adapt(await gpu.readCounters());
    const started = performance.now();
    for (let i = 0; i < 120; i++) gpu.step();
    await device!.queue.onSubmittedWorkDone();
    const wallMs = (performance.now() - started) / 120;
    const profiles: Record<string, number>[] = [];
    if (device!.features.has('timestamp-query')) for (let i = 0; i < 9; i++) {
      const profile = new Promise<Record<string, number>>(resolve => gpu.profileNextStep(resolve));
      gpu.step();
      profiles.push(await profile);
    }
    const counters = await gpu.readCounters();
    assert.equal(counters.overflow, 0, `${scene}: capacity overflow invalidates comparison`);
    return { states, wallMs, gpu: Object.fromEntries(Object.keys(profiles[0] ?? {}).map(k => [k, median(profiles.map(p => p[k]!))])), counters };
  } finally { gpu.destroy(); }
}

for (const scene of (process.argv[3] ? [process.argv[3]] : ['fixed-heavy', 'pile', 'hub', 'sparse-hub', 'dense-tail', 'mixed-sizes'])) {
  const a = await measure(baseline, scene), b = await measure(candidate, scene);
  let differing = 0, maxDifference = 0;
  assert.equal(a.states.length, b.states.length);
  for (let i = 0; i < a.states.length; i++) if (!Object.is(a.states[i], b.states[i])) {
    differing++; maxDifference = Math.max(maxDifference, Math.abs(a.states[i]! - b.states[i]!));
  }
  console.log(JSON.stringify({ scene, stepsCompared: 60, differing, maxDifference, baseline: { wallMs: a.wallMs, gpu: a.gpu, counters: a.counters }, candidate: { wallMs: b.wallMs, gpu: b.gpu, counters: b.counters } }));
  if (differing) process.exitCode = 1;
}

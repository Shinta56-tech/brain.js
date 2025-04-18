import { NeuralNetworkGPU } from './neural-network-gpu';
import { xorTrainingData } from './test-utils';

describe('NeuralNetworkGPU Class: End to End', () => {
  it('can learn xor', () => {
    const net = new NeuralNetworkGPU();
    const status = net.train(xorTrainingData, {
      iterations: Infinity,
      errorThresh: 0.01,
      activation: 'leaky-relu',
      //praxis: 'adamw',
      log: true,
      logPeriod: 1,
    });
    console.log(status);
    expect(status.error).toBeLessThanOrEqual(0.01);
    expect(status.iterations).toBeLessThanOrEqual(5000);
  });
});

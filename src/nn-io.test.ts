import { NeuralNetwork } from './neural-network';
import { NeuralNetworkGPU } from './neural-network-gpu';
import { xorTrainingData } from './test-utils';

describe('NeuralNetwork IO', () => {
  it('json', () => {
    const net = new NeuralNetworkGPU();
    const status = net.train(xorTrainingData, { log: true, logPeriod: 1, activation: "mish", praxis: "adamw", });
    console.log(status);
    const json = net.toJSON2();
    console.log(JSON.stringify(json));
    const net2 = new NeuralNetworkGPU();
    net2.fromJSON2(json);
    console.log(JSON.stringify(net2.toJSON2()));
    const input = xorTrainingData[2].input;
    const output = net2.run(input);
    console.log(input);
    console.log(output);
    const status2 = net2.train(xorTrainingData, { log: true, logPeriod: 1, activation: "mish", praxis: "adamw", });
    console.log(status2);
  });
});

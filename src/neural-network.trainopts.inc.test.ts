import { NeuralNetwork } from './neural-network';

const data = [
  { input: { a: 0 }, output: { c: 0 } },
  { input: { a: 0, b: 1 }, output: { c: 1 } },
  { input: { a: 1, b: 0 }, output: { c: 1 } },
  { input: { a: 1, b: 1 }, output: { c: 1, d: 0 } },
];

const data2 = [
  { input: { a: 0, x: 0 }, output: { c: 0 } },
  { input: { a: 0, b: 1 }, output: { c: 1 } },
  { input: { a: 1, b: 0 }, output: { c: 1 } },
  { input: { a: 1, b: 1 }, output: { c: 1, y: 0 } },
];

describe('NeuralNetwork.train()', () => {
  describe('train() options', () => {
    it('train', () => {
      const net = new NeuralNetwork();
      const res = net.train(data, { activation: 'mish', praxis: 'adamw' });
      const res2 = net.train(data2, { activation: 'mish', praxis: 'adamw' });
      console.log('res', res);
      console.log('res2', res2);
    });
  });
});

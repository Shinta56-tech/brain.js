import { INumberHash, ITrainingDatum } from './lookup';
import { NeuralNetworkCustom } from './neural-network-custom';

describe('NeuralNetworkCustom', () => {
  describe('train', () => {
    describe('フラットオブジェクト4入力1出力のトレーニング', () => {
      // prettier-ignore
      it('トレーニングが成功する', () => {
        const expect_toBetoween = (value: number, min: number, max: number) => {
          expect(value).toBeGreaterThanOrEqual(min);
          expect(value).toBeLessThanOrEqual(max);
        }
        let net;
        net = new NeuralNetworkCustom();
        const trainingOption = {
          logPeriod: 1,
          errorThresh: 0.000001,
        };
        const trainingData1 = [
          {
            input: { a: 1, b: 1, c: 1 },
            output: { e: 1 },
          },
          {
            input: { a: 0, b: 0, c: 0 },
            output: { e: 0 },
          },
        ];
        let state1 = net.train(trainingData1, trainingOption);
        console.log('state1', state1);
        expect((net.outputLookup as INumberHash)['e']).toBe(0);
        expect_toBetoween((net.run({ a: 0, b: 0, c: 0 }) as INumberHash)['e'] as number, 0, 0.1);
        expect_toBetoween((net.run({ a: 1, b: 1, c: 1 }) as INumberHash)['e'] as number, 0.95, 1);
        console.log((net.run({ a: 1, b: 1, c: 1 }) as INumberHash));
        const trainingData2 = [
          {
            input: { f: 1, g: 1 },
            output: { h: 1 },
          },
          {
            input: { f: 0, g: 0 },
            output: { h: 0 },
          },
          {
            input: { a: 1, b: 1, c: 1, f: 0, g: 0 },
            output: { e: 1, h: 0 },
          },
          {
            input: { a: 0, b: 0, c: 0, f: 1, g: 1 },
            output: { e: 0, h: 1 },
          },
          {
            input: { a: 1, b: 1, c: 1, f: 1, g: 1 },
            output: { e: 1, h: 1 },
          },
        ];
        let state2 = net.train(trainingData2, trainingOption);
        console.log('state2', state2);
        expect((net.outputLookup as INumberHash)['h']).toBe(1);
        let data;
        console.log(data = { a: 1, f: 1, g: 1 }, (net.run(data) as INumberHash));
        console.log(data = { a: 1, b: 1, c: 1 }, (net.run(data) as INumberHash));
        console.log(data = { a: 0, b: 0, c: 0, f: 1, g: 1  }, (net.run(data) as INumberHash));
        console.log(data = { a: 1, b: 1, c: 1, f: 0, g: 0  }, (net.run(data) as INumberHash));
        console.log(data = { a: 1, b: 1, c: 1, f: 1, g: 1  }, (net.run(data) as INumberHash));
      });
    });
  });
});

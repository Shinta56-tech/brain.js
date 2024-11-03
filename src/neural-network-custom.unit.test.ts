import { INumberHash, ITrainingDatum } from './lookup';
import { NeuralNetworkCustom } from './neural-network-custom';
import { NeuralNetwork } from './neural-network';

describe('NeuralNetworkCustom', () => {
  describe.skip('train', () => {
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
  describe('exportSON importJSON', () => {
    it(
      'ファイル出力',
      async () => {
        let net = new NeuralNetworkCustom();
        const trainingData = [
          {
            input: { a: 1, b: 1, c: 1 },
            output: { e: 1 },
          },
          {
            input: { a: 0, b: 0, c: 0 },
            output: { e: 0 },
          },
        ];
        let state;
        state = net.train(trainingData, {
          logPeriod: 1,
          errorThresh: 0.000001,
        });
        console.log('state', state);
        console.log(net.run({ a: 1, b: 1, c: 1 }));
        const toJSON = net.toJSON();
        console.log('toJSON', JSON.stringify(toJSON, null, 2));
        await net.exportJSON('dist/exportJSON.json');
        net = new NeuralNetworkCustom();
        await net.importJSON('dist/exportJSON.json');
        console.log('importJSON', JSON.stringify(net.toJSON(), null, 2));
        console.log('result importJSON', net.run({ a: 1, b: 1, c: 1 }));
        net = new NeuralNetworkCustom();
        net.fromJSON(toJSON);
        console.log('result toJSON', net.run({ a: 1, b: 1, c: 1 }));
      },
      60 * 1000
    );
  });
  describe.skip('importJSON', () => {
    it(
      'ファイル入力',
      async () => {
        let net = new NeuralNetworkCustom();
        await net.importJSON('dist/net.json');
      },
      30 * 60 * 1000
    );
  });
});

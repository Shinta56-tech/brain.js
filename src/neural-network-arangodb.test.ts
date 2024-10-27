import { NeuralNetworkArangoDB } from './neural-network-arangodb';

describe('NeuralNetworkArangoDB', () => {
  describe('main function', () => {
    describe('constructor check', () => {
      it('throws', () => {
        expect(() => {
          new NeuralNetworkArangoDB();
        });
      });
    });
  });
});

import { INumberHash } from './lookup';
import { INeuralNetworkState } from './neural-network-types';

export interface INeuralNetworkDatum<InputType, OutputType> {
  input: InputType;
  output: OutputType;
}

export type NeuralNetworkActivation =
  | 'sigmoid'
  | 'relu'
  | 'leaky-relu'
  | 'tanh';

export interface INeuralNetworkTrainOptions {
  activation: NeuralNetworkActivation | string;
  iterations: number;
  errorThresh: number;
  log: boolean | ((status: INeuralNetworkState) => void);
  logPeriod: number;
  leakyReluAlpha: number;
  learningRate: number;
  momentum: number;
  callback?: (status: { iterations: number; error: number }) => void;
  callbackPeriod: number;
  timeout: number;
  praxis?: 'adam';
  beta1: number;
  beta2: number;
  epsilon: number;
}

export interface INeuralNetworkOptions {
  inputSize: number;
  outputSize: number;
  binaryThresh: number;
  hiddenLayers?: number[];
}

export type INeuralNetworkData = number[] | Float32Array | Partial<INumberHash>;

export class NeuralNetworkArangoDB<
  InputType extends INeuralNetworkData,
  OutputType extends INeuralNetworkData
> {
  constructor(
    options: Partial<INeuralNetworkOptions & INeuralNetworkTrainOptions> = {}
  ) {}

  run(input: Partial<InputType>): OutputType {
    return {} as OutputType;
  }

  train(
    data: Array<INeuralNetworkDatum<Partial<InputType>, Partial<OutputType>>>,
    options: Partial<INeuralNetworkTrainOptions> = {}
  ): INeuralNetworkState {
    return {} as INeuralNetworkState;
  }
}

import { Thaw } from 'thaw.js';
import { ITrainingStatus } from './feed-forward';
import { INumberHash, lookup } from './lookup';
import {
  INeuralNetworkBinaryTestResult,
  INeuralNetworkState,
  INeuralNetworkTestResult,
} from './neural-network-types';
import { arrayToFloat32Array } from './utilities/cast';
import { LookupTable } from './utilities/lookup-table';
import { max } from './utilities/max';
import { mse } from './utilities/mse';
import { randos } from './utilities/randos';
import { zeros } from './utilities/zeros';
import fs from 'fs';
import { get } from 'http';
import { createSecureContext } from 'tls';

type NeuralNetworkFormatter =
  | ((v: INumberHash) => Float32Array)
  | ((v: number[]) => Float32Array);

export function getTypedArrayFn(
  value: INeuralNetworkData,
  table: INumberHash | null
): null | NeuralNetworkFormatter {
  if ((value as Float32Array).buffer instanceof ArrayBuffer) {
    return null;
  }
  if (Array.isArray(value)) {
    return arrayToFloat32Array;
  }
  if (!table) throw new Error('table is not Object');
  const { length } = Object.keys(table);
  return (v: INumberHash): Float32Array => {
    const array = new Float32Array(length);
    for (const p in table) {
      if (!table.hasOwnProperty(p)) continue;
      if (typeof v[p] !== 'number') continue;
      array[table[p]] = v[p] || 0;
    }
    return array;
  };
}

export type NeuralNetworkActivation =
  | 'sigmoid'
  | 'relu'
  | 'leaky-relu'
  | 'tanh';

export interface IJSONLayer {
  biases: number[];
  weights: number[][];
}

export interface INeuralNetworkJSON {
  type: string;
  sizes: number[];
  layers: IJSONLayer[];
  inputLookup: INumberHash | null;
  inputLookupLength: number;
  outputLookup: INumberHash | null;
  outputLookupLength: number;
  options: INeuralNetworkOptions;
  trainOpts: INeuralNetworkTrainOptionsJSON;
}

export interface INeuralNetworkOptions {
  inputSize: number;
  outputSize: number;
  binaryThresh: number;
  hiddenLayers?: number[];
}

export function defaults(): INeuralNetworkOptions {
  return {
    inputSize: 0,
    outputSize: 0,
    binaryThresh: 0.5,
  };
}

export interface INeuralNetworkTrainOptionsJSON {
  activation: NeuralNetworkActivation | string;
  iterations: number;
  errorThresh: number;
  log: boolean;
  logPeriod: number;
  leakyReluAlpha: number;
  learningRate: number;
  momentum: number;
  callbackPeriod: number;
  timeout: number | 'Infinity';
  praxis?: 'adam';
  beta1: number;
  beta2: number;
  epsilon: number;
}

export interface INeuralNetworkPreppedTrainingData<T> {
  status: ITrainingStatus;
  preparedData: Array<INeuralNetworkDatumFormatted<T>>;
  endTime: number;
}

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

export function trainDefaults(): INeuralNetworkTrainOptions {
  return {
    activation: 'sigmoid',
    iterations: 20000, // the maximum times to iterate the training data
    errorThresh: 0.005, // the acceptable error percentage from training data
    log: false, // true to use console.log, when a function is supplied it is used
    logPeriod: 10, // iterations between logging out
    leakyReluAlpha: 0.01,
    learningRate: 0.3, // multiply's against the input and the delta then adds to momentum
    momentum: 0.1, // multiply's against the specified "change" then adds to learning rate for change
    callbackPeriod: 10, // the number of iterations through the training data between callback calls
    timeout: Infinity, // the max number of milliseconds to train for
    beta1: 0.9,
    beta2: 0.999,
    epsilon: 1e-8,
  };
}

export type INeuralNetworkData = number[] | Float32Array | Partial<INumberHash>;

// TODO: should be replaced by ITrainingDatum
export interface INeuralNetworkDatum<InputType, OutputType> {
  input: InputType;
  output: OutputType;
}

export interface INeuralNetworkDatumFormatted<T> {
  input: T;
  output: T;
}

export class NeuralNetworkCustom<
  InputType extends INeuralNetworkData,
  OutputType extends INeuralNetworkData
> {
  options: INeuralNetworkOptions = defaults();
  trainOpts: INeuralNetworkTrainOptions = trainDefaults();
  sizes: number[] = [];
  outputLayer = -1;
  biases: Float32Array[] = [];
  weights: Float32Array[][] = []; // weights for bias nodes
  outputs: Float32Array[] = [];
  // state for training
  deltas: Float32Array[] = [];
  changes: Float32Array[][] = [];
  mse_sum = 0;
  mse_length = 0;

  errorCheckInterval = 1;

  inputLookup: INumberHash | null = null;
  inputLookupLength = 0;
  outputLookup: INumberHash | null = null;
  outputLookupLength = 0;

  _formatInput: NeuralNetworkFormatter | null = null;
  _formatOutput: NeuralNetworkFormatter | null = null;

  runInput: (input: Float32Array) => Float32Array = (input: Float32Array) => {
    this.setActivation();
    return this.runInput(input);
  };

  calculateDeltas: (output: Float32Array) => void = (
    output: Float32Array
  ): void => {
    this.setActivation();
    return this.calculateDeltas(output);
  };

  // adam
  biasChangesLow: Float32Array[] = [];
  biasChangesHigh: Float32Array[] = [];
  changesLow: Float32Array[][] = [];
  changesHigh: Float32Array[][] = [];
  iterations = 0;

  constructor(
    options: Partial<INeuralNetworkOptions & INeuralNetworkTrainOptions> = {}
  ) {
    this.options = { ...this.options, ...options };
    this.updateTrainingOptions(options);

    const { inputSize, hiddenLayers, outputSize } = this.options;
    if (inputSize && outputSize) {
      this.sizes = [inputSize].concat(hiddenLayers ?? []).concat([outputSize]);
    }
  }

  private initExtend(): void {
    function margeExtendArray(o: Float32Array, n: Float32Array): Float32Array {
      if (!o) {
        return n;
      }
      if (o.length < n.length) {
        const extendedArray = new Float32Array(n.length);
        extendedArray.set(o);
        extendedArray.set(n.slice(o.length), o.length);
        return extendedArray;
      } else {
        return o;
      }
    }
    function margeExtendArrayArray(
      o: Float32Array[],
      n: any[]
    ): Float32Array[] {
      if (!o) {
        return n;
      }
      if (o.length < n.length) {
        return o.concat(n.slice(o.length));
      } else {
        return o;
      }
    }
    // prettier-ignore
    for (let layerIndex = 0; layerIndex <= this.outputLayer; layerIndex++) {
      const size = this.sizes[layerIndex];
      this.deltas[layerIndex] = margeExtendArray(this.deltas[layerIndex], zeros(size));
      this.outputs[layerIndex] = zeros(size);
      if (layerIndex > 0) {
        this.biases[layerIndex] = margeExtendArray(this.biases[layerIndex], randos(size));
        this.weights[layerIndex] = margeExtendArrayArray(this.weights[layerIndex], new Array(size));
        this.changes[layerIndex] = margeExtendArrayArray(this.changes[layerIndex], new Array(size));
        for (let nodeIndex = 0; nodeIndex < size; nodeIndex++) {
          const prevSize = this.sizes[layerIndex - 1];
          this.weights[layerIndex][nodeIndex] = margeExtendArray(this.weights[layerIndex][nodeIndex], randos(prevSize));
          this.changes[layerIndex][nodeIndex] = margeExtendArray(this.changes[layerIndex][nodeIndex], zeros(prevSize));
        }
      }
    }
  }

  /**
   *
   * Expects this.sizes to have been set
   */
  initialize(): void {
    if (this.sizes.length < 0) {
      throw new Error('Sizes must be set before initializing');
    }

    if (this.outputLayer > 0) {
      this.initExtend();
      return;
    }

    this.outputLayer = this.sizes.length - 1;
    this.biases = new Array(this.outputLayer); // weights for bias nodes
    this.weights = new Array(this.outputLayer);
    this.outputs = new Array(this.outputLayer);

    // state for training
    this.deltas = new Array(this.outputLayer);
    this.changes = new Array(this.outputLayer); // for momentum

    for (let layerIndex = 0; layerIndex <= this.outputLayer; layerIndex++) {
      const size = this.sizes[layerIndex];
      this.deltas[layerIndex] = zeros(size);
      this.outputs[layerIndex] = zeros(size);

      if (layerIndex > 0) {
        this.biases[layerIndex] = randos(size);
        this.weights[layerIndex] = new Array(size);
        this.changes[layerIndex] = new Array(size);

        for (let nodeIndex = 0; nodeIndex < size; nodeIndex++) {
          const prevSize = this.sizes[layerIndex - 1];
          this.weights[layerIndex][nodeIndex] = randos(prevSize);
          this.changes[layerIndex][nodeIndex] = zeros(prevSize);
        }
      }
    }

    this.setActivation();
    if (this.trainOpts.praxis === 'adam') {
      this._setupAdam();
    }
  }

  setActivation(activation?: NeuralNetworkActivation): void {
    const value = activation ?? this.trainOpts.activation;
    switch (value) {
      case 'sigmoid':
        this.runInput = this._runInputSigmoid;
        this.calculateDeltas = this._calculateDeltasSigmoid;
        break;
      case 'relu':
        this.runInput = this._runInputRelu;
        this.calculateDeltas = this._calculateDeltasRelu;
        break;
      case 'leaky-relu':
        this.runInput = this._runInputLeakyRelu;
        this.calculateDeltas = this._calculateDeltasLeakyRelu;
        break;
      case 'tanh':
        this.runInput = this._runInputTanh;
        this.calculateDeltas = this._calculateDeltasTanh;
        break;
      default:
        throw new Error(
          `Unknown activation ${value}. Available activations are: 'sigmoid', 'relu', 'leaky-relu', 'tanh'`
        );
    }
  }

  get isRunnable(): boolean {
    return this.sizes.length > 0;
  }

  run(input: Partial<InputType>): OutputType {
    if (!this.isRunnable) {
      throw new Error('network not runnable');
    }
    let formattedInput: Float32Array;
    if (this.inputLookup) {
      formattedInput = lookup.toArray(
        this.inputLookup,
        (input as unknown) as INumberHash,
        this.inputLookupLength
      );
    } else {
      formattedInput = (input as unknown) as Float32Array;
    }
    this.validateInput(formattedInput);
    const output = this.runInput(formattedInput).slice(0);
    if (this.outputLookup) {
      return (lookup.toObject(
        this.outputLookup,
        output
      ) as unknown) as OutputType;
    }
    return (output as unknown) as OutputType;
  }

  _runInputSigmoid(input: Float32Array): Float32Array {
    this.outputs[0] = input; // set output state of input layer

    let output = null;
    for (let layer = 1; layer <= this.outputLayer; layer++) {
      const activeLayer = this.sizes[layer];
      const activeWeights = this.weights[layer];
      const activeBiases = this.biases[layer];
      const activeOutputs = this.outputs[layer];
      for (let node = 0; node < activeLayer; node++) {
        const weights = activeWeights[node];

        let sum = activeBiases[node];
        for (let k = 0; k < weights.length; k++) {
          sum += weights[k] * input[k];
        }
        // sigmoid
        activeOutputs[node] = 1 / (1 + Math.exp(-sum));
      }
      output = input = activeOutputs;
    }
    if (!output) {
      throw new Error('output was empty');
    }
    return output;
  }

  _runInputRelu(input: Float32Array): Float32Array {
    this.outputs[0] = input; // set output state of input layer

    let output = null;
    for (let layer = 1; layer <= this.outputLayer; layer++) {
      const activeSize = this.sizes[layer];
      const activeWeights = this.weights[layer];
      const activeBiases = this.biases[layer];
      const activeOutputs = this.outputs[layer];
      for (let node = 0; node < activeSize; node++) {
        const weights = activeWeights[node];

        let sum = activeBiases[node];
        for (let k = 0; k < weights.length; k++) {
          sum += weights[k] * input[k];
        }
        // relu
        activeOutputs[node] = sum < 0 ? 0 : sum;
      }
      output = input = activeOutputs;
    }
    if (!output) {
      throw new Error('output was empty');
    }
    return output;
  }

  _runInputLeakyRelu(input: Float32Array): Float32Array {
    this.outputs[0] = input; // set output state of input layer
    const { leakyReluAlpha } = this.trainOpts;
    let output = null;
    for (let layer = 1; layer <= this.outputLayer; layer++) {
      const activeSize = this.sizes[layer];
      const activeWeights = this.weights[layer];
      const activeBiases = this.biases[layer];
      const activeOutputs = this.outputs[layer];
      for (let node = 0; node < activeSize; node++) {
        const weights = activeWeights[node];

        let sum = activeBiases[node];
        for (let k = 0; k < weights.length; k++) {
          sum += weights[k] * input[k];
        }
        // leaky relu
        activeOutputs[node] = Math.max(sum, leakyReluAlpha * sum);
      }
      output = input = activeOutputs;
    }
    if (!output) {
      throw new Error('output was empty');
    }
    return output;
  }

  _runInputTanh(input: Float32Array): Float32Array {
    this.outputs[0] = input; // set output state of input layer

    let output = null;
    for (let layer = 1; layer <= this.outputLayer; layer++) {
      const activeSize = this.sizes[layer];
      const activeWeights = this.weights[layer];
      const activeBiases = this.biases[layer];
      const activeOutputs = this.outputs[layer];
      for (let node = 0; node < activeSize; node++) {
        const weights = activeWeights[node];

        let sum = activeBiases[node];
        for (let k = 0; k < weights.length; k++) {
          sum += weights[k] * input[k];
        }
        // tanh
        activeOutputs[node] = Math.tanh(sum);
      }
      output = input = activeOutputs;
    }
    if (!output) {
      throw new Error('output was empty');
    }
    return output;
  }

  /**
   *
   * Verifies network sizes are initialized
   * If they are not it will initialize them based off the data set.
   */
  verifyIsInitialized(): void {
    //if (this.sizes.length && this.outputLayer > 0) return;

    this.sizes = [];
    this.sizes.push(this.inputLookupLength);
    if (!this.options.hiddenLayers) {
      this.sizes.push(Math.max(3, Math.floor(this.inputLookupLength / 2)));
    } else {
      this.options.hiddenLayers.forEach((size) => {
        this.sizes.push(size);
      });
    }
    this.sizes.push(this.outputLookupLength);

    this.initialize();
  }

  updateTrainingOptions(trainOpts: Partial<INeuralNetworkTrainOptions>): void {
    const merged = { ...this.trainOpts, ...trainOpts };
    this.validateTrainingOptions(merged);
    this.trainOpts = merged;
    this.setLogMethod(this.trainOpts.log);
  }

  validateTrainingOptions(options: INeuralNetworkTrainOptions): void {
    const validations: { [fnName: string]: () => boolean } = {
      activation: () => {
        return ['sigmoid', 'relu', 'leaky-relu', 'tanh'].includes(
          options.activation
        );
      },
      iterations: () => {
        const val = options.iterations;
        return typeof val === 'number' && val > 0;
      },
      errorThresh: () => {
        const val = options.errorThresh;
        return typeof val === 'number' && val > 0 && val < 1;
      },
      log: () => {
        const val = options.log;
        return typeof val === 'function' || typeof val === 'boolean';
      },
      logPeriod: () => {
        const val = options.logPeriod;
        return typeof val === 'number' && val > 0;
      },
      leakyReluAlpha: () => {
        const val = options.leakyReluAlpha;
        return typeof val === 'number' && val > 0 && val < 1;
      },
      learningRate: () => {
        const val = options.learningRate;
        return typeof val === 'number' && val > 0 && val < 1;
      },
      momentum: () => {
        const val = options.momentum;
        return typeof val === 'number' && val > 0 && val < 1;
      },
      callback: () => {
        const val = options.callback;
        return typeof val === 'function' || val === undefined;
      },
      callbackPeriod: () => {
        const val = options.callbackPeriod;
        return typeof val === 'number' && val > 0;
      },
      timeout: () => {
        const val = options.timeout;
        return typeof val === 'number' && val > 0;
      },
      praxis: () => {
        const val = options.praxis;
        return !val || val === 'adam';
      },
      beta1: () => {
        const val = options.beta1;
        return val > 0 && val < 1;
      },
      beta2: () => {
        const val = options.beta2;
        return val > 0 && val < 1;
      },
      epsilon: () => {
        const val = options.epsilon;
        return val > 0 && val < 1;
      },
    };
    for (const p in validations) {
      const v = (options as unknown) as { [v: string]: string };
      if (!validations[p]()) {
        throw new Error(
          `[${p}, ${v[p]}] is out of normal training range, your network will probably not train.`
        );
      }
    }
  }

  /**
   *
   *  Gets JSON of trainOpts object
   *    NOTE: Activation is stored directly on JSON object and not in the training options
   */
  getTrainOptsJSON(): INeuralNetworkTrainOptionsJSON {
    const {
      activation,
      iterations,
      errorThresh,
      log,
      logPeriod,
      leakyReluAlpha,
      learningRate,
      momentum,
      callbackPeriod,
      timeout,
      praxis,
      beta1,
      beta2,
      epsilon,
    } = this.trainOpts;
    return {
      activation,
      iterations,
      errorThresh,
      log:
        typeof log === 'function'
          ? true
          : typeof log === 'boolean'
          ? log
          : false,
      logPeriod,
      leakyReluAlpha,
      learningRate,
      momentum,
      callbackPeriod,
      timeout: timeout === Infinity ? 'Infinity' : timeout,
      praxis,
      beta1,
      beta2,
      epsilon,
    };
  }

  setLogMethod(log: boolean | ((state: INeuralNetworkState) => void)): void {
    if (typeof log === 'function') {
      this.trainOpts.log = log;
    } else if (log) {
      this.trainOpts.log = this.logTrainingStatus;
    } else {
      this.trainOpts.log = false;
    }
  }

  logTrainingStatus(status: INeuralNetworkState): void {
    console.log(
      `iterations: ${status.iterations}, training error: ${status.error}`
    );
  }

  calculateTrainingError(
    data: Array<INeuralNetworkDatumFormatted<Float32Array>>
  ): number {
    let sum = 0;
    for (let i = 0; i < data.length; ++i) {
      sum += this.trainPattern(data[i], true) as number;
    }
    return sum / data.length;
  }

  trainPatterns(data: Array<INeuralNetworkDatumFormatted<Float32Array>>): void {
    for (let i = 0; i < data.length; ++i) {
      this.trainPattern(data[i]);
    }
  }

  trainingTick(
    data: Array<INeuralNetworkDatumFormatted<Float32Array>>,
    status: INeuralNetworkState,
    endTime: number
  ): boolean {
    const {
      callback,
      callbackPeriod,
      errorThresh,
      iterations,
      log,
      logPeriod,
    } = this.trainOpts;

    if (
      status.iterations >= iterations ||
      status.error <= errorThresh ||
      Date.now() >= endTime
    ) {
      return false;
    }

    status.iterations++;

    if (log && status.iterations % logPeriod === 0) {
      status.error = this.calculateTrainingError(data);
      (log as (state: INeuralNetworkState) => void)(status);
    } else if (status.iterations % this.errorCheckInterval === 0) {
      status.error = this.calculateTrainingError(data);
    } else {
      this.trainPatterns(data);
    }

    if (callback && status.iterations % callbackPeriod === 0) {
      callback({
        iterations: status.iterations,
        error: status.error,
      });
    }
    return true;
  }

  prepTraining(
    data: Array<INeuralNetworkDatum<InputType, OutputType>>,
    options: Partial<INeuralNetworkTrainOptions> = {}
  ): INeuralNetworkPreppedTrainingData<Float32Array> {
    this.updateTrainingOptions(options);
    const preparedData = this.formatData(data);
    const endTime = Date.now() + this.trainOpts.timeout;

    const status = {
      error: 1,
      iterations: 0,
    };

    this.verifyIsInitialized();
    //this.validateData(preparedData);
    return {
      preparedData,
      status,
      endTime,
    };
  }

  train(
    data: Array<INeuralNetworkDatum<Partial<InputType>, Partial<OutputType>>>,
    options: Partial<INeuralNetworkTrainOptions> = {}
  ): INeuralNetworkState {
    const { preparedData, status, endTime } = this.prepTraining(
      data as Array<INeuralNetworkDatum<InputType, OutputType>>,
      options
    );

    while (true) {
      if (!this.trainingTick(preparedData, status, endTime)) {
        break;
      }
    }
    return status;
  }

  async trainAsync(
    data: Array<INeuralNetworkDatum<InputType, OutputType>>,
    options: Partial<INeuralNetworkTrainOptions> = {}
  ): Promise<ITrainingStatus> {
    const { preparedData, status, endTime } = this.prepTraining(data, options);

    return await new Promise((resolve, reject) => {
      try {
        const thawedTrain: Thaw = new Thaw(
          new Array(this.trainOpts.iterations),
          {
            delay: true,
            each: () =>
              this.trainingTick(preparedData, status, endTime) ||
              thawedTrain.stop(),
            done: () => resolve(status),
          }
        );
        thawedTrain.tick();
      } catch (trainError) {
        reject(trainError);
      }
    });
  }

  trainPattern(
    value: INeuralNetworkDatumFormatted<Float32Array>,
    logErrorRate?: boolean
  ): number | null {
    // forward propagate
    this.runInput(value.input);

    // back propagate
    this.calculateDeltas(value.output);
    this.adjustWeights();

    if (logErrorRate) {
      const mse = this.mse_sum / this.mse_length;
      this.mse_sum = 0;
      this.mse_length = 0;
      return mse;
      //return mse(this.errors[this.outputLayer]);
    }
    return null;
  }

  _calculateDeltasSigmoid(target: Float32Array): void {
    for (let layer = this.outputLayer; layer >= 0; layer--) {
      const activeSize = this.sizes[layer];
      const activeOutput = this.outputs[layer];
      const activeDeltas = this.deltas[layer];
      const nextLayer = this.weights[layer + 1];

      for (let node = 0; node < activeSize; node++) {
        const output = activeOutput[node];

        let error = 0;
        if (layer === this.outputLayer) {
          error = target[node] - output;
        } else {
          const deltas = this.deltas[layer + 1];
          for (let k = 0; k < deltas.length; k++) {
            error += deltas[k] * nextLayer[k][node];
          }
        }
        if (layer === this.outputLayer) {
          this.mse_sum += error ** 2;
          this.mse_length++;
        }
        activeDeltas[node] = error * output * (1 - output);
      }
    }
  }

  _calculateDeltasRelu(target: Float32Array): void {
    for (let layer = this.outputLayer; layer >= 0; layer--) {
      const currentSize = this.sizes[layer];
      const currentOutputs = this.outputs[layer];
      const nextWeights = this.weights[layer + 1];
      const nextDeltas = this.deltas[layer + 1];
      const currentDeltas = this.deltas[layer];

      for (let node = 0; node < currentSize; node++) {
        const output = currentOutputs[node];

        let error = 0;
        if (layer === this.outputLayer) {
          error = target[node] - output;
        } else {
          for (let k = 0; k < nextDeltas.length; k++) {
            error += nextDeltas[k] * nextWeights[k][node];
          }
        }
        if (layer === this.outputLayer) {
          this.mse_sum += error ** 2;
          this.mse_length++;
        }
        currentDeltas[node] = output > 0 ? error : 0;
      }
    }
  }

  _calculateDeltasLeakyRelu(target: Float32Array): void {
    const alpha = this.trainOpts.leakyReluAlpha;
    for (let layer = this.outputLayer; layer >= 0; layer--) {
      const currentSize = this.sizes[layer];
      const currentOutputs = this.outputs[layer];
      const nextDeltas = this.deltas[layer + 1];
      const nextWeights = this.weights[layer + 1];
      const currentDeltas = this.deltas[layer];

      for (let node = 0; node < currentSize; node++) {
        const output = currentOutputs[node];

        let error = 0;
        if (layer === this.outputLayer) {
          error = target[node] - output;
        } else {
          for (let k = 0; k < nextDeltas.length; k++) {
            error += nextDeltas[k] * nextWeights[k][node];
          }
        }
        if (layer === this.outputLayer) {
          this.mse_sum += error ** 2;
          this.mse_length++;
        }
        currentDeltas[node] = output > 0 ? error : alpha * error;
      }
    }
  }

  _calculateDeltasTanh(target: Float32Array): void {
    for (let layer = this.outputLayer; layer >= 0; layer--) {
      const currentSize = this.sizes[layer];
      const currentOutputs = this.outputs[layer];
      const nextDeltas = this.deltas[layer + 1];
      const nextWeights = this.weights[layer + 1];
      const currentDeltas = this.deltas[layer];

      for (let node = 0; node < currentSize; node++) {
        const output = currentOutputs[node];

        let error = 0;
        if (layer === this.outputLayer) {
          error = target[node] - output;
        } else {
          for (let k = 0; k < nextDeltas.length; k++) {
            error += nextDeltas[k] * nextWeights[k][node];
          }
        }
        if (layer === this.outputLayer) {
          this.mse_sum += error ** 2;
          this.mse_length++;
        }
        currentDeltas[node] = (1 - output * output) * error;
      }
    }
  }

  /**
   *
   * Changes weights of networks
   */
  adjustWeights(): void {
    const { learningRate, momentum } = this.trainOpts;
    for (let layer = 1; layer <= this.outputLayer; layer++) {
      const incoming = this.outputs[layer - 1];
      const activeSize = this.sizes[layer];
      const activeDelta = this.deltas[layer];
      const activeChanges = this.changes[layer];
      const activeWeights = this.weights[layer];
      const activeBiases = this.biases[layer];

      for (let node = 0; node < activeSize; node++) {
        const delta = activeDelta[node];

        for (let k = 0; k < incoming.length; k++) {
          let change = activeChanges[node][k];

          change = learningRate * delta * incoming[k] + momentum * change;

          activeChanges[node][k] = change;
          activeWeights[node][k] += change;
        }
        activeBiases[node] += learningRate * delta;
      }
    }
  }

  _setupAdam(): void {
    this.biasChangesLow = [];
    this.biasChangesHigh = [];
    this.changesLow = [];
    this.changesHigh = [];
    this.iterations = 0;

    for (let layer = 0; layer <= this.outputLayer; layer++) {
      const size = this.sizes[layer];
      if (layer > 0) {
        this.biasChangesLow[layer] = zeros(size);
        this.biasChangesHigh[layer] = zeros(size);
        this.changesLow[layer] = new Array(size);
        this.changesHigh[layer] = new Array(size);

        for (let node = 0; node < size; node++) {
          const prevSize = this.sizes[layer - 1];
          this.changesLow[layer][node] = zeros(prevSize);
          this.changesHigh[layer][node] = zeros(prevSize);
        }
      }
    }

    this.adjustWeights = this._adjustWeightsAdam;
  }

  _adjustWeightsAdam(): void {
    this.iterations++;

    const { iterations } = this;
    const { beta1, beta2, epsilon, learningRate } = this.trainOpts;

    for (let layer = 1; layer <= this.outputLayer; layer++) {
      const incoming = this.outputs[layer - 1];
      const currentSize = this.sizes[layer];
      const currentDeltas = this.deltas[layer];
      const currentChangesLow = this.changesLow[layer];
      const currentChangesHigh = this.changesHigh[layer];
      const currentWeights = this.weights[layer];
      const currentBiases = this.biases[layer];
      const currentBiasChangesLow = this.biasChangesLow[layer];
      const currentBiasChangesHigh = this.biasChangesHigh[layer];

      for (let node = 0; node < currentSize; node++) {
        const delta = currentDeltas[node];

        for (let k = 0; k < incoming.length; k++) {
          const gradient = delta * incoming[k];
          const changeLow =
            currentChangesLow[node][k] * beta1 + (1 - beta1) * gradient;
          const changeHigh =
            currentChangesHigh[node][k] * beta2 +
            (1 - beta2) * gradient * gradient;

          const momentumCorrection =
            changeLow / (1 - Math.pow(beta1, iterations));
          const gradientCorrection =
            changeHigh / (1 - Math.pow(beta2, iterations));

          currentChangesLow[node][k] = changeLow;
          currentChangesHigh[node][k] = changeHigh;
          currentWeights[node][k] +=
            (learningRate * momentumCorrection) /
            (Math.sqrt(gradientCorrection) + epsilon);
        }

        const biasGradient = currentDeltas[node];
        const biasChangeLow =
          currentBiasChangesLow[node] * beta1 + (1 - beta1) * biasGradient;
        const biasChangeHigh =
          currentBiasChangesHigh[node] * beta2 +
          (1 - beta2) * biasGradient * biasGradient;

        const biasMomentumCorrection =
          currentBiasChangesLow[node] / (1 - Math.pow(beta1, iterations));
        const biasGradientCorrection =
          currentBiasChangesHigh[node] / (1 - Math.pow(beta2, iterations));

        currentBiasChangesLow[node] = biasChangeLow;
        currentBiasChangesHigh[node] = biasChangeHigh;
        currentBiases[node] +=
          (learningRate * biasMomentumCorrection) /
          (Math.sqrt(biasGradientCorrection) + epsilon);
      }
    }
  }

  validateData(data: Array<INeuralNetworkDatumFormatted<Float32Array>>): void {
    const inputSize = this.sizes[0];
    const outputSize = this.sizes[this.sizes.length - 1];
    const { length } = data;
    for (let i = 0; i < length; i++) {
      const { input, output } = data[i];
      if (input.length !== inputSize) {
        throw new Error(
          `input at index ${i} length ${input.length} must be ${inputSize}`
        );
      }
      if (data[i].output.length !== outputSize) {
        throw new Error(
          `output at index ${i} length ${output.length} must be ${outputSize}`
        );
      }
    }
  }

  validateInput(formattedInput: Float32Array): void {
    const inputSize = this.sizes[0];
    if (formattedInput.length !== inputSize) {
      throw new Error(
        `input length ${formattedInput.length} must match options.inputSize of ${inputSize}`
      );
    }
  }

  formatData(
    data: Array<INeuralNetworkDatum<InputType, OutputType>>
  ): Array<INeuralNetworkDatumFormatted<Float32Array>> {
    function extendNumberHash(o: INumberHash, n: INumberHash): INumberHash {
      if (!o) {
        return n;
      }
      let length = Object.keys(o).length;
      for (const p in n) {
        if (!o.hasOwnProperty(p)) {
          o[p] = length++;
        }
      }
      return o;
    }

    if (!Array.isArray(data[0].input)) {
      const inputLookup = new LookupTable(data, 'input');
      this.inputLookup = extendNumberHash(
        this.inputLookup as INumberHash,
        inputLookup.table
      );
      this.inputLookupLength = Object.keys(this.inputLookup).length;
    }

    if (!Array.isArray(data[0].output)) {
      const lookup = new LookupTable(data, 'output');
      this.outputLookup = extendNumberHash(
        this.outputLookup as INumberHash,
        lookup.table
      );
      this.outputLookupLength = Object.keys(this.outputLookup).length;
    }
    this._formatInput = getTypedArrayFn(data[0].input, this.inputLookup);
    this._formatOutput = getTypedArrayFn(data[0].output, this.outputLookup);

    // turn sparse hash input into arrays with 0s as filler
    if (this._formatInput && this._formatOutput) {
      const result: Array<INeuralNetworkDatumFormatted<Float32Array>> = [];
      for (let i = 0; i < data.length; i++) {
        result.push({
          input: (this._formatInput as (v: INumberHash) => Float32Array)(
            (data[i].input as unknown) as INumberHash
          ),
          output: (this._formatOutput as (v: INumberHash) => Float32Array)(
            (data[i].output as unknown) as INumberHash
          ),
        });
      }
      return result;
    }
    if (this._formatInput) {
      const result: Array<INeuralNetworkDatumFormatted<Float32Array>> = [];
      for (let i = 0; i < data.length; i++) {
        result.push({
          input: (this._formatInput as (v: INumberHash) => Float32Array)(
            (data[i].input as unknown) as INumberHash
          ),
          output: (data[i].output as unknown) as Float32Array,
        });
      }
      return result;
    }
    if (this._formatOutput) {
      const result: Array<INeuralNetworkDatumFormatted<Float32Array>> = [];
      for (let i = 0; i < data.length; i++) {
        result.push({
          input: (data[i].input as unknown) as Float32Array,
          output: (this._formatOutput as (v: INumberHash) => Float32Array)(
            (data[i].output as unknown) as INumberHash
          ),
        });
      }
      return result;
    }
    return (data as unknown) as Array<
      INeuralNetworkDatumFormatted<Float32Array>
    >;
  }

  addFormat(data: INeuralNetworkDatum<InputType, OutputType>): void {
    if (!Array.isArray(data.input) || typeof data.input[0] !== 'number') {
      this.inputLookup = lookup.addKeys(
        (data.input as unknown) as INumberHash,
        this.inputLookup ?? {}
      );
      if (this.inputLookup) {
        this.inputLookupLength = Object.keys(this.inputLookup).length;
      }
    }
    if (!Array.isArray(data.output) || typeof data.output[0] !== 'number') {
      this.outputLookup = lookup.addKeys(
        (data.output as unknown) as INumberHash,
        this.outputLookup ?? {}
      );
      if (this.outputLookup) {
        this.outputLookupLength = Object.keys(this.outputLookup).length;
      }
    }
  }

  test(
    data: Array<INeuralNetworkDatum<Partial<InputType>, Partial<OutputType>>>
  ): INeuralNetworkTestResult | INeuralNetworkBinaryTestResult {
    const { preparedData } = this.prepTraining(
      data as Array<INeuralNetworkDatum<InputType, OutputType>>
    );
    // for binary classification problems with one output node
    const isBinary = preparedData[0].output.length === 1;
    // for classification problems
    const misclasses = [];
    // run each pattern through the trained network and collect
    // error and misclassification statistics
    let errorSum = 0;
    if (isBinary) {
      let falsePos = 0;
      let falseNeg = 0;
      let truePos = 0;
      let trueNeg = 0;

      for (let i = 0; i < preparedData.length; i++) {
        const output = this.runInput(preparedData[i].input);
        const target = preparedData[i].output;
        const actual = output[0] > this.options.binaryThresh ? 1 : 0;
        const expected = target[0];

        if (actual !== expected) {
          const misclass = preparedData[i];
          misclasses.push({
            input: misclass.input,
            output: misclass.output,
            actual,
            expected,
          });
        }

        if (actual === 0 && expected === 0) {
          trueNeg++;
        } else if (actual === 1 && expected === 1) {
          truePos++;
        } else if (actual === 0 && expected === 1) {
          falseNeg++;
        } else if (actual === 1 && expected === 0) {
          falsePos++;
        }

        errorSum += mse(
          output.map((value, i) => {
            return target[i] - value;
          })
        );
      }

      return {
        error: errorSum / preparedData.length,
        misclasses,
        total: preparedData.length,
        trueNeg,
        truePos,
        falseNeg,
        falsePos,
        precision: truePos > 0 ? truePos / (truePos + falsePos) : 0,
        recall: truePos > 0 ? truePos / (truePos + falseNeg) : 0,
        accuracy: (trueNeg + truePos) / preparedData.length,
      };
    }

    for (let i = 0; i < preparedData.length; i++) {
      const output = this.runInput(preparedData[i].input);
      const target = preparedData[i].output;
      const actual = output.indexOf(max(output));
      const expected = target.indexOf(max(target));

      if (actual !== expected) {
        const misclass = preparedData[i];
        misclasses.push({
          input: misclass.input,
          output: misclass.output,
          actual,
          expected,
        });
      }

      errorSum += mse(
        output.map((value, i) => {
          return target[i] - value;
        })
      );
    }
    return {
      error: errorSum / preparedData.length,
      misclasses,
      total: preparedData.length,
    };
  }

  toJSON(): INeuralNetworkJSON {
    if (!this.isRunnable) {
      this.initialize();
    }
    // use Array.from, keeping json small
    const jsonLayerWeights = this.weights.map((layerWeights) => {
      return layerWeights.map((layerWeights) => Array.from(layerWeights));
    });
    const jsonLayerBiases = this.biases.map((layerBiases) =>
      Array.from(layerBiases)
    );
    const jsonLayers: IJSONLayer[] = [];
    const outputLength = this.sizes.length - 1;
    for (let i = 0; i <= outputLength; i++) {
      jsonLayers.push({
        weights: jsonLayerWeights[i] ?? [],
        biases: jsonLayerBiases[i] ?? [],
      });
    }
    return {
      type: 'NeuralNetwork',
      sizes: [...this.sizes],
      layers: jsonLayers,
      inputLookup: this.inputLookup ? { ...this.inputLookup } : null,
      inputLookupLength: this.inputLookupLength,
      outputLookup: this.outputLookup ? { ...this.outputLookup } : null,
      outputLookupLength: this.outputLookupLength,
      options: { ...this.options },
      trainOpts: this.getTrainOptsJSON(),
    };
  }

  fromJSON(json: INeuralNetworkJSON): this {
    this.options = { ...defaults(), ...json.options };
    if (json.hasOwnProperty('trainOpts')) {
      const trainOpts = {
        ...json.trainOpts,
        timeout:
          json.trainOpts.timeout === 'Infinity'
            ? Infinity
            : json.trainOpts.timeout,
      };
      this.updateTrainingOptions(trainOpts);
    }
    this.sizes = json.sizes;
    this.initialize();

    this.inputLookup = json.inputLookup ? { ...json.inputLookup } : null;
    this.inputLookupLength = json.inputLookupLength;
    this.outputLookup = json.outputLookup ? { ...json.outputLookup } : null;
    this.outputLookupLength = json.outputLookupLength;

    const jsonLayers = json.layers;
    const layerWeights = this.weights.map((layerWeights, layerIndex) => {
      return jsonLayers[layerIndex].weights.map((layerWeights) =>
        Float32Array.from(layerWeights)
      );
    });
    const layerBiases = this.biases.map((layerBiases, layerIndex) =>
      Float32Array.from(jsonLayers[layerIndex].biases)
    );
    for (let i = 0; i <= this.outputLayer; i++) {
      this.weights[i] = layerWeights[i] || [];
      this.biases[i] = layerBiases[i] || [];
    }
    return this;
  }

  /**
   * Output JSON files using streams.
   */
  // prettier-ignore
  async exportJSON(filePath: string): Promise<void> {
    console.log('Start export to ' + filePath);
    const writeStream = fs.createWriteStream(filePath);
    // 書き込みエラーの処理
    writeStream.on('error', (err) => {
      console.error('書き込みエラーが発生しました:', err);
    });
    var tab = '\n';
    // 書き込みを逐次的に行うための関数
    const writeAsync = (data: any, asyncFlag = true) => {
      return new Promise<void>((resolve, reject) => {
        if (asyncFlag) {
          if (!writeStream.write(data)) {
            writeStream.once('drain', resolve);
          } else {
            resolve();
          }
        } else {
          writeStream.write(data, (err) => {
            if (err) {
              reject(err); // エラーがあればリジェクト
            } else {
              resolve(); // 書き込み完了時にリゾルブ
            }
          });
        }
      });
    };
    // 巡回出力
    const output = async (val: any, typename?: string) => {
      if (Array.isArray(val) || (typeof val === 'object' && typename === 'Array')) {
        // Array
        await writeAsync('[', false);
        tab += '\t';
        const likeArray = (!Array.isArray(val) && typeof val === 'object' && typename === 'Array');
        const len = likeArray?Object.keys(val).length:val.length;
        for (let idx = 0; idx < len; idx++) {
          await writeAsync(tab);
          await output(val[idx], typename);
          if (idx < len - 1) {
            await writeAsync(',');
          }
        }
        tab = tab.replace(/\t$/, '');
        await writeAsync(tab + ']', false);
      } else if (typeof val == 'object') {
        // Object
        await writeAsync('{', false);
        tab += '\t';
        const entries = Object.entries(val);
        const len = entries.length;
        var idx = 0;
        for (let [ckey, cval] of entries) {
          idx++;
          if (cval === undefined) continue;
          await writeAsync(tab + `"${ckey}": `);
          await output(cval);
          if (idx < len) {
            await writeAsync(',');
          }
        }
        tab = tab.replace(/\t$/, '');
        await writeAsync(tab + '}', false);
      } else if (typeof val === 'string') {
        // String
        await writeAsync('"' + val + '"');
      } else {
        // Primitive
        val = val === Infinity ? 'Infinity' : val;
        await writeAsync(JSON.stringify(val??null));
      }
    };
    await writeAsync('{');
    tab += '\t';
    await writeAsync(tab + '"type": "NeuralNetwork",');
    await writeAsync(tab + '"sizes": '); await output(this.sizes); await writeAsync(',');
    await writeAsync(tab + '"inputLookup": '); await output(this.inputLookup); await writeAsync(',');
    await writeAsync(tab + '"inputLookupLength": '); await output(this.inputLookupLength); await writeAsync(',');
    await writeAsync(tab + '"outputLookup": '); await output(this.outputLookup); await writeAsync(',');
    await writeAsync(tab + '"outputLookupLength": '); await output(this.outputLookupLength); await writeAsync(',');
    await writeAsync(tab + '"options": '); await output(this.options); await writeAsync(',');
    await writeAsync(tab + '"trainOpts": '); await output(this.getTrainOptsJSON()); await writeAsync(',');
    await writeAsync(tab + '"weights": '); await output(this.weights ?? [], 'Array'); await writeAsync(',');
    await writeAsync(tab + '"biases": '); await output(this.biases ?? [], 'Array')
    tab = tab.replace(/\t$/, '');
    await writeAsync(tab + '}');
    writeStream.end();
    console.log('End export to ' + filePath);
  }

  async importJSON(filepath: string): Promise<void> {
    console.log('Start inport from ' + filepath);
    var json: any; // 解析オブジェクト
    var typename: string; // オブジェクトの型名
    var typenames: string[] = []; // 型名のスタック
    var cur: any; // 現在のオブジェクト
    var curs: any[] = []; // オブジェクトのスタック
    var idx: number | string; // オブジェクトのインデックス
    var idxs: (number | string)[] = []; // インデックスのスタック
    var preIdx: number | string; // 1つ前のインデックス
    var callbacks: any[] = []; // コールバックリスト
    var initIgnoreFlag: boolean = false; // 初期化無効フラグ
    const runCallbacks = () => {
      callbacks = callbacks.filter((v) => v);
      callbacks.forEach(([check, callback], i) => {
        if (check()) {
          callback();
          callbacks[i] = null;
        }
      });
    };
    // オブジェクト構築関数
    const build = (lineStr: string) => {
      const str = lineStr.trim();
      try {
        if (str.match(/("[^:]+?"): ([^:]+?)$/)) {
          // オブジェクト項目
          const [all, key, val] = str.match(
            /("[^:]+?"): ([^:]+?)$/
          ) as RegExpMatchArray;
          if (json === undefined) {
            throw new Error('想定外の行です : ' + lineStr);
          }
          idx = JSON.parse(key);
          if (!cur[idx]) {
            cur[idx] = {};
          }
          build(val);
        } else if (str.match(/[\{\[]/)) {
          // オブジェクト・配列開始
          if (initIgnoreFlag) {
            initIgnoreFlag = false;
            return;
          }
          var now;
          let setIdx;
          typenames.push(typename);
          if (str.match(/\{$/)) {
            // オブジェクト開始
            now = {};
            typename = 'Object';
          } else if (str.match(/\[$/)) {
            // 配列開始
            now = [];
            typename = 'Array';
            setIdx = () => {
              idx = 0;
            };
          }
          if (json === undefined) {
            json = now;
            cur = json;
          } else {
            curs.push(cur);
            if (cur[idx] === undefined) {
              cur[idx] = now;
            }
            cur = cur[idx];
            idxs.push(idx);
          }
          if (setIdx) setIdx();
        } else if (str.match(/[\]\}](,|$)/)) {
          // 配列orオブジェクト終了
          if (json === undefined) {
            throw new Error('想定外の行です : ' + lineStr);
          }
          cur = curs.pop() ?? json;
          preIdx = idx;
          idx = idxs.pop() as number | string;
          typename = typenames.pop() as string;
          if(typename === 'Array') {
            (idx as number)++;
          }
          runCallbacks();
        } else if (str.match(/([\S ]+?)(,|$)/)) {
          // 値
          let [all, v, end] = str.match(/([\S ]+?)(,|$)/) as RegExpMatchArray;
          let val: any = v === '"undefined"' ? undefined : JSON.parse(v);
          val = val === 'Infinity' ? Infinity : val;
          if (json === undefined) {
            json = val;
          } else {
            cur[idx] = val;
            if (typename === 'Array') {
              (idx as number)++;
            }
          }
        } else {
          throw new Error('想定外の行です : ' + lineStr);
        }
      } catch (e) {
        console.error(
          'cur:' + cur,
          'curs:' + curs,
          'idx:' + idx,
          'idxs:' + idxs,
          'str:' + str
        );
        throw e;
      }
    };
    // 初期化設定
    json = this;
    cur = this;
    initIgnoreFlag = true;
    this.outputLayer = -1;
    typename = 'Object';
    // コールバック設定
    callbacks.push(
      ...[
        [
          () => (idx as string) === 'sizes',
          () => {
            this.initialize();
          },
        ],
        [
          () => (idx as string) == 'trainOpts',
          () => {
            this.updateTrainingOptions(this.trainOpts);
          },
        ],
        [
          () => (idx as string) == 'weights',
          () => {
            this.weights[0] = [];
          },
        ],
        [
          () => (idx as string) == 'biases',
          () => {
            this.biases[0] = Float32Array.from([]);
          },
        ],
      ]
    );
    // Promiseを返すようにして、読み込み完了まで待てるようにする
    return new Promise((resolve, reject) => {
      const readStream = fs.createReadStream(filepath, { encoding: 'utf8' });
      let leftover = '';
      let newlineIndex = 0;
      // データが読み込まれた時の処理
      readStream.on('data', (chunk) => {
        var data = leftover + chunk;
        // 改行がある位置を探す
        while ((newlineIndex = data.indexOf('\n')) !== -1) {
          // 改行前の部分を取り出して出力
          const line = data.slice(0, newlineIndex);
          // 行ごとにオブジェクトビルド
          build(line);
          // 改行の次の位置からデータを残す
          data = data.slice(newlineIndex + 1);
        }
        // 改行がなかった部分を次の読み込みに持ち越し
        leftover = data;
      });
      // エラー処理
      readStream.on('error', (err) => {
        console.error('Error reading file:', err);
        reject(err); // エラーがあればPromiseを拒否
      });
      // 読み込み終了時の処理
      readStream.on('end', () => {
        console.log('End inport from ' + filepath);
        readStream.close();
        resolve(); // 読み込みが完了したらPromiseを解決
      });
    });
  }
}

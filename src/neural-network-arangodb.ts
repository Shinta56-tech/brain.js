import { INumberHash } from './lookup';
import { INeuralNetworkState } from './neural-network-types';
import {
  ArangoDBConfig,
  connect,
  query,
  createCollection,
} from './utilities/arangodb';

export interface INeuralNetworkOptions {
  inputSize: number;
  outputSize: number;
  binaryThresh: number;
  hiddenLayers?:
    | number[]
    | ((inputSize: number, outputSize: number) => number[])
    | undefined;
}

export function defaults(): INeuralNetworkOptions {
  return {
    inputSize: 0,
    outputSize: 0,
    binaryThresh: 0.5,
  };
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
  hiddenLayers?:
    | number[]
    | ((inputSize: number, outputSize: number) => number[])
    | undefined;
}

export interface ITrainingStatus {
  iterations: number;
  error: number;
}

export interface INeuralNetworkDatumFormatted<T> {
  input: T;
  output: T;
}

export type INeuralNetworkData = number[] | Float32Array | Partial<INumberHash>;

function randomWeight(): number {
  return Math.random() * 0.4 - 0.2;
}

function mse(errors: number[] | INumberHash): number {
  // mean squared error
  let sum = 0;
  if (Array.isArray(errors)) {
    for (let i = 0; i < errors.length; i++) {
      sum += errors[i] ** 2;
    }
    return sum / errors.length;
  } else {
    for (let key in errors) {
      sum += errors[key] ** 2;
    }
    return sum / Object.keys(errors).length;
  }
}

export class NeuralNetworkArangoDB<
  InputType extends INeuralNetworkData,
  OutputType extends INeuralNetworkData
> {
  //** common properties
  collection: string = 'brainjs_NeuralNetworkArangoDB';
  name: string;
  options: INeuralNetworkOptions = defaults();
  trainOpts: INeuralNetworkTrainOptions = trainDefaults();

  //** neural network properties
  sizes: number[] = [];
  // prettier-ignore
  async _getQuery(field: string, createFunc:(() => number) | undefined, i1: number | string, i2?: number | string, i3?: number | string): Promise<number> {
    const aql = `
      FOR doc IN \`${this.name}.${field}\`
      FILTER doc.i1 == ${JSON.stringify(i1)} ${i2!==undefined?`&& doc.i2 == ${JSON.stringify(i2)}`:''} ${i3!==undefined?`&& doc.i3 == ${JSON.stringify(i3)}`:''}
      RETURN doc.number
    `;
    const result: any = await query(aql);
    if (result.length > 0) {
      return result[0] as number;
    } else {
      const createAql = `
      INSERT {
        i1: ${JSON.stringify(i1)},
        ${i2!==undefined ? `i2: ${JSON.stringify(i2)},` : ""}
        ${i3!==undefined ? `i3: ${JSON.stringify(i3)},` : ""}
        number: ${createFunc?createFunc():0}
      } INTO \`${this.name}.${field}\`
      RETURN NEW.number
    `;
    const createResult: any = await query(createAql);
    return createResult[0] as number;
    }
  }
  // prettier-ignore
  async _setQuery(field: string, number: number, i1: number | string, i2?: number | string, i3?: number | string): Promise<void> {
    const aql = `
      UPSERT {i1: ${JSON.stringify(i1)}${i2!==undefined?`, i2: ${JSON.stringify(i2)}`:''}${i3!==undefined?`, i3: ${JSON.stringify(i3)}`:''}}
      INSERT {number: ${number}, i1: ${JSON.stringify(i1)}${i2!==undefined?`, i2: ${JSON.stringify(i2)}`:''}${i3!==undefined?`, i3: ${JSON.stringify(i3)}`:''}}
      UPDATE {number: ${number}, i1: ${JSON.stringify(i1)}${i2!==undefined?`, i2: ${JSON.stringify(i2)}`:''}${i3!==undefined?`, i3: ${JSON.stringify(i3)}`:''}}
      IN \`${this.name}.${field}\`
    `;
    await query(aql);
  }
  // prettier-ignore
  async _keysQuery<T extends number | string>(field: string, i1?: number | string, i2?: number | string): Promise<T[]> {
    const aql = `
      FOR doc IN \`${this.name}.${field}\`
      ${i1!==undefined ? `FILTER doc.i1 == ${JSON.stringify(i1)}` : ''}${i2!==undefined ? ` && doc.i2 == ${JSON.stringify(i2)}` : ''}
      RETURN doc.${i2!==undefined ? 'i3' : i1!==undefined ? 'i2' : 'i1'}
    `;
    const result: any = await query(aql);
    return result as T[];
  }
  // prettier-ignore
  async _parseQuery(field: string, i1: number | string, i2?: number | string): Promise<OutputType> {
    const aql = `
      FOR doc IN \`${this.name}.${field}\`
      ${i1!==undefined ? `FILTER doc.i1 == ${JSON.stringify(i1)}` : ''}${i2!==undefined ? ` && doc.i2 == ${JSON.stringify(i2)}` : ''}
      RETURN [doc.${i2!==undefined ? 'i3' : i1!==undefined ? 'i2' : 'i1'},doc.number]
    `;
    const result: any = await query(aql);
    if (result.length > 0) {
      if (typeof result[0][0] === "number") {
        return result.reduce((acc: number[], [key, value]: [number, number]) => {
          acc[key] = value;
          return acc;
        }, []) as number[] as OutputType;
      } else {
        return result.reduce((acc: INumberHash, [key, value]: [string, number]) => {
          acc[key] = value;
          return acc;
        }, {}) as INumberHash as OutputType;
      }
    } else {
      return {} as OutputType;
    }
  }

  //** neural network properties for training
  errorCheckInterval = 1;
  inputSize: number = 0;
  outputSize: number = 0;
  outputLayer: number = 0;

  constructor(
    options: Partial<INeuralNetworkOptions & INeuralNetworkTrainOptions> = {},
    name?: string,
    arangoDBConfig?: ArangoDBConfig | undefined
  ) {
    connect(arangoDBConfig);
    this.name = name || 'default';
    this.options = { ...this.options, ...options };
    this._updateTrainingOptions(options);
    const { inputSize, hiddenLayers, outputSize } = this.options;
    if (inputSize && outputSize) {
      this.sizes = [inputSize]
        .concat(Array.isArray(hiddenLayers) ? hiddenLayers ?? [] : [])
        .concat([outputSize]);
    }
  }

  async train(
    data: Array<INeuralNetworkDatum<InputType, OutputType>>,
    options: Partial<INeuralNetworkTrainOptions> = {}
  ): Promise<INeuralNetworkState> {
    for (let fieldName of [
      'bias',
      'weight',
      'output',
      'delta',
      'error',
      'change',
    ]) {
      await createCollection(`${this.name}.${fieldName}`, ['i1', 'i2', 'i3']);
    }
    const endTime = Date.now() + this.trainOpts.timeout;
    const status = {
      error: 1,
      iterations: 0,
    };
    this._prepTraining(data, options);

    while (true) {
      if (!(await this._trainingTick(data, status, endTime))) {
        break;
      }
    }
    return status;
  }

  async run(input: InputType): Promise<OutputType> {
    return (await this._runInput(input)) as OutputType;
  }

  private _prepTraining(
    data: Array<INeuralNetworkDatum<InputType, OutputType>>,
    options: Partial<INeuralNetworkTrainOptions> = {}
  ): void {
    this._updateTrainingOptions(options);
    this._verifyIsInitialized(data);
  }

  private _verifyIsInitialized(
    data: Array<INeuralNetworkDatum<InputType, OutputType>>
  ): void {
    this.sizes = [];
    this.inputSize = data
      .map((d) => Object.keys(d.input))
      .reduce((a, b) => Array.from(new Set([...a, ...b])), []).length;
    this.outputSize = data
      .map((d) => Object.keys(d.output))
      .reduce((a, b) => Array.from(new Set([...a, ...b])), []).length;
    this.sizes.push(this.inputSize);
    if (!this.options.hiddenLayers) {
      this.sizes.push(Math.max(3, this.inputSize / 2));
    } else {
      if (Array.isArray(this.options.hiddenLayers)) {
        this.options.hiddenLayers.forEach((size) => {
          this.sizes.push(size);
        });
      } else if (typeof this.options.hiddenLayers === 'function') {
        this.options
          .hiddenLayers(this.inputSize, this.outputSize)
          .forEach((size) => {
            this.sizes.push(size);
          });
      }
    }
    this.sizes.push(this.outputSize);
    this.outputLayer = this.sizes.length - 1;
  }

  private _updateTrainingOptions(
    trainOpts: Partial<INeuralNetworkTrainOptions>
  ): void {
    const merged = { ...this.trainOpts, ...trainOpts };
    this._validateTrainingOptions(merged);
    this.trainOpts = merged;
    this._setLogMethod(this.trainOpts.log);
  }

  private _validateTrainingOptions(options: INeuralNetworkTrainOptions): void {
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

  private _setLogMethod(
    log: boolean | ((state: INeuralNetworkState) => void)
  ): void {
    if (typeof log === 'function') {
      this.trainOpts.log = log;
    } else if (log) {
      this.trainOpts.log = this._logTrainingStatus;
    } else {
      this.trainOpts.log = false;
    }
  }

  private _logTrainingStatus(status: INeuralNetworkState): void {
    console.log(
      `iterations: ${status.iterations}, training error: ${status.error}`
    );
  }

  private async _trainingTick(
    data: Array<INeuralNetworkDatum<InputType, OutputType>>,
    status: INeuralNetworkState,
    endTime: number
  ): Promise<boolean> {
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
      status.error = await this._calculateTrainingError(data);
      (log as (state: INeuralNetworkState) => void)(status);
    } else if (status.iterations % this.errorCheckInterval === 0) {
      status.error = await this._calculateTrainingError(data);
    } else {
      await this._trainPatterns(data);
    }

    if (callback && status.iterations % callbackPeriod === 0) {
      callback({
        iterations: status.iterations,
        error: status.error,
      });
    }
    return true;
  }

  private async _calculateTrainingError(
    data: Array<INeuralNetworkDatum<InputType, OutputType>>
  ): Promise<number> {
    let sum = 0;
    for (let i = 0; i < data.length; ++i) {
      sum += (await this._trainPattern(data[i], true)) as number;
    }
    return sum / data.length;
  }

  private async _trainPatterns(
    data: Array<INeuralNetworkDatum<InputType, OutputType>>
  ): Promise<void> {
    for (let i = 0; i < data.length; ++i) {
      await this._trainPattern(data[i]);
    }
  }

  private async _trainPattern(
    value: INeuralNetworkDatum<InputType, OutputType>,
    logErrorRate?: boolean
  ): Promise<number | null> {
    // forward propagate
    await this._runInput(value.input, value.output);

    // back propagate
    await this._calculateDeltas(value.output);
    await this._adjustWeights();

    if (logErrorRate) {
      return mse(
        (await this._parseQuery('error', this.outputLayer)) as INumberHash
      );
    }
    return null;
  }

  // prettier-ignore
  private async _runInput(input: InputType, output?: OutputType): Promise<OutputType> {
    this._setActivation();
    return await this._runInput(input, output);
  }

  // prettier-ignore
  private async _activation(layer: number, node: number | string, sum: number): Promise<void> {
    this._setActivation();
    return await this._activation(layer, node, sum);
  }

  private async _calculateDeltas(output: OutputType): Promise<void> {
    this._setActivation();
    return await this._calculateDeltas(output);
  }

  // prettier-ignore
  private async _runInputDefault(input: InputType, output?: OutputType): Promise<OutputType> {
    for (let [key, value] of Object.entries(input)) {
      await this._setQuery('output', value as number, 0, key);
    }
    const activateFunc = async (avtivelayer: number, node: number | string): Promise<void> => {
      let sum = await this._getQuery('bias', randomWeight, avtivelayer, node);
        for (let k of (await this._keysQuery('output', avtivelayer - 1))) {
          sum +=
            (await this._getQuery('weight', randomWeight, avtivelayer, node, k))
            * (await this._getQuery('output', ()=>0, avtivelayer - 1, k));
        }
        // activation
        this._activation(avtivelayer, node, sum);
    };
    for (let layer = 1; layer <= this.outputLayer - 1; layer++) {
      for (let node = 0; node < this.sizes[layer]; node++) {
        await activateFunc(layer, node);
      }
    }
    let outputNodes: string[] = [];
    if (output) {
      outputNodes = Object.keys(output);
    } else {
      outputNodes = await this._keysQuery('output', this.outputLayer);
    }
    for (let node of outputNodes) {
      await activateFunc(this.outputLayer, node);
    }
    let outputObj = await this._parseQuery('output', this.outputLayer);
    if (!outputObj) {
      throw new Error('output was empty');
    }
    return outputObj;
  }

  private _setActivation(activation?: NeuralNetworkActivation): void {
    this._runInput = this._runInputDefault;
    const value = activation ?? this.trainOpts.activation;
    switch (value) {
      case 'sigmoid':
        this._activation = this._activationSigmoid;
        this._calculateDeltas = this._calculateDeltasSigmoid;
        break;
      default:
        throw new Error(
          `Unknown activation ${value}. Available activations are: 'sigmoid', 'relu', 'leaky-relu', 'tanh'`
        );
    }
  }

  // prettier-ignore
  private async _activationSigmoid(layer: number, node: number | string, sum: number): Promise<void> {
    if (typeof node === 'string' && !isNaN(parseInt(node))) {
      node = parseInt(node);
    }
    await this._setQuery('output', 1 / (1 + Math.exp(-sum)), layer, node);
  }

  // prettier-ignore
  private async _calculateDeltasSigmoid(target: OutputType): Promise<void> {
    for (let layer = this.outputLayer; layer >= 0; layer--) {
      const updateFunc = async (node: number | string): Promise<void> => {
        let avtiveOutput = await this._getQuery('output', ()=>0, layer, node);
        let error = 0;
        if (layer === this.outputLayer) {
          if ( (target as INumberHash)[node as string] !== undefined) {
            error = (target as INumberHash)[node as string] - avtiveOutput;
          } else {
            error = 0 - avtiveOutput;
          }
        } else {
          for (let k of await this._keysQuery('output', layer + 1)) {
            error += (await this._getQuery('delta', ()=>0, layer + 1, k)) * (await this._getQuery('weight', randomWeight, layer + 1, k, node));
          }
        }
        await this._setQuery('error', error, layer, node);
        let delta = error * avtiveOutput * (1 - avtiveOutput);
        await this._setQuery('delta', delta, layer, node);
      }
      if (layer === 0 || layer === this.outputLayer) {
        for (let node of await this._keysQuery('output', layer)) {
          await updateFunc(node as string);
        }
      } else {
        for (let node = 0; node < this.sizes[layer]; node++) {
          await updateFunc(node as number);
        }
      }
    }
  }

  // prettier-ignore
  async _adjustWeights(): Promise<void> {
    const { learningRate, momentum } = this.trainOpts;
    for (let layer = 1; layer <= this.outputLayer; layer++) {
      for (let node of (await this._keysQuery('output', layer))) {
        let delta = await this._getQuery('delta', ()=>0, layer, node);
        for (let k of await this._keysQuery('output', layer - 1)) {
          let change = await this._getQuery('change', ()=>0, layer, node, k);
          change = learningRate * delta * (await this._getQuery('output', ()=>0, layer - 1, k)) + momentum * change;
          await this._setQuery('change', change, layer, node, k);
          let weight = await this._getQuery('weight', randomWeight, layer, node, k);
          await this._setQuery('weight', weight + change, layer, node, k);
        }
        let bias = await this._getQuery('bias', randomWeight, layer, node);
        await this._setQuery('bias', bias + learningRate * delta, layer, node);
      }
    }
  }
}

import { INumberHash } from './lookup';
import { INeuralNetworkState } from './neural-network-types';
import {
  ArangoDBConfig,
  connect,
  query,
} from './utilities/arangodb';

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
  hiddenLayers?: number[];
}

export type INeuralNetworkData = number[] | Float32Array | Partial<INumberHash>;

export class NeuralNetworkArangoDB<
  InputType extends INeuralNetworkData,
  OutputType extends INeuralNetworkData
> {
  collection: string = 'brainjs_NeuralNetworkArangoDB';
  name: string;
  options: INeuralNetworkOptions = defaults();
  trainOpts: INeuralNetworkTrainOptions = trainDefaults();
  sizes: number[] = [];
  // prettier-ignore
  async _getQuery(field: string, i1: number | string, i2?: number | string, i3?: number | string): Promise<number> {
    const aql = `
      FOR doc IN ${this.collection}
      FILTER doc.name == "${this.name}" && doc.field == "${field}" && doc.i1 == ${JSON.stringify(i1)} ${i2?`&& doc.i2 == ${JSON.stringify(i2)}`:''} ${i3?`&& doc.i3 == ${JSON.stringify(i3)}`:''}
      RETURN doc.number
    `;
    const result: any = await query(aql);
    if (result.length > 0) {
      return result[0] as number;
    } else {
      return 0;
    }
  };
  // prettier-ignore
  async _setQuery(field: string, number: number, i1: number | string, i2?: number | string, i3?: number | string): Promise<void> {
    const aql = `
      UPSERT {name: "${this.name}", field: "${field}", i1: ${JSON.stringify(i1)}${i2?`, i2: ${JSON.stringify(i2)}`:''}${i3?`, i3: ${JSON.stringify(i3)}`:''}}
      INSERT {name: "${this.name}", field: "${field}", number: ${number}, i1: ${JSON.stringify(i1)}${i2?`, i2: ${JSON.stringify(i2)}`:''}${i3?`, i3: ${JSON.stringify(i3)}`:''}}
      UPDATE {name: "${this.name}", field: "${field}", number: ${number}, i1: ${JSON.stringify(i1)}${i2?`, i2: ${JSON.stringify(i2)}`:''}${i3?`, i3: ${JSON.stringify(i3)}`:''}}
      IN ${this.collection}
    `;
    await query(aql);
  };
  // prettier-ignore
  async _lengthQuery(field: string, i1?: number | string, i2?: number | string): Promise<number> {
    const aql = `
      FOR doc IN ${this.collection}
      FILTER doc.name == "${this.name}" && doc.field == "${field}"${i1?` && doc.i1 == ${JSON.stringify(i1)}`:''}${i2?` && doc.i2 == ${JSON.stringify(i2)}`:''}
      COLLECT WITH COUNT INTO length
      RETURN length
    `;
    const result: any = await query(aql);
    if (result.length > 0) {
      return result[0];
    } else {
      return 0;
    }
  };
  // prettier-ignore
  async _clearQueryfield(field: string, i1?: number | string, i2?: number | string): Promise<void> {
    const aql = `
      FOR doc IN ${this.collection}
      FILTER doc.name == "${this.name}" && doc.field == "${field}"${i1?` && doc.i1 == ${JSON.stringify(i1)}`:''}${i2?` && doc.i2 == ${JSON.stringify(i2)}`:''}
      REMOVE doc IN ${this.collection}
    `;
    await query(aql);
  };
  // prettier-ignore
  async _pushQuery(field: string, number: number, i1?: number | string, i2?: number | string): Promise<void> {
    const aql = `
      LET maxI = (
        FOR doc IN ${this.collection}
        FILTER doc.name == @name && doc.field == @field${i1?` && doc.i1 == ${JSON.stringify(i1)}`:''}${i2?` && doc.i2 == ${JSON.stringify(i2)}`:''}
        SORT doc.${i1?i2?'i3':'i2':'i1'} DESC
        LIMIT 1
        RETURN doc.${i1?i2?'i3':'i2':'i1'}
      )[0]

      LET newI = maxI + 1

      INSERT {
        field: @field,
        number: @number,
        ${i1?'i1: ${i1}, '+i2?'i2: ${i2}, i3: newI':'i2: newI':'i1: newI'}
      } INTO ${this.collection}

      RETURN NEW
    `;
    await query(aql);
  };

  constructor(
    options: Partial<INeuralNetworkOptions & INeuralNetworkTrainOptions> = {},
    name?: string,
    arangoDBConfig?: ArangoDBConfig | undefined,
  ) {
    connect(arangoDBConfig);
    this.name = name || 'default';
    this.options = { ...this.options, ...options };
    this._updateTrainingOptions(options);
    const { inputSize, hiddenLayers, outputSize } = this.options;
    if (inputSize && outputSize) {
      this.sizes = [inputSize].concat(hiddenLayers ?? []).concat([outputSize]);
    }
  }

  run(input: Partial<InputType>): OutputType {
    return {} as OutputType;
  }

  train(
    data: Array<INeuralNetworkDatum<Partial<InputType>, Partial<OutputType>>>,
    options: Partial<INeuralNetworkTrainOptions> = {}
  ): INeuralNetworkState {
    return {} as INeuralNetworkState;
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
}

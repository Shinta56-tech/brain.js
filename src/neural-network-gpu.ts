import {
  alias,
  GPU,
  GPUFunction,
  IKernelFunctionThis,
  IKernelMapRunShortcut,
  IMappedKernelResult,
  KernelOutput,
  Texture,
  utils,
} from 'gpu.js';
import { ITrainingStatus } from './feed-forward';
import { INumberHash, lookup } from './lookup';
import {
  IJSONLayer,
  INeuralNetworkData,
  INeuralNetworkDatum,
  INeuralNetworkJSON,
  INeuralNetworkOptions,
  INeuralNetworkPreppedTrainingData,
  INeuralNetworkTrainOptions,
  NeuralNetwork,
} from './neural-network';
import { release } from './utilities/kernel';

export interface INeuralNetworkGPUDatumFormatted {
  input: KernelOutput;
  output: KernelOutput;
}

export interface INeuralNetworkGPUPreppedTrainingData
  extends INeuralNetworkPreppedTrainingData<KernelOutput> {
  status: ITrainingStatus;
  endTime: number;
}

interface ISizedKernelThis extends IKernelFunctionThis {
  constants: {
    size: number;
  };
}

type WeightedSum = ((
  this: IKernelFunctionThis<{ size: number }>,
  weights: number[][],
  biases: number[],
  inputs: number[]
) => IMappedKernelResult) &
  IKernelMapRunShortcut<{
    outputs: number[];
  }>;

function weightedSumSigmoid(sum: number): number {
  // sigmoid
  return 1 / (1 + Math.exp(-sum));
}

function weightedSumRelu(sum: number): number {
  // relu
  return sum < 0 ? 0 : sum;
}

function weightedSumLeakyRelu(sum: number): number {
  // leaky relu
  return sum < 0 ? 0 : 0.01 * sum;
}

function weightedSumTanh(sum: number): number {
  // tanh
  return Math.tanh(sum);
}

function weightedSumMish(sum: number): number {
  // mish
  return sum * Math.tanh(Math.log(1 + Math.exp(sum)));
}

function calcErrorOutput(output: number, target: number): number {
  const error = target - output;
  return Math.abs(target) > 1 ? error / Math.abs(target) : error;
}

function calcDeltasSigmoid(
  error: number,
  output: number,
  outputPA: number
): number {
  // sigmoid derivative
  return error * output * (1 - output);
}

function calcDeltasRelu(
  error: number,
  output: number,
  outputPA: number
): number {
  // relu derivative
  return output > 0 ? error : 0;
}

function calcDeltasLeakyRelu(
  error: number,
  output: number,
  outputPA: number
): number {
  // leaky relu derivative
  return output > 0 ? error : 0.01 * error;
}

function calcDeltasTanh(
  error: number,
  output: number,
  outputPA: number
): number {
  // tanh derivative
  return (1 - output * output) * error;
}

function calcDeltasMish(
  error: number,
  output: number,
  outputPA: number
): number {
  // mish derivative
  const softX = Math.max(Math.min(outputPA, 80), -80);
  const sp = Math.log(1 + Math.exp(softX));
  const tsp = Math.tanh(sp);
  const sigx = 1 / (1 + Math.exp(-softX));
  const back = tsp + softX * sigx * (1 - tsp * tsp);
  return error * back;
}

function calcError(
  x: number,
  size: number,
  nextWeights: number[][],
  nextDeltas: number[]
): number {
  let error = 0;
  for (let k = 0; k < size; k++) {
    error += nextDeltas[k] * nextWeights[k][x];
  }
  return error;
}

interface ILearningKernelThis extends IKernelFunctionThis {
  constants: {
    momentum: number;
    learningRate: number;
  };
}

type ChangeePropagate = ((
  this: IKernelFunctionThis<{
    learningRate: number;
    momentum: number;
  }>,
  previousOutputs: number[],
  deltas: number[],
  weights: number[][],
  previousChanges: number[][]
) => IMappedKernelResult) &
  IKernelMapRunShortcut<{ weights: number[][]; changes: number[][] }>;

type ChangeePropagateAdam = ((
  this: IKernelFunctionThis<{
    iterations: number;
    beta1: number;
    beta2: number;
    epsilon: number;
    learningRate: number;
  }>,
  previousOutputs: number[],
  deltas: number[],
  weights: number[][],
  currentChangesLow: number[][],
  currentChangesHigh: number[][]
) => IMappedKernelResult) &
  IKernelMapRunShortcut<{
    weights: number[][];
    changesLow: number[][];
    changesHigh: number[][];
  }>;

type ChangeePropagateAdamW = ((
  this: IKernelFunctionThis<{
    iterations: number;
    beta1: number;
    beta2: number;
    epsilon: number;
    learningRate: number;
    weightDecay: number;
  }>,
  previousOutputs: number[],
  deltas: number[],
  weights: number[][],
  currentChangesLow: number[][],
  currentChangesHigh: number[][]
) => IMappedKernelResult) &
  IKernelMapRunShortcut<{
    weights: number[][];
    changesLow: number[][];
    changesHigh: number[][];
  }>;

function calcChanges(
  learningRate: number,
  momentum: number,
  previousChange: number,
  delta: number,
  previousOutput: number
): number {
  return learningRate * delta * previousOutput + momentum * previousChange;
}

function calcChangesLowAdam(
  changeLow: number,
  beta1: number,
  gradient: number
): number {
  return changeLow * beta1 + (1 - beta1) * gradient;
}

function calcChangesHighAdam(
  changeHigh: number,
  beta2: number,
  gradient: number
): number {
  return changeHigh * beta2 + (1 - beta2) * gradient * gradient;
}

function addWeights(change: number, weight: number): number {
  return change + weight;
}

function addWeightsAdam(
  weight: number,
  learningRate: number,
  momentumCorrection: number,
  gradientCorrection: number,
  epsilon: number
): number {
  return (
    weight +
    (learningRate * momentumCorrection) /
      (Math.sqrt(gradientCorrection) + epsilon)
  );
}

function addWeightsAdamW(
  weight: number,
  learningRate: number,
  momentumCorrection: number,
  gradientCorrection: number,
  epsilon: number,
  weightDecay: number
): number {
  return (
    weight * (1 - learningRate * weightDecay) +
    (learningRate * momentumCorrection) /
      (Math.sqrt(gradientCorrection) + epsilon)
  );
}

type BiasesPropagate = (
  biases: KernelOutput,
  deltas: KernelOutput
) => KernelOutput;

type BiasesPropagateAdam = ((
  this: IKernelFunctionThis<{
    iterations: number;
    beta1: number;
    beta2: number;
    epsilon: number;
    learningRate: number;
  }>,
  deltas: number[],
  biases: number[],
  currentBiasChangesLow: number[],
  currentBiasChangesHigh: number[]
) => IMappedKernelResult) &
  IKernelMapRunShortcut<{
    biases: number[];
    biasChangesLow: number[];
    biasChangesHigh: number[];
  }>;

function addBiases(
  this: ILearningKernelThis,
  biases: number[],
  deltas: number[]
): number {
  return (
    biases[this.thread.x] + deltas[this.thread.x] * this.constants.learningRate
  );
}

function addBiasesAdam(
  biase: number,
  biasMomentumCorrection: number,
  biasGradientCorrection: number,
  learningRate: number,
  epsilon: number
): number {
  return (
    biase +
    (learningRate * biasMomentumCorrection) /
      (Math.sqrt(biasGradientCorrection) + epsilon)
  );
}

function calcBiasChangesLowAdam(
  biasChangeLow: number,
  beta1: number,
  biasGradient: number
): number {
  return biasChangeLow * beta1 + (1 - beta1) * biasGradient;
}

function calcBiasChangesHighAdam(
  biasChangeHigh: number,
  beta2: number,
  biasGradient: number
): number {
  return biasChangeHigh * beta2 + (1 - beta2) * biasGradient * biasGradient;
}

// mean squared error, reimplemented for GPU
function mse(this: ISizedKernelThis, errors: number[]): number {
  let sum = 0;
  for (let i = 0; i < this.constants.size; i++) {
    sum += errors[i] ** 2;
  }
  return sum / this.constants.size;
}

export interface INeuralNetworkGPUOptions extends INeuralNetworkOptions {
  mode?: 'cpu' | 'gpu';
}

export type BackPropagateOutput = (
  this: IKernelFunctionThis,
  outputs: KernelOutput,
  targets: KernelOutput,
  outputsPA: KernelOutput
) => { result: KernelOutput; error: KernelOutput };

export type BackPropagateLayer = (
  this: IKernelFunctionThis,
  weights: KernelOutput,
  outputs: KernelOutput,
  deltas: KernelOutput,
  outputsPA: KernelOutput
) => { result: KernelOutput; error: KernelOutput };

export class NeuralNetworkGPU<
  InputType extends INeuralNetworkData,
  OutputType extends INeuralNetworkData
> extends NeuralNetwork<InputType, OutputType> {
  gpu: GPU;

  // texturizeInputData: (value: KernelOutput) => KernelOutput = () => {
  //   throw new Error('not yet setup');
  // };

  forwardPropagate: WeightedSum[] = [];

  backwardPropagate: Array<BackPropagateOutput | BackPropagateLayer> = [];

  changesPropagate: Array<
    ChangeePropagate | ChangeePropagateAdam | ChangeePropagateAdamW
  > = [];

  biasesPropagate: Array<BiasesPropagate | BiasesPropagateAdam> = [];

  getMSE: (error: KernelOutput) => KernelOutput = () => {
    throw new Error('not yet setup');
  };

  _addMSE: (sum: KernelOutput, error: KernelOutput) => KernelOutput = () => {
    throw new Error('not yet setup');
  };

  _divideMSESum: (length: number, sum: KernelOutput) => KernelOutput = () => {
    throw new Error('not yet setup');
  };

  // eslint-disable-next-line @typescript-eslint/ban-ts-comment
  // @ts-expect-error
  outputs: KernelOutput[] = [];
  // eslint-disable-next-line @typescript-eslint/ban-ts-comment
  // @ts-expect-error
  outputsPreActivation: KernelOutput[] = [];
  // eslint-disable-next-line @typescript-eslint/ban-ts-comment
  // @ts-expect-error
  deltas: KernelOutput[] = [];
  // eslint-disable-next-line @typescript-eslint/ban-ts-comment
  // @ts-expect-error
  errors: KernelOutput[] = [];
  // eslint-disable-next-line @typescript-eslint/ban-ts-comment
  // @ts-expect-error
  weights: KernelOutput[] = [];
  // eslint-disable-next-line @typescript-eslint/ban-ts-comment
  // @ts-expect-error
  changes: KernelOutput[] = [];
  // eslint-disable-next-line @typescript-eslint/ban-ts-comment
  // @ts-expect-error
  biases: KernelOutput[] = [];
  // eslint-disable-next-line @typescript-eslint/ban-ts-comment
  // @ts-expect-error
  changesLow: KernelOutput[] = [];
  // eslint-disable-next-line @typescript-eslint/ban-ts-comment
  // @ts-expect-error
  changesHigh: KernelOutput[] = [];
  // eslint-disable-next-line @typescript-eslint/ban-ts-comment
  // @ts-expect-error
  biasChangesLow: KernelOutput[] = [];
  // eslint-disable-next-line @typescript-eslint/ban-ts-comment
  // @ts-expect-error
  biasChangesHigh: KernelOutput[] = [];

  constructor(options: Partial<INeuralNetworkGPUOptions> = {}) {
    super(options);
    this.errorCheckInterval = 100;
    this.gpu = new GPU({ mode: options.mode });
  }

  initialize(): void {
    super.initialize();
    this.buildRunInput();
    this.buildCalculateDeltas();
    this.buildGetChanges();
    this.buildChangeBiases();
    this.buildGetMSE();
  }

  setActivation(): void {}

  // eslint-disable-next-line @typescript-eslint/ban-ts-comment
  // @ts-expect-error
  trainPattern(
    value: INeuralNetworkGPUDatumFormatted,
    logErrorRate?: boolean
  ): KernelOutput | null {
    // forward propagate
    this.runInput(value.input);

    // back propagate
    this.calculateDeltas(value.output);
    this.adjustWeights();

    if (logErrorRate) {
      return this.getMSE(this.errors[this.outputLayer]);
    }
    return null;
  }

  calculateTrainingError(data: INeuralNetworkGPUDatumFormatted[]): number {
    let sum = new Float32Array([0]) as KernelOutput;
    for (let i = 0; i < data.length; ++i) {
      const prevSum = sum;
      const error = this.trainPattern(data[i], true) as KernelOutput;
      sum = this._addMSE(sum, error);
      release(error);
      release(prevSum);
    }
    const result = this._divideMSESum(data.length, sum);
    release(sum);
    const res = (result instanceof Texture
      ? (result.toArray() as number[])
      : (result as number[]))[0];
    return res;
  }

  adjustWeights(): void {
    this.getChanges();
    this.changeBiases();
  }

  buildRunInput(): void {
    let weightedSum: (sum: number) => number;
    switch (this.trainOpts.activation) {
      case 'sigmoid':
        weightedSum = weightedSumSigmoid;
        break;
      case 'relu':
        weightedSum = weightedSumRelu;
        break;
      case 'leaky-relu':
        weightedSum = weightedSumLeakyRelu;
        break;
      case 'tanh':
        weightedSum = weightedSumTanh;
        break;
      case 'mish':
        weightedSum = weightedSumMish;
        break;
      default:
        throw new Error(
          `Unknown activation ${this.trainOpts.activation}. Available activations are: 'sigmoid', 'relu', 'leaky-relu', 'tanh'`
        );
    }
    weightedSum = alias(
      utils.getMinifySafeName(() => weightedSum),
      weightedSum
    );
    for (let layer = 1; layer <= this.outputLayer; layer++) {
      this.forwardPropagate[layer] = this.gpu.createKernelMap(
        {
          outputs: weightedSum,
        },
        function (
          this: ISizedKernelThis,
          weights: number[][],
          biases: number[],
          inputs: number[]
        ): number {
          let sum = biases[this.thread.x];
          for (let k = 0; k < this.constants.size; k++) {
            sum += weights[this.thread.x][k] * inputs[k];
          }

          weightedSum(sum);
          return sum;
        },
        {
          output: [this.sizes[layer]],
          pipeline: true,
          constants: {
            size: this.sizes[layer - 1],
          },
          immutable: true, // false
        }
      ) as WeightedSum;
    }

    // this.texturizeInputData = this.gpu.createKernel(
    //   function (value: number[]): number {
    //     return value[this.thread.x];
    //   },
    //   {
    //     pipeline: true,
    //     immutable: true,
    //   }
    // );
  }

  // eslint-disable-next-line @typescript-eslint/ban-ts-comment
  // @ts-expect-error
  runInput = (input: KernelOutput): KernelOutput => {
    let output;
    this.outputs[0] = input;
    for (let layer = 1; layer <= this.outputLayer; layer++) {
      release(this.outputs[layer]);
      release(this.outputsPreActivation[layer]);
      const outputs = this.forwardPropagate[layer](
        this.weights[layer],
        this.biases[layer],
        input
      );
      this.outputs[layer] = output = input = outputs.outputs;
      this.outputsPreActivation[layer] = outputs.result;
    }
    return output;
  };

  buildCalculateDeltas(): void {
    let calcDeltas: GPUFunction<[number, number, number]>;
    switch (this.trainOpts.activation) {
      case 'sigmoid':
        calcDeltas = calcDeltasSigmoid;
        break;
      case 'relu':
        calcDeltas = calcDeltasRelu;
        break;
      case 'leaky-relu':
        calcDeltas = calcDeltasLeakyRelu;
        break;
      case 'tanh':
        calcDeltas = calcDeltasTanh;
        break;
      case 'mish':
        calcDeltas = calcDeltasMish;
        break;
      default:
        throw new Error(
          `Unknown activation ${this.trainOpts.activation}. Available activations are: 'sigmoid', 'relu', 'leaky-relu', 'tanh', ''mish'`
        );
    }

    calcDeltas = alias(
      utils.getMinifySafeName(() => calcDeltas),
      calcDeltas
    );
    this.gpu.addFunction(calcDeltas);
    for (let layer = this.outputLayer; layer > 0; layer--) {
      if (layer === this.outputLayer) {
        // eslint-disable-next-line @typescript-eslint/ban-ts-comment
        // @ts-expect-error
        this.backwardPropagate[this.outputLayer] = this.gpu.createKernelMap(
          {
            error: calcErrorOutput,
          },
          function (
            this: IKernelFunctionThis,
            outputs: number[],
            targets: number[],
            outputsPA: number[]
          ): number {
            const output = outputs[this.thread.x];
            const target = targets[this.thread.x];
            const outputPA = outputsPA[this.thread.x];
            // eslint-disable-next-line @typescript-eslint/ban-ts-comment
            // @ts-expect-error
            return calcDeltas(
              calcErrorOutput(output, target),
              output,
              outputPA
            );
          },
          {
            output: [this.sizes[this.outputLayer]],
            pipeline: true,
            immutable: true,
          }
        );
      } else {
        // eslint-disable-next-line @typescript-eslint/ban-ts-comment
        // @ts-expect-error
        this.backwardPropagate[layer] = this.gpu.createKernelMap(
          {
            error: calcError,
          },
          function (
            this: ISizedKernelThis,
            nextWeights: number[][],
            outputs: number[],
            nextDeltas: number[],
            outputsPA: number[]
          ): number {
            const output = outputs[this.thread.x];
            const outputPA = outputsPA[this.thread.x];
            // eslint-disable-next-line @typescript-eslint/ban-ts-comment
            // @ts-expect-error
            return calcDeltas(
              calcError(
                this.thread.x,
                this.constants.size,
                nextWeights,
                nextDeltas
              ),
              output,
              outputPA
            );
          },
          {
            output: [this.sizes[layer]],
            pipeline: true,
            constants: {
              size: this.sizes[layer + 1],
            },
            immutable: true,
          }
        );
      }
    }
  }

  calculateDeltas = (target: KernelOutput): void => {
    for (let layer = this.outputLayer; layer > 0; layer--) {
      release(this.deltas[layer]);
      release(this.errors[layer]);

      let output;
      if (layer === this.outputLayer) {
        // eslint-disable-next-line @typescript-eslint/ban-ts-comment
        // @ts-expect-error
        output = this.backwardPropagate[layer](
          this.outputs[layer],
          target,
          this.outputsPreActivation[layer]
        );
        this.errors[layer] = output.error;
      } else {
        // eslint-disable-next-line @typescript-eslint/ban-ts-comment
        // @ts-expect-error
        output = this.backwardPropagate[layer](
          this.weights[layer + 1],
          this.outputs[layer],
          this.deltas[layer + 1],
          this.outputsPreActivation[layer]
        );
      }
      this.deltas[layer] = output.result;
      // this.errors[layer] = output.error;
    }
  };

  buildGetChanges(): void {
    const { praxis } = this.trainOpts;
    if (praxis === 'adam') {
      this._buildGetChangesAdam();
    } else if (praxis === 'adamw') {
      this._buildGetChangesAdamW();
    } else {
      this._buildGetChanges();
    }
  }

  _buildGetChanges(): void {
    for (let layer = 1; layer <= this.outputLayer; layer++) {
      // eslint-disable-next-line @typescript-eslint/ban-ts-comment
      // @ts-expect-error
      this.changesPropagate[layer] = this.gpu.createKernelMap(
        {
          weights: addWeights,
          changes: calcChanges,
        },
        function (
          this: IKernelFunctionThis<{
            learningRate: number;
            momentum: number;
          }>,
          previousOutputs: number[],
          deltas: number[],
          weights: number[][],
          previousChanges: number[][]
        ) {
          const change = calcChanges(
            this.constants.learningRate,
            this.constants.momentum,
            previousChanges[this.thread.y][this.thread.x],
            deltas[this.thread.y],
            previousOutputs[this.thread.x]
          );
          return addWeights(change, weights[this.thread.y][this.thread.x]);
        },
        {
          output: [this.sizes[layer - 1], this.sizes[layer]],
          pipeline: true,
          constants: {
            learningRate: this.trainOpts.learningRate,
            momentum: this.trainOpts.momentum,
          },
          immutable: true,
        }
      );
    }
  }

  _buildGetChangesAdam(): void {
    for (let layer = 1; layer <= this.outputLayer; layer++) {
      // eslint-disable-next-line @typescript-eslint/ban-ts-comment
      // @ts-expect-error
      this.changesPropagate[layer] = this.gpu.createKernelMap(
        {
          changesLow: calcChangesLowAdam,
          changesHigh: calcChangesHighAdam,
          weights: addWeightsAdam,
        },
        function (
          this: IKernelFunctionThis<{
            iterations: number;
            beta1: number;
            beta2: number;
            epsilon: number;
            learningRate: number;
          }>,
          previousOutputs: number[],
          deltas: number[],
          weights: number[][],
          currentChangesLow: number[][],
          currentChangesHigh: number[][]
        ) {
          const gradient =
            deltas[this.thread.y] * previousOutputs[this.thread.x];
          const changeLow = calcChangesLowAdam(
            currentChangesLow[this.thread.y][this.thread.x],
            this.constants.beta1,
            gradient
          );
          const changeHigh = calcChangesHighAdam(
            currentChangesHigh[this.thread.y][this.thread.x],
            this.constants.beta2,
            gradient
          );
          const momentumCorrection =
            changeLow /
            (1 - Math.pow(this.constants.beta1, this.constants.iterations));
          const gradientCorrection =
            changeHigh /
            (1 - Math.pow(this.constants.beta2, this.constants.iterations));
          return addWeightsAdam(
            weights[this.thread.y][this.thread.x],
            this.constants.learningRate,
            momentumCorrection,
            gradientCorrection,
            this.constants.epsilon
          );
        },
        {
          output: [this.sizes[layer - 1], this.sizes[layer]],
          pipeline: true,
          constants: {
            learningRate: this.trainOpts.learningRate,
            beta1: this.trainOpts.beta1,
            beta2: this.trainOpts.beta2,
            epsilon: this.trainOpts.epsilon,
          },
          immutable: true,
        }
      );
    }
  }

  _buildGetChangesAdamW(): void {
    for (let layer = 1; layer <= this.outputLayer; layer++) {
      // eslint-disable-next-line @typescript-eslint/ban-ts-comment
      // @ts-expect-error
      this.changesPropagate[layer] = this.gpu.createKernelMap(
        {
          changesLow: calcChangesLowAdam,
          changesHigh: calcChangesHighAdam,
          weights: addWeightsAdamW,
        },
        function (
          this: IKernelFunctionThis<{
            iterations: number;
            beta1: number;
            beta2: number;
            epsilon: number;
            learningRate: number;
            weightDecay: number;
          }>,
          previousOutputs: number[],
          deltas: number[],
          weights: number[][],
          currentChangesLow: number[][],
          currentChangesHigh: number[][]
        ) {
          const gradient =
            deltas[this.thread.y] * previousOutputs[this.thread.x];
          const changeLow = calcChangesLowAdam(
            currentChangesLow[this.thread.y][this.thread.x],
            this.constants.beta1,
            gradient
          );
          const changeHigh = calcChangesHighAdam(
            currentChangesHigh[this.thread.y][this.thread.x],
            this.constants.beta2,
            gradient
          );
          const momentumCorrection =
            changeLow /
            (1 - Math.pow(this.constants.beta1, this.constants.iterations));
          const gradientCorrection =
            changeHigh /
            (1 - Math.pow(this.constants.beta2, this.constants.iterations));
          return addWeightsAdamW(
            weights[this.thread.y][this.thread.x],
            this.constants.learningRate,
            momentumCorrection,
            gradientCorrection,
            this.constants.epsilon,
            this.constants.weightDecay
          );
        },
        {
          output: [this.sizes[layer - 1], this.sizes[layer]],
          pipeline: true,
          constants: {
            learningRate: this.trainOpts.learningRate,
            beta1: this.trainOpts.beta1,
            beta2: this.trainOpts.beta2,
            epsilon: this.trainOpts.epsilon,
            weightDecay: this.trainOpts.weightDecay,
          },
          immutable: true,
        }
      );
    }
  }

  getChanges(): void {
    const { praxis } = this.trainOpts;
    if (praxis === 'adam') {
      this._getChangesAdam<ChangeePropagateAdam>();
    } else if (praxis === 'adamw') {
      this._getChangesAdam<ChangeePropagateAdamW>();
    } else {
      this._getChanges();
    }
  }

  _getChanges(): void {
    for (let layer = 1; layer <= this.outputLayer; layer++) {
      const weights = this.weights[layer];
      const changes = this.changes[layer];
      const output = (this.changesPropagate[layer] as ChangeePropagate)(
        this.outputs[layer - 1],
        this.deltas[layer],
        weights,
        changes
      );
      release(weights);
      release(changes);
      this.weights[layer] = output.weights;
      this.changes[layer] = output.changes;
      release(output.result);
    }
  }

  _getChangesAdam<
    T extends ChangeePropagateAdam | ChangeePropagateAdamW
  >(): void {
    for (let layer = 1; layer <= this.outputLayer; layer++) {
      const weights = this.weights[layer];
      const changesLow = this.changesLow[layer];
      const changesHigh = this.changesHigh[layer];
      const changesPropagate = this.changesPropagate[layer] as T;
      const output = changesPropagate.setConstants({
        iterations: this.iterations,
      })(
        this.outputs[layer - 1],
        this.deltas[layer],
        weights,
        changesLow,
        changesHigh
      );
      release(weights);
      release(changesLow);
      release(changesHigh);
      this.weights[layer] = output.weights;
      this.changesLow[layer] = output.changesLow;
      this.changesHigh[layer] = output.changesHigh;
      release(output.result);
      if (layer !== this.outputLayer) this.iterations++;
    }
  }

  buildChangeBiases(): void {
    const { praxis } = this.trainOpts;
    if (praxis === 'adam' || praxis === 'adamw') {
      this._buildChangeBiasesAdam();
    } else {
      this._buildChangeBiases();
    }
  }

  _buildChangeBiases(): void {
    for (let layer = 1; layer <= this.outputLayer; layer++) {
      this.biasesPropagate[layer] = this.gpu.createKernel(addBiases, {
        output: [this.sizes[layer]],
        pipeline: true,
        constants: {
          learningRate: this.trainOpts.learningRate,
        },
        immutable: true,
      });
    }
  }

  _buildChangeBiasesAdam(): void {
    for (let layer = 1; layer <= this.outputLayer; layer++) {
      // eslint-disable-next-line @typescript-eslint/ban-ts-comment
      // @ts-expect-error
      this.biasesPropagate[layer] = this.gpu.createKernelMap(
        {
          biases: addBiasesAdam,
          biasChangesLow: calcBiasChangesLowAdam,
          biasChangesHigh: calcBiasChangesHighAdam,
        },
        function (
          this: IKernelFunctionThis<{
            iterations: number;
            beta1: number;
            beta2: number;
            epsilon: number;
            learningRate: number;
          }>,
          deltas: number[],
          biases: number[],
          currentBiasChangesLow: number[],
          currentBiasChangesHigh: number[]
        ) {
          const biasGradient = deltas[this.thread.x];
          const biasChangeLow = calcBiasChangesLowAdam(
            currentBiasChangesLow[this.thread.x],
            this.constants.beta1,
            biasGradient
          );
          const biasChangeHigh = calcBiasChangesHighAdam(
            currentBiasChangesHigh[this.thread.x],
            this.constants.beta2,
            biasGradient
          );
          const biasMomentumCorrection =
            biasChangeLow /
            (1 - Math.pow(this.constants.beta1, this.constants.iterations));
          const biasGradientCorrection =
            biasChangeHigh /
            (1 - Math.pow(this.constants.beta2, this.constants.iterations));

          return addBiasesAdam(
            biases[this.thread.x],
            biasMomentumCorrection,
            biasGradientCorrection,
            this.constants.learningRate,
            this.constants.epsilon
          );
        },
        {
          output: [this.sizes[layer]],
          pipeline: true,
          constants: {
            learningRate: this.trainOpts.learningRate,
            beta1: this.trainOpts.beta1,
            beta2: this.trainOpts.beta2,
            epsilon: this.trainOpts.epsilon,
          },
          immutable: true,
        }
      );
    }
  }

  changeBiases(): void {
    const { praxis } = this.trainOpts;
    if (praxis === 'adam' || praxis === 'adamw') {
      this._changeBiasesAdam();
    } else {
      this._changeBiases();
    }
  }

  _changeBiases(): void {
    for (let layer = 1; layer <= this.outputLayer; layer++) {
      const biases = this.biases[layer];
      this.biases[layer] = (this.biasesPropagate[layer] as BiasesPropagate)(
        biases,
        this.deltas[layer]
      );
      release(biases);
    }
  }

  _changeBiasesAdam(): void {
    for (let layer = 1; layer <= this.outputLayer; layer++) {
      const biases = this.biases[layer];
      const biasChangeeLow = this.biasChangesLow[layer];
      const biasChangeeHigh = this.biasChangesHigh[layer];
      const biasesPropagate = this.biasesPropagate[
        layer
      ] as BiasesPropagateAdam;
      const output = biasesPropagate.setConstants({
        iterations: this.iterations,
      })(this.deltas[layer], biases, biasChangeeLow, biasChangeeHigh);
      release(biases);
      release(biasChangeeLow);
      release(biasChangeeHigh);
      this.biases[layer] = output.biases;
      this.biasChangesLow[layer] = output.biasChangesLow;
      this.biasChangesHigh[layer] = output.biasChangesHigh;
      release(output.result);
    }
  }

  buildGetMSE(): void {
    this.getMSE = this.gpu.createKernel(mse, {
      output: [1],
      constants: {
        size: this.sizes[this.outputLayer],
      },
      pipeline: true,
      immutable: true,
    });
    this._addMSE = this.gpu.createKernel(
      function (value1: number[], value2: number[]): number {
        return value1[0] + value2[0];
      },
      {
        output: [1],
        pipeline: true,
        immutable: true,
      }
    );
    this._divideMSESum = this.gpu.createKernel(
      function (length: number, mseSum: number[]): number {
        const value = mseSum[0];
        if (value > 0) {
          return value / length;
        }
        return 0;
      },
      {
        output: [1],
        // pipeline: false,
      }
    );
  }

  run(input: InputType): OutputType {
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
    const outputTextures = this.runInput(formattedInput);
    const output =
      outputTextures instanceof Texture
        ? outputTextures.toArray()
        : outputTextures;

    if (this.outputLookup) {
      return (lookup.toObject(
        this.outputLookup,
        output as Float32Array
      ) as unknown) as OutputType;
    }

    return (output as unknown) as OutputType;
  }

  // @ts-expect-error the underlying network works as normal, but we are working on the GPU
  prepTraining(
    data: Array<INeuralNetworkDatum<InputType, OutputType>>,
    options: Partial<INeuralNetworkTrainOptions> = {}
  ): INeuralNetworkGPUPreppedTrainingData {
    this.updateTrainingOptions(options);
    const preparedData = this.formatData(data);
    const endTime = Date.now() + this.trainOpts.timeout;

    const status = {
      error: 1,
      iterations: 0,
    };

    this.verifyIsInitialized(preparedData);

    const texturizeOutputData = this.gpu.createKernel(
      function (value: number[]): number {
        return value[this.thread.x];
      },
      {
        pipeline: true,
        immutable: true,
      }
    );

    try {
      return {
        preparedData: preparedData.map((set) => ({
          input: texturizeOutputData.setOutput([set.input.length])(set.input),
          output: texturizeOutputData.setOutput([set.output.length])(
            set.output
          ),
        })),
        status,
        endTime,
      };
    } finally {
      texturizeOutputData.destroy();
    }
  }

  // eslint-disable-next-line @typescript-eslint/ban-ts-comment
  // @ts-expect-error
  toFunction(): (input: InputType) => OutputType {
    throw new Error(
      `${this.constructor.name}-toFunction is not yet implemented`
    );
  }

  toJSON(): INeuralNetworkJSON {
    if (this.sizes === null) {
      this.initialize();
    }
    // use Array.from, keeping json small
    const jsonLayerWeights = this.weights.map((layerWeights) => {
      const res = (layerWeights instanceof Texture
        ? (layerWeights.toArray() as Float32Array[])
        : (layerWeights as Float32Array[])
      ).map((layerWeights) => Array.from(layerWeights));
      return res;
    });
    const jsonLayerBiases = this.biases.map((layerBiases) => {
      const res = Array.from(
        layerBiases instanceof Texture
          ? (layerBiases.toArray() as Float32Array)
          : (layerBiases as Float32Array)
      );
      return res;
    });
    const jsonLayers: IJSONLayer[] = [];
    for (let i = 0; i <= this.outputLayer; i++) {
      jsonLayers.push({
        weights: jsonLayerWeights[i] ?? [],
        biases: jsonLayerBiases[i] ?? [],
      });
    }
    return {
      type: 'NeuralNetworkGPU',
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
}

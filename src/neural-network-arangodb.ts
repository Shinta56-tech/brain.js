import { query, QueryOptions } from './utilities/arangodb';

export class NeuralNetworkArangoDB {
  async testArangoDB(aql: string, options: QueryOptions): Promise<void> {
    await query(aql, options);
  }
}

import { Database } from 'arangojs';
import { QueryOptions } from 'arangojs/database';

export interface ArangoDBConfig {
  url: string;
  databaseName: string;
  auth: {
    username: string;
    password: string;
  };
}

export interface NeuronSchema {
  name: string;
  field: string;
  num: number;
  i1: number | string;
  i2?: number | string;
  i3?: number | string;
}

let db: any; // Database

export const connect = (config?: ArangoDBConfig | undefined): void => {
  if (!config) {
    db = new Database({
      url: 'http://localhost:8529',
      databaseName: 'NetkeibaDBML',
      auth: {
        username: 'root',
        password: 'password',
      },
    });
  } else {
    db = new Database(config);
  }
};

export type QueryResult = number[] | NeuronSchema[];

// prettier-ignore
export const query = async (aql: string, options?: QueryOptions): Promise<QueryResult> => {
  const cursor = await db.query(aql, options);
  const result = await cursor.all();
  return (result as unknown) as QueryResult;
};

export { QueryOptions };

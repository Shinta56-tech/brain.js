import { Database } from 'arangojs';
import { QueryOptions } from 'arangojs/database';
import config from './arangodb.config';

const db = new Database({
  url: config.url,
  databaseName: config.databaseName,
  auth: {
    username: config.username,
    password: config.password,
  },
});

export const query = async (aql: string, options?: QueryOptions) => {
  const cursor = await db.query(aql, options);
  return await cursor.all();
};

export { QueryOptions };

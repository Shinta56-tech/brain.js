// arangodb.test.ts

import { query } from './arangodb';
import { Database } from 'arangojs';

jest.mock('arangojs', () => {
  const mockDatabase = {
    query: jest.fn(),
  };
  return { Database: jest.fn(() => mockDatabase) };
});

describe('query function', () => {
  let dbInstance: any; // Database instance mock

  beforeEach(() => {
    // Reset mock instance before each test
    dbInstance = new Database();
    (dbInstance.query as jest.Mock).mockClear();
  });

  it('should insert a document into TEST collection', async () => {
    const aql = `
      INSERT { _key: '1', name: 'Test Document' } INTO TEST
      RETURN NEW
    `;

    // Mock the query to resolve with a mock result
    (dbInstance.query as jest.Mock).mockResolvedValue({
      all: jest.fn().mockResolvedValue([{ _key: '1', name: 'Test Document' }]),
    });

    const result = await query(aql);
    expect(result).toEqual([{ _key: '1', name: 'Test Document' }]);

    // Verify that the query method was called with the correct AQL
    expect(dbInstance.query).toHaveBeenCalledWith(aql);
  });

  it('should delete a document from TEST collection', async () => {
    const aql = `
      REMOVE '1' IN TEST
      RETURN OLD
    `;

    // Mock the query to resolve with a mock result
    (dbInstance.query as jest.Mock).mockResolvedValue({
      all: jest.fn().mockResolvedValue([{ _key: '1', name: 'Test Document' }]),
    });

    const result = await query(aql);
    expect(result).toEqual([{ _key: '1', name: 'Test Document' }]);

    // Verify that the query method was called with the correct AQL
    expect(dbInstance.query).toHaveBeenCalledWith(aql);
  });

  it('should upsert a document into TEST collection (insert)', async () => {
    const aql = `
      UPSERT { _key: '1' }
      INSERT { _key: '1', name: 'Test Document' }
      UPDATE { _key: '1', name: 'Updated Document' }
      IN TEST
      RETURN NEW
    `;

    // Mock the query to resolve with a mock result for insert
    (dbInstance.query as jest.Mock).mockResolvedValueOnce({
      all: jest.fn().mockResolvedValue([{ _key: '1', name: 'Test Document' }]),
    });

    const resultInsert = await query(aql);
    expect(resultInsert).toEqual([{ _key: '1', name: 'Test Document' }]);

    // Verify that the query method was called with the correct AQL for insert
    expect(dbInstance.query).toHaveBeenCalledWith(aql);
  });

  it('should upsert a document into TEST collection (update)', async () => {
    const aql = `
      UPSERT { _key: '1' }
      INSERT { _key: '1', name: 'Test Document' }
      UPDATE { _key: '1', name: 'Updated Document' }
      IN TEST
      RETURN NEW
    `;

    // Mock the query to resolve with a mock result for update
    (dbInstance.query as jest.Mock).mockResolvedValueOnce({
      all: jest
        .fn()
        .mockResolvedValue([{ _key: '1', name: 'Updated Document' }]),
    });

    const resultUpdate = await query(aql);
    expect(resultUpdate).toEqual([{ _key: '1', name: 'Updated Document' }]);

    // Verify that the query method was called with the correct AQL for update
    expect(dbInstance.query).toHaveBeenCalledWith(aql);
  });
});

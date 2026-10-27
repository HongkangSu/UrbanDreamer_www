"""Bounded, persistent static route tuples shared by independent CPU processes."""
from __future__ import annotations

from contextlib import contextmanager
import fcntl
import hashlib
import json
import os
from pathlib import Path
import sqlite3
import warnings

import numpy as np

ROUTE_CACHE_CAPACITY = 200000
ALGORITHM_VERSION = 'penalty_dijkstra_ordered_adjacency_x3_retry_once_pad_first_v1'


def network_identity(network):
    digest = hashlib.sha256(ALGORITHM_VERSION.encode())
    digest.update(json.dumps([network.edge_ids, network.adjacency_list],
                             separators=(',', ':')).encode())
    digest.update(np.asarray(network.passenger_allowed, dtype=np.uint8).tobytes())
    digest.update(np.asarray(network.free_flow_travel_seconds, dtype='<f8').tobytes())
    return digest.hexdigest()


class SharedRouteCache:
    """WAL readers share one bounded insertion-ordered store per network.

    Hits do not acquire a write lock. Per-key file locks prevent duplicate cold
    searches; different keys may be computed concurrently outside SQLite writes.
    Each process opens its own connection, including after pickle/spawn/fork.
    """

    def __init__(self, directory, identity, capacity=ROUTE_CACHE_CAPACITY):
        self._connection = None
        self._pid = None
        self._lock_fd = None
        if isinstance(capacity, bool) or not isinstance(capacity, int) or capacity < 1:
            raise ValueError('shared route cache capacity must be a positive integer')
        self.path = Path(directory).resolve() / f'{identity}_c{capacity}.sqlite3'
        self.capacity = capacity
        self._warned = False
        self.hits = self.misses = self.computations = 0

    def __getstate__(self):
        return dict(path=self.path, capacity=self.capacity)

    def __setstate__(self, state):
        self.path, self.capacity = state['path'], state['capacity']
        self._connection = self._pid = self._lock_fd = None
        self._warned = False
        self.hits = self.misses = self.computations = 0

    def _open(self):
        if self._connection is not None and self._pid == os.getpid():
            return self._connection
        self.close()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        connection = sqlite3.connect(self.path, timeout=30., isolation_level=None)
        try:
            connection.execute('PRAGMA journal_mode=WAL')
            connection.execute('PRAGMA synchronous=NORMAL')
            connection.execute('CREATE TABLE IF NOT EXISTS routes ('
                'sequence INTEGER PRIMARY KEY, current INTEGER NOT NULL, destination INTEGER NOT NULL, '
                'paths TEXT NOT NULL, UNIQUE(current,destination))')
            connection.execute('CREATE TABLE IF NOT EXISTS size (id INTEGER PRIMARY KEY CHECK(id=1), '
                               'entries INTEGER NOT NULL)')
            connection.execute('INSERT OR IGNORE INTO size SELECT 1, COUNT(*) FROM routes')
            self._lock_fd = os.open(self.path.with_suffix('.locks'), os.O_CREAT | os.O_RDWR, 0o600)
        except Exception:
            connection.close()
            raise
        self._connection, self._pid = connection, os.getpid()
        return connection

    def close(self):
        if self._connection is not None:
            self._connection.close()
        if self._lock_fd is not None:
            os.close(self._lock_fd)
        self._connection = self._pid = self._lock_fd = None

    def _get(self, current, destination):
        row = self._open().execute('SELECT paths FROM routes WHERE current=? AND destination=?',
                                  (current, destination)).fetchone()
        if row is None:
            return None
        paths = json.loads(row[0])
        if (not isinstance(paths, list) or len(paths) != 4 or any(
                not isinstance(path, list) or not path or path[0] != current or path[-1] != destination
                or any(type(edge) is not int or edge < 0 for edge in path) for path in paths)):
            raise ValueError('invalid shared route cache payload')
        return tuple(tuple(path) for path in paths)

    @contextmanager
    def _key_lock(self, current, destination):
        # One stable stripe per pair; no dependence on Python's randomized hash.
        stripe = (current * 1000003 + destination) % 1024
        fcntl.lockf(self._lock_fd, fcntl.LOCK_EX, 1, stripe, os.SEEK_SET)
        try:
            yield
        finally:
            fcntl.lockf(self._lock_fd, fcntl.LOCK_UN, 1, stripe, os.SEEK_SET)

    def _put(self, current, destination, paths):
        connection = self._open()
        payload = json.dumps(paths, separators=(',', ':'))
        connection.execute('BEGIN IMMEDIATE')
        try:
            inserted = connection.execute('INSERT OR IGNORE INTO routes(current,destination,paths) '
                                          'VALUES(?,?,?)', (current, destination, payload)).rowcount
            if inserted:
                connection.execute('UPDATE size SET entries=entries+1 WHERE id=1')
                count = connection.execute('SELECT entries FROM size WHERE id=1').fetchone()[0]
                overflow = count - self.capacity
                if overflow > 0:
                    connection.execute('DELETE FROM routes WHERE sequence IN '
                        '(SELECT sequence FROM routes ORDER BY sequence LIMIT ?)', (overflow,))
                    connection.execute('UPDATE size SET entries=entries-? WHERE id=1', (overflow,))
            connection.execute('COMMIT')
        except Exception:
            connection.execute('ROLLBACK')
            raise

    def get_or_compute(self, current, destination, compute, *, fallback=None):
        fallback = compute if fallback is None else fallback
        try:
            paths = self._get(current, destination)
        except (OSError, sqlite3.Error, ValueError) as error:
            return self._fallback(current, destination, fallback, error)
        if paths is not None:
            self.hits += 1
            return paths
        self.misses += 1
        try:
            with self._key_lock(current, destination):
                try:
                    paths = self._get(current, destination)
                except (OSError, sqlite3.Error, ValueError) as error:
                    return self._fallback(current, destination, fallback, error)
                if paths is None:
                    # Keep topology failures visible; they are not cache failures.
                    self.computations += 1
                    paths = compute(current, destination)
                    try:
                        self._put(current, destination, paths)
                    except (OSError, sqlite3.Error) as error:
                        self._warn(error)
                else:
                    self.hits += 1
                return paths
        except (OSError, sqlite3.Error) as error:
            return self._fallback(current, destination, fallback, error)

    def _warn(self, error):
        if not self._warned:
            warnings.warn(f'Shared route cache unavailable; using local candidates: {error}',
                          RuntimeWarning, stacklevel=3)
            self._warned = True

    def _fallback(self, current, destination, compute, error):
        self._warn(error)
        self.computations += 1
        return compute(current, destination)

    def statistics(self):
        count = self._open().execute('SELECT entries FROM size WHERE id=1').fetchone()[0]
        return dict(path=str(self.path), capacity=self.capacity, entries=count,
                    hits=self.hits, misses=self.misses, computations=self.computations)

    def __del__(self):
        self.close()

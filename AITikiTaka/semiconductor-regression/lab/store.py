"""Project-scoped append-only records and local artifacts."""
from __future__ import annotations

import json
import os
import sqlite3
import uuid
import unicodedata
from datetime import datetime, timezone
from pathlib import Path

from lab.constants import ROOT


def now():
    return datetime.now(timezone.utc).isoformat()


class Store:
    def __init__(self, root=None):
        self.root = Path(root or os.environ.get('TIKITAKA_DATA_DIR', ROOT / 'local_data'))
        self.root.mkdir(parents=True, exist_ok=True)
        self.db = self.root / 'lab.sqlite3'
        with self.connect() as conn:
            conn.execute('''CREATE TABLE IF NOT EXISTS projects
                (id TEXT PRIMARY KEY, name TEXT NOT NULL, created TEXT NOT NULL,
                 settings TEXT NOT NULL, active_model TEXT)''')
            conn.execute('''CREATE TABLE IF NOT EXISTS records
                (id TEXT PRIMARY KEY, project_id TEXT NOT NULL REFERENCES projects(id),
                 kind TEXT NOT NULL, created TEXT NOT NULL, payload TEXT NOT NULL)''')
            conn.execute('CREATE INDEX IF NOT EXISTS records_project ON records(project_id, kind)')

    def connect(self):
        conn = sqlite3.connect(self.db, timeout=30)
        conn.row_factory = sqlite3.Row
        conn.execute('PRAGMA foreign_keys=ON')
        return conn

    def create_project(self, name, settings):
        name = unicodedata.normalize('NFC', name).strip()
        if not name:
            raise ValueError('프로젝트 이름을 입력해주세요.')
        pid = uuid.uuid4().hex
        with self.connect() as conn:
            # Serialize the name check and insert across browser sessions.
            # Existing duplicate projects remain intact with their own records.
            conn.execute('BEGIN IMMEDIATE')
            key = name.casefold()
            if any(unicodedata.normalize('NFC', row['name']).strip().casefold() == key
                   for row in conn.execute('SELECT name FROM projects')):
                raise ValueError('이미 같은 이름의 공정이 있습니다. 기존 공정을 선택하거나 다른 이름을 입력해주세요.')
            conn.execute('INSERT INTO projects VALUES (?,?,?,?,NULL)',
                         (pid, name, now(), json.dumps(settings, ensure_ascii=False)))
        return pid

    def projects(self):
        with self.connect() as conn:
            return [dict(r) for r in conn.execute('SELECT * FROM projects ORDER BY created')]

    def project(self, pid):
        with self.connect() as conn:
            row = conn.execute('SELECT * FROM projects WHERE id=?', (pid,)).fetchone()
        if row is None:
            raise ValueError('프로젝트를 찾을 수 없습니다.')
        result = dict(row)
        result['settings'] = json.loads(result['settings'])
        return result

    def settings(self, pid, settings):
        with self.connect() as conn:
            conn.execute('UPDATE projects SET settings=? WHERE id=?',
                         (json.dumps(settings, ensure_ascii=False, allow_nan=False), pid))

    def directory(self, pid):
        self.project(pid)
        path = self.root / 'projects' / pid
        path.mkdir(parents=True, exist_ok=True)
        return path

    def add(self, pid, kind, payload, rid=None):
        rid = rid or uuid.uuid4().hex
        with self.connect() as conn:
            conn.execute('INSERT INTO records VALUES (?,?,?,?,?)',
                         (rid, pid, kind, now(), json.dumps(payload, ensure_ascii=False, allow_nan=False)))
        return rid

    def records(self, pid, kind):
        with self.connect() as conn:
            rows = conn.execute('SELECT * FROM records WHERE project_id=? AND kind=? ORDER BY created',
                                (pid, kind)).fetchall()
        return [dict(r, payload=json.loads(r['payload'])) for r in rows]

    def record(self, pid, rid, kind):
        with self.connect() as conn:
            row = conn.execute('SELECT * FROM records WHERE project_id=? AND id=? AND kind=?',
                               (pid, rid, kind)).fetchone()
        if row is None:
            raise ValueError('이 프로젝트의 기록이 아닙니다.')
        return dict(row, payload=json.loads(row['payload']))

    def activate(self, pid, model_id):
        record = self.record(pid, model_id, 'model')
        if not record['payload']['eligible']:
            raise ValueError('검증 허용 오차 기준을 충족한 모델만 활성화할 수 있습니다.')
        with self.connect() as conn:
            conn.execute('UPDATE projects SET active_model=? WHERE id=?', (model_id, pid))

    def write(self, pid, relative, data):
        base = self.directory(pid).resolve()
        path = (base / relative).resolve()
        if base not in path.parents:
            raise ValueError('잘못된 저장 경로입니다.')
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(path.suffix + '.tmp')
        temporary.write_bytes(data)
        temporary.replace(path)
        return str(path.relative_to(base))

    def path(self, pid, relative):
        base = self.directory(pid).resolve()
        path = (base / relative).resolve()
        if base not in path.parents:
            raise ValueError('잘못된 파일 경로입니다.')
        return path

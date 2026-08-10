"""Crash-safe repair for LeRobotDataset (v3.0) recordings.

Why this exists
---------------
lerobot writes each episode's data/metadata as a parquet *row group* through a
long-lived ``pyarrow.ParquetWriter`` whose **footer is only flushed at
``finalize()``**. If the recorder dies non-gracefully — SIGKILL, OOM, power
loss, or a native segfault (e.g. an ffmpeg/pyav dylib clash) — ``finalize()``
never runs and *both* parquet files are left without a footer. Every byte of
every saved episode is still on disk, but pyarrow refuses to open a footerless
file, so the whole dataset (all episodes, not just the in-flight one) becomes
unreadable and ``LeRobotDataset.resume`` raises
``ArrowInvalid: Parquet magic bytes not found in footer``.

The pages are intact, so the footer is mechanically reconstructable: walk the
page headers to recover each row group's offsets/sizes, and pair them with the
schema (which we regenerate from the dataset's own ``meta/info.json``). This
module does exactly that, in place, backing up the corrupt file first. It runs
automatically at resume time (see ``operators/teleoperator/recorder.py``), so a
crash can bruise at most the un-saved in-flight episode — never a saved one.

It only *rebuilds footers* over pages that are already on disk; it does not
fabricate episodes. Pair it with ``metadata_buffer_size=1`` on the writer so
every saved episode's metadata row is flushed to disk immediately (otherwise up
to ``metadata_buffer_size-1`` metadata rows live only in memory and are lost on a
hard kill, footer or no footer).
"""
from __future__ import annotations

import json
import logging
import shutil
import struct
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

logger = logging.getLogger(__name__)

STATKEYS = ["min", "max", "mean", "std", "count", "q01", "q10", "q50", "q90", "q99"]

# ─────────────────────────── Thrift compact codec ───────────────────────────
# Just enough of the compact protocol to parse a parquet footer as a generic
# tree and re-serialize a tree we assemble by hand. Byte-exact round-trip is
# covered by the repo's recovery tests.

CT_STOP, CT_BOOL_T, CT_BOOL_F, CT_BYTE, CT_I16, CT_I32, CT_I64 = 0, 1, 2, 3, 4, 5, 6
CT_DOUBLE, CT_BINARY, CT_LIST, CT_SET, CT_MAP, CT_STRUCT = 7, 8, 9, 10, 11, 12


class _Struct:
    __slots__ = ("fields",)

    def __init__(self, fields=None):
        self.fields = fields if fields is not None else []  # list of [fid, ctype, value]

    def get(self, fid, default=None):
        for f in self.fields:
            if f[0] == fid:
                return f[2]
        return default

    def set(self, fid, ctype, value):
        for f in self.fields:
            if f[0] == fid:
                f[1], f[2] = ctype, value
                return
        self.fields.append([fid, ctype, value])
        self.fields.sort(key=lambda f: f[0])


class _List:
    __slots__ = ("etype", "items")

    def __init__(self, etype, items):
        self.etype, self.items = etype, items


class _R:
    def __init__(self, b, i=0):
        self.b, self.i = b, i

    def u8(self):
        v = self.b[self.i]
        self.i += 1
        return v

    def varint(self):
        shift = res = 0
        while True:
            x = self.b[self.i]
            self.i += 1
            res |= (x & 0x7F) << shift
            if not (x & 0x80):
                return res
            shift += 7

    def zz(self):
        n = self.varint()
        return (n >> 1) ^ -(n & 1)


def _read_scalar(r, ct):
    if ct == CT_BOOL_T:
        return True
    if ct == CT_BOOL_F:
        return False
    if ct == CT_BYTE:
        return r.u8()
    if ct in (CT_I16, CT_I32, CT_I64):
        return r.zz()
    if ct == CT_DOUBLE:
        v = struct.unpack("<d", r.b[r.i:r.i + 8])[0]
        r.i += 8
        return v
    if ct == CT_BINARY:
        n = r.varint()
        v = r.b[r.i:r.i + n]
        r.i += n
        return v
    if ct == CT_STRUCT:
        return _read_struct(r)
    if ct in (CT_LIST, CT_SET):
        return _read_list(r)
    if ct == CT_MAP:
        return _read_map(r)
    raise ValueError(f"scalar ctype {ct}")


def _read_list(r):
    h = r.u8()
    size, et = (h >> 4) & 0xF, h & 0xF
    if size == 15:
        size = r.varint()
    items = []
    for _ in range(size):
        items.append(r.u8() == 1 if et in (CT_BOOL_T, CT_BOOL_F) else _read_scalar(r, et))
    return _List(et, items)


def _read_map(r):
    size = r.varint()
    if size == 0:
        return ("map", [])
    kv = r.u8()
    kt, vt = (kv >> 4) & 0xF, kv & 0xF
    return ("map", kt, vt, [(_read_scalar(r, kt), _read_scalar(r, vt)) for _ in range(size)])


def _read_struct(r):
    s = _Struct()
    fid = 0
    while True:
        h = r.u8()
        if h == 0:
            return s
        delta, ct = (h >> 4) & 0xF, h & 0xF
        fid = r.zz() if delta == 0 else fid + delta
        if ct in (CT_BOOL_T, CT_BOOL_F):
            s.fields.append([fid, ct, ct == CT_BOOL_T])
        else:
            s.fields.append([fid, ct, _read_scalar(r, ct)])


class _W:
    def __init__(self):
        self.o = bytearray()

    def varint(self, v):
        while True:
            b = v & 0x7F
            v >>= 7
            if v:
                self.o.append(b | 0x80)
            else:
                self.o.append(b)
                return

    def zz(self, n):
        self.varint(((n << 1) ^ (n >> 63)) & ((1 << 64) - 1))


def _write_scalar(w, ct, val):
    if ct in (CT_BOOL_T, CT_BOOL_F):
        return
    if ct == CT_BYTE:
        w.o.append(val & 0xFF)
    elif ct in (CT_I16, CT_I32, CT_I64):
        w.zz(val)
    elif ct == CT_DOUBLE:
        w.o += struct.pack("<d", val)
    elif ct == CT_BINARY:
        b = val if isinstance(val, (bytes, bytearray)) else str(val).encode()
        w.varint(len(b))
        w.o += b
    elif ct == CT_STRUCT:
        _write_struct(w, val)
    elif ct in (CT_LIST, CT_SET):
        _write_list(w, val)
    else:
        raise ValueError(f"write ctype {ct}")


def _write_list(w, lst):
    n, et = len(lst.items), lst.etype
    if n < 15:
        w.o.append((n << 4) | et)
    else:
        w.o.append((15 << 4) | et)
        w.varint(n)
    for it in lst.items:
        if et in (CT_BOOL_T, CT_BOOL_F):
            w.o.append(1 if it else 2)
        else:
            _write_scalar(w, et, it)


def _write_struct(w, s):
    last = 0
    for fid, ct, val in s.fields:
        delta = fid - last
        if 1 <= delta <= 15:
            w.o.append((delta << 4) | ct)
        else:
            w.o.append(ct)
            w.zz(fid)
        last = fid
        if ct not in (CT_BOOL_T, CT_BOOL_F):
            _write_scalar(w, ct, val)
    w.o.append(0)


def _has_footer(path: Path) -> bool:
    try:
        pq.read_metadata(path)
        return True
    except Exception:
        return False


# ─────────────────────────── page walking ───────────────────────────

def _walk_pages(buf: bytes):
    """Walk contiguous parquet pages from offset 4 to EOF (no footer needed).
    Returns list of dicts: type (0 data / 2 dict), nvals, hdr_off, hdr_end,
    body_end, csize, usize."""
    assert buf[:4] == b"PAR1", "missing leading PAR1 magic"
    i, N, pages = 4, len(buf), []
    while i < N:
        if N - i < 2:
            break
        r = _R(buf, i)
        s = _read_struct(r)
        ptype, usize, csize = s.get(1), s.get(2), s.get(3)
        if ptype is None or csize is None:
            break
        nvals = None
        for sub in (5, 7, 8):  # DataPageHeader / DictPageHeader / DataPageHeaderV2
            sh = s.get(sub)
            if isinstance(sh, _Struct):
                nvals = sh.get(1)
        body_end = r.i + csize
        pages.append({"type": ptype, "usize": usize, "csize": csize,
                      "nvals": nvals, "hdr_off": i, "hdr_end": r.i, "body_end": body_end})
        i = body_end
    return pages, N, i


def _group_row_groups(pages, ncols):
    """Group pages into row groups of ``ncols`` column chunks; each chunk starts
    at a DICT page and includes the following DATA pages."""
    chunks, cur = [], None
    for p in pages:
        if p["type"] == 2:  # DICT begins a new column chunk
            if cur is not None:
                chunks.append(cur)
            cur = [p]
        else:
            if cur is None:
                raise ValueError("data page before any dict page — unexpected encoding")
            cur.append(p)
    if cur is not None:
        chunks.append(cur)
    if len(chunks) % ncols != 0:
        raise ValueError(f"{len(chunks)} column chunks not divisible by {ncols} columns")
    return [chunks[i:i + ncols] for i in range(0, len(chunks), ncols)]


def _chunk_stats(chunk):
    dicts = [p for p in chunk if p["type"] == 2]
    datas = [p for p in chunk if p["type"] == 0]
    if len(dicts) != 1:
        raise ValueError(f"expected 1 dict page per chunk, got {len(dicts)}")
    leaf = sum(p["nvals"] for p in datas)
    comp = sum((p["hdr_end"] - p["hdr_off"]) + p["csize"] for p in chunk)
    uncomp = sum((p["hdr_end"] - p["hdr_off"]) + p["usize"] for p in chunk)
    return {"dict_off": dicts[0]["hdr_off"], "data_off": datas[0]["hdr_off"], "leaf": leaf,
            "uncomp": uncomp, "comp": comp, "start": chunk[0]["hdr_off"], "end": chunk[-1]["body_end"]}


# ─────────────────────────── schema templates ───────────────────────────

_PA_SCALAR = {
    "float32": pa.float32(), "float64": pa.float64(), "float": pa.float32(),
    "int64": pa.int64(), "int32": pa.int32(), "int16": pa.int16(), "int8": pa.int8(),
    "bool": pa.bool_(),
}


def _pa_scalar(dtype: str):
    if dtype not in _PA_SCALAR:
        raise ValueError(f"unmapped feature dtype {dtype!r}")
    return _PA_SCALAR[dtype]


def _data_arrow_schema(features: dict) -> pa.Schema:
    """Parquet schema lerobot writes for the data file: the non-image/video
    features, in declaration order. A shape-(1,) feature is stored as a scalar;
    a multi-element feature as a variable ``list<T>`` (matching the writer, which
    reshapes (1,) features to 1-D and lets ``datasets`` emit list<> for the rest).
    """
    fields = []
    for name, spec in features.items():
        if spec["dtype"] in ("image", "video", "string"):
            continue
        t = _pa_scalar(spec["dtype"])
        shape = tuple(spec.get("shape", [1]))
        fields.append(pa.field(name, t if shape in ((), (1,)) else pa.list_(t)))
    return pa.schema(fields)


def _episodes_arrow_schema(features: dict) -> pa.Schema:
    """Reconstruct the meta/episodes schema deterministically from features.
    Column order mirrors lerobot: base cols, per-camera video meta, then per-
    feature stats (numeric features first, image/video features last), then the
    meta/episodes chunk/file indices."""
    video_keys = [k for k, v in features.items() if v["dtype"] in ("video", "image")]
    numeric_keys = [k for k, v in features.items() if v["dtype"] not in ("video", "image", "string")]

    fields = [
        pa.field("episode_index", pa.int64()),
        pa.field("tasks", pa.list_(pa.string())),
        pa.field("length", pa.int64()),
        pa.field("data/chunk_index", pa.int64()),
        pa.field("data/file_index", pa.int64()),
        pa.field("dataset_from_index", pa.int64()),
        pa.field("dataset_to_index", pa.int64()),
    ]
    for vk in video_keys:
        fields += [
            pa.field(f"videos/{vk}/chunk_index", pa.int64()),
            pa.field(f"videos/{vk}/file_index", pa.int64()),
            pa.field(f"videos/{vk}/from_timestamp", pa.float64()),
            pa.field(f"videos/{vk}/to_timestamp", pa.float64()),
        ]

    def stat_fields(feat, is_image):
        is_int = str(features[feat]["dtype"]).startswith("int")
        if is_image:
            base = pa.list_(pa.list_(pa.list_(pa.float64())))   # per-channel (C,1,1)
            minmax = base
        else:
            base = pa.list_(pa.float64())
            minmax = pa.list_(pa.int64()) if is_int else pa.list_(pa.float64())
        out = []
        for sk in STATKEYS:
            if sk == "count":
                out.append(pa.field(f"stats/{feat}/count", pa.list_(pa.int64())))
            elif sk in ("min", "max"):
                out.append(pa.field(f"stats/{feat}/{sk}", minmax))
            else:
                out.append(pa.field(f"stats/{feat}/{sk}", base))
        return out

    for feat in numeric_keys:
        fields += stat_fields(feat, is_image=False)
    for feat in video_keys:
        fields += stat_fields(feat, is_image=True)

    fields += [
        pa.field("meta/episodes/chunk_index", pa.int64()),
        pa.field("meta/episodes/file_index", pa.int64()),
    ]
    return pa.schema(fields)


def _footer_template(schema: pa.Schema):
    """Write a tiny 2-row parquet with ``schema`` to harvest a consistent
    parquet-schema thrift + per-column metadata templates."""
    arrs = [pa.array([None, None], type=f.type) for f in schema]
    tbl = pa.table(arrs, schema=schema)
    buf = pa.BufferOutputStream()
    pq.write_table(tbl, buf, compression="snappy", use_dictionary=True)
    data = buf.getvalue().to_pybytes()
    assert data[-4:] == b"PAR1"
    flen = struct.unpack("<I", data[-8:-4])[0]
    fmd = _read_struct(_R(data, len(data) - 8 - flen))
    tmpl_cols = fmd.get(4).items[0].get(1).items
    return fmd, tmpl_cols


def _clone(node):
    if isinstance(node, _Struct):
        return _Struct([[f[0], f[1], _clone(f[2])] for f in node.fields])
    if isinstance(node, _List):
        return _List(node.etype, [_clone(x) for x in node.items])
    return node


def _rebuild_footer(buf: bytes, ncols: int, tmpl_fmd, tmpl_cols, keep_row_groups=None):
    """Rebuild a parquet from its intact pages + a schema template.
    keep_row_groups: if set, keep only the first N row groups. Returns bytes."""
    pages, _, walk_end = _walk_pages(buf)
    rgs = _group_row_groups(pages, ncols)
    if keep_row_groups is not None:
        rgs = rgs[:keep_row_groups]

    def cm_for(col_i, st):
        d = tmpl_cols[col_i].get(3)
        cm = _Struct()
        cm.set(1, CT_I32, d.get(1))            # type
        cm.set(2, CT_LIST, _clone(d.get(2)))   # encodings
        cm.set(3, CT_LIST, _clone(d.get(3)))   # path_in_schema
        cm.set(4, CT_I32, d.get(4))            # codec
        cm.set(5, CT_I64, st["leaf"])          # num_values
        cm.set(6, CT_I64, st["uncomp"])        # total_uncompressed_size
        cm.set(7, CT_I64, st["comp"])          # total_compressed_size
        cm.set(9, CT_I64, st["data_off"])      # data_page_offset
        cm.set(11, CT_I64, st["dict_off"])     # dictionary_page_offset
        return cm

    rg_structs, total_rows = [], 0
    for g in rgs:
        stats = [_chunk_stats(c) for c in g]
        rows = stats[-1]["leaf"]  # last column is a scalar -> rows (meta: file_index; data: task_index)
        total_rows += rows
        cols, tot_u, tot_c, first = [], 0, 0, None
        for ci, st in enumerate(stats):
            if first is None:
                first = st["start"]
            tot_u += st["uncomp"]
            tot_c += st["comp"]
            cc = _Struct()
            cc.set(2, CT_I64, 0)                # file_offset (writers use 0)
            cc.set(3, CT_STRUCT, cm_for(ci, st))
            cols.append(cc)
        rg = _Struct()
        rg.set(1, CT_LIST, _List(CT_STRUCT, cols))
        rg.set(2, CT_I64, tot_u)
        rg.set(3, CT_I64, rows)
        rg.set(5, CT_I64, first)
        rg.set(6, CT_I64, tot_c)
        rg_structs.append(rg)

    fmd = _Struct()
    fmd.set(1, CT_I32, tmpl_fmd.get(1))                 # version
    fmd.set(2, CT_LIST, _clone(tmpl_fmd.get(2)))        # schema
    fmd.set(3, CT_I64, total_rows)                      # num_rows
    fmd.set(4, CT_LIST, _List(CT_STRUCT, rg_structs))  # row_groups
    if tmpl_fmd.get(5) is not None:
        fmd.set(5, CT_LIST, _clone(tmpl_fmd.get(5)))   # key_value_metadata (arrow schema)
    if tmpl_fmd.get(6) is not None:
        fmd.set(6, CT_BINARY, tmpl_fmd.get(6))         # created_by
    if tmpl_fmd.get(7) is not None:
        fmd.set(7, CT_LIST, _clone(tmpl_fmd.get(7)))   # column_orders

    cut = rgs[-1][-1][-1]["body_end"] if rgs else 4
    w = _W()
    _write_struct(w, fmd)
    footer = bytes(w.o)
    return buf[:cut] + footer + struct.pack("<I", len(footer)) + b"PAR1", total_rows, [
        sum(_chunk_stats(c)["leaf"] for c in g[-1:]) for g in rgs  # per-rg rows
    ]


def _rows_per_rg(buf: bytes, ncols: int, limit=None):
    pages, _, _ = _walk_pages(buf)
    rgs = _group_row_groups(pages, ncols)
    if limit is not None:
        rgs = rgs[:limit]
    return [_chunk_stats(g[-1])["leaf"] for g in rgs]


def repair_dataset(root: str | Path) -> bool:
    """Repair footerless parquet files under a LeRobotDataset ``root`` in place.

    Returns True if anything was repaired. No-op (returns False) if both parquet
    files already have valid footers. Corrupt files are backed up to
    ``<root>/.corrupt-originals/`` before being overwritten. After a footer
    rebuild, the data file and metadata file are reconciled to the same committed
    episode count (dropping a trailing data row group written just before a crash
    but before its metadata row), and ``meta/info.json`` is updated to match.
    """
    root = Path(root)
    info_path = root / "meta" / "info.json"
    if not info_path.exists():
        return False
    info = json.loads(info_path.read_text())
    features = info["features"]

    data_path = root / "data" / "chunk-000" / "file-000.parquet"
    eps_path = root / "meta" / "episodes" / "chunk-000" / "file-000.parquet"

    data_ok = data_path.exists() and _has_footer(data_path)
    eps_ok = eps_path.exists() and _has_footer(eps_path)
    if data_ok and eps_ok:
        return False

    logger.warning("dataset at %s has footerless parquet(s) — repairing "
                   "(data_ok=%s eps_ok=%s)", root, data_ok, eps_ok)
    backup = root / ".corrupt-originals"

    # --- rebuild the metadata (episodes) footer; it defines the committed count
    eps_schema = _episodes_arrow_schema(features)
    eps_ncols = len(eps_schema)
    if not eps_ok:
        raw = eps_path.read_bytes()
        fmd_t, cols_t = _footer_template(eps_schema)
        fixed, eps_rows, _ = _rebuild_footer(raw, eps_ncols, fmd_t, cols_t)
        _backup_and_write(eps_path, root, raw, fixed, backup)
    else:
        eps_rows = pq.read_metadata(eps_path).num_rows
    committed_eps = eps_rows

    # --- rebuild the data footer, trimming to the committed episode count
    data_schema = _data_arrow_schema(features)
    data_ncols = len(data_schema)
    if not data_ok:
        raw = data_path.read_bytes()
        rows = _rows_per_rg(raw, data_ncols)
        keep = min(committed_eps, len(rows))
        committed_frames = sum(rows[:keep])
        fmd_t, cols_t = _footer_template(data_schema)
        fixed, _, _ = _rebuild_footer(raw, data_ncols, fmd_t, cols_t, keep_row_groups=keep)
        _backup_and_write(data_path, root, raw, fixed, backup)
    else:
        md = pq.read_metadata(data_path)
        committed_frames = md.num_rows

    # --- reconcile info.json to the committed counts
    if info.get("total_episodes") != committed_eps or info.get("total_frames") != committed_frames:
        logger.warning("reconciling info.json: episodes %s->%s, frames %s->%s",
                       info.get("total_episodes"), committed_eps,
                       info.get("total_frames"), committed_frames)
        info["total_episodes"] = committed_eps
        info["total_frames"] = committed_frames
        info["splits"] = {"train": f"0:{committed_eps}"}
        info_path.write_text(json.dumps(info, indent=4))

    logger.warning("repair complete: %d episodes / %d frames readable", committed_eps, committed_frames)
    return True


def _backup_and_write(path: Path, root: Path, original: bytes, fixed: bytes, backup_root: Path) -> None:
    # sanity: the rebuilt file must open before we touch the original on disk
    pq.read_metadata(pa.BufferReader(pa.py_buffer(fixed)))
    dest = backup_root / path.relative_to(root)
    dest.parent.mkdir(parents=True, exist_ok=True)
    if not dest.exists():
        dest.write_bytes(original)
    path.write_bytes(fixed)

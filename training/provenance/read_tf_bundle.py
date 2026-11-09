"""Minimal reader for a TensorFlow tensor-bundle checkpoint (variables.index/.data).

Only what is needed to compare Google's speech_embedding weights with the shipped ONNX:
leveldb table blocks (raw or snappy), BundleEntryProto fields dtype/shape/offset/size.
"""
from __future__ import annotations

import struct
from pathlib import Path

import numpy as np

DTYPES = {1: np.float32, 2: np.float64, 3: np.int32, 9: np.int64}


def varint(buf: bytes, pos: int) -> tuple[int, int]:
    result, shift = 0, 0
    while True:
        b = buf[pos]
        pos += 1
        result |= (b & 0x7F) << shift
        if not b & 0x80:
            return result, pos
        shift += 7


def block_entries(block: bytes):
    n_restarts = struct.unpack("<I", block[-4:])[0]
    end = len(block) - 4 - 4 * n_restarts
    pos, key = 0, b""
    while pos < end:
        shared, pos = varint(block, pos)
        unshared, pos = varint(block, pos)
        vlen, pos = varint(block, pos)
        key = key[:shared] + block[pos:pos + unshared]
        pos += unshared
        yield key, block[pos:pos + vlen]
        pos += vlen


def snappy_decompress(src: bytes) -> bytes:
    n, pos = varint(src, 0)
    out = bytearray()
    while pos < len(src):
        tag = src[pos]
        pos += 1
        kind = tag & 3
        if kind == 0:  # literal
            length = tag >> 2
            if length >= 60:
                extra = length - 59
                length = int.from_bytes(src[pos:pos + extra], "little")
                pos += extra
            length += 1
            out += src[pos:pos + length]
            pos += length
            continue
        if kind == 1:
            length = ((tag >> 2) & 7) + 4
            offset = ((tag >> 5) << 8) | src[pos]
            pos += 1
        elif kind == 2:
            length = (tag >> 2) + 1
            offset = int.from_bytes(src[pos:pos + 2], "little")
            pos += 2
        else:
            length = (tag >> 2) + 1
            offset = int.from_bytes(src[pos:pos + 4], "little")
            pos += 4
        for _ in range(length):
            out.append(out[-offset])
    assert len(out) == n
    return bytes(out)


def read_block(data: bytes, handle: bytes) -> bytes:
    offset, p = varint(handle, 0)
    size, _ = varint(handle, p)
    kind = data[offset + size]
    block = data[offset:offset + size]
    if kind == 0:
        return block
    if kind == 1:
        return snappy_decompress(block)
    raise ValueError(f"unsupported block compression {kind}")


def parse_entry(buf: bytes) -> dict:
    """BundleEntryProto: 1 dtype, 2 shape(TensorShapeProto), 3 shard_id, 4 offset, 5 size, 6 crc32c."""
    out, pos = {"shape": []}, 0
    while pos < len(buf):
        tag, pos = varint(buf, pos)
        field, wire = tag >> 3, tag & 7
        if wire == 0:
            value, pos = varint(buf, pos)
            out[{1: "dtype", 3: "shard", 4: "offset", 5: "size"}.get(field, f"f{field}")] = value
        elif wire == 2:
            n, pos = varint(buf, pos)
            sub = buf[pos:pos + n]
            pos += n
            if field == 2:  # TensorShapeProto: repeated dim (field 2) { size (field 1) }
                sp = 0
                while sp < len(sub):
                    t, sp = varint(sub, sp)
                    ln, sp = varint(sub, sp)
                    dim = sub[sp:sp + ln]
                    sp += ln
                    dp, size = 0, 0
                    while dp < len(dim):
                        dt, dp = varint(dim, dp)
                        if dt >> 3 == 1 and dt & 7 == 0:
                            size, dp = varint(dim, dp)
                        else:
                            break
                    out["shape"].append(size)
        elif wire == 5:
            pos += 4
        elif wire == 1:
            pos += 8
    return out


def read_bundle(prefix: Path) -> dict[str, np.ndarray]:
    index = (prefix.parent / (prefix.name + ".index")).read_bytes()
    data = (prefix.parent / (prefix.name + ".data-00000-of-00001")).read_bytes()
    footer = index[-48:]
    _, p = varint(footer, 0)
    _, p = varint(footer, p)
    index_handle = footer[p:]
    tensors = {}
    for _, handle in block_entries(read_block(index, index_handle)):
        for key, value in block_entries(read_block(index, handle)):
            if not key:
                continue  # header entry
            e = parse_entry(value)
            if e.get("dtype") not in DTYPES:
                continue
            arr = np.frombuffer(data[e.get("offset", 0):e.get("offset", 0) + e["size"]], dtype=DTYPES[e["dtype"]])
            tensors[key.decode()] = arr.reshape(e["shape"]) if e["shape"] else arr
    return tensors


if __name__ == "__main__":
    import sys
    for name, arr in read_bundle(Path(sys.argv[1])).items():
        print(name, arr.shape, arr.dtype)

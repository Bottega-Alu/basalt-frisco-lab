"""The Gemma 4 MoE pack from llama.cpp's BF16 GGUF (`convert.py --gguf gemma-4-26B-A4B-it-BF16.gguf --nvfp4 --hf-dir DIR`).

  experts / experts.layout  the routed experts rounded once to NVFP4 as hy4flash_bf16 does (per matrix the tensor scale
                            amax / (6 x 448), per 16 values the UE4M3 scale of least squared error): `ffn_gate_up_exps`
                            [2 x n_ff_exp rows] split into gate (the first half, as Hugging Face's chunk(2)) and up, then
                            `ffn_down_exps`; blobs [gate | up | down] plus the FP32 tail {s_gate, s_up, s_down, 0}, where
                            s_down carries the router's per-expert scale (`ffn_down_exps.scale`): the router multiplies the
                            renormalized top-k weight by it, which is the expert's down output times it, exactly
  gguf / dense / dense.index  every other tensor: BF16 projections (attention, the dense MLP, the tied token embedding) as
                            Q8_0, the F32 tensors as stored
  tokenizer/*               the GGUF's (SentencePiece-style BPE, tools/tokenizer.py `gemma4`), checked against the Hugging
                            Face directory's tokenizer.json and chat template

`verify` repeats every rounding and compares the bytes with the pack's. Needs numpy.
"""
from __future__ import annotations

import hashlib
import json
import multiprocessing
import os
import re
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import checkpoint as C  # noqa: E402
import checkpoint_qwen35 as Q  # noqa: E402
import model_sections as S  # noqa: E402
from arch.gemma4 import Gemma4  # noqa: E402
from hy4flash_bf16 import QuantDense, _rows_job, bf16_values, check_tokenizer, compare, die_on, pread  # noqa: E402
from model_file import SECTION_ALIGN, TENSOR_ALIGN, ModelFile, ModelWriter, Stream  # noqa: E402

TYPE_ID = {"F32": 0, "Q8_0": 8, "BF16": 30, "NVFP4": 40}
ROWS_VALUES_PER_JOB = 1 << 22
EXPERTS_PER_JOB = 4
DEFAULT_NAME = "Gemma-4-26B-A4B-it NVFP4"
ARCH = Gemma4()


def _round(x: np.ndarray):
    gs = Q.fp4q_global(float(np.abs(x).max()))
    codes, sc = Q.fp4q_rows(x, gs)
    return C.nvfp4_rows(codes, sc).tobytes(), gs


def _expert_job(args):
    """Blobs of experts [e0, e1) of one layer: [gate | up | down] NVFP4 rows and the FP32 tail, s_down times the
    expert's router scale."""
    path, gu_off, down_off, ff, n_embd, scales, e0, e1 = args
    out = []
    gu_n, d_n = 2 * ff * n_embd, n_embd * ff
    for e in range(e0, e1):
        gu = bf16_values(pread(path, gu_off + e * gu_n * 2, gu_n * 2)).reshape(2 * ff, n_embd)
        dn = bf16_values(pread(path, down_off + e * d_n * 2, d_n * 2)).reshape(n_embd, ff)
        if not (np.isfinite(gu).all() and np.isfinite(dn).all()):
            return f"expert {e}: a non-finite value"
        tail = []
        for x in (gu[:ff], gu[ff:], dn):
            b, gs = _round(x)
            out.append(b)
            tail.append(gs)
        tail[2] = np.float32(tail[2]) * np.float32(scales[e])
        out.append(np.array(tail + [0.0], dtype="<f4").tobytes())
    return b"".join(out)


class Plan:
    """What the pack holds, from the BF16 GGUF's tensor table alone."""

    def __init__(self, model: S.SourceModel):
        self.model = model
        T = {n: w[1] for n, w in model.where.items()}
        self.T = T
        self.layers = sorted(int(m.group(1)) for n in T if (m := re.fullmatch(r"blk\.(\d+)\.ffn_gate_up_exps\.weight", n)))
        if not self.layers:
            raise S.ConvertError("the model has no blk.N.ffn_gate_up_exps.weight")
        gu0 = T[f"blk.{self.layers[0]}.ffn_gate_up_exps.weight"]
        self.n_embd, two_ff, self.n_expert = (int(d) for d in gu0.shape)
        self.ff = two_ff // 2
        for l in self.layers:
            gu, dn = T.get(f"blk.{l}.ffn_gate_up_exps.weight"), T.get(f"blk.{l}.ffn_down_exps.weight")
            sc = T.get(f"blk.{l}.ffn_down_exps.scale")
            if gu is None or dn is None or sc is None:
                raise S.ConvertError(f"layer {l}: missing gate_up / down experts or the per-expert scale")
            if gu.type_name != "BF16" or [int(d) for d in gu.shape] != [self.n_embd, 2 * self.ff, self.n_expert]:
                raise S.ConvertError(f"{gu.name} is {gu.type_name} {list(gu.shape)}")
            if dn.type_name != "BF16" or [int(d) for d in dn.shape] != [self.ff, self.n_embd, self.n_expert]:
                raise S.ConvertError(f"{dn.name} is {dn.type_name} {list(dn.shape)}")
            if sc.type_name != "F32" or [int(d) for d in sc.shape] != [self.n_expert]:
                raise S.ConvertError(f"{sc.name} is {sc.type_name} {list(sc.shape)}; expected F32 [{self.n_expert}]")
        self.dims = [(self.ff, self.n_embd), (self.ff, self.n_embd), (self.n_embd, self.ff)]
        self.blob = sum(rows * S.nvfp4_row_bytes(cols) for rows, cols in self.dims) + S.NVFP4_TAIL
        self.expert_bytes = len(self.layers) * self.n_expert * self.blob
        self.layout = [(l, TYPE_ID["NVFP4"], TYPE_ID["NVFP4"], i * self.n_expert * self.blob, self.blob)
                       for i, l in enumerate(self.layers)]
        self.how: dict[str, tuple[str, str]] = {}
        for n, t in T.items():
            if S.is_expert(n):
                continue
            self.how[n] = self.classify(n, t)

    @staticmethod
    def classify(name, t):
        tn, ne0 = t.type_name, int(t.shape[0])
        if tn == "F32":
            return "raw", "F32"
        if tn != "BF16":
            raise S.ConvertError(f"{name} is {tn}; the BF16 GGUF holds F32 and BF16 tensors")
        if ne0 % 32:
            raise S.ConvertError(f"{name}: rows of {ne0} values are not whole Q8_0 blocks")
        return "quant", "Q8_0"

    @staticmethod
    def stored_bytes(t, ty: str) -> int:
        n = int(np.prod([int(d) for d in t.shape]))
        return {"Q8_0": n // 32 * 34, "BF16": n * 2, "F32": n * 4}[ty]

    def expert_scales(self, l) -> np.ndarray:
        path, off, size = self.model.abs_range(f"blk.{l}.ffn_down_exps.scale")
        return np.frombuffer(bytes(pread(path, off, size)), dtype="<f4").copy()

    def expert_jobs(self, l):
        gp, goff, _ = self.model.abs_range(f"blk.{l}.ffn_gate_up_exps.weight")
        dp, doff, _ = self.model.abs_range(f"blk.{l}.ffn_down_exps.weight")
        if gp != dp:
            raise S.ConvertError(f"layer {l}: the expert tensors are in different files")
        sc = self.expert_scales(l)
        return [(gp, goff, doff, self.ff, self.n_embd, sc, e, min(e + EXPERTS_PER_JOB, self.n_expert))
                for e in range(0, self.n_expert, EXPERTS_PER_JOB)]

    def expert_chunks(self, pool):
        for l in self.layers:
            for b in S.bounded_imap(pool, _expert_job, self.expert_jobs(l), 64):
                yield die_on(b)
            print(f"  layer {l} experts", flush=True)

    def parts(self, pool):
        out = []
        for n, (how, ty) in self.how.items():
            path, off, size = self.model.abs_range(n)
            t = self.T[n]
            nb = self.stored_bytes(t, ty)
            if how == "raw":
                out.append((n, ty, (path, off, size)))
                continue
            ne0, ne1 = int(t.shape[0]), int(np.prod([int(d) for d in t.shape[1:]]))
            step = max(1, ROWS_VALUES_PER_JOB // ne0)
            jobs = [(path, off, ne0, r, min(step, ne1 - r), ty) for r in range(0, ne1, step)]
            out.append((n, ty, Stream(nb, (die_on(b) for b in S.bounded_imap(pool, _rows_job, jobs, 32)) if pool else iter(()))))
        return out

    def dense(self):
        return QuantDense(self.model, ARCH, self.how)


QUANTIZED_FROM = {
    "source": "BF16 GGUF (llama.cpp convert_hf_to_gguf.py)",
    "experts": "NVFP4: per matrix tensor scale amax / (6 x 448), per 16 values the UE4M3 scale of least squared error "
               "(as the 27B pack); ffn_gate_up_exps split into gate (first half) and up; blobs [gate | up | down] and the "
               "FP32 tail {s_gate, s_up, s_down x ffn_down_exps.scale[e], 0}",
    "router_per_expert_scale": "folded into s_down (not a tensor of the pack)",
    "dense": "Q8_0 (ggml reference rounding)", "dense_variant": "Q8_0",
}


def build(a, Job, DryWriter, fingerprint, source_record, source_tensors, export_tokenizer) -> None:
    out = Path(a.out)
    partial = Path(str(out) + ".partial")
    dry = getattr(a, "dry_run", False)
    if out.exists() and not a.force:
        raise SystemExit(f"convert: {out} exists (use --force to overwrite)")
    paths = S.expand_shards(Path(a.gguf[0])) if len(a.gguf) == 1 else [Path(p) for p in a.gguf]
    hf_dir = Path(a.hf_dir)
    try:
        print("reading the source GGUF ...", flush=True)
        model = S.SourceModel(paths, frozenset())
        kv, _ = source_tensors(model, ARCH.own_sections)
        if kv.get("general.architecture") != ARCH.gguf_architecture:
            raise S.ConvertError(f"{paths[0].name} is architecture {kv.get('general.architecture')!r}, not gemma4")
        ARCH.check(kv, {})
        hf_meta = ARCH.hf_meta(hf_dir)
        check_tokenizer(kv, hf_dir)
        plan = Plan(model)
        dense = plan.dense()
    except (S.ConvertError, KeyError) as e:
        raise SystemExit(f"convert: {e}")
    if partial.exists():
        partial.unlink()
    name = a.name or DEFAULT_NAME
    w = DryWriter() if dry else ModelWriter(str(partial))
    job = Job(w)
    ok = False
    pool = None if dry else multiprocessing.Pool(a.workers)
    try:
        with tempfile.TemporaryDirectory() as tmp:
            print("planning sections:" if dry else "writing sections:", flush=True)
            for p in export_tokenizer(model.paths[0], Path(tmp)):
                job.file(f"tokenizer/{p.name}", p)
            job.chunks("dense", dense.chunks())
            job.bytes("dense.index", dense.index_text())
            job.chunks("experts", plan.expert_chunks(pool) if pool else iter(()), plan.expert_bytes)
            if not dry and w.entries[-1][2] != plan.expert_bytes:
                raise S.ConvertError(f"experts: {w.entries[-1][2]} bytes written, {plan.expert_bytes} planned")
            job.bytes("experts.layout", S.nvfp4_layout_text(plan.n_expert, plan.expert_bytes, plan.layout))
            t = time.time()
            parts = plan.parts(pool)
            offs = w.add_parts("gguf", [p for _, _, p in parts], TENSOR_ALIGN)
            job._report("gguf", w.entries[-1][2], time.time() - t)
            tensors = {}
            for (n, ty, _), off in zip(parts, offs):
                t_ = plan.T[n]
                tensors[n] = {"type": ty, "type_id": TYPE_ID[ty], "shape": [int(d) for d in t_.shape],
                              "bytes": plan.stored_bytes(t_, ty), "offset": off}
            kv = dict(kv)
            kv["general.architecture"] = ARCH.name
            kv.pop("general.file_type", None)
            sources = [source_record(p) for p in model.paths]
            sources += [{"path": f"hf/{n}", "size": (hf_dir / n).stat().st_size,
                         "sha256": hashlib.sha256((hf_dir / n).read_bytes()).hexdigest()}
                        for n in ("config.json", "tokenizer.json", "chat_template.jinja")]
            meta = {
                "format": "basalt", "version": 2, "name": name,
                "converter": {"tool": "tools/convert/convert.py"},
                "sources": sources,
                "gguf_kv": kv,
                "tensors": tensors,
                "conversions": {"compat_bf16": False, "tensors": dense.records},
                "license": ARCH.license(kv),
                "mtp": None,
                **ARCH.meta_extra(kv, tensors, {}),
                "experts": {"n_expert": plan.n_expert, "n_ff": plan.ff, "n_embd": plan.n_embd, "layers": plan.layers,
                            "blob_bytes": plan.blob, "activation": "gelu_tanh"},
                "quantized_from": QUANTIZED_FROM,
                **hf_meta,
            }
            job.bytes("meta", json.dumps(meta, ensure_ascii=False, separators=(",", ":")).encode("utf-8"))
        w.close()
        ok = True
    except S.ConvertError as e:
        raise SystemExit(f"convert: {e}")
    finally:
        if pool is not None:
            pool.terminate()
        if not ok and not dry:
            try:
                w.f.close()
            except OSError:
                pass
            partial.unlink(missing_ok=True)
    if dry:
        end = w.pos
        print(f"dry run: {'section':<28} {'offset':>16} {'bytes':>16}")
        for n, off, size, _ in w.entries:
            print(f"         {n:<28} {off:>16} {size:>16}")
        print(f"dry run: {len(w.entries)} sections, {end} bytes ({end / (1 << 30):.3f} GiB) -> {out}")
        return
    os.replace(partial, out)
    print(f"done: {len(job.rows)} sections, {out.stat().st_size / (1 << 30):.3f} GiB, {time.time() - job.t0:.1f} s -> {out}")


def verify(pack: str, first_gguf: str, workers: int = 16) -> bool:
    """Every section's SHA-256, then each section's bytes against the roundings of the source, repeated."""
    t0 = time.time()
    problems: list[str] = []

    def bad(msg):
        problems.append(msg)
        print("  FAIL", msg, flush=True)

    m = ModelFile(pack)
    meta = json.loads(m.read("meta"))
    paths = S.expand_shards(Path(first_gguf))
    model = S.SourceModel(paths, frozenset())
    plan = Plan(model)
    print(f"{pack}: {len(m.names())} sections, {Path(pack).stat().st_size} bytes; source {len(paths)} GGUFs, {len(model.where)} tensors")
    for n in m.names():
        off, size = m.section(n)
        ok = m.verify(n)
        print(f"  {n:<24} {off:>16} {size:>16}  {'sha256 ok' if ok else 'sha256 MISMATCH'}")
        if not ok or off % SECTION_ALIGN:
            bad(f"section {n}: " + ("checksum differs" if not ok else "not 2 MiB aligned"))
    file_map = np.memmap(pack, dtype=np.uint8, mode="r")

    def view(name):
        off, size = m.section(name)
        return file_map[off:off + size]

    tens = meta["tensors"]
    if set(tens) != set(plan.how):
        bad(f"meta.tensors lacks {sorted(set(plan.how) - set(tens))[:4]} / has extra {sorted(set(tens) - set(plan.how))[:4]}")
    for n, (how, ty) in plan.how.items():
        e, t = tens.get(n), plan.T[n]
        if e and (e["type"] != ty or [int(x) for x in e["shape"]] != [int(x) for x in t.shape] or
                  e["bytes"] != plan.stored_bytes(t, ty) or e["offset"] % TENSOR_ALIGN):
            bad(f"{n}: meta.tensors says {e['type']} {e['shape']} {e['bytes']} B at {e['offset']}")
    if m.read("experts.layout") != S.nvfp4_layout_text(plan.n_expert, plan.expert_bytes, plan.layout):
        bad("experts.layout differs from the source's expert tensors")
    if m.section("experts")[1] != plan.expert_bytes:
        bad("the experts section has the wrong size")
    dense = plan.dense()
    with multiprocessing.Pool(workers) as pool:
        if compare(view("dense"), 0, dense.chunks(), "dense", bad) != m.section("dense")[1]:
            bad("the dense section has the wrong size")
        if m.read("dense.index") != dense.index_text():
            bad("dense.index differs from the source's tensors")
        print("  dense compared", flush=True)
        compare(view("experts"), 0, plan.expert_chunks(pool), "experts", bad)
        print("  experts compared", flush=True)
        gg = view("gguf")
        for n, ty, part in plan.parts(pool):
            e = tens[n]
            at = e["offset"]
            if isinstance(part, tuple):
                chunks = (bytes(pread(part[0], part[1] + o, min(64 << 20, part[2] - o))) for o in range(0, part[2], 64 << 20))
            else:
                chunks = part.chunks
            if compare(gg, at, chunks, n, bad) != at + e["bytes"]:
                bad(f"{n}: wrong length")
        print("  gguf compared", flush=True)
    kv = meta.get("gguf_kv", {})
    if kv.get("general.architecture") != ARCH.name or meta.get("arch") != ARCH.name:
        bad("the pack does not say gemma4 in general.architecture and arch")
    for f in ("tokenizer/vocab.json", "tokenizer/merges.json", "tokenizer/token_type.json", "tokenizer/tokenizer.json",
              "tokenizer/chat_template.jinja", "meta"):
        if f not in m.sections:
            bad(f"no {f} section")
    if "tokenizer/chat_template.jinja" in m.sections and \
            m.read("tokenizer/chat_template.jinja").decode("utf-8") != kv.get("tokenizer.chat_template"):
        bad("tokenizer/chat_template.jinja differs from the metadata's template")
    print(f"verify: {'OK' if not problems else str(len(problems)) + ' problems'} in {time.time() - t0:.0f} s")
    return not problems

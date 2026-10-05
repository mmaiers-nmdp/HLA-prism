"""
hla_graph.py - block graph of HLA protein alleles.

1. Align the training alleles (MAFFT).
2. Cut the alignment into consecutive blocks. A block closes at the first
   column where the alleles fully observed in it show >= K distinct strings.
3. Each distinct string in a block is a vertex (v1, v2, ...); each allele is a
   path with one vertex per block.
4. Place each test sequence on the graph with the fewest new vertices
   (exact dynamic programming) and add those vertices to the graph.

Run from VS Code: edit SETTINGS and press Run. See README.md.
"""

import json
import os
import shutil
import subprocess
import tempfile
from collections import Counter, defaultdict
from dataclasses import dataclass, field

import numpy as np

# ==========================================================================
# SETTINGS
# ==========================================================================
TRAIN_FASTA = "training.fasta"      # alleles used to build the graph
TEST_FASTA = "test.fasta"           # sequences placed on the graph ("" to skip)
OUT_DIR = "hla_graph_output"        # all results go here
K = 100                             # minimal polymorphism per block
N_TEST = None                       # use only the first N test sequences (None = all)

MAFFT = "mafft"                     # or full path, e.g. r"C:\tools\mafft-win\mafft.bat"
MAFFT_ARGS = "--auto"
BACKBONE_MIN_LEN = 330              # align sequences >= this length first, then add the rest
                                    # (None = one MAFFT run on everything)
THREADS = -1                        # MAFFT threads, -1 = all cores
TRAIN_IS_ALIGNED = False            # True if TRAIN_FASTA is already a gapped alignment
REUSE_ALIGNMENT = True              # reuse OUT_DIR/training.aln.fasta if present
REUSE_GRAPH = False                 # reuse OUT_DIR/graph.json if present
UPDATE_GRAPH_WITH_TEST = True       # add the test sequences' new vertices to the graph
SLACK = 10                          # a new vertex may be this much shorter/longer than block labels
PLOT_TOP = 25                       # vertices drawn individually per block
# ==========================================================================

GAP = "-"
UNKNOWN = set("X*?~")               # residues treated as not observed


# --------------------------------------------------------------------------
# FASTA and alignment
# --------------------------------------------------------------------------

def parse_fasta(lines):
    """Yield (name, seq). IMGT headers '>HLA:HLA00001 A*01:01:01:01 365 bp' give the allele name."""
    name, buf = None, []
    for line in lines:
        line = line.strip()
        if line.startswith(">"):
            if name is not None:
                yield name, "".join(buf).upper()
            parts = line[1:].split()
            name = parts[1] if len(parts) > 1 and parts[0].startswith("HLA:") else parts[0]
            buf = []
        elif line:
            buf.append(line)
    if name is not None:
        yield name, "".join(buf).upper()


def read_fasta(path):
    with open(path) as fh:
        return list(parse_fasta(fh))


def write_fasta(recs, path):
    with open(path, "w") as fh:
        for n, s in recs:
            fh.write(f">{n}\n" + "\n".join(s[i:i + 60] for i in range(0, len(s), 60)) + "\n")


def mafft(args, inp):
    if shutil.which(MAFFT) is None and not os.path.exists(MAFFT):
        raise FileNotFoundError(f"MAFFT not found at '{MAFFT}'; set MAFFT in SETTINGS.")
    out = subprocess.run([MAFFT, *args, "--anysymbol", "--thread", str(THREADS), inp],
                         capture_output=True, text=True)
    if out.returncode != 0:
        raise RuntimeError("MAFFT failed:\n" + out.stderr[-2000:])
    return dict(parse_fasta(out.stdout.splitlines()))


def align(recs):
    """
    Align with MAFFT; identical sequences are aligned once. With BACKBONE_MIN_LEN,
    full-length sequences are aligned first and shorter ones added with
    --add --keeplength (fast, robust to tiny fragments; a fragment residue that
    would need a new column is dropped, and such alleles are reported).
    """
    seqs = {}                                            # sequence -> short id
    for _, s in recs:
        seqs.setdefault(s.replace("*", "X"), f"s{len(seqs)}")
    items = [(i, s) for s, i in seqs.items()]
    with tempfile.TemporaryDirectory() as td:
        p_bb, p_fr, p_aln = (os.path.join(td, f) for f in ("bb.fa", "fr.fa", "bb.aln"))
        cut = BACKBONE_MIN_LEN or 0
        write_fasta([x for x in items if len(x[1]) >= cut], p_bb)
        aligned = mafft(MAFFT_ARGS.split(), p_bb)
        short = [x for x in items if len(x[1]) < cut]
        if short:
            write_fasta(list(aligned.items()), p_aln)
            write_fasta(short, p_fr)
            aligned = mafft(["--add", p_fr, "--keeplength"], p_aln)
    out = [(n, aligned[seqs[s.replace("*", "X")]]) for n, s in recs]
    lost = [n for (n, s), (_, a) in zip(recs, out) if a.replace(GAP, "") != s.replace("*", "X")]
    if lost:
        print(f"warning: {len(lost)} alleles lost residues under --keeplength, e.g. {lost[:5]}")
    return out


def to_arrays(aln):
    """
    M: residues as a (N, C) uint8 matrix.  U: True where a residue is not observed.
    Leading/trailing gaps count as not observed (partial alleles), internal gaps as deletions.
    """
    if len({len(s) for _, s in aln}) != 1:
        raise ValueError("Training sequences are not aligned (different lengths).")
    M = np.array([np.frombuffer(s.encode(), np.uint8) for _, s in aln])
    U = np.isin(M, [ord(c) for c in UNKNOWN])
    gap = M == ord(GAP)
    U |= np.cumprod(gap, 1).astype(bool) | np.cumprod(gap[:, ::-1], 1)[:, ::-1].astype(bool)
    return M, U


def n_distinct(M, U, s, e):
    """Number of distinct ungapped strings among alleles fully observed in columns [s, e)."""
    full = ~U[:, s:e].any(1)
    return len({bytes(r).replace(GAP.encode(), b"") for r in M[full, s:e]})


def segment(M, U, k):
    """
    Greedy left-to-right blocks [s, e): extend e until >= k distinct strings.
    A running 64-bit hash per allele (gaps skipped) makes each step O(N).
    A last block that never reaches k is merged into the previous one.
    """
    N, C = M.shape
    P, gap = np.uint64(1000003), ord(GAP)
    blocks, s = [], 0
    while s < C:
        h, full, poly, e = np.zeros(N, np.uint64), np.ones(N, bool), 0, s
        with np.errstate(over="ignore"):
            while e < C and poly < k:
                col = M[:, e]
                m = col != gap
                h[m] = h[m] * P + col[m].astype(np.uint64) + np.uint64(1)
                full &= ~U[:, e]
                e += 1
                poly = len(np.unique(h[full]))
        blocks.append([s, e, poly])
        s = e
    if len(blocks) > 1 and blocks[-1][2] < k:
        last = blocks.pop()
        blocks[-1][1] = last[1]
        blocks[-1][2] = n_distinct(M, U, blocks[-1][0], blocks[-1][1])
    return [tuple(b) for b in blocks]


def eq(a, t):
    """String equality with X as a wildcard."""
    return len(a) == len(t) and all(x == y or x == "X" or y == "X" for x, y in zip(a, t))


# --------------------------------------------------------------------------
# Graph
# --------------------------------------------------------------------------

@dataclass
class Vertex:
    id: str
    block: int
    label: str                    # ungapped residues ('' = block deleted); X = not observed
    count: int = 0                # alleles passing through it
    added_by: str = None          # test sequence that created it
    partial: bool = False         # built from a partially observed allele


@dataclass
class BlockGraph:
    k: int
    blocks: list                                     # [(start_col, end_col, polymorphism)]
    vertices: dict = field(default_factory=dict)     # id -> Vertex
    by_block: list = field(default_factory=list)     # block -> [vertex ids]
    edges: Counter = field(default_factory=Counter)  # (u, v) -> number of alleles
    paths: dict = field(default_factory=dict)        # allele -> [vertex id or None]
    ambiguous: dict = field(default_factory=dict)    # allele -> {block: n compatible vertices}
    next_id: int = 1

    def new_vertex(self, b, label, **kw):
        v = Vertex(f"v{self.next_id}", b, label, **kw)
        self.next_id += 1
        self.vertices[v.id] = v
        self.by_block[b].append(v.id)
        return v.id

    def add_path(self, name, path):
        self.paths[name] = path
        for v in path:
            if v is not None:
                self.vertices[v].count += 1
        for u, v in zip(path, path[1:]):
            if u and v:
                self.edges[(u, v)] += 1

    # ---------------- build from an alignment ----------------
    @classmethod
    def build(cls, aln, k):
        M, U = to_arrays(aln)
        blocks = segment(M, U, k)
        g = cls(k=k, blocks=blocks, by_block=[[] for _ in blocks])
        assign = [[None] * len(blocks) for _ in aln]
        for b, (s, e, _) in enumerate(blocks):
            sub, sU = M[:, s:e], U[:, s:e]
            full = ~sU.any(1)
            # 1. fully observed alleles: one vertex per distinct string, most frequent first
            labels = {i: bytes(sub[i]).decode().replace(GAP, "") for i in np.flatnonzero(full)}
            freq = Counter(labels.values())
            lab2vid = {lab: g.new_vertex(b, lab) for lab, _ in freq.most_common()}
            for i, lab in labels.items():
                assign[i][b] = lab2vid[lab]
            # 2. partially observed alleles: match a full vertex treating unobserved
            #    residues as wildcards (most frequent if several); otherwise own vertex
            part = np.flatnonzero(~full & ~sU.all(1))
            if not len(part):
                continue
            reps = {bytes(sub[i]): assign[i][b] for i in np.flatnonzero(full)}
            R = np.array([np.frombuffer(x, np.uint8) for x in reps]).reshape(len(reps), e - s)
            rv = list(reps.values())
            for chunk in np.array_split(part, len(part) // 1000 + 1):
                ok = ((sub[chunk][:, None] == R[None]) | sU[chunk][:, None]).all(2)
                for row, i in zip(ok, chunk):
                    cand = sorted({rv[t] for t in np.flatnonzero(row)},
                                  key=lambda v: -freq[g.vertices[v].label])
                    if len(cand) > 1:
                        g.ambiguous.setdefault(aln[i][0], {})[b] = len(cand)
                    if cand:
                        assign[i][b] = cand[0]
                        continue
                    lab = "".join("X" if u else chr(c) for c, u in zip(sub[i], sU[i])
                                  if u or c != ord(GAP)).strip("X") or "X"
                    if lab not in lab2vid:
                        lab2vid[lab] = g.new_vertex(b, lab, partial=True)
                    assign[i][b] = lab2vid[lab]
        for (name, _), p in zip(aln, assign):
            g.add_path(name, p)
        print(f"{len(aln)} alleles, {M.shape[1]} columns -> {len(blocks)} blocks, "
              f"{len(g.vertices)} vertices, {len(g.edges)} edges")
        return g

    # ---------------- place a new sequence ----------------
    def add_sequence(self, name, seq, slack=SLACK, update=True):
        """
        Cut the unaligned `seq` into one substring per block. Each substring either
        matches an existing vertex (cost 0) or becomes a new vertex (cost 1).
        Minimise, in order: new vertices, their total edit distance to the nearest
        existing vertex in the block, and the number of blocks left unobserved.
        Partial sequences may skip blocks at either end, and may start (end) inside
        a block by matching the suffix (prefix) of a vertex.
        """
        from rapidfuzz import process
        from rapidfuzz.distance import Levenshtein

        q = seq.upper().replace(GAP, "").replace("*", "")
        L, B = len(q), len(self.blocks)
        lab = lambda v: self.vertices[v].label

        # per block: vertices by frequency, labels grouped by length, wildcard labels,
        # and the lengths allowed for a new vertex
        order, by_len, wild, new_len = [], [], [], []
        for b in range(B):
            vs = sorted(self.by_block[b], key=lambda v: -self.vertices[v].count)
            d = defaultdict(dict)
            for v in vs:
                d[len(lab(v))].setdefault(lab(v), v)
            order.append(vs)
            by_len.append(d)
            wild.append([v for v in vs if "X" in lab(v)])
            ls = list(d) or [0]
            new_len.append(range(max(0, min(ls) - slack), max(ls) + slack + 1))

        def moves(b, p):
            """Ways to consume block b from query position p: (next_p, cost, kind, payload)."""
            out, used = [], set()
            if p in (0, L):                                   # block not covered by the query
                out.append((p, 0, "skip", None))
            for l, labs in by_len[b].items():                 # exact match (X = wildcard)
                if p + l > L:
                    continue
                s = q[p:p + l]
                v = labs.get(s)
                if v is None:
                    cands = labs.values() if "X" in s else [w for w in wild[b] if len(lab(w)) == l]
                    v = next((w for w in cands if eq(s, lab(w))), None)
                if v:
                    out.append((p + l, 0, "match", v))
                    used.add(l)
            if p == 0:                                        # query starts inside this block
                for v in order[b]:
                    t = lab(v)
                    for l in range(1, min(len(t), L + 1)):
                        if l not in used and eq(q[:l], t[-l:]):
                            out.append((l, 0, "match", v))
                            used.add(l)
            rem = q[p:]                                       # query ends inside this block
            if rem and len(rem) not in used:
                v = next((v for v in order[b] if len(rem) < len(lab(v)) and eq(rem, lab(v)[:len(rem)])), None)
                if v:
                    out.append((L, 0, "match", v))
                    used.add(len(rem))
            for l in new_len[b]:                              # new vertex
                if p + l <= L and l not in used:
                    out.append((p + l, 1, "new", q[p:p + l]))
            return out

        # phase 1: fewest new vertices, forward (f) and backward (g)
        INF = 10 ** 9
        f = np.full((B + 1, L + 1), INF)
        f[0, 0] = 0
        E = {}
        for b in range(B):
            for p in np.flatnonzero(f[b] < INF):
                E[b, p] = moves(b, p)
                for p2, c, *_ in E[b, p]:
                    f[b + 1, p2] = min(f[b + 1, p2], f[b, p] + c)
        if f[B, L] >= INF:                                    # lengths too far off: widen
            if slack > 200:
                raise RuntimeError(f"{name}: cannot be placed on the graph")
            return self.add_sequence(name, seq, slack * 3, update)
        g = np.full((B + 1, L + 1), INF)
        g[B, L] = 0
        for b in range(B - 1, -1, -1):
            for p in np.flatnonzero(f[b] < INF):
                g[b, p] = min((g[b + 1, p2] + c for p2, c, *_ in E[b, p]), default=INF)
        opt = f[B, L]

        # phase 2: tie-breaks, only along moves that keep the optimum
        nearest = {}

        def edit(b, s):
            if (b, s) not in nearest:
                labels = [lab(v) for v in self.by_block[b]]
                hit = process.extractOne(s, labels, scorer=Levenshtein.distance) if labels else None
                nearest[b, s] = (int(hit[1]), self.by_block[b][hit[2]]) if hit else (len(s), None)
            return nearest[b, s]

        best = {(0, 0): ((0, 0, 0), None)}
        for b in range(B):
            for p in range(L + 1):
                if (b, p) not in best:
                    continue
                cost = best[b, p][0]
                for p2, c, kind, x in E[b, p]:
                    if f[b, p] + c + g[b + 1, p2] != opt:
                        continue
                    add = (1, edit(b, x)[0], 0) if kind == "new" else (0, 0, kind == "skip")
                    nc = tuple(a + d for a, d in zip(cost, add))
                    if (b + 1, p2) not in best or nc < best[b + 1, p2][0]:
                        best[b + 1, p2] = (nc, (p, kind, x))

        # traceback
        steps, p = [], L
        for b in range(B, 0, -1):
            p0, kind, x = best[b, p][1]
            steps.append((b - 1, p0, p, kind, x))
            p = p0
        if update and name in self.paths:
            name += "_test"
        path, new = [], []
        for b, s, e, kind, x in reversed(steps):
            if kind == "new":
                d, near = edit(b, x)
                vid = self.new_vertex(b, x, added_by=name) if update else f"new@block{b}"
                new.append(dict(id=vid, block=b, label=x, query_span=[s, e],
                                nearest=near, edit_distance=d))
                path.append(vid)
            else:
                path.append(x)                                # existing vertex, or None if skipped
        if update:
            self.add_path(name, path)
        return dict(name=name, n_new=int(opt), path=path, new_vertices=new,
                    unobserved_blocks=[b for b, v in enumerate(path) if v is None])

    # ---------------- save / load ----------------
    def save(self, prefix):
        """prefix.json (graph), prefix.gfa (Bandage/odgi/vg), prefix_vertices.tsv, prefix_blocks.tsv"""
        with open(prefix + ".json", "w") as fh:
            json.dump(dict(k=self.k, next_id=self.next_id, blocks=self.blocks,
                           vertices={i: vars(v) for i, v in self.vertices.items()},
                           by_block=self.by_block, paths=self.paths, ambiguous=self.ambiguous,
                           edges=[[u, v, c] for (u, v), c in self.edges.items()]), fh)
        with open(prefix + ".gfa", "w") as fh:
            fh.write("H\tVN:Z:1.0\n")
            for v in self.vertices.values():
                fh.write(f"S\t{v.id}\t{v.label or '*'}\tRC:i:{v.count}\n")
            for (u, v), c in self.edges.items():
                fh.write(f"L\t{u}\t+\t{v}\t+\t0M\tRC:i:{c}\n")
            for name, p in self.paths.items():            # GFA paths must be contiguous
                runs = [r.split() for r in " ".join(v or "|" for v in p).split("|") if r.strip()]
                for i, r in enumerate(runs):
                    nm = name if len(runs) == 1 else f"{name}#{i + 1}"
                    fh.write(f"P\t{nm}\t{','.join(x + '+' for x in r)}\t*\n")
        with open(prefix + "_vertices.tsv", "w") as fh:
            fh.write("vertex\tblock\tstart_col\tend_col\tcount\tadded_by\tpartial\tlabel\n")
            for v in self.vertices.values():
                s, e, _ = self.blocks[v.block]
                fh.write(f"{v.id}\t{v.block}\t{s}\t{e}\t{v.count}\t{v.added_by or ''}\t{int(v.partial)}\t{v.label}\n")
        with open(prefix + "_blocks.tsv", "w") as fh:
            fh.write("block\tstart_col\tend_col\twidth\tpolymorphism\tn_vertices\n")
            for b, (s, e, poly) in enumerate(self.blocks):
                fh.write(f"{b}\t{s}\t{e}\t{e - s}\t{poly}\t{len(self.by_block[b])}\n")

    @classmethod
    def load(cls, path):
        with open(path) as fh:
            d = json.load(fh)
        g = cls(k=d["k"], blocks=[tuple(b) for b in d["blocks"]], by_block=d["by_block"],
                paths=d["paths"], next_id=d["next_id"],
                ambiguous={a: {int(b): n for b, n in x.items()} for a, x in d["ambiguous"].items()})
        g.vertices = {i: Vertex(**v) for i, v in d["vertices"].items()}
        g.edges = Counter({(u, v): c for u, v, c in d["edges"]})
        return g

    # ---------------- drawing ----------------
    def plot(self, out_png, top=PLOT_TOP, highlight=()):
        """
        x = block (at its alignment position), y = vertex rank by frequency.
        The `top` vertices per block are drawn, the rest pooled into a grey square.
        Node size ~ alleles, edge width ~ alleles, red = added by test sequences,
        coloured lines = highlighted paths. Upper panel: vertices per block.
        """
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.collections import LineCollection

        xs = [(s + e) / 2 for s, e, _ in self.blocks]
        pos = {}
        for b, vids in enumerate(self.by_block):
            vids = sorted(vids, key=lambda v: -self.vertices[v].count)
            keep = vids[:top] + [v for v in vids[top:] if self.vertices[v].added_by]
            for v in vids:
                pos[v] = (xs[b], keep.index(v) if v in keep else len(keep))

        fig, (ax0, ax) = plt.subplots(2, 1, figsize=(16, 9), sharex=True,
                                      gridspec_kw=dict(height_ratios=[1, 5]))
        ax0.bar(xs, [len(v) for v in self.by_block], width=[0.9 * (e - s) for s, e, _ in self.blocks])
        ax0.set_ylabel("vertices")
        ax0.set_title(f"k={self.k}: {len(self.blocks)} blocks, {len(self.vertices)} vertices, "
                      f"{len(self.paths)} paths")

        seg = Counter()
        for (u, v), c in self.edges.items():
            seg[pos[u], pos[v]] += c
        if seg:
            w = np.sqrt(np.array(list(seg.values())) / max(seg.values()))
            ax.add_collection(LineCollection(list(seg), linewidths=0.3 + 3 * w, colors="#999", alpha=0.35))
        node = Counter()
        for v, xy in pos.items():
            node[xy] += self.vertices[v].count
        for xy, c in node.items():
            pooled = xy[1] >= top                     # pooled 'rest' node (grey square)
            ax.scatter(*xy, s=10 + 300 * np.sqrt(c / max(node.values())), zorder=3,
                       color="#BBB" if pooled else "#4C72B0", marker="s" if pooled else "o")
        for v in self.vertices.values():
            if v.added_by:
                ax.scatter(*pos[v.id], s=60, color="#C44E52", edgecolor="k", zorder=4)
        for name in highlight:
            pts = [pos[v] for v in self.paths.get(name, []) if v]
            if pts:
                ax.plot(*zip(*pts), lw=2.2, zorder=5, label=name)
        if highlight:
            ax.legend(loc="upper right", fontsize=9)
        ax.invert_yaxis()
        ax.set_xlim(0, self.blocks[-1][1])
        ax.set_xlabel("alignment column")
        ax.set_ylabel(f"vertex rank in block (top {top})")
        fig.tight_layout()
        fig.savefig(out_png, dpi=150)
        plt.close(fig)


# --------------------------------------------------------------------------
# Pipeline
# --------------------------------------------------------------------------

def build_graph():
    """Align TRAIN_FASTA (or reuse the saved alignment / graph) and build the graph."""
    graph_json = os.path.join(OUT_DIR, "graph.json")
    if REUSE_GRAPH and os.path.exists(graph_json):
        print(f"Loading {graph_json}")
        return BlockGraph.load(graph_json)
    recs = read_fasta(TRAIN_FASTA)
    print(f"{len(recs)} training sequences in {TRAIN_FASTA}")
    aln_path = os.path.join(OUT_DIR, "training.aln.fasta")
    if TRAIN_IS_ALIGNED:
        aln = recs
    elif REUSE_ALIGNMENT and os.path.exists(aln_path):
        aln = read_fasta(aln_path)
        if [n for n, _ in aln] != [n for n, _ in recs]:
            raise ValueError(f"{aln_path} does not match {TRAIN_FASTA}; delete it.")
        print(f"Reusing {aln_path}")
    else:
        print("Aligning with MAFFT...")
        aln = align(recs)
        write_fasta(aln, aln_path)
    g = BlockGraph.build(aln, K)
    g.save(os.path.join(OUT_DIR, "graph"))
    g.plot(os.path.join(OUT_DIR, "graph.png"))
    return g


def run_test(g):
    """Place the test sequences; returns one row per sequence for sequence_vertices.tsv."""
    recs = read_fasta(TEST_FASTA)[:N_TEST]
    print(f"\nPlacing {len(recs)} test sequences from {TEST_FASTA}")
    reports = []
    for i, (name, seq) in enumerate(recs, 1):
        r = g.add_sequence(name, seq, update=UPDATE_GRAPH_WITH_TEST)
        reports.append(r)
        new = ", ".join(f"{v['id']} (block {v['block']}, {v['edit_distance']} from {v['nearest']})"
                        for v in r["new_vertices"])
        print(f"  [{i}/{len(recs)}] {r['name']}: {new or 'existing vertices only'}")
    with open(os.path.join(OUT_DIR, "test_report.json"), "w") as fh:
        json.dump(reports, fh, indent=1)
    if UPDATE_GRAPH_WITH_TEST:
        g.save(os.path.join(OUT_DIR, "graph_updated"))
        g.plot(os.path.join(OUT_DIR, "graph_updated.png"),
               highlight=[r["name"] for r in reports if r["n_new"]][:5])
    print(f"{sum(r['n_new'] == 0 for r in reports)}/{len(reports)} fully represented, "
          f"{sum(r['n_new'] for r in reports)} new vertices")
    return [(r["name"], "test", r["path"], [v["id"] for v in r["new_vertices"]]) for r in reports]


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    g = build_graph()
    rows = [(n, "train", p, []) for n, p in g.paths.items()]
    if TEST_FASTA and os.path.exists(TEST_FASTA):
        rows += run_test(g)
    # one line per sequence; `vertices` in block order, unobserved blocks listed separately
    out = os.path.join(OUT_DIR, "sequence_vertices.tsv")
    with open(out, "w") as fh:
        fh.write("name\tset\tn_vertices\tn_new\tnew_vertices\tunobserved_blocks\tvertices\n")
        for name, kind, path, new in rows:
            seen = [v for v in path if v]
            miss = [str(b) for b, v in enumerate(path) if v is None]
            fh.write(f"{name}\t{kind}\t{len(seen)}\t{len(new)}\t{','.join(new)}\t"
                     f"{','.join(miss)}\t{','.join(seen)}\n")
    print(f"\nPer-sequence vertices: {out}\nAll outputs: {os.path.abspath(OUT_DIR)}")
    return g


if __name__ == "__main__":
    graph = main()

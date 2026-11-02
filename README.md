# HLA block graph

Represents HLA protein alleles (e.g. class I: HLA-A, -B, -C) as a graph of variable-length sequence blocks. Every allele is a path through the graph, and new sequences can be placed on an existing graph with the fewest possible new vertices.

## Method

**1. Alignment.** The training alleles are aligned with [MAFFT](https://mafft.cbrc.jp/alignment/software/). Identical protein sequences are aligned once. By default, sequences of at least 330 residues (full length) are aligned first, and shorter ones (e.g. exon 2-3 only) are added with `mafft --add --keeplength`, which keeps the full-length alignment columns fixed.

**2. Segmentation.** The alignment is read from left to right. A block is closed at the first column where the alleles fully observed in the block show at least `K` distinct (ungapped) strings. Blocks are therefore short in highly polymorphic regions and long in conserved ones. A final block that never reaches `K` is merged into the previous block.

**3. Graph.** Each distinct string in a block is a vertex (`v1`, `v2`, ...). Each allele is a path of one vertex per block, and consecutive vertices on a path are joined by an edge. Leading and trailing gaps are treated as unobserved rather than deleted, so a partial allele (e.g. exon 2-3 only):

- has no vertex in blocks it does not cover;
- in a block it covers only partly, is assigned to the most frequent full-length vertex that matches its observed residues. If no vertex matches, it gets its own vertex, with `X` marking the unobserved positions.

**4. Placing new sequences.** An unaligned sequence is cut into one substring per block. Each substring either equals an existing vertex (cost 0) or becomes a new vertex (cost 1). An exact dynamic program over (block, sequence position) minimises, in order:

1. the number of new vertices;
2. the total edit distance between each new vertex and the nearest existing vertex in its block;
3. the number of blocks the sequence leaves unobserved.

Partial sequences may skip blocks at either end, and may start or end inside a block by matching a vertex suffix or prefix. The new vertices are added to the graph and reported.

## Installation

This project uses [uv](https://docs.astral.sh/uv/). From the repo root:

```
uv sync
```

This creates `.venv/` and installs the dependencies (numpy, matplotlib, rapidfuzz)
from `pyproject.toml`. Without uv you can still `pip install numpy matplotlib rapidfuzz`.

MAFFT is needed only when the training file is unaligned.

- **Linux / macOS:** `conda install -c bioconda mafft` or `sudo apt install mafft`.
- **Windows:** bioconda has no Windows build. Download the Windows package from the MAFFT site, unzip it to a path without spaces, and set `MAFFT = r"C:\path\to\mafft-win\mafft.bat"` in the settings.

## Repository layout

```
src/hla_graph.py   the tool
data/              input FASTA files (training.fasta, test.fasta)
results/           generated outputs (graph.json, graph.gfa, *.tsv, *.png, ...)
pyproject.toml     uv / dependencies
```

Inputs and outputs are resolved relative to the repo root, so the defaults are
`data/training.fasta`, `data/test.fasta` and `results/`. Override with the
`HLA_DATA_DIR` / `HLA_RESULTS_DIR` environment variables if needed.

## Usage

1. Put the training and test FASTA files in `data/` as `train.fa` and
   `test.fa`. IMGT headers (`>HLA:HLA00001 A*01:01:01:01 365 bp`) are read as
   allele names. For other headers, the first word is used.
2. Edit the `SETTINGS` block at the top of `src/hla_graph.py` if needed (`K` can
   also be set with the `HLA_K` environment variable).
3. Run it:

```
uv run python src/hla_graph.py
```

### Environment overrides

Some settings can be set on the command line without editing the source. Pass
them as a prefix on the **same** command (no semicolon), and always go through
`uv run` so the dependencies in `.venv/` are used:

```
# rebuild the graph with a different polymorphism threshold
HLA_K=50 uv run python src/hla_graph.py

# single MAFFT run (for uniform-length partial data, e.g. exon 2-3 only)
HLA_BACKBONE_MIN_LEN=none uv run python src/hla_graph.py

# higher-quality (slower) alignment, and a custom data/results location
HLA_MAFFT_ARGS="--maxiterate 1000 --localpair" HLA_DATA_DIR=/path/to/data \
  HLA_RESULTS_DIR=/path/to/out uv run python src/hla_graph.py
```

Common mistakes: `HLA_K=50; python ...` (the `;` stops the variable reaching
the command) and plain `python ...` (the deps live in the uv `.venv`, so use
`uv run python ...`).

| Variable | Overrides |
|---|---|
| `HLA_K` | `K`, the minimal polymorphism per block. |
| `HLA_BACKBONE_MIN_LEN` | `BACKBONE_MIN_LEN`; `none`/`0` means a single MAFFT run. |
| `HLA_MAFFT_ARGS` | `MAFFT_ARGS` passed to MAFFT. |
| `HLA_DATA_DIR`, `HLA_RESULTS_DIR` | Input / output folders. |

| Setting | Meaning |
|---|---|
| `TRAIN_FASTA`, `TEST_FASTA` | Input files (default `data/train.fa`, `data/test.fa`). Set `TEST_FASTA = ""` to only build the graph. |
| `OUT_DIR` | Output folder (default `results/`). |
| `K` | Minimal number of distinct strings per block (env: `HLA_K`). |
| `N_TEST` | Place only the first N test sequences (`None` = all). |
| `MAFFT`, `MAFFT_ARGS`, `THREADS` | MAFFT executable and options (`MAFFT_ARGS` env: `HLA_MAFFT_ARGS`). |
| `BACKBONE_MIN_LEN` | Length threshold for the two-step alignment (`None` = a single MAFFT run; env: `HLA_BACKBONE_MIN_LEN`). |
| `TRAIN_IS_ALIGNED` | Set to `True` if the training file is already a gapped alignment. |
| `REUSE_ALIGNMENT` | Reuse `OUT_DIR/training.aln.fasta`, so MAFFT runs only once. |
| `REUSE_GRAPH` | Reuse `OUT_DIR/graph.json` instead of rebuilding the graph. |
| `UPDATE_GRAPH_WITH_TEST` | Add the test sequences' new vertices to the graph. If `False`, only report them. |
| `SLACK` | How much shorter or longer than the existing vertices of a block a new vertex may be. |
| `PLOT_TOP` | Vertices drawn individually per block in the figures. |

The functions can also be used directly:

```python
from hla_graph import read_fasta, align, BlockGraph

g = BlockGraph.build(align(read_fasta("training.fasta")), k=100)
report = g.add_sequence("my_allele", "MAVMAPRTLLLLLSGALALTQTWAGSHSMRYF...")
g.save("my_graph")
g.plot("my_graph.png", highlight=["my_allele"])
g = BlockGraph.load("my_graph.json")
```

## Outputs (`OUT_DIR`)

| File | Content |
|---|---|
| `sequence_vertices.tsv` | One line per training and test sequence: its vertices in block order (`v4,v130,v236,...`), the new vertices it created, and the blocks it does not cover. |
| `graph.json` | The full graph, reloadable with `BlockGraph.load`. |
| `graph.gfa` | GFA 1.0, for [Bandage](https://rrwick.github.io/Bandage/), odgi or vg. Allele paths are `P` lines. |
| `graph_vertices.tsv` | For each vertex: block, alignment columns, allele count, residues, and the test sequence that created it, if any. |
| `graph_blocks.tsv` | For each block: alignment columns, width, polymorphism and number of vertices. |
| `graph.png` | Drawing of the graph (see below). |
| `test_report.json` | For each test sequence: path, new vertices, the query positions they cover, and the nearest existing vertex with its edit distance. |
| `graph_updated.*` | The same files for the graph after adding the test sequences. |
| `training.aln.fasta` | The MAFFT alignment. Delete it if the training file changes. |

The figure has two panels. The upper panel shows the number of vertices per block. The lower panel shows blocks along the alignment, with vertices ranked by frequency:

- node size and edge width are proportional to allele counts;
- grey squares pool the vertices beyond `PLOT_TOP`;
- red nodes were added by test sequences;
- coloured lines are highlighted allele paths.

## Notes

- **Polymorphism counting.** Polymorphism is counted only over alleles fully observed in a block. With many partial alleles in the training set, terminal blocks may therefore need many columns to reach `K`.
- **Ambiguous partial alleles.** A partial allele can match several vertices in a block. It is then assigned to the most frequent one, and the cases are listed under `ambiguous` in `graph.json`.
- **Insertions in short sequences.** With the two-step alignment, residues of a short sequence that would need a new alignment column are dropped, and the affected alleles are printed as a warning. Test sequences are not aligned with MAFFT, so this does not apply to them.
- **Block boundaries are fixed.** Adding test sequences never moves block boundaries. Rebuild the graph to re-segment with them included.
- **Example.** HLA-A (IMGT/HLA), 3,000 training and 30 test alleles, `K = 100`: 14 blocks and about 1,630 vertices. Of the 30 test alleles, 20 needed no new vertex and 10 needed one each. The run took about 20 seconds once the alignment existed.

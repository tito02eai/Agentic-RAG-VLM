# Paper

**Agentic RAG-VLM: Affordance-Aware Retrieval-Augmented Generation with Self-Reflective Planning for Robotic Grasping**

Tao Chen, Lizheng Liu, Jiaxu Wang, Ziyue Jiang, Ruiqi Tian, JiGuang Huo and Zhongxue Gan.

IROS 2026 accepted paper.

| File | Content |
| --- | --- |
| [main_final.pdf](main_final.pdf) | Accepted manuscript, eight pages |
| [main_final.tex](main_final.tex) | LaTeX source restored from the original manuscript source bundle |
| [references.bib](references.bib) | Bibliography |
| [main_final.bbl](main_final.bbl) | Compiled bibliography preserved from the original source bundle |
| [ieeeconf.cls](ieeeconf.cls) | Document class used by the source |
| [figures/](figures/) | Five referenced figures |

The PDF was checked against the manuscript stored with the original acceptance materials. Its SHA256 is `d5bd896d3d183e9df937b3c494b4e10d3f1d8fbe1fdc0f46920e2d90bfd00480`.

## Citation

```bibtex
@inproceedings{chen2026agenticragvlm,
  title = {Agentic RAG-VLM: Affordance-Aware Retrieval-Augmented Generation with Self-Reflective Planning for Robotic Grasping},
  author = {Chen, Tao and Liu, Lizheng and Wang, Jiaxu and Jiang, Ziyue and Tian, Ruiqi and Huo, JiGuang and Gan, Zhongxue},
  booktitle = {2026 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS)},
  year = {2026}
}
```

## Build the source

With a LaTeX distribution containing the paper's packages:

```bash
mkdir -p local/output/paper-build
cp paper/main_final.tex paper/main_final.bbl paper/ieeeconf.cls paper/references.bib local/output/paper-build/
cp -r paper/figures local/output/paper-build/
cd local/output/paper-build
pdflatex main_final.tex
pdflatex main_final.tex
```

The recorded `.bbl` supports this build with the original references. Regenerating the bibliography after editing citations additionally requires the standard `IEEEtran.bst` style, available in a LaTeX distribution with IEEE publisher support.

The checked-in PDF is the accepted source artifact. A local rebuild is a separate output. Paper publication rights remain with the authors and relevant publisher. The code's Apache-2.0 license applies to the software. Current engineering measurements are documented in [Validation](../docs/VALIDATION.md).

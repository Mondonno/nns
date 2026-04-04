# Publication Checklist

## Draft package for mentor

- LaTeX source: `main.tex`
- Bibliography: `references.bib`
- Generated figures:
  - `figures/mnist_dataset_samples.png`
  - `figures/mnist_network_architecture.png`
  - `figures/mnist_training_curves.png`
  - `figures/mnist_confusion_matrix.png`
  - `figures/mnist_experiment_comparison.png`
- Metrics artifact: `figures/mnist_conv2d_metrics.json`
- Compiled PDF: `main.pdf`

## External actions still required

1. Create and confirm a real ORCID account, then replace the author metadata in `main.tex` if the conference template requires it.
2. Send the LaTeX package and compiled PDF to the mentor for language review and AI-detector verification.
3. Contact Krzysztof Siemi{\'n}ski regarding arXiv endorsement.
4. Decide whether the current draft should be revised further before submission to IWUS/CEUR.

## Technical state of the repository

- The article compiles successfully.
- The draft no longer depends on placeholder figures.
- The MNIST experiment is reproducible from `nns/nns/mnist_conv2d.py`.
- The figure set is generated directly into `article/figures/` by the experiment script.
- The library now includes an IDX parser in `nns/core/datasets/mnist_dataset.py`.

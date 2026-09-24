## How to generate

Run these commands from the repository root to generate the standalone interactive HTML and PDF:

```bash
# Requirements: Quarto CLI, Python 3, and LuaLaTeX (included with TeX Live).
# The HTML build downloads the pinned Plotly.js bundle, so it needs internet access.
cd assets/documents/prestentations/segmentation_model_full_conceptual
python3 build_standalone.py
quarto render slides.qmd --to pdf --output segmentation_model_full_conceptual.pdf
```

The outputs are `AtlasMSI_interactive.html` and `segmentation_model_full_conceptual.pdf`.

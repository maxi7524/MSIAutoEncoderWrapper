## How to generate

Run these commands from the repository root to generate the standalone interactive HTML and slide-sized PDF:

```bash
# Requirements: Quarto CLI, Python 3, and Chrome Headless Shell with its Linux runtime libraries.
# If needed, install the browser with: quarto install chrome-headless-shell
# The HTML build downloads the pinned Plotly.js bundle, so it needs internet access.
cd assets/documents/prestentations/segmentation_model_full_conceptual
python3 build_standalone.py
python3 build_pdf.py
```

The outputs are `AtlasMSI_interactive.html` and `segmentation_model_full_conceptual.pdf`.

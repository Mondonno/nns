# ArXiv Submission Steps for This Article

This guide is tailored to this folder and its CEUR template setup.

## 1) Confirm final manuscript content

- Verify title, author name, affiliation, and ORCID in main.tex.
- Confirm the final abstract and keywords.
- Confirm all figures in figures/ are final.

## 2) Build using an ArXiv-safe pipeline

Run from this folder:

```bash
make arxiv-pdf
```

What this does:
- Compiles with pdflatex in non-interactive mode.
- Runs bibtex to refresh bibliography output.
- Produces an up-to-date main.bbl for source submission.

## 3) Run preflight checks

```bash
make arxiv-check
```

This check validates:
- Required files exist: main.tex, ceurart.cls, references.bib, main.bbl, main.xmpdata, pdfa.xmpi.
- PNG figure files are present in figures/.
- main.tex does not contain absolute image paths.

## 4) Create upload bundle

```bash
make arxiv-bundle
```

Bundle output:
- build/arxiv/main-arxiv.zip

Included files:
- Main sources and class files.
- Bibliography source and compiled bibliography output.
- Metadata files used by pdfx.
- Figure PNG files.
- Optional local files if present (for safety): main.abs, cc-by.pdf, ceur-ws-logo.pdf.

## 5) Sanity check the zip before upload

```bash
cd build/arxiv
unzip -l main-arxiv.zip
```

Check that:
- main.tex is at top level.
- figures/*.png are included.
- main.bbl is included.

## 6) Upload to ArXiv

- Upload build/arxiv/main-arxiv.zip as source.
- On compiler selection, start with pdfLaTeX.
- If ArXiv reports engine-specific issues, switch to LuaLaTeX and recompile.

## 7) Verify ArXiv generated PDF and logs

Check:
- No missing files.
- All references resolve.
- All figures render.
- Author metadata is correct.

If there is a compile error, fix locally in main.tex and rebuild:

```bash
make arxiv
```

Then upload the refreshed zip from build/arxiv/.

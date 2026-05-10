# LATEX: pdflatex | xelatex | lualatex
LATEX = lualatex
LATEX_FLAGS = -shell-escape
BIBTEX = bibtex

ARXIV_LATEX = pdflatex
ARXIV_LATEX_FLAGS = -interaction=nonstopmode -halt-on-error
ARXIV_BUILD_DIR = build/arxiv
ARXIV_BUNDLE_NAME = main-arxiv.zip
ARXIV_REQUIRED_FILES = main.tex ceurart.cls references.bib main.bbl main.xmpdata pdfa.xmpi
ARXIV_OPTIONAL_FILES = main.abs cc-by.pdf ceur-ws-logo.pdf
ARXIV_FIGURE_FILES = $(wildcard figures/*.png)

FILES = $(patsubst %.tex, %.pdf, $(wildcard *.tex))

.PHONY: all clean cleanall arxiv-check arxiv-pdf arxiv-bundle arxiv

all: $(FILES)


%.pdf: %.tex
	$(LATEX) $(LATEX_FLAGS) $<
	$(LATEX) $(LATEX_FLAGS) $<
	-$(BIBTEX) `basename $< .tex`
	$(LATEX) $(LATEX_FLAGS) $<
	$(LATEX) $(LATEX_FLAGS) $<
	$(LATEX) $(LATEX_FLAGS) $<
	qpdf --linearize --newline-before-endstream $@ /tmp/$@
	mv /tmp/$@ $@

arxiv-check:
	@echo "Checking ArXiv bundle prerequisites..."
	@missing=0; \
	for f in $(ARXIV_REQUIRED_FILES); do \
		if [ ! -f "$$f" ]; then echo "Missing required file: $$f"; missing=1; fi; \
	done; \
	if [ -z "$(ARXIV_FIGURE_FILES)" ]; then echo "No PNG figures found in figures/."; missing=1; fi; \
	if grep -nE '\\\\includegraphics\\{(/|[A-Za-z]:)' main.tex >/dev/null; then \
		echo "Found absolute figure paths in main.tex."; missing=1; \
	fi; \
	if [ "$$missing" -ne 0 ]; then \
		echo "ArXiv check failed."; exit 1; \
	fi; \
	echo "ArXiv check passed."

arxiv-pdf:
	$(ARXIV_LATEX) $(ARXIV_LATEX_FLAGS) main.tex
	-$(BIBTEX) main
	$(ARXIV_LATEX) $(ARXIV_LATEX_FLAGS) main.tex
	$(ARXIV_LATEX) $(ARXIV_LATEX_FLAGS) main.tex

arxiv-bundle: arxiv-check
	@rm -rf $(ARXIV_BUILD_DIR)
	@mkdir -p $(ARXIV_BUILD_DIR)/figures
	@cp $(ARXIV_REQUIRED_FILES) $(ARXIV_BUILD_DIR)/
	@for f in $(ARXIV_OPTIONAL_FILES); do \
		if [ -f "$$f" ]; then cp "$$f" $(ARXIV_BUILD_DIR)/; fi; \
	done
	@cp $(ARXIV_FIGURE_FILES) $(ARXIV_BUILD_DIR)/figures/
	@cd $(ARXIV_BUILD_DIR) && zip -q -r $(ARXIV_BUNDLE_NAME) ./*
	@echo "Created $(ARXIV_BUILD_DIR)/$(ARXIV_BUNDLE_NAME)"

arxiv: arxiv-pdf arxiv-bundle

clean:
	-rm -f *.{log,toc,tac,aux,dvi,ps,bbl,blg,tmp,nav,out,snm,vrb,rel,cut,abs,*~}
	-rm -rf out _minted* auto
	-rm -rf $(ARXIV_BUILD_DIR)
	for i in $(ALLSUBDIRS); do \
	    (cd $$i; make clean) || exit 1; \
	done
	-rm $(FILES)

cleanall: clean
	-rm -rf out auto
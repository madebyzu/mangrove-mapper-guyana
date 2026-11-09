# figure and table paths in the .tex are written relative to reports/, the
# Overleaf project root, so adding the parent to the search path
final-report: final-report.tex report-template.cls references.bib figures/* tables/*
	TEXINPUTS="..:$$TEXINPUTS" latexmk -pdf -interaction=nonstopmode final-report.tex
	

clean:
	rm -f *.aux
	rm -f *.bbl
	rm -f *.log
	rm -f *.blg
	rm -f *.bcf*
	rm -f *.xml
	rm -f *~
	rm -f *.bak
	rm -f *.out
	rm -f *.fdb_latexmk
	rm -f *.fls
	rm -f *.synctex.gz
	rm -f *.toc
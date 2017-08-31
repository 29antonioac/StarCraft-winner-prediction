all: proyecto.pdf

Rnw := $(wildcard capitulos/*.Rnw)
Prefacios := $(wildcard prefacios/*.tex)
Portadas := $(wildcard portada/*.tex)
Apendices := $(wildcard apendices/*.tex)

proyecto.pdf: $(Apendices) $(Prefacios) $(Portadas) proyecto.tex bibliografia.bib
	pdflatex proyecto.tex && bibtex proyecto.aux && pdflatex proyecto.tex && pdflatex proyecto.tex

proyecto.tex: proyecto.Rnw $(Rnw)
	R -e 'library(knitr);knit("proyecto.Rnw")'

install:
	R -e 'install.packages(c("knitr","ggplot2","reshape2", "xtable","stringi"), repos="http://cran.us.r-project.org")'

clean:
	rm *.aux *.pdf *.tex cache/*

all: Prediction.html


Prediction.html: Prediction.Rmd
	R -e 'library(rmarkdown);render("Prediction.Rmd", output_format="ioslides_presentation");render("Prediction.Rmd",output_format="beamer_presentation")'
	cp Prediction.html ../docs/index.html

install:
	R -e 'install.packages(c("rmarkdown","knitr","ggplot2","reshape2", "xtable","stringi"), repos="http://cran.us.r-project.org")'

clean:
	rm Prediction.html

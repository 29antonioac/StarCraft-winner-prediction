# StarCraft-winner-prediction
Using supervised learning to predict the winner of a match.

To see the report (in Spanish only), compile it using

```
make
```
under the Report folder. You have to have `R` and `knitr` installed
in your system.

Install `R` with your package manager, and dependencies using

```
make install
```

To execute all the code you need `Scala`, `Spark` and `Maven`.

To execute it, open the `StarCraft-Analytics` folder with the `Scala-IDE`,
compile it and execute

```
mvn package
```

under the same folder. After that, you only have to send the `.jar` to the
`Spark` machine using `spark-submit`.

```
spark-submit --class analytics.modelling target/StarCraft-Analytics-0.0.1-SNAPSHOT.jar

spark-submit --class analytics.testing target/StarCraft-Analytics-0.0.1-SNAPSHOT.jar
```

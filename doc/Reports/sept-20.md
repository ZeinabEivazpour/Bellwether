Bellwethers in Heterogeneous Transfer Learning
-----
+ Tried using GENERATE--APPLY--MONITOR on HDP data sets
```
|          | EQ   | JDT  | LC*  | ML   | PDE  |
|----------|------|------|------|------|------|
| Ant      |      |      |      | 0.54 | 0.49 |
| camel    |      |      |      |      |      |
| Ivy      |      |      | 0.54 |      |      |
| Jedit    |      |      |      | 0.6  | 0.49 |
| log4j    |      |      |      |      |      |
| Lucene*  | 0.52 | 0.59 | 0.77 |      | 0.57 |
| Poi      |      |      |      |      |      |
| velocity |      |      | 0.66 |      |      |
| xalan    |      |      |      |      |      |
| xerces   |      | 0.64 |      |      | 0.52 |
```
+ There are very few valid metric matches. Average of 1 or 2 metrics.

|    | pde | eq | jdt | lc | ml |
|----|-----|----|-----|----|----|
| cm |     |    |     | 2  |    |
| jm |     |    |     |    |    |
| kc |     |    |     | 1  |    |
| mc |     | 1  |     |    |    |
| mw |     | 1  |     |    |    |

+ And the metrics change with datasets.
+ In many cases there are no matches at all (using the recommended KSAnalyzer)

+ This week, I found embeddings using community bellwethers, instead of an exhaustive search.

|        | Bellwether |
|--------|------------|
| Apache | Lucene     |
| NASA   | MC         |
| ReLink | Safe       |
| AEEEM  | LC         |


+ Instead of matching target datasets with every data set in the other communities, match instead only to the bellwether of the community.

```
+--------+--------+------------+
+        +        + #matches:  +
|        |        | Bellwether |
+--------+--------+------------+
| AEEEM  | Apache | 9          |
+--------+--------+------------+
|        | NASA   | 3*         |
+--------+--------+------------+
|        | ReLink | 12         |
+--------+--------+------------+
| Apache | AEEEM  | 6          |
+        +--------+------------+
|        | NASA   | 1*         |
+        +--------+------------+
|        | ReLink | 7          |
+--------+--------+------------+
| NASA   | AEEEM  | 1*         |
+        +--------+------------+
|        | Apache | 2*         |
+        +--------+------------+
|        | ReLink | 5          |
+--------+--------+------------+
| ReLink | AEEEM  | 34         |
+        +--------+------------+
|        | Apache | 23         |
+        +--------+------------+
|        | NASA   | 16         |
+--------+--------+------------+

* In cases with no matchings, compare the two community bellwethers directly
```
+ Use the same embeddings for learning from all the projects in that community. 
+ The variance is very low and the embeddings remain the as long as the bellwether remains the same.



![image](https://cloud.githubusercontent.com/assets/1433964/18678079/e34565dc-7f28-11e6-9965-d171e1275c0e.png)

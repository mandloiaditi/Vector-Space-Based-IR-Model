
### install hashedindex package available at pypi using 
```
pip install hashedindex
```

## 1) to create inverted index and bigram index for a given file
```
python index_creation.py filepath
```
e.g.

```
python index_creation.py wiki_22
```

Following files will be generated(name of index files hard-coded in index_creation.py)

index.txt
bigram_index.txt
docids.txt


## 2) to test a query 

define a query in query.txt file

### to run query on vector space model:
```python test_queries.py filepath_of_query filepath_of_inverted_index filepath_of_bigramindex filepath_of_docids
```
i.e.
```python test_queries.py query.txt index.txt bigram_index.txt docids.txt
```

### to run query on model based on champion list:
```
python test_queries.py query.txt index.txt bigram_index.txt docids.txt 1
```

### to run query on model based on phrasal query with vector space:
```
python test_queries.py query.txt index.txt bigram_index.txt docids.txt 2
```


Sample queries to be put in query.txt file:
(one query at a time)
```
MIAA members and history
```
```
International Airports 
```

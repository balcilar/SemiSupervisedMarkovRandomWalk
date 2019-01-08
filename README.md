# Semi Supervised Classification with Markov Random Walks

This repository consist of the Python implementation of following research paper [1]. Two different test cases were also provided to test the method and expain how to you can use it in your specific tasks.

Test case1 is classification of the voters either is republician or democrat party. There are a total of 440 voters  with 196 Democrats and 244 Republicans. The data consists of answers (Yes/No/Absent) of 11 different question. We randomly select 20% of them train which means the data we can know the voters pary (democrat or republician) and we pretended the rest of them %80 are unknown. By the provided code, we can reach %72 accuracy on classification such a huge amount of unknown data (80%).

To run this demo just run provided script. 
```
$ python test1.py
```
Test case2 is on iris dataset. Although there are 3 different class, we just focued on two of them. We assume first 20 element of +1 class are known data and second 30 element of +1 class are unknown data. The same for first 20 element of -1 class are known and following 30 element of -1 class are unknown. With provided 40 number of data, we can classfy 60 unknown element's class. As a result all unknown elements are classify correctly. As a result we reported the test cases and their probabilities of belonging +1 class.  
To run this demo just run provided script. 
```
$ python test2.py
```



## Reference

[1] Szummer, Martin, and Tommi Jaakkola. "Partially labeled classification with Markov random walks." Advances in neural information processing systems. 2002.

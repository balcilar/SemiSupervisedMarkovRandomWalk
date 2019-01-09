# Semi Supervised Classification with Markov Random Walks

Semi-Supervised methods are very important if there are enough data but very very small amount of them is labelled. Hence we cannot use supervised method on unlabelled data, within semi supervised techniques, we can get rid of that problem. This repository consist of the Python implementation of following research paper [1]. Two different test cases were also provided to test the method and expain how you can use it in your specific tasks.

Test case1 is classification of the voters either voter is republician or democrat party. There are a total of 440 voters  with 196 Democrats and 244 Republicans. The data consists of answers (Yes/No/Absent) of 11 different question. We randomly select 20% of them train which means we just know 20% of the voters party (democrat or republician) and we pretend the rest of them %80 are unknown. By the provided code, we can reach %72 accuracy on classification such a huge amount of unknown data (80%).

To run this demo just launch provided script. 

```
$ python test1.py
```

Test case2 is on iris dataset. Although there are 3 different class originally, we just focused on two of them. We assume first 20 element of +1 class are known data and second 30 element of +1 class are unknown data. The same for first 20 element of -1 class are known and following 30 element of -1 class are unknown. With provided 40 number of data, we can classfy 60 unknown element's class. As a result all unknown elements are classify correctly. As a result we reported the test cases and their probabilities of belonging +1 class.  

To run this demo just launch provided script. 

```
$ python test2.py
```



## Reference

[1] Szummer, Martin, and Tommi Jaakkola. "Partially labeled classification with Markov random walks." Advances in neural information processing systems. 2002.

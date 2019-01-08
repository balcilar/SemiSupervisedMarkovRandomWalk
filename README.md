# Semi Supervised Classification with Markov Random Walks

This repository consist of the Pythn implementation of following research paper [1]. Two different test cases were also provided to test the method and expain how to you it in your specific tasks.

Test case1 is classification of the voters either is republician or democrat party. There are a total of 440 voters  with 196 Democrats and 244 Republicans. The data consists of answers (Yes/No/Absent) of 11 different question. We randomly select 20% of them train which means the data we can know the voters pary (democrat or republician) and we pretended the rest of them %80 are unknown. By the provided code, we can reach %72 accuracy on classification such a huge amount of unknown data (80%).

To run this demo just run provided script. 
```
$ python test1.py
```
Test case2 is 




## Reference

[1] Szummer, Martin, and Tommi Jaakkola. "Partially labeled classification with Markov random walks." Advances in neural information processing systems. 2002.

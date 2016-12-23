# About this code
These are a helper class and a helper function that I found really usefull

The idea is to easily express dependencies between pieces of code and then
be able to exploit that capability to create simpler architectures

The `depends_on.py` file contains the implementation of `GraphDependency`
and the `depends_on` function.

The `english_proficiency.py` file exhibits one example of a feature extraction 
heuristic I coded one afternoon 
(Thanks [@slezica](https://github.com/slezica)!) to try to get a sense
of some english proficiency.

The `pipeline.py` file exhibits another example to build a pipeline where 
stages depends with one another.
 
 In both cases there's a computational graph to be solved, this implementation
 makes sure that each stage runs just one time.
 
 I hope you find this useful!
 
 
# How to run it
 
~~The code is self contained, you shouldn't need to install anything.~~
You do need to install [networkx](https://networkx.github.io/)(
`pip install networkx`)


~~However~~ Additionaly, if you want to run the `english_proficiency.py` example
you should download [this](http://norvig.com/ngrams/count_1w.txt) file,
install the `nltk` package and download the stopwords 
(`import nltk; nltk.download("stopwords")`)

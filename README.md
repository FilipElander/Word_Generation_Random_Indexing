# Word_Generation_Randome_Indexing
creating word vectors using Random Indexing technique. Function returning k nearest neighbors with distances for each word in `dataset`.


Function returning k nearest neighbors with distances for each word in `words`


To describe how the function operates, imagine you want to find 5 nearest neighbors for the words
"Harry" and "Potter" using cosine distance (which can be computed as 1 - cosine similarity).
For that you would need to call `self.find_nearest(["Harry", "Potter"], k=5, metric='cosine')`.
The output of the function would then be the following list of lists of tuples (LLT)
(all words and distances are just example values):
\verbatim
[[('Harry', 0.0), ('Hagrid', 0.07), ('Snape', 0.08), ('Dumbledore', 0.08), ('Hermione', 0.09)],
[('Potter', 0.0), ('quickly', 0.21), ('asked', 0.22), ('lied', 0.23), ('okay', 0.24)]]
\endverbatim
The i-th element of the LLT would correspond to k nearest neighbors for the i-th word in the `words`
list, provided as an argument. Each tuple contains a word and a similarity/distance metric.
The tuples are sorted either by descending similarity or by ascending distance.

@param      words   A list of words, for which the nearest neighbors should be returned
@param      k       A number of nearest neighbors to be returned
@param      metric  A similarity/distance metric to be used (defaults to cosine distance)

@return     A list of list of tuples in the format specified in the function description
    

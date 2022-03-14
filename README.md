# NMF-initialization-methods

#### overall objective ####
This project aims to investigate algorithms representative of general categories
of initialization methods in matrix factorization, which include random-based
initializations, structured initializations, and evolutionary and natural based
initializations.  We will evaluate our algorithms on the classic swimmer
datasets,  for its simplicity provides convenient visualization for the outcome
of matrix factorization in capturing data latencies.

* *Random-based Initializations*
    * uniform random (default)
    * gaussian random
    * laplacian random
    * poisson random

* *Structured Initialization*
    * deterministic approach: double SVD, nonnegative ICA
    * clustering: k-means initialization

* *Evolutionary and Natural Based Initialization*
    * genetic algorithms
    * particle-swarm optimization
    * fish school search

# 1-Spectral clustering

This package contains a Matlab implementation of *1-Spectral Clustering* using 
the *inverse power method (IPM)* for nonlinear eigenproblems.

Given a graph with weight matrix W, the goal is to find a partitioning of the 
graph which minimizes a given optimality criterion. Currently supported 
criteria are the Ratio and Normalized Cheeger Cut, the Ratio and Normalized
Cut as well as the Symmetric and Normalized Vertex Expansion.

The *inverse power method (IPM)* for nonlinear eigenproblems is used to compute 
a non-constant solution of the associated nonlinear eigenproblem, which then 
yields a bipartition of the graph. A multipartitioning is then obtained using 
a recursive splitting scheme.


## Documentation

#### Installation

The implementation uses mex-files to solve the inner problem of the IPM. 
Compile them by typing 'make' in the Matlab command line.
To solve the subproblems for the vertex expansion criteria 'sve' and 'nve', 
additionally the Matlab Optimization toolbox needs to be installed.


#### Usage

    [clusters,scores,eigvec,lambda] = OneSpectralClustering(W,crit,k,numRuns,verbosity);


#### Input 
    
    W               Sparse weight matrix. Has to be symmetric.
    crit            The graph partitioning criterion to be optimized.
                    Available choices are
                            'rcut' - Ratio Cut 
                            'ncut' - Normalized Cut 
                            'rcc'  - Ratio Cheeger Cut
                            'ncc'  - Normalized Cheeger Cut
                            'sve'  - Symmetric Vertex Expansion
                            'nve'  - Normalized Vertex Expansion
    k               Number of clusters.
    numRuns         Number of additional times the multipartitioning scheme
                    is performed with random initializations (default is 10). 
    verbosity       Controls how much information is displayed. 
                    Levels 0 (silent) - 4 (very verbose), default is 2.


#### Output
    
    clusters        mx(k-1) matrix containing in each column the computed
                    clustering for each partitioning step.
    scores          struct containing the scores for different criteria (rcut, 
                    ncut etc.) as (k-1)x1 vector, representing the result after
                    each partitioning step.
    eigvec          mx1 vector containing the second nonlinear eigenvector
    lambda          corresponding eigenvalue

The final clustering is obtained via clusters(:,end), the corresponding 
values of the optimization criterion are the last elements in the 
corresponding vector in the scores struct, e.g. for rcut: scores.rcut(end).



## References

    M. Hein and T. Buehler.
    An Inverse Power Method for Nonlinear Eigenproblems with Applications 
    in 1-Spectral Clustering and Sparse PCA.
    In Advances in Neural Information Processing Systems 23 (NIPS 2010).
    Extended version available online at http://arxiv.org/abs/1012.0774. 

    T. Bühler. 
    A flexible framework for solving constrained ratio problems in machine learning. 
    Ph.D. Thesis, Saarland University, 2015. 
    Available at http://scidok.sulb.uni-saarland.de/volltexte/2015/6159/.	


 
## License

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <http://www.gnu.org/licenses/>.

If you use this code for your publication, please include a reference 
to the paper "An inverse power method for nonlinear eigenproblems with 
applications in 1-spectral clustering and sparse PCA".
 
 

## Contact

Copyright 2010-20 Thomas Bühler and Matthias Hein.
Machine Learning Group, Saarland University, Germany.
(http://www.ml.uni-saarland.de).

# 1-Spectral clustering

This package contains a Matlab implementation of *1-Spectral Clustering*.
Given a graph with weight matrix W, the *inverse power method (IPM)* for 
nonlinear eigenproblems is used to compute a non-constant eigenvector 
of the graph 1-Laplacian, which then yields a bipartition of the graph. 
A multipartitioning is then obtained using a recursive splitting scheme.

Current version: V1.1



## Installation

The implementation uses a mex-file to solve the inner problem of the
IPM. Compile it using 

    mex -largeArrayDims solveInnerProblem.cpp 

in the matlab command line.



## Usage

    [clusters,cuts,cheegers] = OneSpectralClustering(W,crit,k,numOuter,numInner,verbosity);

#### Input 
    
    W            Sparse weight matrix. Has to be symmetric.
    crit         The multipartition criterion to be optimized. Available 
                 choices are
                        'ncut' - Normalized Cut, 
                        'ncc' - Normalized Cheeger Cut,
                        'rcut' - Ratio Cut, 
                        'rcc' - Ratio Cheeger Cut
    k            number of clusters


#### Input (optional)

If no additional parameters are specified, the multipartitioning scheme
is performed once, where each subpartitioning problem is initialized with
the second eigenvector of the standard graph Laplacian.

The quality of the obtained partitioning can be improved by performing 
additional runs of the multipartitioning scheme (parameter numOuter)
with multiple random initializations at each level (parameter numInner).


    numOuter    number of additional times the multipartitioning scheme is 
                performed (default is 0); 
    numInner    for the additional runs of the multipartitioning scheme: 
                number of random initializations at each level (default is 0).
    verbosity   Controls how much information is displayed. (Levels 0-3,
                default is 2).

#### Output
    
    clusters    mx(k-1) matrix containing in each column the computed
                clustering for each partitioning step.
    cuts        (k-1)x1 vector containing the Ratio/Normalized Cut values after 
                each partitioning step.
    cheegers    (k-1)x1 vector containing the Ratio/Normalized Cheeger Cut 
                values after each partitioning step.

The final clustering is obtained via clusters(:,end), the corresponding 
cut/cheeger values via cuts(end), cheegers(end).



## References

M. Hein and T. Buehler.
*An Inverse Power Method for Nonlinear Eigenproblems with Applications 
in 1-Spectral Clustering and Sparse PCA*.
In Advances in Neural Information Processing Systems 23 (NIPS 2010).
(Extended version available online at http://arxiv.org/abs/1012.0774) 


 
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

Copyright 2010-11 Thomas Bühler and Matthias Hein
(tb/hein@cs.uni-saarland.de). 
Machine Learning Group, Saarland University, Germany
(http://www.ml.uni-saarland.de).

function [evec,eval,flag]=computeStandardEigenvector(W,normalized,deg,verbosity)
% Computes the second eigenvector of the standard graph Laplacian
%
% Usage: 
%   [evec,eval,flag]=computeStandardEigenvector(W,normalized,deg,verbosity)
%
% Input:
%   W           - Sparse symmetric weight matrix.
%   normalized  - True/false for normalized/unnormalized Graph Laplacian.
%   deg         - Degrees of vertices as column vector.
%   verbosity   - If >=1, log output to console.
%
% Output:
%   evec        - Second eigenvector of normalized/unnormalized Laplacian.
%   eval        - Second eigenvalue of normalized/unnormalized Laplacian.
%   flag        - True if eigenvector computation converged.
%
% (C)2010-18 Thomas Buehler and Matthias Hein
% Machine Learning Group, Saarland University, Germany
% http://www.ml.uni-saarland.de

    % deg has to be the degree vector (also in unnormalised case)
    num=size(W,1);
    D=spdiags(deg,0,num,num);
    opts.disp=0;
    opts.tol = 1E-10;
    opts.maxit=800;
    %opts.issym = 1;
    
    flag=true;
    try
        if (normalized)
            [eigvecs,eigvals]= eigs(D-W, D,2,'SA',opts);
        else
            [eigvecs,eigvals]= eigs(D-W, 2,'SA',opts);
        end
        evec=eigvecs(:,2);
        eval=eigvals(2,2);
    catch exc
        flag=false;
        if(verbosity>=1) 
            disp('WARNING! COMPUTATION OF SECOND EIGENVECTOR OF THE STANDARD GRAPH LAPLACIAN NOT SUCESSFUL!');
            disp(exc.message);
        end
        evec=randn(num,1);
        eval=1;
    end   
end

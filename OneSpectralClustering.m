function [clusters,cuts,cheegers,eigvec,lambda] = OneSpectralClustering(W,criterion,k,numRuns,verbosity)
% Performs 1-Spectral Clustering as described in the paper
%
% M. Hein and T. Bühler
% An Inverse Power Method for Nonlinear Eigenproblems with Applications in 1-Spectral Clustering and Sparse PCA
% In Advances in Neural Information Processing Systems 23 (NIPS 2010)
% Available online at http://arxiv.org/abs/1012.0774
%
% Usage:	
%    [clusters,cuts,cheegers] = OneSpectralClustering(W,criterion,k,numRuns,verbosity);
%
% Input: 
%   W           - Sparse weight matrix. Has to be symmetric.
%   criterion   - The multipartition criterion to be optimized. Available
%                 choices are
%                   'ncut' - Normalized Cut, 
%                   'ncc' - Normalized Cheeger Cut,
%                   'rcut' - Ratio Cut, 
%                   'rcc' - Ratio Cheeger Cut
%   k           - number of clusters
%
% Input(optional):
%   numRuns     - number of additional times the multipartitioning scheme
%                 is performed with random initializations (default is 10). 
%   verbosity   - Controls how much information is displayed. 
%                 Levels 0 (silent) - 4 (very verbose), default is 2.
%
% Output:
%   clusters    - mx(k-1) matrix containing in each column the computed
%                 clustering for each partitioning step.
%   cuts        - (k-1)x1 vector containing the Ratio/Normalized Cut values 
%                 after each partitioning step.
%   cheegers    - (k-1)x1 vector containing the Ratio/Normalized Cheeger 
%                 Cut values after each partitioning step.
%   eigvec      - mx1 vector containing the second eigenvector of the 1-Laplacian
%   lambda      - corresponding eigenvalue
%
% 
% The final clustering is obtained via clusters(:,end), the corresponding 
% cut/Cheeger cut values via cuts(end), cheegers(end).
%
% If more flexibility is desired (e.g. turn off second eigenvector 
% initialization, uncouple multicut-criterion from thresholding criterion), 
% call the subroutine computeMultiPartitioning directly. 
%
%
% This program is free software: you can redistribute it and/or modify
% it under the terms of the GNU General Public License as published by
% the Free Software Foundation, either version 3 of the License, or
% (at your option) any later version.
% 
% (C)2010-18 Thomas Buehler and Matthias Hein
% Machine Learning Group, Saarland University, Germany
% http://www.ml.uni-saarland.de

    if(nargin<5)
        verbosity=2;
    end
    if (nargin< 4)
        numRuns=10;
    end
    
    assert(k>=2,'Wrong usage. Number of clusters has to be at least 2.');
    assert(k<=size(W,1), 'Wrong usage. Number of clusters is larger than size of the graph.');
    assert(isnumeric(W) && issparse(W),'Wrong usage. W should be sparse and numeric.');
    assert(sum(sum(W~=W'))==0,'Wrong usage. W should be symmetric.');
    assert(numRuns>=0, 'Wrong usage. numRuns has to be non-negative.');
    
    switch(lower(criterion))
        case 'ncut'    
            criterion_inner=1; normalized=true;
        case 'ncc'     
            criterion_inner=2; normalized=true;
        case 'rcut'    
            criterion_inner=1; normalized=false;
        case 'rcc'     
            criterion_inner=2; normalized=false;
        otherwise
            error('Wrong usage. Unknown clustering criterion. Available clustering criteria are Ncut/NCC/Rcut/RCC.');
    end
    
    if (verbosity>=1)
        if(criterion_inner==1)
            critstring='Cut';
        else
            critstring ='Cheeger Cut';
        end
        if(normalized)
            critstring=sprintf('Normalized %s',critstring);
        else
            critstring=sprintf('Ratio %s',critstring);
        end
        fprintf('Optimization criterion: %s\n',critstring);
        fprintf('Number of clusters: %d\n',k);
        tempstring='Number of runs: 1 (second eigenvector initialization)';
        if (numRuns==0)
            fprintf(strcat(tempstring,'.'));
        else
            fprintf(strcat(tempstring,sprintf(' + %d (random initialization). ',numRuns)));
        end
        fprintf('\n');
    end

    if (verbosity>=3 && numRuns>0) fprintf('Starting run with second eigenvector initialization.\n'); end;
        
    try
        [clusters,cuts,cheegers,eigvec,lambda] = computeMultiPartitioning(W,normalized,k,true,0,criterion_inner,criterion_inner,verbosity);

        if (verbosity >=2)
            fprintf('Result (eigenvector): %s',  displayCurrentObjective(cuts(end), cheegers(end), normalized));
        end
    catch exc
        %disp(exc.identifier);
        if (strcmp(exc.identifier,'OneSpect:cutinf'))
            if (numRuns>0)
                cuts=inf;
                cheegers=inf;
                fprintf(strcat('WARNING!\t',exc.message,'\n'));
                fprintf('Proceeding with run with random initializations.\n\n');
            else
                cuts=inf;
                cheegers=inf;
                clusters=NaN(size(W,1),1);
                eigvec=NaN(size(W,1),1);
                lambda=NaN;
                fprintf(strcat('ERROR!\t',exc.message,'\n'));
                fprintf('Rerun with additional random initializations.\n');
                return;
            end
        else
            rethrow(exc);
        end
    end
    
    for l=1:numRuns
        if (verbosity>=3) fprintf('Starting run with random initialization %d of %d.\n',l,numRuns); end;
        
        [clusters_temp,cuts_temp,cheegers_temp,eigvec_temp,lambda_temp] = computeMultiPartitioning(W,normalized,k,false,1,criterion_inner,criterion_inner,verbosity);
        
        if (verbosity >=2)
            fprintf('Result (random %d of %d): %s', l, numRuns, displayCurrentObjective(cuts_temp(end), cheegers_temp(end), normalized));
        end

        % if the final cut/cheeger cut is better according to the given
        % criterion, take this partition
        if ((criterion_inner==1 && cuts_temp(end)<cuts(end)) || (criterion_inner==2 && cheegers_temp(end)<cheegers(end)))
            [clusters,cuts,cheegers]=deal(clusters_temp,cuts_temp,cheegers_temp);
        end
        % for the eigenvector and eigenvalue, we have to look at the first
        % partition
        if ((criterion_inner==1 && cuts_temp(1)<cuts(1)) || (criterion_inner==2 && cheegers_temp(1)<cheegers(1)))
            [eigvec,lambda]=deal(eigvec_temp,lambda_temp);
        end
        
    end

    if (verbosity >=1)
        fprintf('Best result: %s', displayCurrentObjective(cuts(end), cheegers(end), normalized));
    end
end
    

% Displays the current objective value
function objective = displayCurrentObjective(cut_temp,cheeger_temp,normalized)
    
    if (normalized)
        objective = sprintf('Normalized Cut: %.8g - Normalized Cheeger Cut: %.8g\n',cut_temp,cheeger_temp); 
    else
        objective = sprintf('Ratio Cut: %.8g - Ratio Cheeger Cut: %.8g\n',cut_temp,cheeger_temp); 
    end
    
end          

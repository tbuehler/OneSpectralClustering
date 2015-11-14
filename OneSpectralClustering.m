function [clusters,cuts,cheegers] = OneSpectralClustering(W,criterion,k,numOuter,numInner,verbosity)
% Performs 1-Spectral Clustering as described in the paper
%
% M. Hein and T. Bühler
% An Inverse Power Method for Nonlinear Eigenproblems with Applications in 1-Spectral Clustering and Sparse PCA
% In Advances in Neural Information Processing Systems 23 (NIPS 2010)
% Available online at http://arxiv.org/abs/1012.0774
%
% Usage, fast version:	
%    [clusters,cuts,cheegers] = OneSpectralClustering(W,criterion,k);
% Usage, slower version, but improved partitioning (recommended):
%    [clusters,cuts,cheegers] = OneSpectralClustering(W,criterion,k,numOuter,numInner);
%    [clusters,cuts,cheegers] = OneSpectralClustering(W,criterion,k,numOuter,numInner,verbosity);
%
% Input: 
%   W: Sparse weight matrix. Has to be symmetric.
%   criterion: The multipartition criterion to be optimized. Available
%   choices are
%               'ncut' - Normalized Cut, 
%               'ncc' - Normalized Cheeger Cut,
%               'rcut' - Ratio Cut, 
%               'rcc' - Ratio Cheeger Cut
%   k: number of clusters
%
% If no additional parameters are specified, the multipartitioning scheme
% is performed once, where each subpartitioning problem is initialized with
% the second eigenvector of the standard graph Laplacian (fast version).
% 
% The quality of the obtained partitioning can be improved by performing 
% additional runs of the multipartitioning scheme (parameter numOuter)
% with multiple random initializations at each level (parameter numInner).
%
% Input(optional):
%   numOuter: number of additional times the multipartitioning scheme is 
%   performed (default is 0); 
%   numInner: for the additional runs of the multipartitioning scheme: 
%   number of random initializations at each level (default is 0).
%   verbosity: Controls how much information is displayed. Levels 0-3,
%   default is 2.
%
% Output:
%   clusters: mx(k-1) matrix containing in each column the computed
%   clustering for each partitioning step.
%   cuts: (k-1)x1 vector containing the Ratio/Normalized Cut values after 
%   each partitioning step.
%   cheegers: (k-1)x1 vector containing the Ratio/Normalized Cheeger Cut 
%   values after each partitioning step.
%
% The final clustering is obtained via clusters(:,end), the corresponding 
% cut/cheeger values via cuts(end), cheegers(end).
%
% If more flexibility is desired (e.g. turn off second eigenvector 
% initialization, uncouple multicut-criterion from thresholding criterion), 
% call the subroutine computeMultiPartitioning directly. 
%
% (C)2010-11 Thomas Buehler and Matthias Hein
% Machine Learning Group, Saarland University, Germany
% http://www.ml.uni-saarland.de

    if(nargin<6)
        verbosity=2;
    end
    if (nargin< 4)
        numOuter=0;
        numInner=0;
    end
    
    assert(k>=2,'Wrong usage. Number of clusters has to be at least 2.');
    assert(k<=size(W,1), 'Wrong usage. Number of clusters is larger than size of the graph.');
    assert(isnumeric(W) && issparse(W),'Wrong usage. W should be sparse and numeric.');
    assert(sum(sum(W~=W'))==0,'Wrong usage. W should be symmetric.');
    assert(isempty(find(diag(W)~=0,1)),'Wrong usage. Graph contains self loops. W has to have zero diagonal.');
	assert(~(numOuter>0 && numInner==0), sprintf('Wrong usage. numOuter=%d but numInner=%d. numInner has to be positive.',numOuter,numInner));
    
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
        tempstring='Performing 1 run initialized with second eigenvector of standard graph Laplacian';
        if (numOuter==0)
            fprintf(strcat(tempstring,'.\n'));
        else
            if(numOuter==1)
                fprintf(strcat(tempstring,'\nand 1 additional run with random initializations. '));
            else
                fprintf(strcat(tempstring,sprintf('\nand %d additional runs with random initializations. ',numOuter)));
            end
            fprintf(' Number of random initializations: %d \n ',numInner);
        end
        fprintf('\n');
    end

    if (verbosity>=1 && numOuter>0) fprintf('STARTING RUN WITH SECOND EIGENVECTOR INITIALIZATION.\n'); end;
        
    try
        [clusters,cuts,cheegers] = computeMultiPartitioning(W,normalized,k,true,0,criterion_inner,criterion_inner,verbosity);
    catch exc
        %disp(exc.identifier);
        if (strcmp(exc.identifier,'OneSpect:cutinf'))
            if (numOuter>0)
                cuts=inf;
                cheegers=inf;
                fprintf(strcat('WARNING!\t',exc.message,'\n'));
                fprintf('Proceeding with run with random initializations.\n\n');
            else
                cuts=inf;
                cheegers=inf;
                clusters=NaN(size(W,1),1);
                fprintf(strcat('ERROR!\t',exc.message,'\n'));
                fprintf('Rerun with additional random initializations.\n');
                return;
            end
        else
            rethrow(exc);
        end
    end
            
    
    for l=1:numOuter
        if (verbosity>=1) fprintf('STARTING RUN WITH RANDOM INITIALIZATIONS %d OF %d.\n', l,numOuter); end;
        
        [clusters_temp,cuts_temp,cheegers_temp] = computeMultiPartitioning(W,normalized,k,false,numInner,criterion_inner,criterion_inner,verbosity);

        if ((criterion_inner==1 && cuts_temp(end)<cuts(end)) || (criterion_inner==2 && cheegers_temp(end)<cheegers(end)))
            [clusters,cuts,cheegers]=deal(clusters_temp,cuts_temp,cheegers_temp);
        end
    end
    

    fprintf('Best result:\n');
    if (normalized)
        fprintf('Normalized Cut: %.8g   Normalized Cheeger Cut: %.8g\n',cuts(end),cheegers(end)); 
    else
        fprintf('Ratio Cut: %.8g   Ratio Cheeger Cut: %.8g\n',cuts(end),cheegers(end)); 
    end

end
    
            
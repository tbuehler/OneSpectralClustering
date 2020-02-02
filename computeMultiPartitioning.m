function [clusters, scores, eigvec, lambda] = computeMultiPartitioning(W, normalized, k, init2nd, crit, verbosity)
% Computes a multipartitioning of the data given by a similarity matrix W 
% by recursively computing bipartitions using eigenvectors of the 1-Laplacian.
%
% Usage:	
%        [clusters, scores, eigvec, lambda] = computeMultiPartitioning(W, normalized, k, init2nd, crit, verbosity)
%
% Input:
%   W           - Sparse weight matrix. Has to be symmetric.
%   normalized  - True for Ncut/NCC/NVE, false for Rcut/RCC/SVE.
%   k           - Number of clusters.
%   init2nd     - If true, the solver for the nonlinear eigenproblem is
%                 initialized with the thresholded 2nd eigenvector of the 
%                 standard graph Laplacian, otherwise with a random vector.
%   crit        - 1: Rcut/Ncut, 2: RCC/NCC, 3: SVE/NVE
%   verbosity   - Controls how much information is displayed.
%                 Levels 0 (silent) - 4 (very verbose).
%
% Output:
%   clusters    - mx(k-1) matrix containing in each column the computed 
%                 clustering for each partitioning step.
%   scores      - struct containing the scores for different criteria (rcut, 
%                 ncut etc.) as (k-1)x1 vector, representing the result after
%                 each partitioning step.
%   eigvec      - mx1 vector containing 2nd eigenvector of the 1-Laplacian
%   lambda      - corresponding eigenvalue
%
%
% This program is free software: you can redistribute it and/or modify
% it under the terms of the GNU General Public License as published by
% the Free Software Foundation, either version 3 of the License, or
% (at your option) any later version.
%
% (C)2010-2020 Thomas Buehler and Matthias Hein
% Machine Learning Group, Saarland University, Germany
% http://www.ml.uni-saarland.de
 
    num = size(W,1);
    
    assert(k>=2, 'Wrong usage. Number of clusters has to be at least 2.');
    assert(k<=num, 'Wrong usage. Number of clusters is larger than size of the graph.');
    assert(isnumeric(W) && issparse(W), 'Wrong usage. W should be sparse and numeric.');
    assert(sum(sum(W~=W'))==0, 'Wrong usage. W should be symmetric.');
    assert((crit==1 || crit==2 || crit==3), 'Wrong usage. Available criteria are 1: Ncut/Rcut, 2: NCC/RCC, 3:SVE/NVE');
    if (crit==3); assert(k<=2, 'Wrong usage. k>2 not supported for SVE/NVE criterion.'); end

    threshold_type = -1; 

    clusters = zeros(num,k-1);
    cuts = zeros(1,k-1);
    cheegers = zeros(1,k-1);
    cutParts = zeros(1,k);
    
    deg = full(sum(W,2)); 
    if normalized
        gdeg = deg;
    else
        gdeg = ones(num,1);
    end
        
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %%% Perform the first partitioning step into 2 clusters %%%
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    [comp,connected,sizes] = connectedComponents(W);
    if(~connected)
        if(verbosity>=1); disp('WARNING! GRAPH IS NOT CONNECTED!'); end
        if(verbosity>=3); disp('... Optimal Cut achieved by separating connected components.'); end
        allClusters = balanceConnectedComponents(comp, sizes, W, normalized);
        eigvec = zeros(length(allClusters), 1);
        [cut, cheeger, cutPart1, cutPart2, lambda] = deal(0);
    else
        if(verbosity>=3); disp('... Starting to compute partitioning into 2 parts.'); end 
       
        % prepare start vector
        if(init2nd)
            if(verbosity>=3); disp('...... Computing second eigenvector of standard graph Laplacian.'); end
            [start, ~, success] = computeStandardEigenvector(W, normalized, deg, verbosity);
            if (success) 
                if(crit<3)
                    start = createClustersGeneral(start, W, normalized, threshold_type, crit, gdeg, false);
                else
                    start = opt_thresh_vertex_expansion(start, W, normalized);
                end
                if (sum(start)>sum(start==0)); start = 1-start; end
                start = start/sum(start);
                if(verbosity>=3); disp('...... Solving nonlinear eigenproblem with eigenvector initialization.'); end
            else
                error('OneSpect:cutinf', 'Clustering with 2nd eigenvector initialization failed at first level');
            end
        else 
            start = randn(num,1);
            if(verbosity>=3); disp('...... Solving nonlinear eigenproblem with random initialization.'); end
        end

        % compute the nonlinear eigenvector and resulting clusters
        [vmin, fval] = computeEigenvectorGeneral(W, start, normalized, crit, verbosity>=4);       
        if (crit<3)
            [allClusters, cut, cheeger, cutPart1, cutPart2] = createClustersGeneral( ...
                                                              vmin, W, normalized, threshold_type, crit, gdeg, false);
        else
            [allClusters, cheeger] = opt_thresh_vertex_expansion(vmin, W, normalized);
            cut = cheeger; % all this stuff is not used currently
            cutPart1 = cheeger;
            cutPart2 = cheeger;
        end
        eigvec = vmin;
        lambda = fval(end);
    end
    assert(cut<inf);
    
    allClusters = allClusters+1;
    clusters(:,1) = allClusters;
    cuts(:,1) = cut;
    cheegers(:,1) = cheeger;
    cutParts(1) = cutPart1;
    cutParts(2) = cutPart2;
    subCutParts = zeros(k,2);
    subClusters = cell(1,k);

    if(verbosity>=3)
        fprintf('...... Finished Clustering into 2 parts.\n');
        fprintf('...... Final result: %s', displayCurrentObjective(cut, cheeger, normalized, crit));
    end

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %%% Perform the 2nd to (k-1)th partitioning step %%%
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    for l=3:k
        if(verbosity>=3); fprintf('... Starting to compute partitioning into %d parts.\n', l); end

        bestCut = inf;
        bestCheeger = inf;
        % in each step we need to check for each of the current l-1 clusters
        % whether splitting it will give the best overall objective
        for m=1:l-1
            index_m = find(allClusters==m);
            
            if (~isempty(subClusters{m}))
                % we have already solved this subproblem	
                clusters_in_m = subClusters{m};
                cutPart1 = subCutParts(m,1);
                cutPart2 = subCutParts(m,2);
            elseif(length(index_m)==1)
                % the current cluster has size 1 it cannot be further divided
                clusters_in_m = [];
                cutPart1 = inf;
                cutPart2 = inf;
                subClusters{m} = clusters_in_m;
                subCutParts(m,1:2) = [cutPart1 cutPart2];
            elseif(length(index_m)==2)
                % edge case of size 2 cluster
                clusters_in_m = [0;1];
                cutPart1 = deg(index_m(1))-W(index_m(1),index_m(1));
                cutPart2 = deg(index_m(2))-W(index_m(2),index_m(2));
                if normalized
                    if (cutPart1>0); cutPart1 = cutPart1/deg(index_m(1)); end
                    if (cutPart2>0); cutPart2 = cutPart2/deg(index_m(2)); end
                end                    
                subClusters{m} = clusters_in_m;
                subCutParts(m,1:2) = [cutPart1 cutPart2];    
            else 
                % otherwise we have to compute the partition   
                if(verbosity>=3); fprintf('...... Checking partitioning of subgraph %d.\n', m); end
                [clusters_in_m, cutPart1, cutPart2] = findBestPartitionInM(W, index_m, normalized, gdeg, cutParts, ...
                                                                           crit, init2nd, verbosity, deg, m, l, num);
                subClusters{m} = clusters_in_m;
                subCutParts(m,1:2) = [cutPart1 cutPart2]; 
            end
            
            % compute the best cut possible by partitioning of current subgraph
            [cut, cheeger] = computeCutCheeger(cutParts, cutPart1, cutPart2, m, l);
            if(verbosity>=3)
                fprintf('...... Best result achievable by partitioning of subgraph %d:\n', m);
                fprintf('...... %s', displayCurrentObjective(cut, cheeger, normalized, crit));
            end
            
            % check if partitoning of the current subgraph gives better cut
            if (crit==1 && cut<bestCut) || (crit==2 && cheeger<bestCheeger)
                [bestCut, bestCheeger, bestCutPart1, bestCutPart2, best_m] = deal(cut, cheeger, cutPart1, cutPart2, m);
                clusters_new = allClusters;
                clusters_new(index_m) = (l-m)*clusters_in_m+clusters_new(index_m);
                assert(bestCut>=0 && bestCheeger>=0);
            end
            
            % if we have already found a partition with cut 0, we can skip the other subgraphs
            if (bestCut==0); break; end
        end
        assert (bestCut<inf);

        % Update
        allClusters = clusters_new;
        cuts(1,l-1) = bestCut;
        cheegers(1,l-1) = bestCheeger;
        clusters(:,l-1) = allClusters;
        cutParts(best_m) = bestCutPart1;
        cutParts(l) = bestCutPart2;
        
        % Check that we have the right number of clusters
        assert(length(unique(allClusters))==l);
        
        % Reset subcutparts and subclusters;
        subCutParts(best_m,:) = 0;
        subClusters{best_m} = [];
        subCutParts(l,:) = 0;
        subClusters{l} = [];
          
        % Print out current objective
        if(verbosity>=3)
            fprintf('...... Finished Clustering into %d parts (decided to partition subgraph %d).\n', l, best_m);
            fprintf('...... Final result: %s', displayCurrentObjective(bestCut, bestCheeger, normalized, crit));
        end
    end

    if (crit<3)
        if (normalized)
            scores = struct('ncut', cuts, 'ncc', cheegers);
        else
            scores = struct('rcut', cuts, 'rcc', cheegers);
        end
    else
        if (normalized)
            scores = struct('nve', cuts);
        else
            scores = struct('sve', cuts);
        end
    end
end


% Find the best partition which can be obtained by partitioning subgraph m.
function [clusters_in_m, cutPart1, cutPart2] = findBestPartitionInM(W, index_m, normalized, gdeg, cutParts, ...
                                                                    crit, init2nd, verbosity, deg, m, l, num)

    % extract subgraph and its connected components
    Wm = W(index_m, index_m);
    size_m = size(Wm, 1);
    [comp, connected, sizes] = connectedComponents(Wm);
    
    cutPart1 = inf;
    cutPart2 = inf;
    
    if (~connected) 
        if(verbosity>=3); disp('...... Subgraph is not connected.'); end
        % check partition which has connected component as one cluster and rest as second cluster
        for m1=1:length(sizes)   
            if(cutPart1+cutPart2>0)
                if(verbosity>=3)
                    fprintf('...... Checking partition found by isolating component %d of %d.\n', m1, length(sizes));
                end
                clusters_in_m_tmp = double(comp==m1);

                cluster_m2 = zeros(num,1);
                cluster_m2(index_m) = clusters_in_m_tmp;
                cutPart2_tmp = computeCutValue(cluster_m2, W, normalized, gdeg); 

                cluster_m1 = zeros(num,1);
                cluster_m1(index_m) = (clusters_in_m_tmp==0);
                cutPart1_tmp = computeCutValue(cluster_m1, W, normalized, gdeg); 

                if(verbosity>=3) 
                    [cut_tmp, cheeger_tmp] = computeCutCheeger(cutParts, cutPart1_tmp, cutPart2_tmp, m, l);
                    fprintf('...... %s', displayCurrentObjective(cut_tmp, cheeger_tmp, normalized, crit)); 
                end

                if (crit==1 && cutPart1_tmp+cutPart2_tmp<cutPart1+cutPart2 ||  ...
                            crit==2 && max(cutPart1_tmp,cutPart2_tmp)<max(cutPart1,cutPart2))
                    [cutPart1, cutPart2, clusters_in_m] = deal(cutPart1_tmp, cutPart2_tmp, clusters_in_m_tmp);
                    assert(length(clusters_in_m)==length(index_m));
                end
                assert(logical(exist('clusters_in_m','var')));
            end 
        end
    end
    if(cutPart1+cutPart2>0)
        for m1=1:length(sizes)
            index_comp = find(comp==m1);
            
            if (length(index_comp)>1)
                % the size of the current connected component is larger than 1, try to partition it
                Wm_comp = sparse(Wm(index_comp,index_comp));
                Wm_comp2 = Wm_comp;
                if(2*max(sum(Wm_comp.^2))<eps)
                    Wm_comp2 = Wm_comp/max(max(Wm_comp));
                end
                if(~connected && verbosity>=3)
                    fprintf('...... Computing partitioning of component %d of %d of subgraph %d.\n', m1, length(sizes), m);
                end
                if (~connected)
                    index_rest = find(comp~=m1); % all other components in the current cluster
                    cut_rest = sum(sum(W(index_m(index_rest), setdiff(1:num, index_m(index_rest)))));
                    size_rest = sum(gdeg(index_m(index_rest)));
                else
                    cut_rest = 0;
                    size_rest = 0;
                end

                % prepare start vector
                if(init2nd)
                    if(verbosity>=3); disp('...... Computing second eigenvector of standard graph Laplacian.'); end
                    [start_comp, ~, success] = computeStandardEigenvector(Wm_comp, normalized, ...
                                                                               deg(index_m(index_comp)), verbosity);
                    if(success)
                        start_m = createSubClusters(start_comp, Wm_comp, normalized, deg, crit, index_comp, ...
                                                    index_m, cut_rest, size_rest, size_m);
                        start_comp = start_m(index_comp);
                        assert(size(start_comp,1)==size(Wm_comp,1));
                        if (sum(start_comp)>sum(start_comp==0)); start_comp = 1-start_comp; end
                        start_comp = start_comp/sum(start_comp);
                        if(verbosity>=3); disp('...... Solving nonlinear eigenproblem with eigenvector initialization.'); end
                    else
                        error('OneSpect:cutinf', 'Clustering with 2nd eigenvector initialization failed at level %d.', l-1);
                    end
                else
                    start_comp = randn(size(Wm_comp,1),1);
                    if(verbosity>=3); disp('...... Solving nonlinear eigenproblem with random initialization.'); end
                end

                % compute the nonlinear eigenvector and resulting clustering
                vmin_comp = computeEigenvectorGeneral(Wm_comp2, start_comp, normalized, crit, verbosity>=4);
                [clusters_in_m_tmp, cutPart1_tmp, cutPart2_tmp] = createSubClusters(vmin_comp, Wm_comp, normalized, deg, ...
                                                                  crit, index_comp, index_m, cut_rest, size_rest, size_m);

                % Display current objective
                if(verbosity>=3) 
                    [cut_tmp, cheeger_tmp] = computeCutCheeger(cutParts, cutPart1_tmp, cutPart2_tmp, m, l);
                    fprintf('...... %s', displayCurrentObjective(cut_tmp, cheeger_tmp, normalized, crit)); 
                end

                % Check if we're better
                if (crit==1 && cutPart1_tmp+cutPart2_tmp<cutPart1+cutPart2 || ...
                         crit==2 && max(cutPart1_tmp,cutPart2_tmp)<max(cutPart1,cutPart2))
                    [cutPart1, cutPart2, clusters_in_m] = deal(cutPart1_tmp, cutPart2_tmp, clusters_in_m_tmp);
                    assert(length(clusters_in_m)==length(index_m));
                end
            end
        end
    end
end


% Computes Rcut/Ncut and Cheeger Cut values
function [cut, cheeger] = computeCutCheeger(cutParts, cutPart1, cutPart2, m, l)

    cut = sum(cutParts)-cutParts(m)+cutPart1+cutPart2;
    cheeger = max([cutParts((1:l-1)~=m) cutPart1 cutPart2]);
end


% Displays the current objective value
function objective = displayCurrentObjective(cut_tmp, cheeger_tmp, normalized, crit)
    if (crit<3)
        if (normalized)
            objective = sprintf('Normalized Cut: %.8g - Normalized Cheeger Cut: %.8g\n', cut_tmp, cheeger_tmp); 
        else
            objective = sprintf('Ratio Cut: %.8g - Ratio Cheeger Cut: %.8g\n', cut_tmp, cheeger_tmp); 
        end
    else
        if (normalized)
            objective = sprintf('Normalized Vertex Expansion: %.8g\n', cut_tmp); 
        else
            objective = sprintf('Symmetric Vertex Expansion: %.8g\n', cut_tmp); 
        end
    end
end


% Creates two clusters by thresholding the vector vmin_comp obtained on a
% connected component of a subgraph. Given the two clusters A and B on the 
% connected component, we consider two different ways of dealing with
% the remaining connected components C of the subgraph: the first is to 
% merge C with A, the second to merge C with B. The method takes the one 
% yielding the lower Cut/Cheeger cut.
function [allClustersInClusterM, cutPart1, cutPart2] = createSubClusters(vmin_comp, W_comp, normalized, deg, crit, ...
                                                                         index_comp, index_m, cut_rest, size_rest, size_m)
      
    % input parameter deg has to be the degree vector (also in unnormalised case)
    if (size(deg,1)>1); deg = deg'; end % make deg a row vector
    
    connected = (cut_rest==0 && size_rest==0);
    [~, index] = sort(vmin_comp);
    W_sorted = W_comp(index,index);

    % calculate cuts
    deg_comp = deg(index_m(index_comp));
    volumes_threshold = cumsum(deg_comp(index));
    triup = triu(W_sorted,1);
    tmpcuts_threshold = volumes_threshold - 2*cumsum(full(sum(triup))) - cumsum(full(diag(W_sorted)))';
    tmpcuts_threshold2 = (volumes_threshold(end)-volumes_threshold) - ...
                          (sum(sum(W_sorted))-2*cumsum(full(sum(triup,2)))' - cumsum(full(diag(W_sorted)))');            

    % it may happen that (due to numerical imprecision) the tmpcuts
    % are a small factor of epsilon below zero.
    tmpcuts_threshold(tmpcuts_threshold<0) = 0;
    tmpcuts_threshold2(tmpcuts_threshold2<0) = 0;

    % divide by size/volume
    if(normalized)
        % those are the cuts obtained by merging A with C
        cutparts1_threshold = (tmpcuts_threshold(1:end-1)+cut_rest)./(volumes_threshold(1:end-1)+size_rest);
        cutparts1_threshold(isnan(cutparts1_threshold)) = 0;
        cutparts2_threshold = tmpcuts_threshold2(1:end-1)./(volumes_threshold(end)-volumes_threshold(1:end-1));
        cutparts2_threshold(isnan(cutparts2_threshold)) = 0;

        if (~connected) % only do this if C is not empty
            % those are the cuts obtained by merging B with C
            cutparts1b_threshold = tmpcuts_threshold(1:end-1)./volumes_threshold(1:end-1);
            cutparts1b_threshold(isnan(cutparts1b_threshold)) = 0;
            cutparts2b_threshold = (tmpcuts_threshold2(1:end-1)+cut_rest)./ ...
                                   ((volumes_threshold(end)-volumes_threshold(1:end-1))+size_rest);
            cutparts2b_threshold(isnan(cutparts2b_threshold)) = 0;
        end
    else
        % those are the cuts obtained by merging A with C
        sizes_threshold = cumsum(ones(1,size(vmin_comp,1)));
        sizes_threshold = sizes_threshold(1:end-1);
        cutparts1_threshold = (tmpcuts_threshold(1:end-1)+cut_rest)./(sizes_threshold+size_rest);
        cutparts2_threshold = tmpcuts_threshold2(1:end-1)./(size(vmin_comp,1)-sizes_threshold);

        if (~connected) % only do this if C is not empty
            % those are the cuts obtained by merging B with C 
            cutparts1b_threshold = tmpcuts_threshold(1:end-1)./sizes_threshold;
            cutparts2b_threshold = (tmpcuts_threshold2(1:end-1)+cut_rest)./((size(vmin_comp,1)-sizes_threshold)+size_rest);
        end
    end

    %calculate total cuts
    if(crit==1)
        cuts_threshold = cutparts1_threshold+cutparts2_threshold;
        [cut1,threshold_index] = min(cuts_threshold);
        comp_case = 1;
        if (~connected)
            cuts_threshold_b = cutparts1b_threshold+cutparts2b_threshold;
            [cut1b,threshold_index_b] = min(cuts_threshold_b);
            if (cut1b<cut1); comp_case = 2; end
        end
    else
        cheegers_threshold = max(cutparts1_threshold,cutparts2_threshold);
        [cheeger1,threshold_index] = min(cheegers_threshold);
        comp_case = 1;
        if (~connected)
            cheegers_threshold_b = max(cutparts1_threshold,cutparts2_threshold);
            [cheeger1b,threshold_index_b] = min(cheegers_threshold_b);
            if (cheeger1b<cheeger1); comp_case = 2; end
        end
    end

    if(comp_case==1)
        cutPart1 = cutparts1_threshold(threshold_index);
        cutPart2 = cutparts2_threshold(threshold_index);
        allClustersInClusterM_comp = ones(length(vmin_comp), 1);
        allClustersInClusterM_comp(index(1:threshold_index)) = 0;
        allClustersInClusterM = zeros(size_m,1);
        allClustersInClusterM(index_comp) = allClustersInClusterM_comp;
    else
        cutPart1 = cutparts1b_threshold(threshold_index_b);
        cutPart2 = cutparts2b_threshold(threshold_index_b);
        allClustersInClusterM_comp = ones(length(vmin_comp), 1);
        allClustersInClusterM_comp(index(1:threshold_index_b)) = 0;
        allClustersInClusterM = ones(size_m,1);
        allClustersInClusterM(index_comp) = allClustersInClusterM_comp;
    end
end


% Tries to separate the connected components into two clusters which have
% roughly the same cardinality/volume
function comp2 = balanceConnectedComponents(comp, sizes, W, normalized)

    % for normalized variant, compute volume for every connected component
    if(normalized)
        deg = sum(W);
        volumes = zeros(length(sizes),1);
        for l=1:length(sizes)
            volumes(l) = sum(deg(comp==l));
        end
        sizes = volumes;
    end
            
    % fill up clusters, trying to balance the size
    [sizes_sort, ind] = sort(sizes, 'descend');
    size_a = 0;
    size_b = 0;
    ind_a = [];
    for l=1:length(sizes_sort)
        if(size_a<=size_b) 
            size_a = size_a + sizes_sort(l);
            ind_a = [ind_a ind(l)];
        else
            size_b = size_b + sizes_sort(l);
        end
    end
    comp2 = double(ismember(comp, ind_a));
end

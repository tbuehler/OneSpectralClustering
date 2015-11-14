function [clusters,cuts,cheegers] = computeMultiPartitioning(W,normalized,k,init2nd,numTrials,criterion_threshold,criterion_multicluster,verbosity)
% Computes a multipartitioning of the data given by a similarity matrix W 
% by recursively computing bipartitions using eigenvectors of the 1-Laplacian.
%
% Usage:	[clusters,cuts,cheegers] 
%           = computeMultiPartitioning(W,normalized,k,init2nd,numTrials,criterion_threshold,criterion_multicluster,verbosity)
%
% Input:
%   W: Sparse weight matrix. Has to be symmetric.
%   normalized: true for Ncut/NCC, false for Rcut/RCC
%   k: number of clusters
%   init2nd: true if you want to perform one initialization with the thresholded
%   second eigenvector of the standard graph Laplacian.
%   numTrials: number of additional runs with different random 
%   initializations at each level
%   criterion_threshold: 1: Rcut/Ncut, 2: RCC/NCC
%   criterion_multicluster: 1: Rcut/Ncut, 2: RCC/NCC
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
% (C)2010-11 Thomas Buehler and Matthias Hein
% Machine Learning Group, Saarland University, Germany
% http://www.ml.uni-saarland.de
 
    num=size(W,1);
    
    assert(k>=2,'Wrong usage. Number of clusters has to be at least 2.');
    assert(k<=num, 'Wrong usage. Number of clusters is larger than size of the graph.');
    assert(isnumeric(W) && issparse(W),'Wrong usage. W should be sparse and numeric.');
    assert(sum(sum(W~=W'))==0,'Wrong usage. W should be symmetric.');
    assert(isempty(find(diag(W)~=0,1)),'Wrong usage. Graph contains self loops. W has to have zero diagonal.');
    assert((criterion_threshold==1 || criterion_threshold==2) && (criterion_multicluster==1 || criterion_multicluster==2), 'Wrong usage. Unknown clustering criterion. Available clustering criteria are 1: Ncut/Rcut, 2: NCC/RCC.');
    assert(init2nd || numTrials>0, 'Wrong usage. If second eigenvector initialization is turned off, numTrials has to be positive.');
    
  	threshold_type = -1; 

    clusters=zeros(num,k-1);
    cuts=zeros(1,k-1);
    cheegers=zeros(1,k-1);
    cutParts=zeros(1,k);
    
    deg=full(sum(W,2)); %% will be needed in createSubClusters2 (also in unnormalized case)
    if normalized
        deg2=deg;%full(sum(W));
    else
        deg2=ones(num,1);
    end
        
    cut=inf;
    cheeger=inf;
    
    % Check if graph is connected
    [comp,connected,sizes]=connectedComponents(W);
    if(~connected)
        if(verbosity>=1) disp('WARNING! GRAPH IS NOT CONNECTED!');end
        if(verbosity>=2) disp('Optimal Cut achieved by separating connected components.');end
        allClusters = balanceConnectedComponents(comp,sizes,W,normalized);
        [cut,cheeger,cutPart1,cutPart2]=deal(0);
    else
        if(verbosity>=2) disp('Computing partitioning.'); end
        if(init2nd)
            % Computing (thresholded) second eigenvector of graph Laplacian.
            if(verbosity>=2) disp('...Computing second eigenvector of standard graph Laplacian.'); end
            [start,flag]=computeStandardEigenvector(W,normalized,deg,verbosity);
            if(flag) % if eigenvector computation succeeded
                start = createClustersGeneral(start,W,normalized,threshold_type,criterion_threshold,deg2,true);
                if (sum(start)>sum(start==0)) start=1-start; end
                start=start/sum(start);
    

                % Computing partitioning initialized with second eigenvector.
                if(verbosity>=2) disp(['...Computing nonlinear eigenvector of graph 1-Laplacian. ',...
                    'Initialization with second eigenvector of standard graph Laplacian.']);  end
                vmin=computeEigenvectorGeneral(W,start,normalized,verbosity>=3);
                [allClusters, cut,cheeger,cutPart1,cutPart2] =  createClustersGeneral(vmin,W,normalized,threshold_type,criterion_threshold,deg2,true);

                % Display current objective
                if(verbosity>=2) displayCurrentObjective(cut,cheeger,normalized); end
            else
                cut=inf;
                cheeger=inf;
            end
        end

        % Computing partitioning with random initializations
        for l=1:numTrials
            if(verbosity>=2) fprintf('...Computing nonlinear eigenvector of graph 1-Laplacian. Random initialization %d of %d.\n',l,numTrials); end
            start=randn(num,1);
            vmin=computeEigenvectorGeneral(W,start,normalized,verbosity>=3);
            [allClusters_temp, cut_temp,cheeger_temp,cutPart1_temp,cutPart2_temp] =  createClustersGeneral(vmin,W,normalized,threshold_type,criterion_threshold,deg2,true);

            % Display current objective
            if(verbosity>=2) displayCurrentObjective(cut_temp,cheeger_temp,normalized); end

            % Check if we're better
            if ((criterion_threshold==1 && cut_temp<cut) || (criterion_threshold==2 && cheeger_temp<cheeger))
                [allClusters, cut,cheeger,cutPart1,cutPart2] = deal(allClusters_temp, cut_temp,cheeger_temp,cutPart1_temp,cutPart2_temp);
            end
        end
    end
    
    
    if (cut==inf)
        error('OneSpect:cutinf','Clustering initialized with second eigenvector of standard graph Laplacian failed at first level');
    end
    
    allClusters=allClusters+1;
    clusters(:,1)=allClusters;
   
    cuts(:,1)=cut;
    cheegers(:,1)=cheeger;

    cutParts(1)=cutPart1;
    cutParts(2)=cutPart2;

    subCutParts=zeros(k,2);
    subClusters=cell(1,k);

    if(verbosity>=1)
        fprintf('Finished Clustering into 2 parts.\n');
        displayCurrentObjective(cut,cheeger,normalized);
        fprintf('\n');
    end
    
    %Perform the 2nd to (k-1)th partitioning step
    for l=3:k
        bestCut=inf;
        bestCheeger=inf;
        % in each step consider each of the current l-1 clusters
        for m=1:l-1

            index_m=find(allClusters==m);
            
            % if we have already solved this subproblem	
            if (~isempty(subClusters{m}))
				allClustersInCluster_m = subClusters{m};
                cutPart1 = subCutParts(m,1);
                cutPart2 = subCutParts(m,2);
            % if the current cluster has size 1 it cannot be further divided
            elseif(length(index_m)==1)
                allClustersInCluster_m=[];
                cutPart1=inf;
                cutPart2=inf;
                subClusters{m}=allClustersInCluster_m;
                subCutParts(m,1:2)=[cutPart1 cutPart2];
            elseif(length(index_m)==2)
                allClustersInCluster_m=[0;1];
                cutPart1=deg(index_m(1))-W(index_m(1),index_m(1));
                cutPart2=deg(index_m(2))-W(index_m(2),index_m(2));
                if normalized
                    if cutPart1>0
                        cutPart1=cutPart1/deg(index_m(1));
                    end
                    if cutPart2>0
                        cutPart2=cutPart2/deg(index_m(2));
                    end
                end                    
                subClusters{m}=allClustersInCluster_m;
                subCutParts(m,1:2)=[cutPart1 cutPart2];    
            % otherwise we have to compute the partition
            else    
                if(verbosity>=2) fprintf('Computing partitioning of subgraph %d.\n',m);end
                
                % extract subgraph and its connected components
                Wm=W(index_m,index_m);
                size_m=size(Wm,1);
                [comp,connected,sizes]=connectedComponents(Wm);
                
                if(verbosity>=2 && ~connected) disp('...Subgraph is not connected.'); end
                cutPart1=inf;
                cutPart2=inf;
                
                % if subgraph is not connected
                if (~connected) 
                                                  
                    %check partition which has connected component as one 
                    %cluster and rest as second cluster
                    for m1=1:length(sizes)   
                        %if(true)
                        if(cutPart1+cutPart2>0)
                            if(verbosity>=2) fprintf('...Checking partition found by isolating connected component %d of %d.\n',m1,length(sizes)); end

                            allClustersInCluster_m_temp = double(comp==m1);

                            cluster_m2=zeros(size(allClusters,1),1);
                            cluster_m2(index_m)=allClustersInCluster_m_temp;
                            cutPart2_temp = computeCutValue(cluster_m2,W,normalized,deg2); 

                            cluster_m1=zeros(size(allClusters,1),1);
                            cluster_m1(index_m)=(allClustersInCluster_m_temp==0);
                            cutPart1_temp = computeCutValue(cluster_m1,W,normalized,deg2); 

                            % Display current objective
                            if(verbosity>=2) 
                                [cut_temp,cheeger_temp]=computeCutCheeger(cutParts,cutPart1_temp,cutPart2_temp,m,l);
                                displayCurrentObjective(cut_temp,cheeger_temp,normalized); 
                            end

                            %Check if we're better						
                            if (criterion_threshold==1 && cutPart1_temp+cutPart2_temp<cutPart1+cutPart2 || criterion_threshold==2 && max(cutPart1_temp,cutPart2_temp)<max(cutPart1,cutPart2))
                                [cutPart1,cutPart2,allClustersInCluster_m]=deal(cutPart1_temp,cutPart2_temp,allClustersInCluster_m_temp);
                                assert(length(allClustersInCluster_m)==length(index_m));
                            end
                            assert(logical(exist('allClustersInCluster_m','var')));
                        end 
                    end
                end
                %if(true)
                if(cutPart1+cutPart2>0)
                    for m1=1:length(sizes)
                        index_comp=find(comp==m1);
                        % if the size of the current connected component is larger than 1, try to partition it
                        if (length(index_comp)>1)
                            Wm_comp=sparse(Wm(index_comp,index_comp));
                            Wm_comp2=Wm_comp;
                            if(2*max(sum(Wm_comp.^2))<eps)
                                Wm_comp2=Wm_comp/max(max(Wm_comp));
                            end
                            if(~connected && verbosity>=2) fprintf('...Computing partitioning of connected component %d of %d of subgraph %d.\n',m1,length(sizes), m); end
                            if (~connected)
                                index_rest=find(comp~=m1); % all other components in the current cluster
                                cut_rest=sum(sum(W(index_m(index_rest),setdiff(1:num,index_m(index_rest)))));
                                size_rest=sum(deg2(index_m(index_rest)));
                            else
                                cut_rest=0;
                                size_rest=0;
                            end
                            % Computing partitioning initialized with second eigenvector.
                            if(init2nd)
                                % Computing (thresholded) second eigenvector of graph Laplacian.
                                if(verbosity>=2) disp('...Computing second eigenvector of standard graph Laplacian.'); end
                                [start_comp, flag]=computeStandardEigenvector(Wm_comp,normalized,deg(index_m(index_comp)),verbosity);
                                if(flag)
                                    start_m =  createSubClusters2(start_comp,Wm_comp,normalized,deg,criterion_threshold,index_comp,index_m,cut_rest,size_rest,size_m);
                                    start_comp=start_m(index_comp);
                                    assert(size(start_comp,1)==size(Wm_comp,1));
                                    if (sum(start_comp)>sum(start_comp==0)) start_comp=1-start_comp; end
                                    start_comp=start_comp/sum(start_comp);

                                    % Computing partitioning initialized with second eigenvector.
                                    if(verbosity>=2) disp(['...Computing nonlinear eigenvector of graph 1-Laplacian. ', ...
                                        'Initialization with second eigenvector of standard graph Laplacian.']); end
                                    vmin_comp=computeEigenvectorGeneral(Wm_comp2,start_comp,normalized,verbosity>=3);
                                    [allClustersInCluster_m_temp, cutPart1_temp, cutPart2_temp] =  createSubClusters2(vmin_comp,Wm_comp,normalized,deg,criterion_threshold,index_comp,index_m,cut_rest,size_rest,size_m);

                                    % Display current objective
                                    if(verbosity>=2) 
                                        [cut_temp,cheeger_temp]=computeCutCheeger(cutParts,cutPart1_temp,cutPart2_temp,m,l);
                                        displayCurrentObjective(cut_temp,cheeger_temp,normalized); 
                                    end
                                else
                                    cutPart1_temp=inf;
                                    cutPart2_temp=inf;
                                end

                                % Check if we're better
                                if (criterion_threshold==1 && cutPart1_temp+cutPart2_temp<cutPart1+cutPart2 || criterion_threshold==2 && max(cutPart1_temp,cutPart2_temp)<max(cutPart1,cutPart2))
                                    [cutPart1,cutPart2,allClustersInCluster_m]=deal(cutPart1_temp,cutPart2_temp,allClustersInCluster_m_temp);
                                    assert(length(allClustersInCluster_m)==length(index_m));
                                end

                            end
                            %Computing partitioning with random initializations
                            for k=1:numTrials

                                %Computing partitioning with random initializations
                                if(verbosity>=2) fprintf('...Computing nonlinear eigenvector of graph 1-Laplacian. Random initialization %d of %d. \n',k,numTrials); end
                                start_comp=randn(size(Wm_comp,1),1);
                                vmin_comp=computeEigenvectorGeneral(Wm_comp2,start_comp,normalized,verbosity>=3);
                                [allClustersInCluster_m_temp, cutPart1_temp, cutPart2_temp] =  createSubClusters2(vmin_comp,Wm_comp,normalized,deg,criterion_threshold,index_comp,index_m,cut_rest,size_rest,size_m);

                                % Display current objective
                                if(verbosity>=2) 
                                    [cut_temp,cheeger_temp]=computeCutCheeger(cutParts,cutPart1_temp,cutPart2_temp,m,l);
                                    displayCurrentObjective(cut_temp,cheeger_temp,normalized); 
                                end

                                %Check if we're better						
                                if (criterion_threshold==1 && cutPart1_temp+cutPart2_temp<cutPart1+cutPart2 || criterion_threshold==2 && max(cutPart1_temp,cutPart2_temp)<max(cutPart1,cutPart2))
                                    [cutPart1,cutPart2,allClustersInCluster_m]=deal(cutPart1_temp,cutPart2_temp,allClustersInCluster_m_temp);
                                    assert(length(allClustersInCluster_m)==length(index_m));
                                end
                            end

                        end

                    end
                end
                % store current best partition
			    subClusters{m}=allClustersInCluster_m;
				subCutParts(m,1:2)=[cutPart1 cutPart2]; 
   
            end
            
            % print out best cut possible by partitioning of current subgraph
            [cut,cheeger]=computeCutCheeger(cutParts,cutPart1,cutPart2,m,l);
            if(verbosity>=2)
                fprintf('Best result achievable by partitioning of subgraph %d:\n',m);
                displayCurrentObjective(cut,cheeger,normalized);
                fprintf('\n');
            end
            
			% check if partitoning of the current subgraph gives better cut
            if (criterion_multicluster==1 && cut<bestCut) || (criterion_multicluster==2 && cheeger<bestCheeger)
                [bestCut,bestCheeger,bestCutPart1,bestCutPart2,best_m]= deal(cut,cheeger,cutPart1,cutPart2,m);
                clusters_new=allClusters;
                clusters_new(index_m)=(l-m)*allClustersInCluster_m+clusters_new(index_m);

                assert(bestCut>=0 && bestCheeger>=0);
            end
            
            % if we have already found a partition with cut 0, we don't
            % need to consider the other subgraphs
            if bestCut==0
                break;
            end
        end
        
        if(bestCut==inf)
             error('OneSpect:cutinf','Clustering initialized with second eigenvector of standard graph Laplacian failed at level %d.',l-1);
        end
        
        % Update
        allClusters=clusters_new;
        cuts(1,l-1)=bestCut;
        cheegers(1,l-1)=bestCheeger;
        clusters(:,l-1)=allClusters;
        
        cutParts(best_m)=bestCutPart1;
        cutParts(l)=bestCutPart2;
        
        % Check that we have the right number of clusters
        assert(length(unique(allClusters))==l);
        
        % Reset subcutparts and subclusters;
        subCutParts(best_m,:)=0;
        subClusters{best_m}= [];
        subCutParts(l,:)=0;
        subClusters{l}= [];
          
        % Print out current objective
        if(verbosity>=1)
            fprintf('Decided to partition subgraph %d. Finished Clustering into %d parts.\n',best_m,l);
            displayCurrentObjective(bestCut,bestCheeger,normalized);
            fprintf('\n');
        end

    end

    
end

% Computes Rcut/Ncut and Cheeger Cut values
function [cut,cheeger]=computeCutCheeger(cutParts,cutPart1,cutPart2,m,l)

    cut= sum(cutParts)-cutParts(m)+cutPart1+cutPart2;
    cheeger=max([cutParts((1:l-1)~=m) cutPart1 cutPart2]);
                                
end

% Computes the thresholded 2nd eigenvector of the standard graph Laplacian
function [start,flag]=computeStandardEigenvector(W,normalized,deg,verbosity)

    % deg has to be the degree vector (also in unnormalised case)
    num=size(W,1);
    D=spdiags(deg,0,num,num);
    opts.disp=0;
    opts.tol = 1E-10;
    opts.maxit=20;
    opts.issym = 1;
    
    flag=true;
    try
        if (normalized)
            [eigvec,eigval]= eigs(D-W, D,2,'SA',opts);
        else
            [eigvec,eigval]= eigs(D-W, 2,'SA',opts);
        end
        start=eigvec(:,2);
    catch exc
        flag=false;
        if(verbosity>=1) 
            disp('WARNING! COMPUTATION OF SECOND EIGENVECTOR OF THE STANDARD GRAPH LAPLACIAN NOT SUCESSFUL!');
            disp(exc.message);
        end
        start=zeros(num,1);
    end
    
    
end

% Displays the current objective value
function displayCurrentObjective(cut_temp,cheeger_temp,normalized)
    
    if (normalized)
        fprintf('...Normalized Cut: %.8g   Normalized Cheeger Cut: %.8g\n',cut_temp,cheeger_temp); 
    else
        fprintf('...Ratio Cut: %.8g   Ratio Cheeger Cut: %.8g\n',cut_temp,cheeger_temp); 
    end
    
end


% Creates two clusters by thresholding the vector vmin_comp obtained on a
% connected component of a subgraph. Given the two clusters on the 
% connected component, there are two ways of constructing the final clusters 
% on the subgraph, as we can keep each of the clusters on the connected 
% component as cluster and merge the other one with the remaining connected 
% components. The method takes the one yielding the lower Cut/Cheeger.
function [allClustersInClusterM, cutPart1,cutPart2] =  createSubClusters2(vmin_comp,W_comp,normalized,deg,criterion_threshold,index_comp,index_m,cut_rest,size_rest,size_m)
      
        % input parameter deg has to be the degree vector (also in unnormalised case)
        %deg=full(sum(W));
        %Make deg a row vector;
        if (size(deg,1)>1) 
            deg=deg';
        end
        
        [vminM_sorted, index]=sort(vmin_comp);
        [vminU,indexU]=unique(vminM_sorted);
        
        
        W_sorted=W_comp(index,index);

        % calculate cuts
        deg_comp=deg(index_m(index_comp));
        volumes_threshold=cumsum(deg_comp(index));
        triup=triu(W_sorted);
        tempcuts_threshold=volumes_threshold - 2*cumsum(full(sum(triup)));
        tempcuts_threshold2=(volumes_threshold(end)-volumes_threshold) - (sum(sum(W_sorted))-2*cumsum(full(sum(triup,2)))');            

        % it may happen that (due to numerical imprecision) the tempcuts
        % are a small factor of epsilon below zero.
        tempcuts_threshold(tempcuts_threshold<0)=0;
        tempcuts_threshold2(tempcuts_threshold2<0)=0;
        
        tempcuts_threshold=tempcuts_threshold(indexU);
        tempcuts_threshold2=tempcuts_threshold2(indexU);
        volumes_threshold=volumes_threshold(indexU);
        
        
        % divide by size/volume
        if(normalized)
            cutparts1_threshold=(tempcuts_threshold(1:end-1)+cut_rest)./(volumes_threshold(1:end-1)+size_rest);
            cutparts1_threshold(isnan(cutparts1_threshold))=0;
            cutparts2_threshold=tempcuts_threshold2(1:end-1)./(volumes_threshold(end)-volumes_threshold(1:end-1));
            cutparts2_threshold(isnan(cutparts2_threshold))=0;
            
            cutparts1b_threshold=tempcuts_threshold(1:end-1)./volumes_threshold(1:end-1);
            cutparts1b_threshold(isnan(cutparts1b_threshold))=0;
            cutparts2b_threshold=(tempcuts_threshold2(1:end-1)+cut_rest)./((volumes_threshold(end)-volumes_threshold(1:end-1))+size_rest);
            cutparts2b_threshold(isnan(cutparts2b_threshold))=0;
        else
            sizes_threshold=cumsum(ones(1,size(vmin_comp,1)-1));
            sizes_threshold=sizes_threshold(indexU(1:end-1));
            cutparts1_threshold=(tempcuts_threshold(1:end-1)+cut_rest)./(sizes_threshold+size_rest);
            cutparts2_threshold=tempcuts_threshold2(1:end-1)./(size(vmin_comp,1)-sizes_threshold);
            
            cutparts1b_threshold=tempcuts_threshold(1:end-1)./sizes_threshold;
            cutparts2b_threshold=(tempcuts_threshold2(1:end-1)+cut_rest)./((size(vmin_comp,1)-sizes_threshold)+size_rest);
        end

        
        
        %calculate total cuts
        if(criterion_threshold==1)
            cuts_threshold=cutparts1_threshold+cutparts2_threshold;
            [cut1,threshold_index]=min(cuts_threshold);
            
            cuts_threshold_b=cutparts1b_threshold+cutparts2b_threshold;
            [cut1b,threshold_index_b]=min(cuts_threshold_b);
            
            comp_case=1;
            if (cut1b<cut1) 
                comp_case=2;
            end
        else
            cheegers_threshold=max(cutparts1_threshold,cutparts2_threshold);
            [cheeger1,threshold_index]=min(cheegers_threshold);
            
            cheegers_threshold_b=max(cutparts1_threshold,cutparts2_threshold);
            [cheeger1b,threshold_index_b]=min(cheegers_threshold_b);
            
            comp_case=1;
            if (cheeger1b<cheeger1) 
                comp_case=2;
            end
        end

        if(comp_case==1)
            cutPart1=cutparts1_threshold(threshold_index);
            cutPart2=cutparts2_threshold(threshold_index);

            allClustersInClusterM_comp= (vmin_comp>vminU(threshold_index));
        
            allClustersInClusterM= zeros(size_m,1);
            allClustersInClusterM(index_comp)=allClustersInClusterM_comp;
        else
            cutPart1=cutparts1b_threshold(threshold_index_b);
            cutPart2=cutparts2b_threshold(threshold_index_b);

            allClustersInClusterM_comp= (vmin_comp>vminU(threshold_index_b));
        
            allClustersInClusterM= ones(size_m,1);
            allClustersInClusterM(index_comp)=allClustersInClusterM_comp;
        end
        
        

end


% Tries to separate the connected components into two clusters which have
% roughly the same cardinality/volume
function comp2 = balanceConnectedComponents(comp,sizes,W,normalized)

    % for normalized variant, compute the volume for every connected
    % component
    if(normalized)
        deg=sum(W);
        volumes=zeros(length(sizes),1);
        for l=1:length(sizes)
            volumes(l)=sum(deg(comp==l));
        end
        sizes=volumes;
    end
            
    % fill up clusters, trying to balance the size
    [sizes_sort,ind]=sort(sizes,'descend');
    size_a=0;
    size_b=0;
    ind_a=[];
    for l=1:length(sizes_sort)
        if(size_a<=size_b) 
            size_a=size_a+sizes_sort(l);
            ind_a=[ind_a ind(l)];
        else size_b=size_b+sizes_sort(l);
        end
    end
    comp2=double(ismember(comp,ind_a));
end
      
        

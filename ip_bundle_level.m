function [x_best,ub,cur_delta,it]=ip_bundle_level(start,params,eps1, solver_lp,solver_qp,verbose,obj_subg)
% Solves the inner problem in the IPM using bundle level method
%
% (C)2010 Thomas Buehler 
% Machine Learning Group, Saarland University, Germany
% http://www.ml.uni-saarland.de

debug=false;

start=start/norm(start,inf);
dim=length(start);


%% Initialization
x=start;
x_lb=start;
it=0;
maxit=200;
lambda= 1/(2+sqrt(2));
max_bundle_size=50;
A=[];
ub=inf;
lb=-inf;
lev =lambda* ub + (1-lambda)*lb;
converged=false;

alpha_start=0;
alpha=[];

obj_x_lb=inf;
obj_lp=inf;
obj_qp=inf;


%% main loop
while(~converged && it<maxit)
    
    
    % compute objective and element of subdifferential at x_lb
    [obj_x, subg_x] = obj_subg(x,params);
    
    % upperbound
    if obj_x<ub
        ub=obj_x;
        x_best=x;
    end
    if obj_x_lb<ub
        ub=obj_x_lb;
        x_best=x_lb;
    end
    
    eps2=1E-11;
    
    
    % cur_Delta is computed relative to gap at first iteration
    if (it==0)
        cur_delta=inf;
        rel_gap=inf;
    elseif (it==1)
        cur_delta=ub-lb;
        max_cur_delta=cur_delta;
        rel_gap=1;
    else
        cur_delta=ub-lb;
        rel_gap=cur_delta/max_cur_delta;
    end
    
    
    % check if converged
    if(rel_gap<eps1 && ub <eps2)
        converged=true;
    end
    
    
    
    % display results
    if(verbose)
        if (mod(it,20)==0 || it==1 || converged)
            disp([' it= ', num2str(it) , ...
                ' k = ',num2str(size(A,1)), ...
                ' cur_delta=',num2str(cur_delta), ...
                ' rel_gap= ', num2str(rel_gap), ...
                ' lb=',num2str(lb), ...
                ' ub=',num2str(ub), ...
                ' lev=', num2str(lev), ...
                ' feasible=', num2str(norm(x,inf)<=1), ...
                ' obj_x=', num2str(obj_x),  ...
                ' obj_x_lb=', num2str(obj_x_lb),  ...
                ' obj_lp=',num2str(obj_lp), ...
                ' obj_qp=',num2str(obj_qp)  ]);
        end
    end
    
    if (~ converged)
        
        % throw out all the constraint which were not active
        %  if (length(alpha)>0)
        %      ind=find(alpha ~=0);
        %      alpha=alpha(ind);
        %      A=A(ind,:);
        %  end
        
        % restrict to at most kmax bundle elements
        ix=max(length(alpha)-max_bundle_size+2,1);
        alpha=alpha(ix:end);
        A=A(ix:end,:);
        
        
        
        if (~isempty(alpha))
            alpha_start=[alpha; alpha(end)];
        else
            alpha_start=0;
        end
        
        A=[A; subg_x'];
        %tic;
        
        %% first subproblem (LP)
        % compute lower bound lb (by minimizing current piecewise linear approximation)
        if (size(A,1)==1)
            x_lb= -sign(A)';
            lb_new=-sum(abs(A));
            
        else
            
            
            switch(solver_lp)
                
                case 'cvx'
                    t1=tic;
                    cvx_begin quiet
                    variables x_lb(dim);
                    minimize(max(A*x_lb)); % we already use here that the objective is one-homogeneous
                    subject to
                    x_lb<=1;
                    x_lb >= -1;
                    cvx_end
                    obj_lp=cvx_optval;
                    
                    lb=obj_lp;
                    
                    toc1=toc(t1);
                    
                    
                case 'linprog'
                    
                    opts.Display='off';
                    if (debug)
                        opts.Display='iter';
                    end
                    %  opts.Algorithm='simplex';
                    opts.Algorithm='interior-point';
                    %  tic1=tic;
                    [f,obj_lp,exitflag,output,lambda1]=linprog([zeros(dim,1); 1],[A -ones(size(A,1),1)],zeros(size(A,1),1),[],[],[-ones(dim,1);-inf],[ones(dim,1);inf],[],opts);
                    
                    x_lb=f(1:end-1);
                    
                    lb_new=obj_lp;
                    %toc1=toc(tic1);
                    
                    
                case 'mosek'
                    
                    % solve using mosek
                    tic3=tic;
                    
                    c= [zeros(dim,1); 1];
                    Ainp=[A -ones(size(A,1),1)];
                    blc=-inf(size(A,1),1);
                    buc=zeros(size(A,1),1);
                    blx=[-ones(dim,1);-inf];
                    bux=[ones(dim,1);inf];
                    
                    [res] = msklpopt(c,Ainp,blc,buc,blx,bux,[],'minimize echo(0)');
                    sol = res.sol;
                    
                    obj_lp=sol.bas.xx(end);
                    x_lb=sol.bas.xx(1:end-1);
                    
                    toc3=toc(tic3);
                    
                    
                    %disp(['Obj Simpl =',num2str(obj_lp),' Obj bar = ', num2str(obj_lp2),' Ob mosek = ',num2str(obj_lp3)]);
                    
                    %disp(['Tim1 Simp =',num2str(toc1),'Tim2 int =', num2str(toc2), ' Tim3 Mosek =',  num2str(toc3)]);
                    % if (mod(it,10)==0)
                    % disp(['Obj LP=',num2str(obj_lp),' Time = ', num2str(toc3)]);
                    
                    % end
                    lb_new=obj_lp;
                    
                case 'barrier'
                    %         %construct feasible point for lp
                    %         x_lp_feas=(1-2*eps1)*rand(dim,1)+eps1;
                    %         c_lp_feas=max(A*x_lp_feas)+ eps1;
                    %
                    %         assert(sum(x_lp_feas<=1)==length(x_lp_feas));
                    %         assert(sum(x_lp_feas>=0)==length(x_lp_feas));
                    %         assert(sum(A*x_lp_feas<=c_lp_feas)==size(A,1));
                    %
                    %         [x_lp,obj_orig,obj_new,iter_all]=lp_ABL_Linf_barrier(A,eps1,x_lp_feas,c_lp_feas);
                    %
                    %         % check if x_lp feasible
                    %         assert(sum(x_lp<=1)==length(x_lp));
                    %         assert(sum(x_lp>=0)==length(x_lp));
                    %         assert(sum(A*x_lp<=c_lp_feas)==size(A,1));
                    %
                    %         x_ub=x_lp;
                    %         lb=obj_orig;
                    
                otherwise
                    assert(0);
            end
            %  disp(['lower bound =', num2str(lb)]);
        end
        
        obj_x_lb= obj_subg(x_lb,params);
        
        lb=max(lb,lb_new);
        % compute level
        lev = (1-lambda)*lb + lambda* ub;
        
        if lb>ub % this may happen due to numerical imprecision
            converged=true; 
            continue
        end
      
        assert(lev>=lb_new);
        
        %% second subproblem (QP)
        % compute new x
        tic_qp=tic;
        switch(solver_qp)
            case 'cvx'
                %tic1=tic;
                %solution of qp via cvx
                %compute new x
                cvx_begin quiet
                cvx_precision high
                variables y(dim);
                minimize(0.5*sum((y-x).^2));
                subject to
                A*y<=lev;
                abs(y)<=1;
                cvx_end
                x_new=y;
                obj_qp=cvx_optval;
                %toc1=toc(tic1);
            case 'lsqlin'
                tic1=tic;
                opts.MaxIter=10000;
                
                opts.Display='off';
                if (debug)
                    opts.Display='iter';
                end
                warning('off','all');
                [x_new, resnorm]= lsqlin(eye(dim),x,A,lev*ones(size(A,1),1),[],[],-ones(dim,1),ones(dim,1),[],opts);
                obj_qp=0.5*resnorm;
                warning('on','all');
                toc1=toc(tic1);
                
                % eliminate in each step the ones where alpha=0
                % how does this affect LP problem?
                
            case 'fista_matlab'
                At=A';
                L=norm(A,'fro')^2;
                tic2=tic;
                [v_proj,primal_obj,dual_obj,alpha,num_it]=qp_bundle_level_Linf_fista(x,A,At,lev,eps1,L,alpha_start);
                
                toc2=toc(tic2);
                %alpha_start=[alpha; alpha(end)];
                
            case 'fista_mex'
                At=A';
                L=norm(A,'fro')^2;
                tic3=tic;alpha_start2=alpha_start;
                [v_proj2,primal_obj2,dual_obj2,alpha, it2,x_new,obj_qp]=mex_qp_bundle_level_Linf_fista(x,A,At,lev,eps1,L,alpha_start);
                toc3=toc(tic3);
                
                %  disp(['Obj lsl=', num2str(0.5*resnorm),' Fista=', num2str(primal_obj),' Fista2=', num2str(primal_obj2), ' Time lsql=',num2str(toc1), ' Fista=', num2str(toc2), ' Fista2=', num2str(toc3)]);
                %alpha_start=[alpha; alpha(end)];
                
                
                %  disp([' Obj CVX = ', num2str(cvx_optval),' Obj lsqlin=', num2str(0.5*resnorm), ' Time CVX = ',num2str(toc1), ' Time lsqlin =', num2str(toc2)]);
                
                %     case 'barrier'
                %
                %         solution of qp via barrier method
                %         construct feasible point for qp
                %         eps2=min(0.5,0.5* (lev-lb)/(norm(subg_x_lb,1)^2*length(x)));
                %         assert(eps2>0);
                %         x_qp_feas=(1-2*eps2)* x_lp+ eps2;
                %         assert(sum(x_qp_feas >= 0)==length(x_qp_feas));
                %         assert(sum(x_qp_feas<=1)==length(x_qp_feas));
                %         assert(sum(A*x_qp_feas<=lev)==size(A,1));
                %
                %         [x,obj_orig,obj_new,iter_all]=qp_ABL_Linf_barrier(A,x_old,lev,eps1,x_qp_feas);
                %
                %         check if x feasible
                %         assert(sum(x<=1)==length(x));
                %         assert(sum(x>=0)==length(x));
                %         assert(sum(A*x2<=lev)==size(A,1));
                %
                %         disp(['obj cvx=', num2str(cvx_optval),' obj barrier=', num2str(obj_orig)]);
                %         assert(norm(lb-obj_orig)<1E-3);
                %
                %     case 'dykstra'
                %
                %         solution of qp via dykstra algorithm
                %         [x,obj_orig,obj_new,iter_all]=qp_ABL_Linf_dykstra(A,x_old,lev,eps1,x_qp_feas);
                %
                %
                %     case 'fista'
                %
                %         solution of qp via fista
                %         L=sqrt(norm(A*A','fro')^2);
                %
                %         [x,obj_orig,obj_new,iter_all]=qp_ABL_Linf_V2(x_old,A,lev,A',eps1,L,randn(size(A,1),1)); %qp_ABL_Linf_V2(A,x_old,lev,eps1,x_qp_feas);
                %
                %         check if x feasible
                %         assert(sum(x<=1)==length(x));
                %         assert(sum(x>=0)==length(x));
                %         assert(sum(A*x<=lev)==size(A,1));
            otherwise
                assert(0);
                
        end
        toc_qp=toc(tic_qp);
        
        %if (mod(it,10)==0)
        %disp(['Time to solve qp: ',num2str(toc_qp)]);
        %end
        
        
        x=x_new;
        it=it+1;
      
        
    end
end

end

function [f,obj,it] = ip_cutting_plane(start,params,eps1,solver_lp,obj_subg)
% Solves the inner problem in the IPM using the cutting-plane method
%
% (C)2010 Thomas Buehler 
% Machine Learning Group, Saarland University, Germany
% http://www.ml.uni-saarland.de

    %% some constants
    debug = false;
    itmax = 100;
    dim = length(start);

    %% initialization
    lb = -inf;
    ub = inf;
    A_old = [];
    f = start/norm(start,inf);
    it = 0;
    converged = false;

    %% main loop
    while (~converged && it <itmax)
        
        %% compute objective and element of subdifferential at f
        [obj, subg] = obj_subg(f,params);
        
        %% update upper bound on optimal point
        ub = min(ub, obj);
        
        %% update relative gap between ub and lb
        if (it==0)
            rel_gap = inf;
        elseif (it==1)
            max_cur_delta = ub-lb;
            rel_gap = 1;
        else
            rel_gap = (ub-lb)/max_cur_delta;
        end
        
        %% check if converged
        if rel_gap<eps1
            converged = true;
        end
        
        %% display stuff
        if (mod(it,10)==0 || it==1 || converged)
            fprintf('............ it=%i rel_gap=%.5f obj=%.15g lb=%.15g norm(subg^T*f-obj)=%.15g\n', ...
                    it, rel_gap, obj, lb, norm(subg'*f-obj));
        end
        
        %% update model + solve inner problem
        if  (~converged)
            A = [A_old;subg'];

            [f, lb_new] = solve_LP(A, dim, solver_lp, debug);
 
            if lb_new<lb
                fprintf('WARNING: lb_new=%.5g lb=%.5g diff=%.5g\n', lb_new, lb, lb_new-lb);
            end
            
            A_old = A;
            lb = lb_new;
            it = it+1;
        end
    end
end


function [f, lb_new] = solve_LP(A, dim, solver_lp, debug)
% solve subproblem (LP): compute lower bound lb by minimizing current 
% piecewise linear approximation)
    
    if (size(A,1)==1)
        f = -sign(A)';
        lb_new = -sum(abs(A));
    else
        switch solver_lp
            case 'cvx'
                cvx_begin quiet
                variables y(dim);
                
                minimize(max(A*y));
                subject to
                abs(y) <= 1;
                cvx_end
                
                f = y;
                lb_new = cvx_optval;
            case 'linprog'
                opts.Display = 'off';
                if (debug)
                    opts.Display = 'iter';
                end
                
                [f, obj_model, exitflag, output, lambda] = linprog( ...
                    [zeros(dim,1); 1], [A -ones(size(A,1),1)], zeros(size(A,1),1), ...
                    [], [], [-ones(dim,1);-inf], [ones(dim,1);inf], [], opts);
                
                f = f(1:end-1);
                lb_new = max(A*f); % this is better than taking obj_model
            case 'mosek'
                c = [zeros(dim,1); 1];
                Ainp = [A -ones(size(A,1),1)];
                blc = -inf(size(A,1),1);
                buc = zeros(size(A,1),1);
                blx = [-ones(dim,1);-inf];
                bux = [ones(dim,1);inf];
                
                [res] = msklpopt(c,Ainp,blc,buc,blx,bux,[],'minimize echo(0)');
                sol = res.sol;
                
                lb_new = sol.bas.xx(end);
                f = sol.bas.xx(1:end-1);
            otherwise
                error("Invalid solver for LP");
        end
    end
end

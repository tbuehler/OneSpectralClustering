function [x_best, ub, it] = ip_bundle_level(start, params, eps1, solver_lp, solver_qp, verbose, obj_subg)
% Solves the inner problem in the IPM using a bundle level method
%
% (C)2010 Thomas Buehler 
% Machine Learning Group, Saarland University, Germany
% http://www.ml.uni-saarland.de

    %% some constants
    debug = false;
    dim = length(start);
    maxit = 200;40;%200;
    lambda = 1/(2+sqrt(2));
    max_bundle_size = 50;
    eps2 = 1E-11;

    %% initialization
    x = start/norm(start,inf);
    x_lb = x;
    A = [];
    ub = inf;
    lb = -inf;
    lev = lambda*ub + (1-lambda)*lb;
    alpha = [];
    obj_x_lb = inf;
    obj_qp = inf;
    it = 0;
    converged = false;

    %% main loop
    while(~converged && it<maxit)
        
        %% compute objective and an element of the subdifferential at x
        [obj_x, subg_x] = obj_subg(x,params);
        
        %% upper bound on optimal point is given by best objective so far
        if obj_x<ub
            ub = obj_x;
            x_best = x;
        end
        if obj_x_lb<ub
            ub = obj_x_lb;
            x_best = x_lb;
        end
        assert(norm(x,inf)<=1, "x is not feasible");
                       
        %% compute the gap between ub and lb, relative to gap at first iteration
        if (it==0)
            rel_gap = inf;
        elseif (it==1)
            max_gap = ub-lb;
            rel_gap = 1;
        else
            rel_gap = (ub-lb)/max_gap;
        end
                
        %% check if converged
        if(rel_gap<eps1 && ub<eps2)
            converged = true;
        end
        %% stop if the objective is negative
        %if (it>20 && ub<0)
        %    converged=true;
        %end
        
        %% display results
        if(verbose)
           if (mod(it,20)==0 || it==1 || converged)
                fprintf('............ it=%i k=%i rel_gap=%.5f lb=%.5g ub=%.5g lev=%.5g obj_x=%.5g obj_x_lb=%.5g obj_qp=%.5g\n', ...
                it, size(A,1), rel_gap, lb, ub, lev, obj_x, obj_x_lb, obj_qp);
           end
        end
        
        if (~converged)
            % throw out all the constraint which were not active
            %if (length(alpha)==max_bundle_size)
            %      ind = find(alpha ~=0);
            %      alpha = alpha(ind);
            %      A = A(ind,:);
            %end
            
            % restrict to at most kmax bundle elements
            ix = max(length(alpha)-max_bundle_size+2, 1);
            alpha = alpha(ix:end);
            A = A(ix:end, :);
            
            if (isempty(alpha))
                alpha_start = 0;
            else
                alpha_start = [alpha; alpha(end)];
            end
            A = [A; subg_x'];
            
            %% solve first subproblem (LP): compute lower bound lb by minimizing 
            %% current piecewise linear approximation)
            [x_lb, lb_new] = solve_LP(A, dim, solver_lp, debug);
            if (isempty(x_lb)) % inner problem did not converge
                converged = true;
                continue
            end
            obj_x_lb = obj_subg(x_lb,params);
            
            %% update lower bound
            lb = max(lb,lb_new);
            if lb>ub % this may happen due to numerical imprecision
                converged = true; 
                continue
            end

            %% compute level
            lev = (1-lambda)*lb + lambda*ub;
            assert(lev>=lb_new);

            %% solve second subproblem (QP) and compute new x
            [x, obj_qp, alpha] = solve_QP(A, x, dim, lev, solver_qp, alpha_start, eps1, debug);

            it = it+1;
        end
    end
end


function [x_lb, lb_new] = solve_LP(A, dim, solver_lp, debug)
% solve first subproblem (LP): compute lower bound lb by minimizing current 
% piecewise linear approximation)

    if (size(A,1)==1)
        x_lb = -sign(A)';
        lb_new = -sum(abs(A));
    else
        switch(solver_lp)
            case 'cvx'
                cvx_begin quiet
                variables x_lb(dim);
                minimize(max(A*x_lb));
                subject to
                x_lb <= 1;
                x_lb >= -1;
                cvx_end

                lb_new = cvx_optval;
            case 'linprog'
                opts.Display = 'off';
                if (debug)
                    opts.Display = 'iter';
                end
                opts.Algorithm = 'interior-point';
                [f, lb_new, exitflag, ~, ~] = linprog( ...
                    [zeros(dim,1); 1], [A -ones(size(A,1),1)], zeros(size(A,1),1), ...
                    [], [], [-ones(dim,1);-inf], [ones(dim,1);inf], [], opts);
                if (exitflag<=0)
                    opts.Algorithm = 'dual-simplex';
                    [f, lb_new, ~, ~, ~] = linprog( ...
                        [zeros(dim,1); 1], [A -ones(size(A,1),1)], zeros(size(A,1),1), ...
                        [], [], [-ones(dim,1);-inf], [ones(dim,1);inf], [], opts);
                end                
                %fprintf('IP: lb_new=%.5g time=%.3g exit=%i\tDS:  lb_new=%.5g time=%.3g exit=%i   diff=%.5g timediff=%.5g\n',...
                %        lb_new1, el1, exitflag1, lb_new, el2, exitflag2, lb_new1-lb_new, el1-el2);
                x_lb = f(1:end-1);
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
                x_lb = sol.bas.xx(1:end-1);
            otherwise
                error("Invalid solver for LP");
        end
    end
end


function [x_new, obj_qp, alpha] = solve_QP(A, x, dim, lev, solver_qp, alpha_start, eps1, debug)
%% solve second subproblem (QP) and compute new x

    switch(solver_qp)
        case 'cvx'
            cvx_begin quiet
            cvx_precision high
            variables y(dim);
            minimize(0.5*sum((y-x).^2));
            subject to
            A*y <= lev;
            abs(y) <= 1;
            cvx_end

            x_new = y;
            obj_qp = cvx_optval;
        case 'lsqlin'
            opts.MaxIter = 10000;
            opts.Display = 'off';
            if (debug)
                opts.Display = 'iter';
            end
            warning('off','all');
            [x_new, resnorm] = lsqlin(eye(dim), x, A, lev*ones(size(A,1),1), ...
                                      [], [], -ones(dim,1), ones(dim,1), [], opts);
            obj_qp = 0.5*resnorm;
            warning('on','all');
        case 'fista_mex'
            At = A';
            L = norm(A,'fro')^2;
            [alpha, x_new, obj_qp] = mex_qp_bundle_level_Linf_fista(x, A, At, lev, eps1, L, alpha_start);
        otherwise
            error("Invalid solver for QP");        
    end
end

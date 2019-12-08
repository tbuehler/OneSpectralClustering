function [f,obj,cur_delta,it]=ip_cutting_plane(start,params,eps1,solver_lp,obj_subg)
% Solves the inner problem in the IPM using the cutting-plane method
%
% (C)2010 Thomas Buehler 
% Machine Learning Group, Saarland University, Germany
% http://www.ml.uni-saarland.de

debug=false;

it=0;
itmax=100;
converged=false;
lb=-inf;
ub=inf;
A_old=[];
f=start/norm(start,inf);
dim=length(start);

while (~converged && it <itmax)
    
    
    % compute objective and element of subdifferential
    [obj, subg] = obj_subg(f,params);
    
    ub=min(ub,obj);
    
    
    % update delta
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
    if rel_gap<eps1
        converged=true;
    end
    
    % display stuff
    if (mod(it,10)==0 || it==1 || converged)
        disp(['it= ', num2str(it) , ' cur_delta=',num2str(cur_delta), ' rel_gap=',num2str(rel_gap),'   obj=',num2str(obj),' lb=',num2str(lb),' norm(subg^T*f-obj)= ',num2str(norm(subg'*f-obj)) ]);
    end
    
    
    % udpate model + solve inner problem
    if  (~converged)
        
        A=[A_old;subg'];
        
        
        %         cvx_begin quiet
        %             variables y(dim) t;
        %             minimize(t);
        %             subject to
        %                 A*y+b_new<=t;
        %                 norm(y) <=1;
        %         cvx_end
        %          cvx_begin quiet
        %             variables y(dim);
        %             %minimize(max(A*y+b_new));
        %             minimize(max(A*y)); % we already use here that the ovbjective is one-homogeneous
        %             subject to
        %                 norm(y) <=1;
        %                 y >= 0;
        %         cvx_end
        
        %t1=tic;
        % cvx_begin quiet
        %    variables y(dim);
        %    %minimize(max(A*y+b_new));
        %    minimize(max(A*y)); % we already use here that the ovbjective is one-homogeneous
        %    subject to
        %        abs(y) <=1;
        % cvx_end
        %toc1=toc(t1);
        %t2=tic;
        
        
        if (size(A,1)==1)
            f= -sign(A)';
            lb_new=-sum(abs(A));
        else
            
            switch solver_lp
                
                case 'cvx'
                    cvx_begin quiet
                    variables y(dim);
                    
                    minimize(max(A*y)); % we already use here that the ovbjective is one-homogeneous
                    subject to
                    abs(y) <=1;
                    cvx_end
                    
                    f=y;
                    lb_new=cvx_optval;
                    
                case 'linprog'
                    
                    opts.Display='off';
                    if (debug)
                        opts.Display='iter';
                    end
                    
                    [f,obj_model,exitflag,output,lambda]=linprog([zeros(dim,1); 1],[A -ones(size(A,1),1)],zeros(size(A,1),1),[],[],[-ones(dim,1);-inf],[ones(dim,1);inf],[],opts);
                    
                    f=f(1:end-1);
                    
                    lb_new= max(A*f);% this is better than taking obj_model
                    
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
                    f=sol.bas.xx(1:end-1);
                    
                    toc3=toc(tic3);
                    
                    
                    %disp(['Obj Simpl =',num2str(obj_lp),' Obj bar = ', num2str(obj_lp2),' Ob mosek = ',num2str(obj_lp3)]);
                    
                    %disp(['Tim1 Simp =',num2str(toc1),'Tim2 int =', num2str(toc2), ' Tim3 Mosek =',  num2str(toc3)]);
                    
                    
                    if (mod(it,10)==0)
                        disp(['Obj LP=',num2str(obj_lp),' Time = ', num2str(toc3)]);
                    end
                    lb_new=obj_lp;
                    
                    
                    
                otherwise
                    assert(0);
            end
        end
        
        if lb_new< lb
            disp(['WARNING: lb_new=',num2str(lb_new), 'lb=',num2str(lb),'  diff =', num2str( lb_new-lb)]);
        end
        % assert(obj_model>=obj_model_old); % because the model gets closer to the original function
        
        A_old=A;
        %  b_old=b_new;
        lb=lb_new;
        
        it=it+1;
        
    end
    
end

end

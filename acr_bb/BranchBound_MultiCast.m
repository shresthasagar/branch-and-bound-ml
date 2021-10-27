function [w_opt,final_lb,final_ub,lb_seq,ub_seq]=BranchBound_MultiCast_QP(H,max_iternum,epsilon)
% ACR-BB Algorithm for single group multicast beamforming
% min  w'w
% s.t. |H(:,j)'w|>=1, j=1,...,m
% H is an N * M complex matrix,
% max_iternum is the maximum number of iterations
% epsilon is the relative error tolerance
%

[n,m]=size(H);
l=zeros(m,1);
u=l+2*pi;


[x,val,feas]=QpRelaxation_MulticastBeamforming(H,l,u);
A=zeros(max_iternum+100,m*2+n+1);
A(1,:)=[x',l',u',val];
used=1;
lbest=val;
xr=x/min(abs(H'*x));
ub=norm(xr)^2;
ubest=ub;
w_opt=xr;
lb_seq=lbest;
ub_seq=ubest;
if (ubest-lbest)/abs(ubest)<epsilon
    final_lb=lbest;
    final_ub=ubest;
    return;
end


iter=2;
con=1;

while iter<=max_iternum
    iter
    xc=A(con,1:n)';
    lc=A(con,(n+1):(n+m) )';
    uc=A(con,(n+m+1):(n+m*2) )';

    [zc,cd]=min(abs(H'*xc));
    
    xchild_left_lb=lc;
    xchild_left_ub=uc;
    xchild_right_lb=lc;
    xchild_right_ub=uc;
    tr=(lc(cd)+uc(cd))/2;
    xchild_left_ub(cd)=tr;
    xchild_right_lb(cd)=tr;
    
    if con < used
        A(con,:)=A(used,:);
        used=used-1;
    else
        used=used-1;
    end
    
    [x,val,feas]=QpRelaxation_MulticastBeamforming(H,xchild_left_lb,xchild_left_ub);
    if feas>0
        xr=x/min(abs(H'*x));
        ub=norm(xr)^2;
        if ub<ubest
            ubest=ub;
            w_opt=xr;
        end
        A(used+1,:)=[x',xchild_left_lb',xchild_left_ub',val];
        used=used+1;
    end
    
    [x,val,feas]=QpRelaxation_MulticastBeamforming(H,xchild_right_lb,xchild_right_ub);
    if feas>0
        xr=x/min(abs(H'*x));
        ub=norm(xr)^2;
        if ub<ubest
            ubest=ub;
            w_opt=xr;
        end
        A(used+1,:)=[x',xchild_right_lb',xchild_right_ub',val];
        used=used+1;
    end

    [lbest,con]=min(A(1:used,n+m*2+1));
    lb_seq(iter)=lbest;
    ub_seq(iter)=ubest;
    iter=iter+1;
    if (ubest-lbest)/abs(ubest)<epsilon
        final_lb=lbest;
        final_ub=ubest;
        return;
    end
end

final_lb=lbest;
final_ub=ubest;
return;

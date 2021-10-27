function [w,lb,feas]=QpRelaxation_MulticastBeamforming(H,l,u)
% min  w'w
% s.t. |H(:,j)'w|>=1, j=1,...,m
% We assume Image(w_n)=0;

[n,m]=size(H);
dim=2*n;
tcons=( (u-l)<= 2*pi-0.01 );
tcons(1)=0;
ncons= sum(tcons);
nfree=ncons*2;
nvar=nfree+dim;
neq=ncons*2+1;
nieq=ncons*3+1;
Q= eye(nvar)*2;
Q(1:nfree,1:nfree)=zeros(nfree);

c=zeros(nvar,1);

A=zeros(nieq,nvar);
b=zeros(nieq,1);
Aeq=zeros(neq,nvar);
beq=zeros(neq,1);


id=1;
ix=1;
for t=2:m
    if tcons(t)==0
        continue;
    end
    h=H(:,t);
    hr=real(h);
    hi=imag(h);
    Aeq(ix,id*2-1)=-1;
    Aeq(ix,(nfree+1):nvar)=[hr',hi'];
    ix=ix+1;
    Aeq(ix,id*2)=-1;
    Aeq(ix,(nfree+1):nvar)=[-hi',hr'];
    ix=ix+1;
    id=id+1;
end
h=H(:,1);
hr=real(h);
hi=imag(h);
Aeq(ix,(nfree+1):nvar)=[-hi',hr'];

t=0;
ix=1;
for j=2:m
    if  tcons(j)==1
        t=t+1;
    else
        continue;
    end
    x1=cos(l(j));
    y1=sin(l(j));
    x2=cos(u(j));
    y2=sin(u(j));    
    x3=cos((l(j)+u(j))/2);
    y3=sin((l(j)+u(j))/2);
    cc=(y1-y2)+i*(x1-x2);
    
    A(ix , t*2-1)=y1-y2;
    A(ix , t*2)= -x1+x2;
    b(ix)=(y1*x2-y2*x1);
    zr=(real(cc* (x3+i*y3))+y2*x1-y1*x2);
    if zr>0
        A(ix,:)=-A(ix,:);
        b(ix)=-b(ix);
    end
    ix=ix+1;
    
    A(ix , t*2-1)=y1;
    A(ix , t*2)= -x1;
    b(ix)=0;
    zr=x3*y1-y3*x1;
    if zr>0
        A(ix,:)=-A(ix,:);
        b(ix)=-b(ix);
    end
    ix=ix+1;
    
    A(ix , t*2-1)=y2;
    A(ix , t*2)= -x2;
    b(ix)=0;
    zr=x3*y2-y3*x2;
    if zr>0
        A(ix,:)=-A(ix,:);
        b(ix)=-b(ix);
    end
    ix=ix+1;
end
A(ix,(nfree+1):nvar)=-[hr',hi'];
b(ix)=-1;

optnew=optimset('Display','off','LargeScale','off');
[x,lb,flag]=quadprog(Q,c,A,b,Aeq,beq,[],[],[],optnew);

wr=x( (nfree+1):nvar);
w=wr(1:n)+i*wr(n+1:2*n);
if flag==-2
    feas=-1;
else
    feas=1;
end

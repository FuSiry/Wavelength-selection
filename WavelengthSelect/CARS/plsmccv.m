function CV=plsmccv(X,y,A,method,N,ratio,OPT)
%+++ Monte Carlo Cross Validation for PLS regression.
%+++ Input:  X: m x n  (Sample matrix)
%            y: m x 1  (measured property)
%            A: The maximal number of latent variables for
%            cross-validation
%            N: The number of Monte Carlo Simulation.
%        ratio: The ratio of calibration samples to the total samples.
%       method: pretreatment method. Contains: autoscaling,
%               pareto,minmax,center or none.
%          OPT: =1 : print process.
%               =0 : don't print process.
%+++ Output: Structural data: CV
%+++ Ref: Q.S. Xu, Y.Z. Liang, 2001.Chemo Lab,1-11
%+++ Supervisor: Yizeng Liang, yizeng_liang@263.net
%+++ Edited by H.D. Li,Nov.13, 2008.



if nargin<7;OPT=1;end
if nargin<6;ratio=0.8;end
if nargin<5;N=1000;end
if nargin<4;method='center';end
if nargin<3;A=2;end



[Mx,Nx]=size(X);
A=min([size(X) A]);
nc=floor(Mx*ratio);
nv=Mx-nc;
yytest=[];yycal=[];
YR=[];YC=[];Trace=[];
TrainIndex=[];
TestIndex=[];
for i=1:N
    index=randperm(Mx);
    calk=index(1:nc);testk=index(nc+1:Mx);
    Xcal=X(calk,:);ycal=y(calk);
    Xtest=X(testk,:);ytest=y(testk);
    
    %   data pretreatment
    [Xs,xpara1,xpara2]=pretreat(Xcal,method);
    [ys,ypara1,ypara2]=pretreat(ycal,method);   
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    [B,W,T,P,Q]=pls_nipals(Xs,ys,A,0);   % no pretreatment.
 
    yp=[];yc=[];
    for j=1:A
        B=W(:,1:j)*Q(1:j);
        %+++ calculate the coefficient linking Xcal and ycal.
        C=ypara2*B./xpara2';
        coef=[C;ypara1-xpara1*C;];
        %+++ predict
        Xteste=[Xtest ones(size(Xtest,1),1)];
        Xcale=[Xcal ones(size(Xcal,1),1)];
        ycal_p=Xcale*coef;       ytest_p=Xteste*coef;
        yp=[yp ytest_p];         yc=[yc ycal_p];     
    end
    YC=[YC;yc];yycal=[yycal;ycal];    
    YR=[YR;yp];yytest=[yytest;ytest];
    
    e1=sqrt(sum((yc-repmat(ycal,1,A)).^2)/length(ycal));
    e2=sqrt(sum((yp-repmat(ytest,1,A)).^2)/length(ytest));
    Trace=[Trace;[e1 e2]];
    TrainIndex=[TrainIndex;calk];
    TestIndex=[TestIndex;testk]; 
    if OPT==1;fprintf('The %dth sampling for MCCV finished.\n',i);end;
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
error_cal=YC-repmat(yycal,1,A);

error_test=YR-repmat(yytest,1,A);
PRESS=sum(error_test.^2);
cv=sqrt(PRESS/N/nv);

[RMSECV,index]=min(cv);index=index(1);
SST=sumsqr(yytest-mean(yytest));
for i=1:A
  SSE=sumsqr(YR(:,i)-yytest);
  Q2(i)=1-SSE/SST;
end
Ypred=YR(:,index);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%+++ output  %%%%%%%%%%%%%%%%
CV.method=method;
CV.MC_para=[N ratio];
CV.cv=cv;
CV.minRMSECV=RMSECV;
CV.Q2_all=Q2;
CV.Q2_max=Q2(index);
CV.Ypred=Ypred;
CV.optPC=index;
CV.TrainIndex=TrainIndex;
CV.TestIndex=TestIndex;
CV.RMSEF=Trace(:,1:A);
CV.RMSEP=Trace(:,A+1:end);
CV.error_cal=error_cal;
CV.error_test=error_test;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
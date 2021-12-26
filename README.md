# Wavelength-selection
Wavelength selection in spectroscopy
# 前言

<font color=#999AAA >
NIRS是介于可见光和中红外光之间的电磁波，其波长范围为（1100∼2526 nm。
由于近红外光谱区与有机分子中含氢基团（OH、NH、CH、SH）振动的合频和
各级倍频的吸收区一致，通过扫描样品的近红外光谱，可以得到样品中有机分子含氢
基团的特征信息，常被作为获取样本信息的一种有效的载体。
基于NIRS的检测方法具有方便、高效、准确、成本低、可现场检测、不
破坏样品等优势，被广泛应用于各类检测领域。但
近红外光谱存在谱带宽、重叠较严重、吸收信号弱、信息解析复杂等问题，与常用的
化学分析方法不同，仅能作为一种间接测量方法，无法直接分析出被测样本的含量或
类别，它依赖于化学计量学方法，在样品待测属性值与近红外光谱数据之间建立一个
关联模型(或称校正模型，Calibration Model) ，再通过模型对未知样品的近红外光谱
进行预测来得到各性质成分的预测值。现有近红外建模方法主要为经典建模
（预处理+波长筛选进行特征降维和突出，再通过pls、svm算法进行建模）以及深度学习方法（端到端的建模，对预处理、波长选择等依赖性很低）</font>

<hr style=" border:solid; width:100px; height:1px;" color=#000000 size=1">


<font  size=5 color=bule >本篇主要讲述常见的波长选择算法(目前是matlab版本的，python版本有时间重写一下)

# 一、SPA算法
连续投影算法（successive projections algorithm， SPA） 是前向特征变量选择方法。SPA利用向量的投影分析，通过将波长投影到其他波长上，比较投影向量大小，以投影向量最大的波长为待选波长，然后基于矫正模型选择最终的特征波长。SPA选择的是含有最少冗余信息及最小共线性的变量组合。

```python
function chain = projections_qr(X,k,M)

% Projections routine for the Successive Projections Algorithm using the
% built-in QR function of Matlab
%
% chain = projections(X,k,M)
%
% X --> Matrix of predictor variables (# objects N x # variables K)
% k --> Index of the initial column for the projection operations
% M --> Number of variables to include in the chain
%
% chain --> Index set of the variables resulting from the projection operations

X_projected = X;

norms = sum(X_projected.^2);    % Square norm of each column vector
norm_max = max(norms); % Norm of the "largest" column vector

X_projected(:,k) = X_projected(:,k)*2*norm_max/norms(k); % Scales the kth column so that it becomes the "largest" column

[dummy1,dummy2,order] = qr(X_projected,0); 
chain = order(1:M)';
```

# 二、UVE算法
无信息变量去除算法(uninformative variable elimination,UVE)能够去除对建模共效率较小的波长变量，选出特征波长变量，被去除的波长变量我们称之为无信息变量。无信息变量去除算法的建立是基于偏最小二乘(partial least squares,PLS)算法。去除无信息变量，减少了建模所用的变量个数，降低了模型复杂性。为了选择无信息变量，UVE算法通过对PLS模型中添加一组与原始变量数量相同的白噪声变量，然后基于PLS模型的交叉留一法得到每个变量对应的回归系数，包括噪声变量。用每个变量系数的稳定值除以标准差，将他们的商与随机变量矩阵得到的稳定值做比较，删除那些与随机变量一样对建模无效的波长变量。
```python
function [B,C,P,T,U,R,R2X,R2Y]=plssim(X,Y,A,S,XtX);

[n,px] = size(X); [n,m] = size(Y);   				% size of the input data matrices
if nargin<5, S = []; end, if isempty(S), S=(Y'*X)'; end		% if XtX not inputted, S=[]; always when S=[] then S=(Y'*X)'
if nargin<4, XtX=[]; end					% if S is not inputted, XtX=[];
if isempty(XtX) & n>3*px, XtX = X'*X; end			% when XtX=[] and X is very "tall", the booster XtX is calculated
if nargin<3, A=10; end, A = min([A px n-1]);			% if A is not inputted, then the defaul A is min[10 px n-1]
T = zeros(n ,A); U = T;						% initialization of variables
R = zeros(px,A); P = R; V = R;
C = zeros(m ,A); 
R2Y = zeros(1,A);
z = zeros(m,1); v = zeros(px,1);
if n>px, S0 = S; end, StS = S'*S;				% SIMPLS algorithm
nm1 = n-1;
tol = 0;
for a = 1:A
  StS = StS-z*z'; 
  [Q,LAMBDA] = eig(StS); 
  [lambda,j] = max(diag(LAMBDA)); 
  q = Q(:,j(1));
  r = S*q;
  t = X*r;
  if isempty(XtX), p = (t'*X)'; else p = XtX*r; end
  if n>px, d = sqrt(r'*p/nm1); else d = sqrt(t'*t/nm1); end
  if d<tol, 
	disp(' ')
        disp('WARNING: the required number of factors (A) is too high !')
	disp('Less PLS factors were extracted from the data in the PLSSIM program !') 
	disp(' ')
	break,
, 	else tol=max(tol,d/1e5);
  end
  v = p-V(:,1:max(1,a-1))*(p'*V(:,1:max(1,a-1)))'; v = v/sqrt(v'*v); 
  z = (v'*S)'; 
  S = S-v*z'; 
								% save results
  V(:,a) = v;
  R(:,a) = r/d; 						% X weights
  P(:,a) = p/(d*nm1); 						% X loadings
  T(:,a) = t/d;							% X scores
  U(:,a) = Y*q;							% Y scores
  C(:,a) = q*(lambda(1)/(nm1*d)); 				% Y loadings
  R2Y(1,a) =  lambda(1)/d;					% Y-variance accounted for
end
clear StS V LAMBDA Q p q r t v z;
if d<tol,
 A=a-1; a=A; T=T(:,1:A); U=U(:,1:A); R=R(:,1:A); P=P(:,1:A); C=C(:,1:A);
end
while a>1
  U(:,a) = U(:,a)-T(:,1:a-1)*(U(:,a)'*T(:,1:a-1)/nm1)'; 
  a=a-1; 
end
B = R*C';							% B-coefficients of the regression Y on X
if isempty(XtX), sumX2=sum(X.^2); else sumX2 = sum(diag(XtX)); end
R2X = 100*nm1/sum(sumX2)*cumsum(sum(P.^2)); 
R2Y = 100/nm1/sum(sum(Y.^2))*cumsum(R2Y(1:A).^2);
```
# 三、LAR算法
LAR(Least Angel Regression)，Efron于2004年提出的一种变量选择的方法，类似于向前逐步回归(ForwardStepwise)的形式，是lasso regression的一种高效解法。向前逐步回归(Forward Stepwise)不同点在于，Forward Stepwise每次都是根据选择的变量子集，完全拟合出线性模型，计算出RSS，再设计统计量（如AIC）对较高的模型复杂度作出惩罚，而LAR是每次先找出和因变量相关度最高的那个变量, 再沿着LSE的方向一点点调整这个predictor的系数，在这个过程中，这个变量和残差的相关系数会逐渐减小，等到这个相关性没那么显著的时候，就要选进新的相关性最高的变量，然后重新沿着LSE的方向进行变动。而到最后，所有变量都被选中，就和LSE相同了。

```python
function [b info] = lar(X, y, stop, storepath, verbose)
%% Input checking
% Set default values.
if nargin < 5
  verbose = false;
end
if nargin < 4
  storepath = true;
end
if nargin < 3
  stop = 0;
end
if nargin < 2
  error('SpaSM:lar', 'Input arguments X and y must be specified.');
end

%% LARS variable setup
[n p] = size(X);
maxVariables = min(n-1,p); % Maximum number of active variables

useGram = false;
% if n is approximately a factor 10 bigger than p it is faster to use a
% precomputed Gram matrix rather than Cholesky factorization when solving
% the partial OLS偏最小二乘法 problem. Make sure the resulting Gram matrix is not
% prohibitively large.
if (n/p) > 10 && p < 1000
  useGram = true;
  Gram = X'*X;
end

% set up the LAR coefficient vector
if storepath
  b = zeros(p, p+1);
else
  b = zeros(p, 1);
  b_prev = b;
end

mu = zeros(n, 1); % current "position" as LARS travels towards lsq solution

I = 1:p; % inactive set
A = []; % active set
if ~useGram
  R = []; % Cholesky factorization R'R = X'X where R is upper triangular
end

stopCond = 0; % Early stopping condition boolean
step = 1; % step count

if verbose
  fprintf('Step\tAdded\tActive set size\n');
end

%% LARS main loop
% while not at OLS solution or early stopping criterion is met
while length(A) < maxVariables && ~stopCond
  r = y - mu;

  % find max correlation
  c = X(:,I)'*r; % X的每一维与当前残差的相关系数
  [cmax cidx] = max(abs(c));

  % add variable
  if ~useGram
    R = cholinsert(R,X(:,I(cidx)),X(:,A));
  end
  if verbose
    fprintf('%d\t\t%d\t\t%d\n', step, I(cidx), length(A) + 1);
  end
  A = [A I(cidx)];
  I(cidx) = [];
  c(cidx) = []; % 删除原来的这一项后其他各项的相关系数（也即删除了cidx这一维的相关系数（因为已经把第cidx维加进了active set）），后面的项补上来。上一行同理

  % partial OLS solution and direction from current position to the OLS
  % solution of X_A
  if useGram
    b_OLS = Gram(A,A)\(X(:,A)'*y); % same as X(:,A)\y, but faster
  else
%     b_OLS = R\(R'\(X(:,A)'*y)); % same as X(:,A)\y, but faster
    b_OLS = X(:,A)\y; %\是matlab里面的左除。用来求（以你问题为例）X*a=y这个线性方程组的（最小二乘）解   因为运算的时候有时会出现说 警告: '矩阵为奇异工作精度'，所以把上面那行换成这个形式了，尽管运行会慢一点  参考：http://www.ilovematlab.cn/thread-301697-1-1.html和http://zhidao.baidu.com/link?url=GzrsyLqNSkvuryZvr4UAMpxi4PIxjpUtJ9HmJ2nL8jVadDjz5CRMbpoxfmk-mRJZ6bsvqgsPHbV7pslq248SpURQkYUmSNhYBh2VoPYSUQe
  end
  d = X(:,A)*b_OLS - mu;

  if isempty(I)
    % if all variables active, go all the way to the OLS solution
    gamma = 1;
  else
    % compute length of walk along equiangular direction
    cd = (X(:,I)'*d);
    gamma = [ (c - cmax)./(cd - cmax); (c + cmax)./(cd + cmax) ];
    gamma = min(gamma(gamma > 0)); % 取gamma正数部分的最小值
  end

  % update beta
  if storepath
    b(A,step + 1) = b(A,step) + gamma*(b_OLS - b(A,step)); % update beta
  else
    b_prev = b;
    b(A) = b(A) + gamma*(b_OLS - b(A)); % update beta
  end

  % update position
  mu = mu + gamma*d;
  
  % increment step counter
  step = step + 1;

  % Early stopping at specified bound on L1 norm of beta
  if stop > 0
    if storepath
      t2 = sum(abs(b(:,step)));
      if t2 >= stop
        t1 = sum(abs(b(:,step - 1)));
        s = (stop - t1)/(t2 - t1); % interpolation factor 0 < s < 1
        b(:,step) = b(:,step - 1) + s*(b(:,step) - b(:,step - 1));
        stopCond = 1;
      end
    else
      t2 = sum(abs(b));
      if t2 >= stop
        t1 = sum(abs(b_prev));
        s = (stop - t1)/(t2 - t1); % interpolation factor 0 < s < 1
        b = b_prev + s*(b - b_prev);
        stopCond = 1;
      end
    end
  end
    
  % Early stopping at specified number of variables
  if stop < 0
    stopCond = length(A) >= -stop;
  end
end

% trim beta
if storepath && size(b,2) > step
  b(:,step + 1:end) = [];
end

%% Compute auxilliary measures
if nargout == 2 % only compute if asked for
  info.steps = step - 1;
  b0 = pinv(X)*y; % regression coefficients of low-bias model
  penalty0 = sum(abs(b0)); % L1 constraint size of low-bias model低偏置模型
  indices = (1:p)';
  
  if storepath % for entire path
    q = info.steps + 1;
    info.df = zeros(1,q);
    info.Cp = zeros(1,q);
    info.AIC = zeros(1,q);
    info.BIC = zeros(1,q);
    info.s = zeros(1,q);
    sigma2e = sum((y - X*b0).^2)/n;
    for step = 1:q
      A = indices(b(:,step) ~= 0); % active set激活集，有效集，也就是每一列不为0的那几个数的索引
      % compute godness of fit measurements Cp, AIC and BIC
      r = y - X(:,A)*b(A,step); % residuals残差
      rss = sum(r.^2); % residual sum-of-squares残差平方和
      info.df(step) = step - 1;
      info.Cp(step) = rss/sigma2e - n + 2*info.df(step);
      info.AIC(step) = rss + 2*sigma2e*info.df(step);
      info.BIC(step) = rss + log(n)*sigma2e*info.df(step);
      info.s(step) = sum(abs(b(A,step)))/penalty0;
    end
    
  else % for single solution
    info.s = sum(abs(b))/penalty0;
    info.df = info.steps;
  end

```
# 四、Cars算法
竞争性自适应重加权采样法（competitive adapative reweighted sampling， CARS）是一种结合蒙特卡洛采样与PLS模型回归系数的特征变量选择方法，模仿达尔文理论中的 ”适者生存“ 的原则（Li et al., 2009）。CARS 算法中，每次通过自适应加权采样（adapative reweighted sampling， ARS）保留PLS模型中 回归系数绝对值权重较大的点作为新的子集，去掉权值较小的点，然后基于新的子集建立PLS模型，经过多次计算，选择PLS模型交互验证均方根误差（RMSECV）最小的子集中的波长作为特征波长
```python
function F=carspls(X,y,A,fold,method,num) 
%+++ CARS: Competitive Adaptive Reweighted Sampling method for variable selection.
%+++ X: The data matrix of size m x p
%+++ y: The reponse vector of size m x 1
%+++ A: the maximal principle to extract.
%+++ fold: the group number for cross validation.
%+++ num: the  number of Monte Carlo Sampling runs.
%+++ method: pretreatment method.
%+++ Hongdong Li, Dec.15, 2008.
%+++ Advisor: Yizeng Liang, yizeng_liang@263.net
%+++ lhdcsu@gmail.com
%+++ Ref:  Hongdong Li, Yizeng Liang, Qingsong Xu, Dongsheng Cao, Key
%    wavelengths screening using competitive adaptive reweighted sampling 
%    method for multivariate calibration, Anal Chim Acta 2009, 648(1):77-84


tic;
%+++ Initial settings.
if nargin<6;num=50;end;
if nargin<5;method='center';end;
if nargin<4;fold=5;end;
if nargin<3;A=2;end;

%+++ Initial settings.
[Mx,Nx]=size(X);
A=min([Mx Nx A]);
index=1:Nx;
ratio=0.9;
r0=1;
r1=2/Nx;
Vsel=1:Nx;
Q=floor(Mx*ratio);
W=zeros(Nx,num);
Ratio=zeros(1,num);

%+++ Parameter of exponentially decreasing function. 
b=log(r0/r1)/(num-1);  a=r0*exp(b);

%+++ Main Loop
for iter=1:num
     
     perm=randperm(Mx);   
     Xcal=X(perm(1:Q),:); ycal=y(perm(1:Q));   %+++ Monte-Carlo Sampling.
     
     PLS=pls(Xcal(:,Vsel),ycal,A,method);    %+++ PLS model
     w=zeros(Nx,1);coef=PLS.coef_origin(1:end-1,end);
     w(Vsel)=coef;W(:,iter)=w; 
     w=abs(w);                                  %+++ weights
     [ws,indexw]=sort(-w);                      %+++ sort weights
     
     ratio=a*exp(-b*(iter+1));                      %+++ Ratio of retained variables.
     Ratio(iter)=ratio;
     K=round(Nx*ratio);  
     
     
     w(indexw(K+1:end))=0;                      %+++ Eliminate some variables with small coefficients.  
     
     Vsel=randsample(Nx,Nx,true,w);                 %+++ Reweighted Sampling from the pool of retained variables.                 
     Vsel=unique(Vsel);              
     fprintf('The %dth variable sampling finished.\n',iter);    %+++ Screen output.
 end

%+++  Cross-Validation to choose an optimal subset;
RMSEP=zeros(1,num);
Q2_max=zeros(1,num);
Rpc=zeros(1,num);
for i=1:num
   vsel=find(W(:,i)~=0);
 
   CV=plscvfold(X(:,vsel),y,A,fold,method,0);  
   RMSEP(i)=CV.RMSECV;
   Q2_max(i)=CV.Q2_max;   
   
   Rpc(i)=CV.optPC;
   fprintf('The %d/%dth subset finished.\n',i,num);
end
[Rmin,indexOPT]=min(RMSEP);
Q2_max=max(Q2_max);




%+++ save results;
time=toc;
%+++ output
F.W=W;
F.time=time;
F.cv=RMSEP;
F.Q2_max=Q2_max;
F.minRMSECV=Rmin;
F.iterOPT=indexOPT;
F.optPC=Rpc(indexOPT);
Ft.ratio=Ratio;
F.vsel=find(W(:,indexOPT)~=0)';



function sel=weightsampling_in(w)
%Bootstrap sampling
%2007.9.6,H.D. Li.

w=w/sum(w);
N1=length(w);
min_sec(1)=0; max_sec(1)=w(1);
for j=2:N1
   max_sec(j)=sum(w(1:j));
   min_sec(j)=sum(w(1:j-1));
end
% figure;plot(max_sec,'r');hold on;plot(min_sec);
      
for i=1:N1
  bb=rand(1);
  ii=1;
  while (min_sec(ii)>=bb | bb>max_sec(ii)) & ii<N1;
    ii=ii+1;
  end
    sel(i)=ii;
end      % w is related to the bootstrap chance

%+++ subfunction:  booststrap sampling
% function sel=bootstrap_in(w);
% V=find(w>0);
% L=length(V);
% interval=linspace(0,1,L+1);
% for i=1:L;
%     rn=rand(1);
%     k=find(interval<rn);
%     sel(i)=V(k(end));    
% end
```

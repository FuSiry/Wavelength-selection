%% sub-function1
function min_RMSECV  = pls_val(Xcal,y_train,var_sel)

Ncomp = 10;%主因子数
x_train = Xcal(:,var_sel);
NumOfVar = size(var_sel,1);
       
if Ncomp<=NumOfVar
    [~,~,~,~,BETA,~,MSE]  = plsregress(x_train,y_train,Ncomp,'cv',size(x_train,1));
    RMSECV = sqrt(MSE(2,2:Ncomp+1));
    [min_RMSECV,LV] = min(RMSECV);
else
    [~,~,~,~,BETA,~,MSE] = plsregress(x_train,y_train,NumOfVar,'cv',size(x_train,1));
    RMSECV = sqrt(MSE(2,2:NumOfVar+1));
    [min_RMSECV,LV] = min(RMSECV);
end


%% PLS做回归 sub-function1
function [min_RMSEC,min_RMSEP,min_R2c,min_R2p,indexcmp] = pls_val(x_train,y_train,x_test,y_test)
Ncomp = 20;%主因子数
RMSEP = 10^3*ones(Ncomp,1);
for ncomp = 4:Ncomp   % 不同的主成份数下计算RMSEC/P和Square_R
    [XL,YL,XS,YS,BETA] = plsregress(x_train,y_train,ncomp,'cv',size(x_train,1)); %根据训练集得出回归系数BETA
    yt_fit_c = [ones(size(x_train,1),1) x_train]*BETA;  %根据BETA求 校正集的预测值
    yt_fit_p = [ones(size(x_test,1),1) x_test]*BETA;    %根据BETA求 预测集的预测值
    %compute the coeffectives
    RMSEC(ncomp) = sqrt(sum((yt_fit_c-y_train).^2)/length(y_train)); % 等同于sqrt((yt_fit_c-y_train)'*(yt_fit_c-y_train)/size(y_train,1));
    RMSEP(ncomp) = sqrt(sum((yt_fit_p-y_test).^2)/length(y_test));  % 等同于RMSEP(ncomp) = sqrt((yt_fit_p-y_test)'*(yt_fit_p-y_test)/size(y_test,1));
    R2c(ncomp) = 1-sum((yt_fit_c-y_train).^2)/(sum((y_train-mean(y_train)).^2));   % 等同于 RC(ncomp) = sqrt(1-((yt_fit_c-y_train)'*(yt_fit_c-y_train))/((y_train-mean(y_train))'*(y_train-mean(y_train))));
    R2p(ncomp) = 1-sum((yt_fit_p-y_test).^2)/(sum((y_test-mean(y_test)).^2)); % RP(ncomp) = sqrt(1-((yt_fit_p-y_test)'*(yt_fit_p-y_test))/((y_test-mean(y_test))'*(y_test-mean(y_test))));
end
[min_RMSEP,indexcmp] = min(RMSEP)  
min_RMSEC = RMSEC(indexcmp)    
min_R2c=R2c(indexcmp)   
min_R2p=R2p(indexcmp) 
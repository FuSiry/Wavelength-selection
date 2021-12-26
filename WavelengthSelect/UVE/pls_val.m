%% PLS���ع� sub-function1
function [min_RMSEC,min_RMSEP,min_R2c,min_R2p,indexcmp] = pls_val(x_train,y_train,x_test,y_test)
Ncomp = 20;%��������
RMSEP = 10^3*ones(Ncomp,1);
for ncomp = 4:Ncomp   % ��ͬ�����ɷ����¼���RMSEC/P��Square_R
    [XL,YL,XS,YS,BETA] = plsregress(x_train,y_train,ncomp,'cv',size(x_train,1)); %����ѵ�����ó��ع�ϵ��BETA
    yt_fit_c = [ones(size(x_train,1),1) x_train]*BETA;  %����BETA�� У������Ԥ��ֵ
    yt_fit_p = [ones(size(x_test,1),1) x_test]*BETA;    %����BETA�� Ԥ�⼯��Ԥ��ֵ
    %compute the coeffectives
    RMSEC(ncomp) = sqrt(sum((yt_fit_c-y_train).^2)/length(y_train)); % ��ͬ��sqrt((yt_fit_c-y_train)'*(yt_fit_c-y_train)/size(y_train,1));
    RMSEP(ncomp) = sqrt(sum((yt_fit_p-y_test).^2)/length(y_test));  % ��ͬ��RMSEP(ncomp) = sqrt((yt_fit_p-y_test)'*(yt_fit_p-y_test)/size(y_test,1));
    R2c(ncomp) = 1-sum((yt_fit_c-y_train).^2)/(sum((y_train-mean(y_train)).^2));   % ��ͬ�� RC(ncomp) = sqrt(1-((yt_fit_c-y_train)'*(yt_fit_c-y_train))/((y_train-mean(y_train))'*(y_train-mean(y_train))));
    R2p(ncomp) = 1-sum((yt_fit_p-y_test).^2)/(sum((y_test-mean(y_test)).^2)); % RP(ncomp) = sqrt(1-((yt_fit_p-y_test)'*(yt_fit_p-y_test))/((y_test-mean(y_test))'*(y_test-mean(y_test))));
end
[min_RMSEP,indexcmp] = min(RMSEP)  
min_RMSEC = RMSEC(indexcmp)    
min_R2c=R2c(indexcmp)   
min_R2p=R2p(indexcmp) 
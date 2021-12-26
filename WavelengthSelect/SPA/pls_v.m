function [min_RMSEC,min_RMSEP,min_Square_R2C,min_Square_R2P,indexcmp] = pls_v(Xcal,y_train,Xval,y_test,var_sel)
Ncomp = 10;%主因子数
x_train = Xcal(:,var_sel);
x_test = Xval(:,var_sel);
NumOfVar = size(var_sel,1);
 
RMSEP = 10^3*ones(Ncomp,1);
for ncomp = 2:Ncomp         
    if ncomp<=NumOfVar
        [XL,YL,XS,YS,BETA] = plsregress(x_train,y_train,ncomp,'cv',size(x_train,1));
        yt_fit_c = [ones(size(x_train,1),1) x_train]*BETA;
        yt_fit_p = [ones(size(x_test,1),1) x_test]*BETA;
        %compute the coeffectives        
        RMSEC(ncomp) = sqrt((yt_fit_c-y_train)'*(yt_fit_c-y_train)/size(y_train,1));
        RMSEP(ncomp) = sqrt((yt_fit_p-y_test)'*(yt_fit_p-y_test)/size(y_test,1));
        Square_R2C(ncomp) = 1-((yt_fit_c-y_train)'*(yt_fit_c-y_train))/((y_train-mean(y_train))'*(y_train-mean(y_train)));
        Square_R2P(ncomp) = 1-((yt_fit_p-y_test)'*(yt_fit_p-y_test))/((y_test-mean(y_test))'*(y_test-mean(y_test))); 

    else
        [XL,YL,XS,YS,BETA] = plsregress(x_train,y_train,NumOfVar,'cv',size(x_train,1));
        yt_fit_c = [ones(size(x_train,1),1) x_train]*BETA;
        yt_fit_p = [ones(size(x_test,1),1) x_test]*BETA;
        %compute the coeffectives        
        RMSEC(NumOfVar) = sqrt((yt_fit_c-y_train)'*(yt_fit_c-y_train)/size(y_train,1));
        RMSEP(NumOfVar) = sqrt((yt_fit_p-y_test)'*(yt_fit_p-y_test)/size(y_test,1));
        Square_R2C(NumOfVar) = 1-((yt_fit_c-y_train)'*(yt_fit_c-y_train))/((y_train-mean(y_train))'*(y_train-mean(y_train)));
        Square_R2P(NumOfVar) = 1-((yt_fit_p-y_test)'*(yt_fit_p-y_test))/((y_test-mean(y_test))'*(y_test-mean(y_test)));
        break;
    end
end
[min_RMSEP,indexcmp] = min(RMSEP);
min_RMSEC = RMSEC(indexcmp);
min_Square_R2C = Square_R2C(indexcmp);
min_Square_R2P = Square_R2P(indexcmp);
function  vars = assignIntervals(boshudian, n_Vars)
%%�������Ĺ����ǰѽ�������ײ�������Ϊn_Vars�����䣬����¼ÿ������Ĳ���λ��
%%  boshudian: 1 * nά 
%%������������
vars = zeros(n_Vars,2);
n_boshu = size(boshudian,2);
%%����ÿ������Ļ������
Width_base = floor(n_boshu/n_Vars);
%%�ж�������������Ƿ����
if n_Vars > n_boshu
   disp(' ')
   disp('�������ô������������ܴ��ڲ��������������˳�')
   disp(' ')
   return
end
%%����ʣ��Ĳ�������
More_waves = n_boshu - Width_base*n_Vars;
%%������Ĳ�������ƽ���ָ�ǰMore_waves�����䣬ÿ�������ڻ�����ȵĻ������ټ�1��������
if More_waves 
    w1 = Width_base + 1;    %ǰMore_waves����������Ĳ�������
    w2 = Width_base ;      %������������Ĳ�������
else
    w2 = Width_base ;      %������������Ĳ�������
end
%%����ÿ���������ʼ������ͽ���������

if More_waves       %%������ڶ���Ĳ�����
    for i = 1: More_waves
        vars(i, 1) = (i - 1)*w1 + 1;
        vars(i, 2) = i*w1;
    end
    for j = More_waves + 1 : n_Vars
        vars(j, 1) = (j - 1)*w2 + 1 + More_waves;
        vars(j, 2) = j*w2 + More_waves;
    end
else            %%��������ڶ���Ĳ�����
    for i = 1:n_Vars
        vars(i, 1) = (i - 1)*w2 + 1; 
        vars(i, 2) = i*w2;
    end
end

end
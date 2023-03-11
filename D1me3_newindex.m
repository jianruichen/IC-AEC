%function [Precision_K,Recall_K,MRR_K,Hit_K,NDCG_K]=D1me3_newindex(mycase,recommend_number)
mycase=1;recommend_number=10;
method='commondu';neibor_num=40;
%% ����
Ucluster_number=2;
alpha=0.6;
beta=0.2;
S=100; 
c=2.5;
d=2;%2.2 1.8
%% ����
load data1\item
if mycase==1
   load data1\u1base
   load data1\u1test
elseif mycase==2
   load data1\u2base
   load data1\u2test
elseif mycase==3
   load data1\u3base
   load data1\u3test
elseif mycase==4
   load data1\u4base
   load data1\u4test
elseif mycase==5
   load data1\u5base
   load data1\u5test
end
m=size(u1,1);
number_user=max(u1(:,1));
number_movies=1682;
score_matrix=zeros(number_user,number_movies);%���־���,��Ϊ�û�����Ϊ��Ŀ
for i=1:m
    score_matrix(u1(i,1),u1(i,2))=u1(i,3);
end
score_matrix10=score_matrix;
score_matrix10(score_matrix10~=0)=1;
type_number=size(item,2);%��Ӱ������
like_degree=zeros(number_user,type_number);%ϲ���ĳ̶�
kesai=zeros(number_user,number_movies);
att=zeros(number_user,number_user);
Attention=zeros(number_user,number_user);
for i=1:number_user
    usee_item=find(score_matrix(i,:)~=0);%��i���û������ĵ�Ӱ
   for k=1:type_number
      if isempty(usee_item)
         like_degree(i,k)=0;
      else
         like_degree(i,k)=sum(item(usee_item,k))/length(usee_item);
         %�˴��ɸ�Ϊ��������/������K�����͵ĵ�Ӱ������
      end
   end  
   for l=1:number_movies
       kesai(i,l)=sum(like_degree(i,:).^item(l,:));
       %kesai(i,l)=prod(like_degree(i,:).^item(l,:));
   end
   for p=1:number_user
       com_item=find(score_matrix(i,:)~=0 & score_matrix(p,:)~=0);
       if isempty(com_item)
           att(i,p)=0;
       else
          com_k=norm(kesai(i,com_item)-kesai(p,com_item),2);
          att(i,p)=1/(1+com_k);
       end
   end
end
for i=1:number_user
    for j=1:number_user
      temp=length(find(att(i,:)==0));
      if att(i,j)==0
          Attention(i,j)=0;
       else
          Attention(i,j)=exp(att(i,j))/(sum(exp(att(i,:)))-temp);%û�й�ͬ���۵���Ŀ���û���
      end  
    end
end
%% ��ʼ�ݻ�
%degree_u=sum(score_matrix10,2)';
y=zeros(S,number_user);
%y(1,:)=(degree_u-min(degree_u))/(max(degree_u)-min(degree_u));
y(1,:)=rand(1,number_user)*pi/2;
K1=zeros(S,number_user);
K2=zeros(S,number_user);
for e=1:S 
    for i=1:number_user
       usermean=sum(Attention(i,:))/length(find(Attention(i,:)~=0)); 
       K1(e,i)=c*(y(e,i));
       K2(e,i)=-d*(y(e,i));
       bigger_mean=find(Attention(i,:)>=usermean);  
       a =sum(K1(e,i).*Attention(i,bigger_mean).*sin((y(e,i)-y(e,bigger_mean))));%��e�ε�������j���û��͵�i���û��Ĳ�               
       smaller_mean=find(Attention(i,:)<usermean);  
       b =sum(K2(e,i).*Attention(i,smaller_mean).*sin((y(e,i)-y(e,smaller_mean))));%������sin
       y(e+1,i)=alpha*(y(e,i))-0.5*beta*(y(e,i).^2)+a+b;
       %y(e+1,i)=a+b;
        %if norm(y(t+1:t+y_dim,i)-y(t-y_dim+1:t,i))<epsilon %1e-2
        %break%||
        %end
    end   
end
plot(1:S+1,y(1:S+1,1:number_user),'LineWidth',2);
if isnan(y(end,1:number_user))
    Precision_K=0;    
    Recall_K=0; 
    MRR_K=0; 
    Hit_K=0; 
    NDCG_K=0; 
else
a=kmeans(y(end,1:number_user)',Ucluster_number);
for i=1:max(a)%%%%
      temp=find(a==i);%ͬһ���������û�
      data=score_matrix(temp,:);%Ҫ�ҵ�ͬ������Ŀ��Ӧ�����־���
      dex{i}=temp;  %%�����±�������������±꣬����洢��ָ�����û�����Ŀ����Ժ��ָ��
    %%%%%�ò�ͬ�����������ƶȾ���
       switch lower(method)
             case 'cosine'
            sim_matrix{i}=SimilitudItems(data','cosine');
             case 'correlation'
            sim_matrix{i}=SimilitudItems(data','correlation'); 
             case 'adjustedcosine'
            sim_matrix{i}=SimilitudItems(data','adjustedcosine');
             case 'optimizedcos'
            sim_matrix{i}=SimilitudItems(data','optimizedcos');
            case 'ducosine'
            sim_matrix{i}=SimilitudItems(data','ducosine');
            case 'commondu'
            sim_matrix{i}=SimilitudItems(data','commondu');
            case 'singledu'
            sim_matrix{i}=SimilitudItems(data','singledu');
        end
    sim_matrix{i}=sim_matrix{i}./repmat(sqrt(sum(sim_matrix{i}.^2,2)),1,size(sim_matrix{i},2));
end
pp=find(u2(:,2)>max(u1(:,2)));
[m,temp3]=size(u2);
Predict_score=zeros(m,1); 
for i=1:size(pp,1)
    user=u2(pp(i),1);
    [temp3,BB]=find(score_matrix(user,:)~=0);    %���û������е�Ӱ��������
    aver_score=mean(score_matrix(user,BB));
    Predict_score(pp(i))=round(aver_score);
end
for i=1:m
    if ismember(i,setdiff([1:m],pp))
    user=u2(i,1);
    item=u2(i,2);
    no2=a(user);    %���ǵ�user���û����������ı��
    user1=find(dex{no2}==user);
    %%%%�����Ǽ�������û���Эͬ����
    up_score_matrix=score_matrix(dex{no2},:); %%��ͬ�������û������о����ó���
    %up_sim_matrix=sim_matrix{no2}; %%��������Ӧ���û����ƶȾ���
    %%%%%%%%%%  �û����ܶԸ�����û�����֣�Ҳ������һ���п���Ϊ��
    [temp3,BB]=find(up_score_matrix(user1,:)~=0);%���û������е�Ӱ��������
    aver_score=mean(up_score_matrix(user1,BB));
    P_u=find(up_score_matrix(:,item)~=0);  %����Ŀ����Ŀ���û�
    if isempty(P_u)
        Predict_score(i)=round(aver_score);%%����������û���û�жԴ�������Ŀ�����֣����ø��û���ƽ������
    else
        %%������һ���ǳ��ؼ���һ��Ҫ��ʵָ�������
        P_u_sim=sim_matrix{no2}(user1,P_u);  %%����������û����������û������ƶ�
        [temp3,index1]=sort(P_u_sim,2,'descend');    %���н�������
        num1=size(index1,2);
        if num1>=neibor_num
            neibor=(P_u(index1(1:neibor_num)));
        else
            neibor=(P_u(index1));
        end
        sum1=0;
        sum2=0;
        for j=1:size(neibor,1)
            [temp3,BB]=find(up_score_matrix(neibor(j),:)~=0);
            a_score(j)=mean(up_score_matrix(neibor(j),BB));
            sum1 = sum1+sim_matrix{no2}(user1,neibor(j))*(up_score_matrix(neibor(j),item)-a_score(j));
            sum2 = sum2+sim_matrix{no2}(user1,neibor(j));
        end
        if sum2==0   
            Predict_score(i)=round(aver_score); %�ų���ĸΪ������
        else
            Predict_score(i)=round(aver_score+sum1/sum2);
        end
    end
    %ȷ��Ԥ��ֵΪ1~5��������
    if Predict_score(i)>5
        Predict_score(i)=5;
    elseif Predict_score(i)<1
        Predict_score(i)=1;
    elseif isnan(Predict_score(i))
        Predict_score(i)=round(aver_score);
    end
    end
end
%% ���� MAE��RMSE
Eval=abs(u2(:,3)-Predict_score);
RMSE=sqrt(Eval'*Eval/m);
MAE=sum(Eval)/m;
%% �Ƽ�����
test__user=unique(u2(:,1));%���Լ��е��û�
number_user1=length(test__user);%���Լ��е��û�����459
for i=1:number_user1
    user_ID=test__user(i);%�û�ID
    cluster_number=a(user_ID); %Ŀ���û�������
    Tsim_matrix1=sim_matrix{cluster_number};%һ�����������ƶȾ���
    neigh_user=find(a==cluster_number);%һ���������û���ԭ��ID
    incom_number=find(neigh_user==user_ID);%Ŀ���û��������ڵ��к�
    sim_vector=Tsim_matrix1(incom_number,:);%Ŀ���û������ƶ�����
    [sim_Rank,index0]=sort(sim_vector,'descend');%sim_Rank�����ƶ�����index��sim_Rank��ֵ�������ڵ��к�
    if length(neigh_user)>=neibor_num
       best_neibor=neigh_user(index0(1:neibor_num));%Ŀ���û��������ID
    else
       best_neibor=neigh_user;%���յ������
    end
    end_neighnumber=length(best_neibor);
    for j=1:end_neighnumber
        neighU_item{j}=find(score_matrix(j,:)~=0);%����ڿ����ĵ�Ӱ    
    end
    targetU_item=find(score_matrix(user_ID,:)~=0);%Ŀ���û������ĵ�Ӱ
    set=neighU_item{1};
    for k=2:end_neighnumber
    set=union(set,neighU_item{k});%�ھ��û����������еĵ�ӰID
    end
    recommend_item{i}=setdiff(set,targetU_item);%Ԥ�Ƽ�����ĿID
    recommend_num=length(recommend_item{i});%Ԥ�Ƽ�����Ŀ��
    predict_recom{i}=zeros(recommend_num,1);%�����û�i�Ƽ�����Ŀ��Ԥ�����
    %% Ԥ���Ƽ�����Ŀ������
    aver_scoret=mean(score_matrix(user_ID,targetU_item));%ƽ����
    for t=1:recommend_num
        sum3=0;%����
        sum4=0;%��ĸ
      for l=1:end_neighnumber
          if isempty(best_neibor)
             sum3=0;
          else
             BBtest=(score_matrix(best_neibor(l),:)~=0);%����ڿ�������ĿID
             a_scoret(l)=mean(score_matrix(best_neibor(l),BBtest));%����ڵ�ƽ����
          sum3=sum3+Tsim_matrix1(incom_number,index0(l))*...
            (score_matrix(best_neibor(l),recommend_item{i}(t))-a_scoret(l));
          sum4=sum4+Tsim_matrix1(incom_number,index0(l));
          end
      end
      if sum4==0   
        predict_recom{i}(t)=(aver_scoret); %�ų���ĸΪ������
      else
        predict_recom{i}(t)=(aver_scoret+(sum3/sum4));
      end
      if predict_recom{i}(t)>5
        predict_recom{i}(t)=5;
      elseif predict_recom{i}(t)<1
        predict_recom{i}(t)=1;
      elseif isnan(predict_recom{i}(t))
        predict_recom{i}(t)=(aver_scoret);
      end
    end
   [itemscore_Rank,index3]=sort(predict_recom{i},'descend');%�Ƽ���Ŀ������
   if recommend_num>=recommend_number
       Frecommend_item{i}=recommend_item{i}(index3(1:recommend_number));%�û�i����ʵ�Ƽ���ĿID
   else 
       Frecommend_item{i}=recommend_item{i};
   end
   Frecommend_number{i}=length(Frecommend_item{i});
end
%% ָ�����  
Precision_u=zeros(number_user1,1);
MRR_u=zeros(number_user1,1);
Recall_u=zeros(number_user1,1);
Hit_u=zeros(number_user1,1);
NDCG_u=zeros(number_user1,1);
for i=1:number_user1
    user_ID=test__user(i);%�û�ID
    usert_itemrow=find(u2(:,1)==user_ID);%�û���Ӧ������
    usert_item=u2(usert_itemrow,2);%�û���ʵ��������Ŀ����
    true_item=intersect(usert_item,Frecommend_item{i});%�Ƽ��Ե���Ŀ
    true_number=length(true_item);%�Ƽ��Ե���Ŀ��
    %% Precision@K 
    Precision_u(i,1)=true_number/Frecommend_number{i};%�Ƽ��б����˼��� 
    %% Recall@K
    Recall_u(i,1)=true_number/length(usert_item);%��ʵ���б����˼���
    %% MRR 
    row_data=u2(usert_itemrow,:);
    [u2score_Rank,index4]=sort(Predict_score(usert_itemrow,1),'descend');%���Լ��е���ʵ��Ŀ��Ԥ����������
    maxscore_item=row_data(index4(1),2);%������һ����Ŀ
    if ismember(maxscore_item,Frecommend_item{i})
        MRR_u(i,1)=1/find(Frecommend_item{i}==maxscore_item);
    else
        MRR_u(i,1)=0;
    end
    %% Hitrate
    if isempty(true_item)
        Hit_u(i,1)=0;
    else
        Hit_u(i,1)=1;
    end
    %% NDCG
    min_number=min(recommend_number,length(usert_item));
    Z_u=0;
    for g=1:min_number
        Z_u=Z_u+(1/log2(g+1));
    end
    DCG_u=0;
    for f=1:Frecommend_number{i}
        if ismember(Frecommend_item{i}(f),usert_item)
            derta=1;
        else
            derta=0;
        end
        DCG_u=DCG_u+((2^derta-1)/log2(f+1));
    end
    NDCG_u(i,1)=DCG_u/Z_u;
end
Precision_K=mean(Precision_u);    
Recall_K=mean(Recall_u); 
MRR_K=mean(MRR_u); 
Hit_K=mean(Hit_u); 
NDCG_K=mean(NDCG_u);   
F_K=2*Recall_K*Precision_K/(Recall_K+Precision_K);
end
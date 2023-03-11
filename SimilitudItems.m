function [D]=SimilitudItems(data,method)
%INPUT:
%data: rows should be observations while colunms will be variables
%method: metric
%copyright (c) 2010 CONCHA.
%concha.gong@gmail.com
%nl_d=0.9;
D=zeros(size(data,2),size(data,2));
data1=data;
data1(data1~=0)=1;
degree=sum(data1,2);
switch lower(method)
    case 'cosine'
        for i=1:size(data,2)
            for j=i+1:size(data,2)
                D(i,j)=data(:,i)'*data(:,j)/(norm(data(:,i),2)*norm(data(:,j),2));
            end
            if isnan(D(i,j))
                D(i,j)=0;
            end
            D(i,j)=abs(D(i,j));
        end
    case 'correlation'
        for i=1:size(data,2)
            for j=i+1:size(data,2)
                temp=find(data(:,i)~=0 & data(:,j)~=0);
                Rui=data(temp,i);
                Ruj=data(temp,j);
                Ri=mean(data(:,i));
                Rj=mean(data(:,j));
                D(i,j)=(Rui-Ri)'*(Ruj-Rj)/(norm(Rui-Ri)*norm(Ruj-Rj));
                if isnan(D(i,j))
                    D(i,j)=0;
                end
                D(i,j)=abs(D(i,j));
            end
        end
    case 'adjustedcosine'
        for i=1:size(data,2)
            for j=i+1:size(data,2)
                temp=find(data(:,i)~=0 & data(:,j)~=0);
                Rui=data(temp,i);
                Ruj=data(temp,j);
                Ru=mean(data(temp,:)')';
                D(i,j)=(Rui-Ru)'*(Ruj-Ru)/(norm(Rui-Ru)*norm(Ruj-Ru));
                if isnan(D(i,j))
                    D(i,j)=0;
                end
                D(i,j)=abs(D(i,j));
            end
        end
        case 'ducosine'
        for i=1:size(data,2)
            for j=i+1:size(data,2)
                degree(i,1)=sum(data1(:,i));%列和
                degree(j,1)=sum(data1(:,j));
                D(i,j)=data(:,i)'*data(:,j)/(norm(data(:,i),2)*norm(data(:,j),2)*degree(i,1)*degree(j,1));
            end
            if isnan(D(i,j))
                D(i,j)=0;
            end
            D(i,j)=abs(D(i,j));
        end
        case 'commondu'
        for i=1:size(data,2)
            for j=i+1:size(data,2)
                temp1=find(data(:,i)>=3);
                temp2=find(data(:,j)>=3);%找到每个超边里的喜欢的项目
                temp3=find(data(:,i)<3 & data(:,i)>0);
                temp4=find(data(:,j)<3 & data(:,j)>0);%找到每个超边里的不喜欢的项目
                com_like=intersect(temp1,temp2);
                com_notlike=intersect(temp3,temp4);
                like(i,j)=sum(1./degree(com_like));
                notlike(i,j)=sum(1./degree(com_notlike));
              D(i,j)=notlike(i,j)+like(i,j);
             %D(i,j)=nl_d*notlike(i,j)+(1-nl_d)*like(i,j);
            end
            if isnan(D(i,j))
                D(i,j)=0;
            end
            D(i,j)=abs(D(i,j));
        end
        case 'singledu'
        for i=1:size(data,2)
            for j=i+1:size(data,2)
                temp1=find(data(:,i)~=0);
                temp2=find(data(:,j)~=0);%找到每个超边里的喜欢的项目
                com_rate=intersect(temp1,temp2);
                D(i,j)=sum(1./degree(com_rate));
            end
            if isnan(D(i,j))
                D(i,j)=0;
            end
            D(i,j)=abs(D(i,j));
        end
        case 'commonducosine'
          for i=1:size(data,2)
            for j=i+1:size(data,2)
                temp1=find(data(:,i)>=3);
                temp2=find(data(:,j)>=3);%找到每个超边里的喜欢的项目
                temp3=find(data(:,i)<3 & data(:,i)>0);
                temp4=find(data(:,j)<3 & data(:,j)>0);%找到每个超边里的不喜欢的项目
                com_like=intersect(temp1,temp2);
                com_notlike=intersect(temp3,temp4);
                like1(i,j)=sum(1./degree(com_like));
                like2(i,j)=data(com_like,i)*data(com_like,j)'/...
                 (norm(data(com_like,i),2)*norm(data(both_like,j),2));
                notlike1(i,j)=sum(1./degree(com_notlike));
                notlike2(i,j)=data(com_notlike,i)*data(com_notlike,j)'/...
                 (norm(data(com_notlike,i),2)*norm(data(com_notlike,j),2));
              D(i,j)=notlike2(i,j)*notlike1(i,j)+like2(i,j)*like1(i,j);
             %D(i,j)=nl_d*notlike(i,j)+(1-nl_d)*like(i,j);
            end
          end
            if isnan(D(i,j))
                D(i,j)=0;
            end
               D(i,j)=abs(D(i,j));    
end
D=D'+D;
            
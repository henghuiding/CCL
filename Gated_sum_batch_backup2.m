function [h, dzdw1, dzdw2] = Gated_sum_batch_backup2(X, w1, w2, DzDy)
[m, n, k, batch] = size(X{1});

num=size(X,2);
hh=zeros(m,n,num,batch,'single');%relative confidence
w1 = gather(w1);
w2 = gather(w2);

gates=cell(1,num);
gates1=cell(1,num);
gates2=cell(1,num);
gates3=cell(1,num);
x=cell(1,num);
% pred=cell(1,num);
for i=1:num
    x{i}=gather(X{i});
    E = exp(bsxfun(@minus, x{i}, max(x{i},[],3))) ;
    L = sum(E,3) ;
    Y = bsxfun(@rdivide, E, L) ;% SoftMax
    %   [gates{i},pred{i}]=max(gather(X{i+num}),[],3);
    %   gates{i}=bsxfun(@power,1-gates{i},-1);
    tmp=sort(Y,3);
    tmpMax=tmp(:,:,end,:);
    tmpSed=tmp(:,:,end-1,:);
    if i==1
        if max(tmp(:))<0.5
            gates1{i}=1;
            gates2{i}=1;
            hh(:,:,i,:)=tmpMax;
        else
            gates1{i}=tmpMax;
            gates1{i}(tmpMax<0.5)=0.5;
            gates1{i}=bsxfun(@rdivide,0.5,1-gates1{i});
            gates1{i}=bsxfun(@power,gates1{i},2);
            gates2{i}=tmpMax-tmpSed;
            gates2{i}(gates2{i}<0.4)=0.4;
            gates2{i}=bsxfun(@rdivide,0.6,1-gates2{i});
            gates2{i}=bsxfun(@power, gates2{i}, 2);
            tmpMax(tmpMax<0.1)=0.1;
            tmpMax(tmpMax>0.5)=0.5;
            hh(:,:,i,:)=tmpMax;
        end
    else
        if max(tmp(:))<0.3
            gates1{i}=1;
            gates2{i}=1;
            hh(:,:,i,:)=tmpMax;
        else
            gates1{i}=tmpMax;
            gates1{i}(tmpMax<0.3)=0.3;
            gates1{i}=bsxfun(@rdivide,0.7,1-gates1{i});
            gates1{i}=bsxfun(@power,gates1{i},2);
            gates2{i}=tmpMax-tmpSed;
            gates2{i}(gates2{i}<0.05)=0.05;
            gates2{i}=bsxfun(@rdivide,0.95,1-gates2{i});
            gates2{i}=bsxfun(@power, gates2{i}, 2);
            tmpMax(tmpMax<0.1)=0.1;
            tmpMax(tmpMax>0.5)=0.5;
            hh(:,:,i,:)=tmpMax;
        end
    end
    gates1{i}(gates1{i}>5)=5;
    gates2{i}(gates2{i}>5)=5;
end
E = exp(bsxfun(@minus, hh, max(hh,[],3))) ;
L = sum(E,3) ;
Y = bsxfun(@rdivide, E, L) ;
for i=1:num
    gates3{i}=Y(:,:,i,:);
    gates3{i}=bsxfun(@power, gates3{i}, 0.5);
    gates3{i}(gates3{i}>5)=5;
%     end
    gates{i}=w2(i,1)*gates1{i}+w2(i,2)*gates2{i}+w2(i,3)*gates3{i};
%     if max(gates{i}(:))>10
%         fprintf('gates:%d,i=%d\n',max(gates{i}(:)),i);
%     end
    gates{i}(gates{i}<0.5)=0.5;
    gates{i}(gates{i}>5)=5;
    x{i}=permute(x{i}, [3 4 1 2]);
    gates{i}=permute(gates{i}, [3 4 1 2]);
end
% gates{1}=1;
% nh = k;
%% forward pass
if nargin == 3
%     h=zeros(k,batch,m,n,'single');
    h=bsxfun(@times,gates{1},bsxfun(@times,x{1},w1(:,:,1)))+bsxfun(@times,gates{2},bsxfun(@times,x{2},w1(:,:,2)));
    for ii=3:num
        h=bsxfun(@times,gates{i},bsxfun(@times,x{ii},w1(:,:,ii)))+h;
    end

h=permute(h, [3 4 1 2]);
h = gpuArray(h);
end

%% backward pass
if nargin > 3

        dzdy=gather(DzDy);
        dzdy=permute(dzdy, [3 4 1 2]);
        dzdw1=cell(1,num);
        dzdw2=zeros(num,3);
%         Dgate=zeros(k,batch,m,n);
    for ii=1:num
        h{ii}=bsxfun(@times,gates{ii},bsxfun(@times,dzdy,w1(:,:,ii)));
        h{ii}=permute(h{ii},[3 4 1 2]);
        h{ii}=gpuArray(h{ii});
        
        Dgate=bsxfun(@times,w1(:,:,ii),bsxfun(@times,dzdy,x{ii}));
        Dgate=permute(Dgate,[3,4,1,2]);
        dzdw2(ii,1)=sum(mean(mean(mean(bsxfun(@times,Dgate,gates1{ii}),1),2),3),4);
        dzdw2(ii,2)=sum(mean(mean(mean(bsxfun(@times,Dgate,gates2{ii}),1),2),3),4);
        dzdw2(ii,3)=sum(mean(mean(mean(bsxfun(@times,Dgate,gates3{ii}),1),2),3),4);
        dzdw1{ii}=sum(mean(mean(permute(bsxfun(@times,gates{ii},bsxfun(@times,dzdy,x{ii})),[3,4,1,2]),1),2),4);
        dzdw1{ii}=reshape(dzdw1{ii},size(dzdw1{ii},3),1);
    end
    dzdw1=cat(3,dzdw1{:});
    dzdw1=gpuArray(dzdw1);
    dzdw2=gpuArray(dzdw2);

end
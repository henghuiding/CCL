function [h, dzdw] = Gated_sum_batch_backup(X, w, DzDy)
num=size(X,2)/2;
% gate=(num-1)*gather(X{num});
gates=cell(1,num);
x=cell(1,num);
% pred=cell(1,num);
for i=1:num
    x{i}=gather(X{i});
    %   [gates{i},pred{i}]=max(gather(X{i+num}),[],3);
    tmp=sort(gather(X{i+num}),3);   
    gates{i}=tmp(:,:,end,:)-tmp(:,:,end-1,:);
    if i==1
        if max(tmp(:))<0.3
            gates{i}=1;
        else
            gates{i}(gates{i}<0.38)=0.38;
            gates{i}= bsxfun(@rdivide,0.6,1-gates{i});
            gates{i}= bsxfun(@power, gates{i}, 2);
        end
        
    else
        if max(tmp(:))<0.05
            gates{i}=1;
        else
            gates{i}(gates{i}<0.05)=0.05;
            gates{i}= bsxfun(@rdivide,0.95,1-gates{i});
            gates{i}= bsxfun(@power, gates{i}, 5);          
        end
    end
    
    %   max(gates{i}(:))
    gates{i}(gates{i}<0.7)=0.7;
    gates{i}(gates{i}>10)=10;
    x{i}=permute(x{i}, [3 4 1 2]);
    gates{i}=permute(gates{i}, [3 4 1 2]);
end
[m, n, k, batch] = size(X{1});
% gates{1}=1;
w = gather(w);

nh = k;
%% forward pass
if nargin == 2
%     h=zeros(m,n,k,batch,'single');
    h=bsxfun(@times,gates{1},bsxfun(@times,x{1},w(:,:,1)))+bsxfun(@times,gates{2},bsxfun(@times,x{2},w(:,:,2)));
    for ii=3:num
        h=bsxfun(@times,gates{i},bsxfun(@times,x{ii},w(:,:,ii)))+h;
    end

h=permute(h, [3 4 1 2]);
h = gpuArray(h);
end

%% backward pass
if nargin > 2

        dzdy=gather(DzDy);
        dzdy=permute(dzdy, [3 4 1 2]);
        dzdw=cell(1,num-1);
        Dgate=zeros(k,batch,m,n);
    for ii=1:num
        h{ii}=bsxfun(@times,gates{ii},bsxfun(@times,dzdy,w(:,:,ii)));
        h{ii}=permute(h{ii},[3 4 1 2]);
        h{ii}=gpuArray(h{ii});
        
        h{ii+num}=Dgate;
%         h{ii+num}(pred{ii})=mean(bsxfun(@times,x{ii},bsxfun(@times,dzdy,w(:,:,ii))),1);
        h{ii+num}=permute(h{ii},[3 4 1 2]);
        h{ii+num}=gpuArray(h{ii});
        
        dzdw{ii}=sum(mean(mean(permute(bsxfun(@times,gates{ii},bsxfun(@times,dzdy,x{ii})),[3,4,1,2]),1),2),4);
        dzdw{ii}=reshape(dzdw{ii},size(dzdw{ii},3),1);
    end
    dzdw=cat(3,dzdw{:});
    dzdw=gpuArray(dzdw);

end
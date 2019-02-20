function [h, dzdw] = LabelConflict_batch_very_slowly(X, w, DzDy)
x=gather(X{1});
[m, n, k, batch] = size(X{1});
w = gather(w);

nClass = k;
%% forward pass
if nargin == 2
     h=zeros(m,n,k,batch,'single');
    for i=1:nClass
        h(:,:,i,:)=x(:,:,i,:);
        for j=1:nClass
            if i~=j
                h(:,:,i,:)=h(:,:,i,:)+x(:,:,j,:)*w(i,j);
            end
        end
    end
    h = gpuArray(h);
end

%% backward pass
if nargin > 2

    dzdy=gather(DzDy);
    dzdw=zeros(nClass,nClass,'single');
    for i=1:nClass
        h(:,:,i,:)=dzdy(:,:,i,:);
        for j=1:nClass
            if i~=j
                h(:,:,i,:)=h(:,:,i,:)+dzdy(:,:,j,:)*w(j,i);
                dzdw(i,j)=sum(mean(mean(bsxfun(@times,dzdy(:,:,i,:),x(:,:,j,:)),1),2));
            end
        end
    end
    h=gpuArray(h);
    dzdw=gpuArray(dzdw);
end
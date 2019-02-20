function [h, dzdw] = LabelConflict_batch_backup(X, w, DzDy)
x=gather(X{1});
x1=permute(x, [3 4 1 2]);
[m, n, k, batch] = size(X{1});
w = gather(w);

nClass = k;
h=zeros(k,batch,m,n,'single');
% w=w-diag(diag(w))+diag(ones(size(w,1),1));
w(logical(eye(size(w)))) = 1;
%% forward pass
if nargin == 2
    for i=1:nClass/7
        h(i,:,:,:)=sum(bsxfun(@times,w(i,:)',x1),1);
        h(i+3,:,:,:)=sum(bsxfun(@times,w(i+3,:)',x1),1);
        h(i+6,:,:,:)=sum(bsxfun(@times,w(i+6,:)',x1),1);
        h(i+9,:,:,:)=sum(bsxfun(@times,w(i+9,:)',x1),1);
        h(i+12,:,:,:)=sum(bsxfun(@times,w(i+12,:)',x1),1);
        h(i+15,:,:,:)=sum(bsxfun(@times,w(i+15,:)',x1),1);
        h(i+18,:,:,:)=sum(bsxfun(@times,w(i+18,:)',x1),1);
    end
    h=permute(h,[3 4 1 2]);
    h = gpuArray(h);
end

%% backward pass
if nargin > 2

    dzdy=gather(DzDy);
    dzdy1=permute(dzdy, [3 4 1 2]);
    dzdw=zeros(nClass,nClass,'single');
    for i=1:nClass/7
        h(i,:,:,:)=sum(bsxfun(@times,w(:,i),dzdy1),1);
        h(i+3,:,:,:)=sum(bsxfun(@times,w(:,i+3),dzdy1),1);
        h(i+6,:,:,:)=sum(bsxfun(@times,w(:,i+6),dzdy1),1);
        h(i+9,:,:,:)=sum(bsxfun(@times,w(:,i+9),dzdy1),1);
        h(i+12,:,:,:)=sum(bsxfun(@times,w(:,i+12),dzdy1),1);
        h(i+15,:,:,:)=sum(bsxfun(@times,w(:,i+15),dzdy1),1);
        h(i+18,:,:,:)=sum(bsxfun(@times,w(:,i+18),dzdy1),1);
        for j=1:nClass
            if i~=j
%                 h(:,:,i,:)=h(:,:,i,:)+dzdy(:,:,j,:)*w(j,i);%wrong one h(:,:,i,:)=h(:,:,i,:)+dzdy(:,:,j,:)*w(i,j);
                dzdw(i,j)=sum(mean(mean(bsxfun(@times,dzdy(:,:,i,:),x(:,:,j,:)),1),2));
                dzdw(i+3,j)=sum(mean(mean(bsxfun(@times,dzdy(:,:,i+3,:),x(:,:,j,:)),1),2));
                dzdw(i+6,j)=sum(mean(mean(bsxfun(@times,dzdy(:,:,i+6,:),x(:,:,j,:)),1),2));
                dzdw(i+9,j)=sum(mean(mean(bsxfun(@times,dzdy(:,:,i+9,:),x(:,:,j,:)),1),2));
                dzdw(i+12,j)=sum(mean(mean(bsxfun(@times,dzdy(:,:,i+12,:),x(:,:,j,:)),1),2));
                dzdw(i+15,j)=sum(mean(mean(bsxfun(@times,dzdy(:,:,i+15,:),x(:,:,j,:)),1),2));
                dzdw(i+18,j)=sum(mean(mean(bsxfun(@times,dzdy(:,:,i+18,:),x(:,:,j,:)),1),2));
            end
        end
    end
    h=permute(h,[3 4 1 2]);
    h=gpuArray(h);
    dzdw=gpuArray(dzdw);
end
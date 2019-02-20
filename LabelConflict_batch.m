function [h, dzdw] = LabelConflict_batch(X, w, DzDy)
x=gather(X{1});
x1=permute(x, [3 4 1 2]);
[m, n, k, batch] = size(X{1});
w = gather(w);

nClass = k;
h=zeros(k,batch,m,n,'single');
% w(logical(eye(size(w)))) = 1;
%% forward pass
if nargin == 2
    for i=1:6
        h(i,:,:,:)=sum(bsxfun(@times,w(i,:)',x1),1);
        h(i+6,:,:,:)=sum(bsxfun(@times,w(i+6,:)',x1),1);
        h(i+12,:,:,:)=sum(bsxfun(@times,w(i+12,:)',x1),1);
        h(i+18,:,:,:)=sum(bsxfun(@times,w(i+18,:)',x1),1);
        h(i+24,:,:,:)=sum(bsxfun(@times,w(i+24,:)',x1),1);
        h(i+30,:,:,:)=sum(bsxfun(@times,w(i+30,:)',x1),1);
        h(i+36,:,:,:)=sum(bsxfun(@times,w(i+36,:)',x1),1);
        h(i+42,:,:,:)=sum(bsxfun(@times,w(i+42,:)',x1),1);
        h(i+48,:,:,:)=sum(bsxfun(@times,w(i+48,:)',x1),1);
        if i~=6
        h(i+54,:,:,:)=sum(bsxfun(@times,w(i+54,:)',x1),1);
        end
    end
    h=permute(h,[3 4 1 2]);
    h = gpuArray(h);
end

%% backward pass
if nargin > 2

    dzdy=gather(DzDy);
    dzdy1=permute(dzdy, [3 4 1 2]);
    dzdw=zeros(nClass,nClass,'single');
    for i=1:6
        h(i,:,:,:)=sum(bsxfun(@times,w(:,i),dzdy1),1);
        h(i+6,:,:,:)=sum(bsxfun(@times,w(:,i+6),dzdy1),1);
        h(i+12,:,:,:)=sum(bsxfun(@times,w(:,i+12),dzdy1),1);
        h(i+18,:,:,:)=sum(bsxfun(@times,w(:,i+18),dzdy1),1);
        h(i+24,:,:,:)=sum(bsxfun(@times,w(:,i+24),dzdy1),1);
        h(i+30,:,:,:)=sum(bsxfun(@times,w(:,i+30),dzdy1),1);
        h(i+36,:,:,:)=sum(bsxfun(@times,w(:,i+36),dzdy1),1);        
        h(i+42,:,:,:)=sum(bsxfun(@times,w(:,i+42),dzdy1),1);
        h(i+48,:,:,:)=sum(bsxfun(@times,w(:,i+48),dzdy1),1);
        if i~=6
        h(i+54,:,:,:)=sum(bsxfun(@times,w(:,i+54),dzdy1),1);
        end
        for j=1:nClass
                dzdw(i,j)=sum(mean(mean(bsxfun(@times,dzdy(:,:,i,:),x(:,:,j,:)),1),2));
                dzdw(i+6,j)=sum(mean(mean(bsxfun(@times,dzdy(:,:,i+6,:),x(:,:,j,:)),1),2));
                dzdw(i+12,j)=sum(mean(mean(bsxfun(@times,dzdy(:,:,i+12,:),x(:,:,j,:)),1),2));
                dzdw(i+18,j)=sum(mean(mean(bsxfun(@times,dzdy(:,:,i+18,:),x(:,:,j,:)),1),2));
                dzdw(i+24,j)=sum(mean(mean(bsxfun(@times,dzdy(:,:,i+24,:),x(:,:,j,:)),1),2));
                dzdw(i+30,j)=sum(mean(mean(bsxfun(@times,dzdy(:,:,i+30,:),x(:,:,j,:)),1),2));
                dzdw(i+36,j)=sum(mean(mean(bsxfun(@times,dzdy(:,:,i+36,:),x(:,:,j,:)),1),2));                
                dzdw(i+42,j)=sum(mean(mean(bsxfun(@times,dzdy(:,:,i+42,:),x(:,:,j,:)),1),2));
                dzdw(i+48,j)=sum(mean(mean(bsxfun(@times,dzdy(:,:,i+48,:),x(:,:,j,:)),1),2));
                if i~=6
                dzdw(i+54,j)=sum(mean(mean(bsxfun(@times,dzdy(:,:,i+54,:),x(:,:,j,:)),1),2));
                end
        end
    end
    h=permute(h,[3 4 1 2]);
    h=gpuArray(h);
    dzdw=gpuArray(dzdw);
end
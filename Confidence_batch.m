function [Y] = Confidence_batch(X, DzDy)
[W,H,nClass,Batch]=size(X{1});
num=size(X,2);
h=zeros(W,H,num,Batch);
for i=1:size(X,2)
    h(:,:,i,:)=max(gather(X{i}),[],3);
end
E = exp(bsxfun(@minus, h, max(h,[],3))) ;
L = sum(E,3) ;
Y = bsxfun(@rdivide, E, L) ;
%% forward pass
if nargin == 1
    Y = gpuArray(Y);
end

%% backward pass
if nargin > 1   
   Y=gpuArray([]);
end
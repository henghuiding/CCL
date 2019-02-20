function [h] = Gated_sum_batch(X, DzDy)

num=size(X,2);

%% forward pass
if nargin == 1
    h=bsxfun(@times,X{1},X{num}(:,:,1,:))+bsxfun(@times,X{2},X{num}(:,:,2,:))+bsxfun(@times,X{3},X{num}(:,:,3,:))+bsxfun(@times,X{4},X{num}(:,:,4,:));
    for i=5:num-1
        h=bsxfun(@times,X{i},X{num}(:,:,i,:))+h;
    end
end

%% backward pass
if nargin > 1
    for i=1:num-1
        h{i}=bsxfun(@times,DzDy,X{num}(:,:,1,:));
        h{num}(:,:,i,:)=sum(bsxfun(@times,DzDy,X{i}),3);
    end

end

end
function [h] = Gated_sum_batch_2(X, DzDy)

num=size(X,2)-1;
mask=X{num+1};

for ii=2:num-1
    X{ii} = bsxfun(@times, X{ii}, mask);
end
%% forward pass
if nargin == 1
    if num>=4
        h=bsxfun(@times,X{1},X{num}(:,:,1,:))+bsxfun(@times,X{2},X{num}(:,:,2,:))+bsxfun(@times,X{3},X{num}(:,:,3,:))+bsxfun(@times,X{4},X{num}(:,:,4,:));
        for i=5:num-1
            h=bsxfun(@times,X{i},X{num}(:,:,i,:))+h;
        end
    else
        h=bsxfun(@times,X{1},X{num}(:,:,1,:));
        for i=2:num-1
            h=bsxfun(@times,X{i},X{num}(:,:,i,:))+h;
        end
    end
end

%% backward pass
if nargin > 1
    for i=1:num-1
        h{i}=bsxfun(@times,DzDy,X{num}(:,:,1,:));
        if i>1
            h{i}=bsxfun(@times,h{i},mask);
        end
        h{num}(:,:,i,:)=mean(bsxfun(@times,DzDy,X{i}),3);
    end
    h{num+1}=0*mask;
end

end

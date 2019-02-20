function [h, dzdw] = Gated_sum_batch2(X, w, DzDy)

num=size(X,2);
X{num}=bsxfun(@times,(num-1),X{num});
X{num}=bsxfun(@times,X{num},w);

%% forward pass
if nargin == 2
    h=bsxfun(@times,X{1},X{num}(:,:,1,:))+bsxfun(@times,X{2},X{num}(:,:,2,:))+bsxfun(@times,X{3},X{num}(:,:,3,:))+bsxfun(@times,X{4},X{num}(:,:,4,:));
    for i=5:num-1
        h=bsxfun(@times,X{i},X{num}(:,:,i,:))+h;
    end
end

%% backward pass
if nargin > 2
    for i=1:num-1
        h{i}=bsxfun(@times,DzDy,X{num}(:,:,1,:));
        h{num}(:,:,i,:)=mean(bsxfun(@times,DzDy,X{i}),3);
    end  
  dzdw=sum(mean(mean(bsxfun(@times,h{num},X{num}),1),2),4);
  h{num}=bsxfun(@times,h{num},w);
  h{num}=bsxfun(@times,(num-1),h{num});
end
end

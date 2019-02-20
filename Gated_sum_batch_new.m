function [h, dzdw1, dzdw2] = Gated_sum_batch_new(X, w1, w2, DzDy)
[m, n, k, batch] = size(X{1});

num=size(X,2)/2;
% hh=zeros(m,n,num,batch,'single');%relative confidence
w1 = gather(w1);
w2 = gather(w2);

gates=cell(1,num);
gates1=cell(1,num);
gates2=cell(1,num);
ci1=cell(1,num);
ci2=cell(1,num);
tmpMax=cell(1,num);
tmpSed=cell(1,num);
x=cell(1,num);
% pred=cell(1,num);
numPixelsPerImage = m*n ;
numPixels = numPixelsPerImage * batch ;
imageVolume = numPixelsPerImage * k ;

nn = reshape(0:numPixels-1,[m,n,1,batch]) ;
offset = 1 + mod(nn, numPixelsPerImage) + ...
    imageVolume * fix(nn / numPixelsPerImage) ;

for i=1:num
    x{i}=gather(X{i});
    Y=gather(X{i+num}); 
%     E = exp(bsxfun(@minus, x{i}, max(x{i},[],3))) ;
%     L = sum(E,3) ;
%     Y{i} = bsxfun(@rdivide, E, L) ;% SoftMax
    [tmpMax{i}, pred1]=max(Y,[],3);
    ci1{i} = offset + numPixelsPerImage * max(pred1 - 1,0) ;
    Y(ci1{i})=0;
    [tmpSed{i}, pred2]=max(Y,[],3);
    ci2{i} = offset + numPixelsPerImage * max(pred2 - 1,0) ;
    gates1{i}=exp(tmpMax{i})*0.5;
    gates2{i}=exp(tmpMax{i}-tmpSed{i})*0.5;
end
for i=1:num
    gates{i}=w2(i,1)*gates1{i}+w2(i,2)*gates2{i};
    if max(gates{i}(:))>3
        fprintf('%dth gates max waring:%d\n',i,max(gates{i}(:)))
    end
    x{i}=permute(x{i}, [3 4 1 2]);
    gates{i}=permute(gates{i}, [3 4 1 2]);
    gates{i}=1;
end

%% forward pass
if nargin == 3
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
        dzdw2=zeros(num,2);
        D=zeros(m,n,k,batch);
        DD=ones(m,n,k,batch);
        Dgates1=cell(1,num);
        Dgates2=cell(1,num);
%         Dgate=zeros(k,batch,m,n);
    for ii=1:num
        h{ii}=bsxfun(@times,gates{ii},bsxfun(@times,dzdy,w1(:,:,ii)));
        h{ii}=permute(h{ii},[3 4 1 2]);
        h{ii}=gpuArray(h{ii});
        
        Dgate=bsxfun(@times,w1(:,:,ii),bsxfun(@times,dzdy,x{ii}));
        Dgate=permute(Dgate,[3,4,1,2]);
        
        
        dzdw2(ii,1)=sum(mean(mean(mean(bsxfun(@times,Dgate,gates1{ii}),1),2),3),4);
        Dgates1{ii}=bsxfun(@times,Dgate,w2(ii,1));
        Dgates1{ii}=sum(bsxfun(@times,Dgates1{ii},gates1{ii}),3);
        Dgates1{ii}=bsxfun(@times,Dgates1{ii},DD);
        htmp=D;
        htmp(ci1{ii})=1;
        htmp=bsxfun(@times,htmp,Dgates1{ii});
    
        dzdw2(ii,2)=sum(mean(mean(mean(bsxfun(@times,Dgate,gates2{ii}),1),2),3),4);
        Dgates2{ii}=bsxfun(@times,Dgate,w2(ii,2));
        Dgates2{ii}=sum(bsxfun(@times,Dgates2{ii},gates1{ii}),3);
        Dgates2{ii}=bsxfun(@times,Dgates2{ii},DD);
        
        h{ii+num}=D;
        h{ii+num}(ci1{ii})=1;
        h{ii+num}(ci2{ii})=-1;
        h{ii+num}=bsxfun(@times,h{ii+num},Dgates2{ii})+htmp;
        h{ii+num}=gpuArray(h{ii+num});
%         h{ii+num}=h{ii+num}*0;
        if max(h{ii+num}(:))>3
            fprintf('%dth max waring:%d',i+num,max(h{ii+num}(:)))
        end
        dzdw1{ii}=sum(mean(mean(permute(bsxfun(@times,gates{ii},bsxfun(@times,dzdy,x{ii})),[3,4,1,2]),1),2),4);
        dzdw1{ii}=reshape(dzdw1{ii},size(dzdw1{ii},3),1);
    end
    dzdw1=cat(3,dzdw1{:});
    dzdw1=gpuArray(dzdw1);
    dzdw2=gpuArray(dzdw2);

end

end
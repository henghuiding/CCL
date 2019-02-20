function w = rnn_initialize(nh,num,Params)
% %% fine grained
% % SouthEast Plane
% w{1} = 1e-2*eye(nh, nh, 'single');
% w{2} = 1e-2*eye(nh, nh, 'single');
% w{3} = 1e-2*eye(nh, nh, 'single');
% 
% % NorthWest Plane 
% w{4} = 1e-2*eye(nh, nh, 'single');
% w{5} = 1e-2*eye(nh, nh, 'single');
% w{6} = 1e-2*eye(nh, nh, 'single');
% 
% % SouthWest Plane
% w{7} = 1e-2*eye(nh, nh, 'single');
% w{8} = 1e-2*eye(nh, nh, 'single');
% w{9} = 1e-2*eye(nh, nh, 'single');
% 
% % NorthEast Plane
% w{10} = 1e-2*eye(nh, nh, 'single');
% w{11} = 1e-2*eye(nh, nh, 'single');
% w{12} = 1e-2*eye(nh, nh, 'single');
% 
% w = cat(3, w{:});
for i=1:num
    if i==1
        ding=1.2;
    else
%         ding=(Params + (rand(1)-.6)/5);
        ding=Params;
    end
%     w{i}=ding+1e-2*randn(nh,1,'single');
    w{i}=ding*ones(nh,1,'single');
end
w=cat(3,w{:});



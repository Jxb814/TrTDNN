function [Tau_path,net_best,E,Eva] = STTD_train(net,data,method)
net_best = net;
Err_tr = cell(method.attempts,1);
Err_va = cell(method.attempts,1);
TT = cell(method.attempts,1);
WW1 = cell(method.attempts,1);
WW2 = cell(method.attempts,1);
B1 = cell(method.attempts,1);
B2 = cell(method.attempts,1);
st1 = net.nD;
st2 = st1+net.input_num*net.nN;
st3 = st2+net.nN*net.output_num;
st4 = st3+net.nN;
Nvars=net.nD+(net.input_num+1)*net.nN+(net.nN+1)*net.output_num;
Theta_path = cell(method.attempts,1);
for r = 1:method.attempts
    X_tr_tilde = zeros(net.input_num,data.batchsize);
    dX_tilde = zeros(net.input_num,data.batchsize);
    X_va_tilde = zeros(net.input_num,data.batchsize);
    X_tr_tilde_all = zeros(net.input_num,data.nbatches*data.batchsize);
    Y_tr_tilde_all = zeros(net.output_num,data.nbatches*data.batchsize);
    dX_tilde_all = zeros(net.input_num,data.nbatches*data.batchsize);
    
    err_va_min = Inf;
    v = 0;
    W1 = 2*rand(net.nN,net.input_num)-1;
    W2 = 2*rand(net.output_num,net.nN)-1;
    b1 = 0*rand(net.nN,1);
    b2 = 0*rand(net.output_num,1);
    tau = rand(1,net.nD);
    err_tr = NaN(method.epoch,1);
    err_va = NaN(method.epoch,1);
    Tau = NaN(method.epoch,net.nD);
    Theta = NaN(method.epoch,Nvars);
    for i = 1:method.epoch
        Tau(i,:) = tau;
        WW1{r} = W1;
        WW2{r} = W2;
        B1{r} = b1;
        B2{r} = b2;
        B2{r} = b2;
        sigma = round(tau/data.dt);
        
        % check validation error
        for ii = 1:net.nD
            X_va_tilde(ii,:)= data.X_va_tilde(net.shiftlimit+1-sigma(ii):end-sigma(ii));
        end
        S1 = W1*X_va_tilde+b1;
        X2 = net.f(S1);
        Yhat_va_tilde = W2*X2+b2;  % prediction       
        err_va(i) = sqrt(sum((Yhat_va_tilde-data.Y_va_tilde).^2)/data.batchsize);
        
        if strcmp(method.alg,'MGD')||strcmp(method.alg,'MLM')
            k = randi([1 data.nbatches],1,1);
            for ii = 1:net.nD
                X_tr_tilde(ii,:)= data.TDATA{k}(1,data.id_tr{k}+net.shiftlimit-sigma(ii));
%                 % this is using the scaled derivative of inputs between [0,1] 
%                 dX_tilde(ii,:) = data.TDATA{k}(2,data.id_tr{k}+net.shiftlimit-sigma(ii));
                % we need the derivative of scaled input not between [0,1]
                dx = diff(data.TDATA{k}(1,data.id_tr{k}+net.shiftlimit-sigma(ii)))/data.dt;
                dX_tilde(ii,:) = [dx dx(end)];
            end
            S1 = W1*X_tr_tilde+b1;
            X2 = net.f(S1);
            Yhat_tr_tilde = W2*X2+b2;  % prediction
            if strcmp(method.alg,'MGD')
                dE = 2/data.batchsize*(Yhat_tr_tilde-data.Y_tr_tilde{k});
                g_b2 = sum(dE);
                g_W2 = X2*dE';
                db1 = W2'.*net.df(S1).*dE;
                g_b1 = sum(db1,2);
                g_W1 = db1*X_tr_tilde';
                g_t = -sum(W1.*(db1*dX_tilde'));
            else
                dE = sqrt(2/data.batchsize)*(Yhat_tr_tilde-data.Y_tr_tilde{k});
                db2 = ones(1,data.batchsize);
                dW2 = X2.*db2;
                db1 = W2'.*net.df(S1).*db2;
                for ii = 1:data.batchsize
                    dW1(:,ii)= reshape(db1(:,ii)*X_tr_tilde(:,ii)',[],1);
                    g_tau =sum(-W1.*(db1(:,ii)*dX_tilde(:,ii)'),1);
                    dtau(:,ii) = sum(reshape(g_tau(1:end),net.nD,net.states_num),2);
                end
                grad = [dtau;dW1;dW2;db1;db2]';
            end
            err_tr(i) = sqrt(sum((Yhat_tr_tilde-data.Y_tr_tilde{k}).^2)/data.batchsize);  % RMSE
        elseif strcmp(method.alg,'GD')||strcmp(method.alg,'LM')
            for k = 1:data.nbatches
                for ii = 1:net.nD
                    X_tr_tilde_all(ii,1+(k-1)*data.batchsize:k*data.batchsize)= data.TDATA{k}(1,data.id_tr{k}+net.shiftlimit-sigma(ii));
                    dX_tilde_all(ii,1+(k-1)*data.batchsize:k*data.batchsize) = data.TDATA{k}(2,data.id_tr{k}+net.shiftlimit-sigma(ii));
                end
                Y_tr_tilde_all(1+(k-1)*data.batchsize:k*data.batchsize)=data.Y_tr_tilde{k};
            end
            S1 = W1*X_tr_tilde_all+b1;
            X2 = net.f(S1);
            Yhat_tr_tilde_all = W2*X2+b2;  % prediction
            if strcmp(method.alg,'GD')
                dE = 2/data.batchsize/data.nbatches*(Yhat_tr_tilde_all-Y_tr_tilde_all);
                g_b2 = sum(dE);
                g_W2 = X2*dE';
                db1 = W2'.*net.df(S1).*dE;
                g_b1 = sum(db1,2);
                g_W1 = db1*X_tr_tilde_all';
                g_t = -sum(W1.*(db1*dX_tilde_all'));
            else
                dE = sqrt(2/data.nbatches/data.batchsize)*(Yhat_tr_tilde_all-Y_tr_tilde_all);
                db2 = ones(1,data.nbatches*data.batchsize);
                dW2 = X2.*db2;
                db1 = W2'.*net.df(S1).*db2;
                for ii = 1:data.nbatches*data.batchsize
                    dW1(:,ii)= reshape(db1(:,ii)*X_tr_tilde_all(:,ii)',[],1);
                    g_tau =sum(-W1.*(db1(:,ii)*dX_tilde_all(:,ii)'),1);
                    dtau(:,ii) = sum(reshape(g_tau(1:end),net.nD,net.states_num),2);
                end
                grad = [dtau;dW1;dW2;db1;db2]';
            end
            err_tr(i) = sqrt(sum((Yhat_tr_tilde_all-Y_tr_tilde_all).^2)/data.batchsize/data.nbatches);  % RMSE
        end
        
        if strcmp(method.alg,'GD')||strcmp(method.alg,'MGD')
            % Gradient Descent
            W1 = W1-method.a0*g_W1;
            W2 = W2-method.a0*g_W2';
            b1 = b1-method.a0*g_b1;
            b2 = b2-method.a0*g_b2;
            tau = max(tau-method.a0_s*g_t,0);
        elseif strcmp(method.alg,'LM')||strcmp(method.alg,'MLM')
            % LMA: tau,W1,W2,b1,b2
            theta = [tau,reshape(W1,1,[]),reshape(W2,1,[]),reshape(b1,1,[]),reshape(b2,1,[])]';
            Theta(i,:)=theta';
            theta = theta-(grad'*grad+method.mu*eye(Nvars))\grad'*dE';
            theta(1:net.nD) = max(theta(1:net.nD),0);
            tau = theta(1:net.nD)';
            W1 = reshape(theta(st1+1:st2),net.nN,net.input_num);
            W2 = reshape(theta(st2+1:st3),net.output_num,net.nN);
            b1 = theta(st3+1:st4);
            b2 = theta(st4+1:end);
        end
        
        if mod(i,method.gap)==1 && i>method.win+method.gap && mean(err_va(i-method.win:i))>=err_va_min
            v = v+1;
        elseif mod(i,method.gap)==1 && i>method.win+method.gap && mean(err_va(i-method.win:i))<err_va_min
            v = 0;
            err_va_min = mean(err_va(i-method.win+1:i));
        end
        if v>method.vset
            break;
        end
        if any(tau-data.dt*net.shiftlimit>=0)
            break;
        end
    end
    Err_tr{r} = err_tr(1:i);
    Err_va{r} = err_va(1:i);
    TT{r} = Tau(1:i,:);
    Theta_path{r} = Theta(1:i,:);
end

meanE = zeros(1,method.attempts);
for r = 1:method.attempts
    B = ~isnan(Err_tr{r});
    id = find(B,1,'last');
    if id>method.win
        meanE(r) =mean(Err_tr{r}(id-method.win+1:id));
    else
        meanE(r) =mean(Err_tr{r});
    end
end
[~,I] = min(meanE);
E = Err_tr{I};
Eva = Err_va{I};
Tau_path = TT{I};
net_best.W1 = WW1{I};
net_best.W2 = WW2{I};
net_best.b1 = B1{I};
net_best.b2 = B2{I};
net_best.tau = Tau_path(end,:);
net_best.theta_path=Theta_path{I};
end

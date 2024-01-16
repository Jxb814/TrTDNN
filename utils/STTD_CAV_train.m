function [Tau_path,net_best,E,Eva] = STTD_CAV_train(net,data,method)
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
for r = 1:method.attempts
    X_tr_tilde = zeros(net.input_num,data.tr_size);
    dX_tilde = zeros(net.input_num,data.tr_size);
    X_va_tilde = zeros(net.input_num,data.va_size);
    
    tr_in_par = data.tr_in;
    tr_dx_par = data.tr_dx_in;
    va_in_par = data.va_in;
    err_va_min = Inf;
    v = 0;
    W1 = 2*rand(net.nN,net.input_num)-1;
    W2 = 2*rand(net.output_num,net.nN)-1;
    b1 = 0*rand(net.nN,1);
    b2 = 0*rand(net.output_num,1);
    tau = 2*rand(1,net.nD);
    
    g_t = zeros(1,net.nD);
    dW1 = zeros(net.input_num*net.nN,data.tr_size);
    dtau = zeros(net.nD,data.tr_size);

    err_tr = NaN(1,method.epoch);
    err_va = NaN(1,method.epoch);
    Tau = NaN(method.epoch,net.nD);
    for i = 1:method.epoch
        Tau(i,:) = tau;
        WW1{r} = W1;
        WW2{r} = W2;
        B1{r} = b1;
        B2{r} = b2;
        sigma = round(tau/data.dt);
        
        % check validation error
        X_va_tilde(1,:) = va_in_par(2,net.shiftlimit+1:end);
        for ii = 1:net.nD
            X_va_tilde(2+net.states_num*(ii-1):1+net.states_num*ii,:) = va_in_par(:,net.shiftlimit-sigma(ii)+1:end-sigma(ii));
        end
        S1 = W1*X_va_tilde+b1;
        X2 = net.f(S1);
        Yhat_va_tilde = W2*X2+b2;  % prediction       
        err_va(i) = sqrt(sum((Yhat_va_tilde-data.va_out).^2)/data.va_size);
        
        % training error
        X_tr_tilde(1,:)=tr_in_par(2,net.shiftlimit+1:end);
        for ii = 1:net.nD
            X_tr_tilde(2+net.states_num*(ii-1):1+net.states_num*ii,:) = tr_in_par(:,net.shiftlimit-sigma(ii)+1:end-sigma(ii));
            dX_tilde(2+net.states_num*(ii-1):1+net.states_num*ii,:) = tr_dx_par(:,net.shiftlimit-sigma(ii)+1:end-sigma(ii));
        end
        S1 = W1*X_tr_tilde+b1;
        X2 = net.f(S1);
        Yhat_tr_tilde = W2*X2+b2;  % prediction
        
        if strcmp(method.alg,'GD')
            dE = 2/data.tr_size*(Yhat_tr_tilde-data.tr_out);
            g_b2 = sum(dE);
            g_W2 = X2*dE';
            db1 = W2'.*net.df(S1).*dE;
            g_b1 = sum(db1,2);
            g_W1 = db1*X_tr_tilde';
            g_t = -sum(W1.*(db1*dX_tilde'));
            g_t = sum(reshape(g_t(2:end),net.nD,net.states_num),2)';
        elseif strcmp(method.alg,'LM')
            dE = sqrt(2/data.tr_size)*(Yhat_tr_tilde-data.tr_out);
            db2 = ones(1,data.tr_size);
            dW2 = X2.*db2;
            db1 = W2'.*net.df(S1).*db2;
            for ii = 1:data.tr_size
                dW1(:,ii)= reshape(db1(:,ii)*X_tr_tilde(:,ii)',[],1);
                g_tau =sum(-W1.*(db1(:,ii)*dX_tilde(:,ii)'),1);
                dtau(:,ii) = sum(reshape(g_tau(2:end),net.nD,net.states_num),2);
            end
            grad = [dtau;dW1;dW2;db1;db2]';
        end
        err_tr(i) = sqrt(sum((Yhat_tr_tilde-data.tr_out).^2)/data.tr_size);  % RMSE
        
        if strcmp(method.alg,'GD')
            % Gradient Descent
            W1 = W1-method.a0*g_W1;
            W2 = W2-method.a0*g_W2';
            b1 = b1-method.a0*g_b1;
            b2 = b2-method.a0*g_b2;
            tau = max(tau-method.a0_s*g_t,0);
        elseif strcmp(method.alg,'LM')
            % LMA: tau,W1,W2,b1,b2
            theta = [tau,reshape(W1,1,[]),reshape(W2,1,[]),reshape(b1,1,[]),reshape(b2,1,[])]';
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
% id = minE_id{I};
E = Err_tr{I};
Eva = Err_va{I};
Tau_path = TT{I};
net_best.W1 = WW1{I};
net_best.W2 = WW2{I};
net_best.b1 = B1{I};
net_best.b2 = B2{I};
net_best.tau = Tau_path(end,:);
end
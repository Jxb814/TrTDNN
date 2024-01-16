function [data,net] = load_AutoSys_data(data,net)
data.batchsize = length(1:round(data.dt/data.min_dt):data.max_batchsize)-1; 
net.shiftlimit = floor(net.max_shiftlimit*data.min_dt/data.dt);
History = cell(1,length(data.constant));
for ii = 1:length(data.constant)
    History{ii} = strcat('.\\AutoSys_data\AutoSys_',data.type,'_',num2str(10*data.constant(ii)));
end
valid =strcat('.\AutoSys_data\AutoSys_',data.type,'_',num2str(100*data.c_va));
data.nbatches = length(History);
data.DATA = cell(data.nbatches,1);
data.Y_tr = cell(data.nbatches,1);
data.id_tr = cell(data.nbatches,1);
for n = 1:data.nbatches
    training = History{n};
    load(training);
    x = [data.constant(n)*ones(1,net.shiftlimit),D(1,1:round(data.dt/data.min_dt):data.max_batchsize)];
    dx = diff(x)/data.dt;
    data.DATA{n} = [x(1:end-1);dx];   
    data.id_tr{n} = 1:data.batchsize;
    data.Y_tr{n} = dx(1+net.shiftlimit:end);
end

UB = reshape(max(cell2mat(data.DATA),[],2),2,[]);
data.x_max = max(UB(1,:));
data.dx_max = max(UB(2,:));
LB = reshape(min(cell2mat(data.DATA),[],2),2,[]);
data.x_min = min(LB(1,:));
data.dx_min = min(LB(2,:));
data.sc_x = data.x_max-data.x_min;
data.sc_dx = data.dx_max-data.dx_min;
data.fun_scale_x = @(x)(x-data.x_min)/data.sc_x;
data.fun_scaleback_x = @(x) x*data.sc_x+data.x_min;
data.fun_scale_dx = @(x)(x-data.dx_min)/data.sc_dx;
data.fun_scaleback_dx = @(x) x*data.sc_dx+data.dx_min;

% % a separate set as validation set
load(valid);
x = [data.c_va*ones(1,net.shiftlimit),D(1,1:round(data.dt/data.min_dt):data.max_batchsize)];
dx = diff(x)/data.dt;
x = x(1:end-1);
data.X_va_tilde = data.fun_scale_x(x);
data.Y_va_tilde = data.fun_scale_dx(dx(net.shiftlimit+1:end));

% training data normalization
data.TDATA=cell(data.nbatches,1);
data.Y_tr_tilde = cell(data.nbatches,1);
for n = 1:data.nbatches
    data.TDATA{n}(1,:) = data.fun_scale_x(data.DATA{n}(1,:));
    data.TDATA{n}(2,:) = data.fun_scale_dx(data.DATA{n}(2,:));
    data.Y_tr_tilde{n} = data.fun_scale_dx(data.Y_tr{n});
end
end
%%% Robustness test, for no-fixed delay
clear;
clc;
close all
addpath('utils');
plottingPreferences()
%%
trials = 100;
data.type = 'type6';   % type1: nD=1; type6: nD = 3; others: nD = 2
method.alg = 'LM';   % 'GD','LM','MGD','MLM'
method.attempts = 5;   % attempts
method.epoch = 1e4;
method.vset = 10;   % control the consecutive validation violation times
if strcmp(method.alg,'MGD')||strcmp(method.alg,'GD')
    method.win = 100;    % averaging window for validation
    method.a0 = 0.1;   % 0.1 for ReLU/sigmoid and 0.01 for tanh
    method.a0_s =0.1;
elseif strcmp(method.alg,'MLM')||strcmp(method.alg,'LM')
    method.win = 10;
    method.mu = 1e-2;
end

method.gap = method.win;
if strcmp(data.type,'type1')
    net.nD = 2;% # of delays, does not include the non-delay 
else
    net.nD = 3;
end
net.states_num = 1;
net.nN = 3;       % number of neurons in hidden layer
net.input_num = net.states_num*net.nD;    % layer 3
net.output_num = 1;    % layer 3

% % % activation function
% net.f = @(x) (x>0).*x;  % ReLu
% net.df = @(x) (x>0);

net.f = @(x) 1./(1+exp(-x)); % sigmoid
net.df = @(x) net.f(x).*(1-net.f(x));

% net.f = @(x) tanh(x);
% net.df = @(x) 1-tanh(x).^2;


% load and process data
net.max_shiftlimit = 200;
data.max_batchsize = 500;
data.min_dt = 0.01;
data.dt = 0.01;
data.constant = 1:0.1:1.6;
data.c_va = 1.25;
[data,net] = load_AutoSys_data(data,net);
%%
% train and save
Tau_path = cell(trials,1);
Net_best = cell(trials,1);
Ltr_best = cell(trials,1);
Lva_best = cell(trials,1);
tic;
parfor r = 1:trials
    [Tau_path{r},Net_best{r},Ltr_best{r},Lva_best{r}] = STTD_train(net,data,method);
end
toc

filename = strcat('Robust_',method.alg,'_nN',num2str(net.nN),'_attempt',...
    num2str(method.attempts),'_',data.type,'_',num2str(trials),'runs');
save(filename,'Tau_path','Net_best','Ltr_best','Lva_best','data','method','net','trials')
%% read data
% resultfile = 'AutoSys_results\Robust_GD_nN40_attempt20_type6.mat';
% resultfile = 'Robust_GD_nN40_attempt20_type6.mat';
% load(resultfile)
error_end = zeros(1,trials+2);
tau_end = NaN(3,trials+2);
for r = 1:trials
    error_end(r) = Ltr_best{r}(end);
    tau_end(:,r) = Tau_path{r}(end,:);
end
% % normalize the error according to mean(xdot) only for GD/LM
% den = 0;
% for n = 1:data.nbatches
%     den = den+mean(data.Y_tr_tilde{n}.^2);
% end
% den = sqrt(den/data.nbatches);
den = 1;
Nerror_end = error_end/den;
% NEtr_ave = mean(Nerror_end)

% thredhold = quantile(Nerror_end,1);
% thredhold = 0.0162; % 0.0055,0.0162
thredhold = 0.02;
Nerror_end(end) = thredhold;
% low_bound = 0;
error_sat = min(Nerror_end,thredhold);
plottingPreferences()
if strcmp(data.type,'type3')
    true_tau = [1,0.5,0]; 
elseif strcmp(data.type,'type6')
    true_tau = [1, 0.5, 1.5];
end
% fig = figure(100);
% set(gcf, 'Position',  [500, 100, 620, 600])
% x = [ture_tau,ture_tau];
% y = [circshift(ture_tau,[0,-1]),circshift(ture_tau,[0,-2])];
% z = [circshift(ture_tau,[0,-2]),circshift(ture_tau,[0,-1])];
% 
% ax1 =subplot(3,3,[1 2 3 4 5 6]);
% plotObjs = scatter3(x,y,z,50,'o','MarkerEdgeColor',[1 0 0],'LineWidth',2);
% hold on;
% padd = scatter3(tau_end(1,:),tau_end(2,:),tau_end(3,:),25,error_sat,'filled');
% plotObjs = [plotObjs,padd];
% colormap;
% colorbar('TickLabelInterpreter','latex');
% 
% % caxis([0 0.03])
% hold off;
% xlabel('$\tau_{x,1}$')
% ylabel('$\tau_{x,2}$')
% zlabel('$\tau_{x,3}$')
% 
% title(ax1,'view 3D');
% view(ax1,[135,20])
% 
% ax2 =subplot(3,3,7);
% grid on;
% box on;
% % caxis([0 0.03])
% ax3 =subplot(3,3,8);
% grid on;
% box on;
% % caxis([0 0.03])
% ax4 =subplot(3,3,9);
% grid on;
% box on;
% % caxis([0 0.03])
% 
% copyobj(plotObjs,ax2);
% view(ax2,[90,0])
% title(ax2,'view(90,0)');
% 
% copyobj(plotObjs,ax3);
% view(ax3,[0,90])
% title(ax3,'view(0,90)');
% 
% copyobj(plotObjs,ax4);
% view(ax4,[0,0])
% title(ax4,'view(0,0)');
% 
% AxesH    = findobj(fig, 'Type', 'Axes');
% XLabelHC = get(AxesH, 'XLabel');
% XLabelH  = [XLabelHC{:}];
% set(XLabelH, 'String', '$\tau_{x,1}$')
% YLabelHC = get(AxesH, 'YLabel');
% YLabelH  = [YLabelHC{:}];
% set(YLabelH, 'String', '$\tau_{x,2}$')
% ZLabelHC = get(AxesH, 'ZLabel');
% ZLabelH  = [ZLabelHC{:}];
% set(ZLabelH, 'String', '$\tau_{x,3}$')
% 
% Link = linkprop([ax1, ax2, ax3, ax4],{'XLim', 'YLim', 'ZLim'});
% setappdata(gcf, 'StoreTheLink', Link);
% xlim([-0.1,1.6])
% ylim([-0.1,1.6])
% zlim([-0.1,1.6])
% %%

figure(101)
set(gcf, 'Position',  [500, 100, 320, 300])
% set(gcf, 'Renderer', 'painters');
x = [true_tau,true_tau];
y = [circshift(true_tau,[0,-1]),circshift(true_tau,[0,-2])];
z = [circshift(true_tau,[0,-2]),circshift(true_tau,[0,-1])];
scatter3(x,y,z,50,'o','MarkerEdgeColor',[1 0 0],'LineWidth',2);
hold on;
scatter3(tau_end(1,:),tau_end(2,:),tau_end(3,:),25,error_sat,'filled');
colormap;
colorbar('TickLabelInterpreter','latex');
if strcmp(data.type,'type3')
    title(['System I, ',method.alg]);
else strcmp(data.type,'type6')
    title(['System II, ',method.alg]);
end
axis equal;
view([135,20])
xlim([-0.1,1.6])
ylim([-0.1,1.6])
zlim([-0.1,1.6])
xlabel('$\tau_{1}$')
ylabel('$\tau_{2}$')
zlabel('$\tau_{3}$')

%% Error plots
[~,I] = min(error_end(1:end-2));
% I = 1; 
net = Net_best{I};
E = Ltr_best{I};
Eva = Lva_best{I};
fun_scale_dx = data.fun_scale_dx;
fun_scaleback_dx = data.fun_scaleback_dx;
fun_scale_x = data.fun_scale_x;
W1 = net.W1;
W2 = net.W2;
b1 = net.b1;
b2 = net.b2;
tau = net.tau;
f = net.f;
shiftlimit = net.shiftlimit;
dt = data.dt;
nD = net.nD;
win = method.win;
batchsize = data.batchsize;

kernel = ones(win,1) / win;
% kernel = 1;
E_ave = filter(kernel, 1, E);
Eva_ave = filter(kernel, 1, Eva);
plottingPreferences()
figure(1)
set(gcf, 'Position',  [500, 100, 250, 150])
semilogy(win:length(E_ave),E_ave(win:end),'b',win:length(Eva_ave),Eva_ave(win:end),'m','LineWidth',1.5)
title(['Loss = ',num2str(E_ave(end))])
xlabel('Iterations')
ylabel('$\sqrt{L}$')
legend('Training','Validation')
% ylim([0.002,0.2])
ytick=10.^(-3:0);
set(gca, 'YTick', ytick)
xlim([0,length(E_ave)])
E_ave(end)

figure(2)
set(gcf, 'Position',  [500, 100, 250, 150])
hold on;
for ii = 1:nD
    plot(1:length(Tau_path{I}(:,ii)),Tau_path{I}(:,ii),'LineWidth',1.5);
    
end
plot(1:length(Tau_path{I}(:,1)),true_tau(1)*ones(1,length(Tau_path{I}(:,1))),'k--','LineWidth',1)
plot(1:length(Tau_path{I}(:,2)),true_tau(2)*ones(1,length(Tau_path{I}(:,2))),'k--','LineWidth',1)
plot(1:length(Tau_path{I}(:,3)),true_tau(3)*ones(1,length(Tau_path{I}(:,3))),'k--','LineWidth',1)
hold off;
xlabel('Iterations')
ylabel('$\tau$')
xlim([0,length(E_ave)])
box on;
title(['Delay = ',num2str(tau)])
tau

% simulation of training
tspan = [0 5];
tt = 0:dt:5;
T_nominal = [1, 0.5];
c = 1.6;
hist =@(t) c;

kk = find(data.constant == c);
if strcmp(data.type,'type1')
    m_nominal  = @(t,x,xdelay) -x - xdelay(2)^3;
elseif strcmp(data.type,'type2')
    m_nominal = @(t,x,xdelay) -x + xdelay(1)^2 - xdelay(2)^3;
elseif strcmp(data.type,'type3')
    m_nominal  = @(t,x,xdelay) -x + xdelay(1)*xdelay(2) - xdelay(2)^3;
elseif strcmp(data.type,'type4')
    m_nominal  = @(t,x,xdelay) xdelay(1)^2 - xdelay(2)^3;
elseif strcmp(data.type,'type5')
    m_nominal  = @(t,x,xdelay) xdelay(1)*xdelay(2) - xdelay(2)^3;
elseif strcmp(data.type,'type6')
    T_nominal = [1, 0.5, 1.5];
    m_nominal  = @(t,x,xdelay) -xdelay(3)+xdelay(1)^2 - xdelay(2)^3;
end

%% Simulation of nominal system with dde23 
sol_23 = dde23(@(t,x,xdelay)m_nominal(t,x,xdelay),T_nominal,hist,tspan);
x_dde23 = [c*ones(1,net.shiftlimit),deval(sol_23,tt)];


if tau(3)==0
    m = @(t,x,xdelay) fun_scaleback_dx(W2*f(W1*fun_scale_x([xdelay(1);xdelay(2);x])+b1)+b2);
    NN_dde23 = dde23(@(t,x,Z)m(t,x,Z),tau(1:2),hist,tspan);
elseif tau(2)==0
    m = @(t,x,xdelay) fun_scaleback_dx(W2*f(W1*fun_scale_x([xdelay(1);x;xdelay(2)])+b1)+b2);
    NN_dde23 = dde23(@(t,x,Z)m(t,x,Z),tau(1:2:3),hist,tspan);
elseif tau(1)==0
    m = @(t,x,xdelay) fun_scaleback_dx(W2*f(W1*fun_scale_x([x;xdelay(1);xdelay(2)])+b1)+b2);
    NN_dde23 = dde23(@(t,x,Z)m(t,x,Z),tau(1:2),hist,tspan);
else
    m = @(t,x,xdelay) fun_scaleback_dx(W2*f(W1*fun_scale_x([xdelay(1);xdelay(2);xdelay(3)])+b1)+b2);
    NN_dde23 = dde23(@(t,x,Z)m(t,x,Z),tau,hist,tspan);
end

x_NN_dde23 = [c*ones(1,shiftlimit),deval(NN_dde23,tt)];

sigma = round(tau/dt);
state =[];
for ii = 1:nD
    state = [state;data.TDATA{kk}(1,shiftlimit-sigma(ii)+1:end-sigma(ii))];
end
s = W1*state+b1;
fs = f(s);
y = W2*fs+b2;
L_dx_tr = sqrt(mean((fun_scaleback_dx(y)-fun_scaleback_dx(data.Y_tr_tilde{kk})).^2))
L_x_tr = sqrt(mean((x_NN_dde23(end-batchsize:end)-x_dde23(end-batchsize:end)).^2))

% L_dx_tr = sqrt(mean((y-data.Y_tr_tilde{kk}).^2))
% L_x_tr = sqrt(mean((fun_scale_x(x_NN_dde23(end-batchsize:end))...
%     -fun_scale_x(x_dde23(end-batchsize:end))).^2))

figure(4)
set(gcf, 'Position',  [500, 100, 300, 250])
subplot(2,1,1)
plot(tt(1:batchsize),fun_scaleback_dx(data.Y_tr_tilde{kk}),'Color','#EDB120','LineWidth',2)
hold on;
plot(tt(1:batchsize),fun_scaleback_dx(y),'k-.','LineWidth',1)
legend('Training data','TTDNN','location','best')
ylabel('$\dot{x}$')
hold off;
ylim([-4,1.5])
title('Static mapping of $\dot{x}(t)$')

subplot(2,1,2)
hold on;
plot(tt(1:batchsize),x_dde23(shiftlimit+1:shiftlimit+batchsize),'Color','#EDB120','LineWidth',2)
plot(tt(1:batchsize),x_NN_dde23(shiftlimit+1:shiftlimit+batchsize),'k-.','LineWidth',1)
% legend('clean data-dde23','neural network simulation','location','best')
hold off;
box on;
ylim([-2,2])
ylabel('$x$')
xlabel('$t$')
title(['Simulation $x(t)$, histoty = ',num2str(c)])

% simulation of testing
c_ts = 1.35;
hist =@(t) c_ts;

% Simulation of nominal system with dde23 
sol_23 = dde23(@(t,x,xdelay)m_nominal(t,x,xdelay),T_nominal,hist,tspan);
x = deval(sol_23,tt);
x_dde23 = [c_ts*ones(1,shiftlimit),x(1:end-1)];
dx_dde23 = diff(x_dde23)/dt;
x_dde23 = x_dde23(1:end-1);
x_dde23_t = fun_scale_x(x_dde23);
Y_ts_tilde = fun_scale_dx(dx_dde23(shiftlimit+1:end));

if tau(3)==0
    m = @(t,x,xdelay) fun_scaleback_dx(W2*f(W1*fun_scale_x([xdelay(1);xdelay(2);x])+b1)+b2);
    NN_dde23 = dde23(@(t,x,Z)m(t,x,Z),tau(1:2),hist,tspan);
elseif tau(2)==0
    m = @(t,x,xdelay) fun_scaleback_dx(W2*f(W1*fun_scale_x([xdelay(1);x;xdelay(2)])+b1)+b2);
    NN_dde23 = dde23(@(t,x,Z)m(t,x,Z),tau(1:2:3),hist,tspan);
elseif tau(1)==0
    m = @(t,x,xdelay) fun_scaleback_dx(W2*f(W1*fun_scale_x([x;xdelay(1);xdelay(2)])+b1)+b2);
    NN_dde23 = dde23(@(t,x,Z)m(t,x,Z),tau(1:2),hist,tspan);
else
    m = @(t,x,xdelay) fun_scaleback_dx(W2*f(W1*fun_scale_x([xdelay(1);xdelay(2);xdelay(3)])+b1)+b2);
    NN_dde23 = dde23(@(t,x,Z)m(t,x,Z),tau,hist,tspan);
end
x = deval(NN_dde23,tt);
x_NN_dde23 = [c_ts*ones(1,shiftlimit),x(1:end-2)];

sigma = round(tau/dt);
state =[];
for ii = 1:nD
    state = [state;x_dde23_t(1,shiftlimit-sigma(ii)+1:end-sigma(ii))];
end
s = W1*state+b1;
fs = f(s);
y = W2*fs+b2;
Yhat = fun_scaleback_dx(y);
% original scale
L_dx_ts = sqrt(mean((fun_scaleback_dx(y)-fun_scaleback_dx(Y_ts_tilde)).^2))
L_x_ts = sqrt(mean((x_NN_dde23(end-batchsize:end)-x_dde23(end-batchsize:end)).^2))
% scaled to [0,1]
% L_dx_ts = sqrt(mean((y-Y_ts_tilde).^2))
% L_x_ts = sqrt(mean((fun_scale_x(x_NN_dde23(end-batchsize:end))...
%     -fun_scale_x(x_dde23(end-batchsize:end))).^2))

figure(6)
set(gcf, 'Position', [500, 100, 300, 250])
subplot(2,1,1)
plot(tt(1:end-2),fun_scaleback_dx(Y_ts_tilde),'Color','#4DBEEE','LineWidth',2)
hold on;
plot(tt(1:end-2),fun_scaleback_dx(y),'k-.','LineWidth',1)
legend('Testing data','TTDNN','location','best')
hold off;
ylim([-4,1.5])
ylabel('$\dot{x}$')
title('Static mapping of $\dot{x}(t)$')

subplot(2,1,2)
plot(tt(1:end-2),x_dde23(shiftlimit+1:end),'Color','#4DBEEE','LineWidth',2)
hold on;
plot(tt(1:end-2),x_NN_dde23(shiftlimit+1:end),'k-.','LineWidth',1)
% legend('clean data-dde23','neural network simulation','location','best')
hold off;
ylim([-2,2])
ylabel('$x$')
xlabel('$t$')
title(['Simulation $x(t)$, histoty = ',num2str(c_ts)])
% 
% %% get exact integration of the output derivative
% X = zeros(500,1);
% X(1)=c_ts;
% for i=1:499
%     X(i+1) = X(i)+Yhat(i)*dt;
% end
% 
% figure
% set(gcf, 'Position',  [500, 100, 500, 500])
% subplot(3,1,1)
% plot(tt(1:end-2),fun_scaleback_dx(Y_ts_tilde),'Color','#4DBEEE','LineWidth',2);
% hold on;
% plot(tt(1:end-2),fun_scaleback_dx(y),'k-.','LineWidth',1)
% title(['Testing: static mapping, constant history:', num2str(c)]);
% ylabel('$\dot{x}$')
% xlim([0,tt(end)])
% 
% subplot(3,1,2)
% plot(tt(1:end-2),x_dde23(shiftlimit+1:end),'Color','#4DBEEE','LineWidth',2);
% hold on;
% plot(tt(1:end-1),X,'k-.','LineWidth',1)
% title('Exact integration of the static mapping');
% ylabel('$x$')
% xlim([0,tt(end)])
% 
% subplot(3,1,3)
% plot(tt(1:end-2),x_dde23(shiftlimit+1:end),'Color','#4DBEEE','LineWidth',2);
% hold on;
% plot(tt(1:end-2),x_NN_dde23(shiftlimit+1:end),'k-.','LineWidth',1);
% legend('Data','Trainable','Location','Best')
% ylabel('$x$')
% title('Testing: closed-loop simulation');
% xlim([0,tt(end)])
% ylim([-5,2])
% xlabel('$t$')
% set(findall(gcf,'-property','FontSize'),'FontSize',12)
% 
% %%
% surface
% nonlinearity for three terms
Rg = [-1.6 1.6];
x1 = Rg(1):0.001:Rg(2);
x2 = zeros(size(x1));
V = zeros(4,length(x1));
O1 = -x1;  % non-delay term is not zero, the others are zeros
O2 = -x1.^3; % tau = 0.5 is not zero but the others are zeros
if strcmp(data.type,'type6')
    O3 = x1.^2; % tau = 1 is not zero but the other two are zeros
    for n = 1:length(x1)
        V(1,n) = fun_scaleback_dx(W2*f(W1*fun_scale_x([x1(n);x2(n);x2(n)])+b1)+b2);
        V(2,n) = fun_scaleback_dx(W2*f(W1*fun_scale_x([x2(n);x1(n);x2(n)])+b1)+b2);
        V(3,n) = fun_scaleback_dx(W2*f(W1*fun_scale_x([x2(n);x2(n);x1(n)])+b1)+b2);
    end
    figure(7)
    set(gcf, 'Position',  [500, 100, 320, 280])
    p1 = plot(x1,O1,'Color','#0072BD','LineWidth',1.5);
    hold on;
    p2 = plot(x1,O2,'Color','#D95319','LineWidth',1.5);
    p3 = plot(x1,O3,'Color','#EDB120','LineWidth',1.5);
%     p4 = plot(x1,V(1,:),':','Color','#0072BD','LineWidth',1.5);
%     p5 = plot(x1,V(2,:),':','Color','#D95319','LineWidth',1.5);
%     p6 = plot(x1,V(3,:),':','Color','#EDB120','LineWidth',1.5);
    p4 = plot(x1,V(1,:),'k:','LineWidth',2);
    p5 = plot(x1,V(2,:),'r:','LineWidth',2);
    p6 = plot(x1,V(3,:),'b:','LineWidth',2);
    hold off;
    ylim([-5,5])
    xlabel('$x$')
    ylabel('$F(x)$')
    title('Approximation by trained FNN')
    leg1=legend([p1 p2 p3],'$-x$','$-x^3_{\tau_3}$','$x^2_{\tau_2}$','Location','North');
    ah1=axes('position',get(gca,'position'),'visible','off');
    leg2=legend(ah1,[p4 p5 p6],'$F(x_{\tau_a},0,0)$',...
        '$F(0,x_{\tau_b},0)$','$F(0,0,x_{\tau_c})$','Location','South');
end
box on

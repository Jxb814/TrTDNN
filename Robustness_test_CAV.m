%% Robustness test for CAV
clear;
clc;
close all
addpath('utils');
plottingPreferences()

%% Data process
% load('Car_data\processed_8vehicles_114049_233')
% Dataname = 'Data233';
load('Car_data\processed_8vehicles_121029_500')
Dataname = 'Data500';
% divide into three sets - for car case only, already scaled and has derivative
Total_points = length(output);

% set up the maximum allowed delay
taumax = 3;
dt = tsim(2)-tsim(1);
net.shiftlimit = round(taumax/dt);
data.dt = dt;

tr_ratio = 0.7;
va_ratio = 0.15;

tr_len = round(Total_points*0.7);
va_len = round(Total_points*0.15);
ts_len = Total_points-tr_len-va_len;

data.tr_in = input(:,1:tr_len);
data.tr_dx_in = input_dx(:,1:tr_len);
data.tr_out = output(net.shiftlimit+1:tr_len);
data.tr_size = length(data.tr_out);
data.va_in = input(:,tr_len+1:tr_len+va_len);
data.va_out = output(net.shiftlimit+tr_len+1:tr_len+va_len);
data.va_size = length(data.va_out);
%% network configuration
trials = 500;
method.alg = 'LM';   % 'GD','LM'
method.attempts = 10;   % attempts
method.epoch = 1e5;
method.vset = 10;
method.win = 10;
method.mu = 1e-2;
method.a0 = 0.1;
method.a0_s = 0.1;
method.gap = method.win;
net.nD = 1;
net.states_num = length(input(:,1));
net.nN = 3;       % number of neurons in hidden layer
net.input_num = net.states_num*net.nD+1;    % layer 3
net.output_num = 1;    % layer 3
% net.f = @(x) (x>0).*x;  % ReLu
% net.df = @(x) (x>0);
net.f = @(x) 1./(1+exp(-x)); % sigmoid
net.df = @(x) net.f(x).*(1-net.f(x));

% train and save
Tau_path = cell(trials,1);
Net_best = cell(trials,1);
Ltr_best = cell(trials,1);
Lva_best = cell(trials,1);
tic;
parfor r = 1:trials
    [Tau_path{r},Net_best{r},Ltr_best{r},Lva_best{r}] = STTD_CAV_train(net,data,method);
end
toc

filename = strcat('Robust_CAV_',method.alg,'_nN',num2str(net.nN),'_attempt',num2str(method.attempts),'_',Dataname,'_',num2str(trials));
save(filename,'Tau_path','Net_best','Ltr_best','Lva_best','data','method','net','trials')

%%
error_end = zeros(1,trials);
tau_end = NaN(net.nD,trials);
for r = 1:trials
    error_end(r) = Ltr_best{r}(end);
    tau_end(:,r) = Tau_path{r}(end,:);
end

% distribution/histgram
figure(100)
set(gcf, 'Position',  [500, 100, 350, 150])
edges = 0.6:0.002:0.7;
% yyaxis left
histogram(tau_end,edges,'Normalization','probability');
ylim([0,0.6])
ylabel('$p$')
% yyaxis right
% scatter(tau_end,error_end,10,'filled','MarkerFaceAlpha',.5);
% ylabel('')
% ylim([0.035,0.055])
box on;
xlabel('$\tau$')
xlim([0.6,0.7])

%% simulation
[~,I] = min(error_end(1:end));
% I = 4; 
net = Net_best{I};
E = Ltr_best{I};
Eva = Lva_best{I};
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
Tau = Tau_path{I};


kernel = ones(win,1) / win;
E_ave = filter(kernel, 1, E);
Eva_ave = filter(kernel, 1, Eva);

figure(1)
set(gcf, 'Position',  [500, 100, 250, 150])
semilogy(win:length(E_ave),E_ave(win:end),'b',win:length(Eva_ave),Eva_ave(win:end),'m','LineWidth',1.5)
% title(['Loss = ',num2str(E_ave(end))])
xlim([0,600])
ylim([0.028,0.14])
legend('Training','Validation')
xlabel('Iterations')
ylabel('$\sqrt{L}$')
% xtick=10.^(0:5);
% set(gca, 'XTick', xtick)

figure(2)
set(gcf, 'Position',  [500, 100, 250, 150])
plot(1:length(Tau),Tau,'k','LineWidth',1.5)
legend('$\tau$')
xlim([0,600])
ylim([0,1.5])
xlabel('Iterations')
ylabel('$\tau$')
% title(['Final Learned Delay = ',num2str(Tau(end,:))])

%%
hmin = xmin(1);
hmax = xmax(1);
vmin = xmin(2);
vmax = xmax(2);
amin = xmin(3);
amax = xmax(3);
f = net.f;
% f = @(x) (x>0).*x;  % ReLU
taumax = 3;
dt = tsim(2)-tsim(1);
shiftlimit = round(taumax/dt);

Total_points = length(output);
tr_len = round(Total_points*0.7);
va_len = round(Total_points*0.15);

% t_st = tsim(shiftlimit+tr_len+va_len+1);
t_st = tsim(1);
t_ed = tsim(1)+40;
t_ed = tsim(end);

% closed-loop simulation
tt = t_st:dt:t_ed;
tau = Tau(end,:);
select=@(vL,kL)vL(kL);

if contains(Dataname,'500')
    % respond to 1 leader
    input_raw = [fun_scaleback(input(1,:),hmin,hmax);
        fun_scaleback(input(2,:),vmin,vmax);
        fun_scaleback(input(3,:),vmin,vmax)];
    vL = @(t)interp1(tsim,input_raw(3,:),t,'linear','extrap');
    hist = @(t)[interp1(tsim,input_raw(1,:),t,'linear','extrap');
        interp1(tsim,input_raw(2,:),t,'linear','extrap')];
    beta = 0.5;
    betasum = beta;
    % zero-delay for first states only
    m = @(t,x,xtau) [vL(t)-x(2);
        fun_scaleback((W2*f(W1*[fun_scale(x(2),vmin,vmax);...
        fun_scale(xtau(1),hmin,hmax);fun_scale(xtau(2),vmin,vmax);fun_scale(vL(t-tau),vmin,vmax)]...
        +b1)+b2),amin,amax)];
    sol_NN = dde23(@(t,x,Z)m(t,x,Z),tau,hist,[t_st t_ed]);
elseif contains(Dataname,'233')
    % respond to 3 leaders
    input_raw = [fun_scaleback(input(1,:),hmin,hmax);
        fun_scaleback(input(2,:),vmin,vmax);
        fun_scaleback(input(3,:),vmin,vmax);
        fun_scaleback(input(4,:),vmin,vmax);
        fun_scaleback(input(5,:),vmin,vmax)];
    vL = @(t)[interp1(tsim,input_raw(3,:),t,'linear','extrap');
        interp1(tsim,input_raw(4,:),t,'linear','extrap');
        interp1(tsim,input_raw(5,:),t,'linear','extrap')];
    hist = @(t)[interp1(tsim,input_raw(1,:),t,'linear','extrap');
        interp1(tsim,input_raw(2,:),t,'linear','extrap')];
    beta = [0.2,0.3,0.3];
    betasum = sum(beta);
    % one same delay for all states and inputs
    m = @(t,x,xtau) [select(vL(t),1)-x(2);
        fun_scaleback((W2*f(W1*[fun_scale(x(2),vmin,vmax);...
        fun_scale(xtau(1),hmin,hmax);...
        fun_scale(xtau(2),vmin,vmax);...
        fun_scale(vL(t-tau),vmin,vmax)]...
        +b1)+b2),amin,amax)];
    sol_NN = dde23(@(t,x,Z)m(t,x,Z),tau,hist,[t_st t_ed]);
end
    
% fixed parameters for CAV
gamma=0.01;         % [-] tyre rolling resistance coefficient
g=9.81;             % [m/s^2] gravitatioinal constant
a=gamma*g;          % [m/s^2]
Cd=0.34;            % [-] air drag coefficient
A=2.32;             % [m^2] frontal area
rho=1.23;           % [kg/m^3] air density at 25 degree
k=0.5*Cd*rho*A;     % [kg/m]
m=1700;             % [kg] mass of the vehicle
P=75000;
c=k/m;              % [1/m]

sigma=0.6;
hst=5;
hgo=55;
v_max=30;
a_min=7;
a_max=3;
alpha = 0.4;

% range policy and saturations for CAV
V=@(h)v_max*(hgo<=h) + v_max*(h-hst)/(hgo-hst).*(hst<h & h<hgo);
W=@(vL)v_max*(v_max<=vL)+vL.*(vL<v_max);
% alimt=@(v)min([a_max+0*v;P/m./abs(v)]);
% sat=@(v,u)(u<-amin).*(-amin)+(-amin<=u & u<=alimt(v)).*u+(alimt(v)<u).*alimt(v);
sat=@(u)(u<-a_min).*(-a_min)+(-a_min<=u & u<=a_max).*u+(a_max<u).*a_max;
% control input for CAV
u=@(h,v,vL)alpha*V(h)-(alpha+betasum)*v+beta*W(vL);
% nominal model
model=@(t,x,xdelay)[select(vL(t),1)-x(2);
                    -a-c*x(2)^2+sat(u(xdelay(1),xdelay(2),vL(t-sigma)))];
sol_Nominal = dde23(@(t,x,Z)model(t,x,Z),sigma,hist,[t_st t_ed]);

figure(4)
set(gcf, 'Position',  [400, 100, 500, 550])
subplot(3,1,1)
ts_data = hist(tt);
scatter(tt,ts_data(1,:),5,'r')
hold on;
plot(sol_Nominal.x,sol_Nominal.y(1,:),'Color','#4DBEEE','LineWidth',2.5)
plot(sol_NN.x,sol_NN.y(1,:),'k-.','LineWidth',1.5)
plot(tsim(tr_len+1)*ones(2),[0,100],'k--','LineWidth',1)
plot(tsim(tr_len+1+va_len)*ones(2),[0,100],'k--','LineWidth',1)
ylim([0,100])
ylabel('$h$ [m]')
xlim([tt(1),tt(end)])
box on;
legend('Data','Nominal','NN','Location','best','NumColumns',3)
title('Headway')

subplot(3,1,2)
scatter(tt,ts_data(2,:),5,'r','DisplayName','CAV-Data')
hold on;
plot(sol_Nominal.x,sol_Nominal.y(2,:),'Color','#4DBEEE','LineWidth',2.5,'DisplayName','CAV-Nominal')
plot(sol_NN.x,sol_NN.y(2,:),'k-.','LineWidth',1.5,'DisplayName','CAV-NN')
plot(tsim(tr_len+1)*ones(2),[0,35],'k--','LineWidth',1)
plot(tsim(tr_len+1+va_len)*ones(2),[0,35],'k--','LineWidth',1)
ylim([0,35])
ylabel('$v$ [m/s]')
box on;
xlim([tt(1),tt(end)])
title('Velocity')

subplot(3,1,3)
output_raw = fun_scaleback(output,amin,amax);
ts_out = interp1(tsim,output_raw,tt,'linear','extrap');
scatter(tt,ts_out,5,'r')
hold on
plot(sol_Nominal.x,sol_Nominal.yp(2,:),'Color','#4DBEEE','LineWidth',2.5)
plot(sol_NN.x,sol_NN.yp(2,:),'k-.','LineWidth',1.5)
% legend('Data','NN','Nominal','Location','best')
plot(tsim(tr_len+1)*ones(2),[-10,5],'k--','LineWidth',1)
plot(tsim(tr_len+1+va_len)*ones(2),[-10,5],'k--','LineWidth',1)
ylim([-10,5])
xlim([tt(1),tt(end)])
ylabel('$a$ [m/s$^2$]')
xlabel('$t$ [s]')
box on;
title('Acceleration')

[sol_nomi,sol_nomi_p] = deval(sol_Nominal,tt);
RMSE_nomi = sqrt(sum((hist(tt)-sol_nomi).^2,2)/length(sol_nomi))
[sol_nn,sol_nn_p] = deval(sol_NN,tt);
RMSE_nn = sqrt(sum((hist(tt)-sol_nn).^2,2)/length(sol_nn))
out_data = interp1(tsim,output_raw,tt,'linear','extrap');
RMSE_nomi_a = sqrt(sum((out_data-sol_nomi_p(2,:)).^2,2)/length(sol_nn))
RMSE_nn_a = sqrt(sum((out_data-sol_nn_p(2,:)).^2,2)/length(sol_nn))


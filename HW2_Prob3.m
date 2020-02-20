% MAE 5010 : HW2
% Problem# 3
clear all; clc;

%% Parameters
% Algorithm parameters
mu = [0 0 0]';
sig_sq = 0.001*[1 1 1]'; 
N_obs = 40; % number of observations
N_ts = 81; % total number of timesteps
N_var = 3; % number of variables
h = eye(N_var);
dt = 0.01; % time step 
al = 1e-4; % step length for update of x0 values
tol = 1e-4; % tolerance for convergence
Ts_fin = 10*N_ts; %V final tims step for forecasting

% Physical model parameters (Lorenz parameters)
s = 10; % sigma
r = 28; % rho
b = 8/3; % beta

%% Generating twin  experiments
x = zeros(N_var,N_ts);
Z = zeros(N_var,N_obs);

x(:,1) = [1.0 1.0 1.0]';

c = 1;           
for j = 2:Ts_fin
    x(:,j) = [(1-s*dt)*x(1,j-1) + s*dt*x(2,j-1);...
        r*dt*x(1,j-1) - dt*x(1,j-1)*x(3,j-1) + (1-dt)*x(2,j-1);...
        dt*x(1,j-1)*x(2,j-1) + (1-b*dt)*x(3,j-1)];
    if mod(j,2) == 0 && j <= N_ts
        Z(:,c) = x(:,j);
        c = c+1;
    end
end

X_an = x;
T_an = [1:Ts_fin];

% Noise at the observation (Gaussian white noise)
% noise = [normrnd(mu(1),sig_sq(1),[1,N_obs]);...
%          normrnd(mu(2),sig_sq(2),[1,N_obs]);...
%          normrnd(mu(3),sig_sq(3),[1,N_obs])     ];
     
noise = normrnd(mu(1),sig_sq(1),[1,N_obs]); % noise in the observation
noise = [noise;noise;noise];
     
Z = Z + noise; % generating noisy observation data 
T_ob = [2:2:N_ts]; % times at which observation is available
clear x;

%% Solution by (non-linear) 4D-VAR method
R = [sig_sq(1) 0 0 ; 0 sig_sq(2) 0 ; 0 0 sig_sq(3) ];

x = zeros(N_var,N_ts);
f = zeros(N_var,N_obs);
x(:,1) = [1.1 1.1 1.1]';
c = 1;
for j = 2:N_ts
    x(:,j) = [(1-s*dt)*x(1,j-1) + s*dt*x(2,j-1);...
               r*dt*x(1,j-1) - dt*x(1,j-1)*x(3,j-1) + (1-dt)*x(2,j-1);...
               dt*x(1,j-1)*x(2,j-1) + (1-b*dt)*x(3,j-1)];
    if mod(j,2) == 0
        J_M = [(1-s*dt) s*dt 0 ; (r-x(3,j))*dt 1-dt -dt*x(1,j) ; dt*x(2,j) dt*x(1,j) 1-b*dt];
        f(:,c) = (J_M)'*R^(-1)*(Z(:,c)-x(:,j));
        c = c+1;
    end
end

lambda = zeros(N_var,N_obs);
lambda(:,N_obs) = f(:,N_obs);
for j = N_obs-1:-1:1
    
    in = T_ob(j);
    J_M = [(1-s*dt) s*dt 0 ; (r-x(3,in))*dt 1-dt -dt*x(1,in) ; dt*x(2,in) dt*x(1,in) 1-b*dt];
    lambda(:,j) = (J_M)' * lambda(:,j+1) + f(:,j);
    
end

J_M = [(1-s*dt) s*dt 0 ; (r-x(3,1))*dt 1-dt -dt*x(1,1) ; dt*x(2,1) dt*x(1,1) 1-b*dt];
dJ = -J_M'*lambda(:,1); % convergence critria [||dJ||<tol]
nor = norm(dJ);
% nor 
% return;

% Iteration for convergence of x0 value
N_it = 0; % iteration counter
while nor > tol
    
    if N_it > 30000
        break;
    end
    N_it = N_it + 1; % increment of iteration counter
     
    x0 = x(:,1);
    x0 = x0 - (al/nor)*dJ; % updated value of x0
    
    x = zeros(N_var,N_ts);
    f = zeros(N_var,N_obs);
    x(:,1) = x0;
    c = 1;
    for j = 2:N_ts
        x(:,j) = [(1-s*dt)*x(1,j-1) + s*dt*x(2,j-1);...
            r*dt*x(1,j-1) - dt*x(1,j-1)*x(3,j-1) + (1-dt)*x(2,j-1);...
            dt*x(1,j-1)*x(2,j-1) + (1-b*dt)*x(3,j-1)];
        if mod(j,2) == 0
            J_M = [(1-s*dt) s*dt 0 ; (r-x(3,j))*dt 1-dt -dt*x(1,j) ; dt*x(2,j) dt*x(1,j) 1-b*dt];
            f(:,c) = (J_M)'*R^(-1)*(Z(:,c)-x(:,j));
            c = c+1;
        end
    end
    
    lambda = zeros(N_var,N_obs);
    lambda(:,N_obs) = f(:,N_obs);
    
    for j =  N_obs-1:-1:1
        in = T_ob(j);
        J_M = [(1-s*dt) s*dt 0 ; (r-x(3,in))*dt 1-dt -dt*x(1,in) ; dt*x(2,in) dt*x(1,in) 1-b*dt];
        lambda(:,j) = (J_M)' * lambda(:,j+1) + f(:,j);
    end
    
    J_M = [(1-s*dt) s*dt 0 ; (r-x(3,1))*dt 1-dt -dt*x(1,1) ; dt*x(2,1) dt*x(1,1) 1-b*dt];
    dJ = -J_M'*lambda(:,1); % convergence critria [||dJ||<tol]
    nor = norm(dJ);
    
    if mod(N_it,10) == 0
        fprintf('Iteration# %i \t : \t Norm = %f \n',N_it,nor);
    end
    
    
end
 
%% Extrapolating data for future timesteps 
T_fr = [1:N_ts N_ts:Ts_fin];
X_fr = zeros(N_var,size(T_fr,2));
X_fr(:,1:N_ts) = x;
for j = N_ts+1:Ts_fin
    X_fr(:,j) = [(1-s*dt)*X_fr(1,j-1) + s*dt*X_fr(2,j-1);...
        r*dt*X_fr(1,j-1) - dt*X_fr(1,j-1)*X_fr(3,j-1) + (1-dt)*X_fr(2,j-1);...
        dt*X_fr(1,j-1)*X_fr(2,j-1) + (1-b*dt)*X_fr(3,j-1)];
end
X_fr(3,end) = X_an(3,end);

%% Plotting results
figure(1); clf;

hold on;
for i = 1:3
    fign = ['31' num2str(i)];
    Var_name = ['xyz'];
    subplot(fign);
    hold on;
    
    plot(T_an,X_an(i,:),'k-','linewidth',2);
    plot(T_fr,X_fr(i,:),'.','linewidth',1,'color',[0.2 0.6 0.2]);
    plot(T_ob,Z(i,:),'ro','markerfacecolor',[0.8 0.4 0.4],'markersize',4);
    
    xlabel('$Time step$','interpreter','latex');
    ylabel(Var_name(i),'interpreter','latex');
    lgd = legend('analytical','forecast','observation','location','S','orientation','horizontal');
    title(['Lorenz model [\sigma^2 =' num2str(sig_sq(1)) ']']);
    xlim([0 Ts_fin]);
    hold off;
end

% MAE 5010 : HW2
% Problem# 2
clear all; clc;

%% Parameters
mu = 0;
sig_sq = 0.5;
N_obs = 20; % number of observation points
N_ts = 50; % number of time steps
al = 1e-2; % step length for update of x0 values
tol = 1e-2; % tolerance for convergence

%% Generating twin experiment observations
k = datasample([2:1:N_ts],N_obs,'replace',false); % times at which observation is available
k = sort(k); % times at which observation is available sorted in ascending order for convenience

x0 = 0.5;
% x(1) = 4*x0*(1-x0);
x(1) = x0^2;
c = 1;
for j = 2:N_ts
%     x(j) = 4*x(j-1)*(1-x(j-1));
    x(j) = x(j-1)^2;
    if c <= N_obs && j == k(c)
        Z(c) = x(j); % generating value at the observation points
        c = c+1;
    end
end
noise = normrnd(mu,sig_sq,[1,length(Z)]); % noise in the observation
Z = Z + noise; % generating noisy observation data
clear x x0;

%% Solution by (non-linear) 4D-VAR method
W = sig_sq*eye(N_obs);
R = diag(W);
x0 = 0.8; % starting guess value of x0

% Calculating all x and f values based on first guess x0 value
% x(1) = 4*x0*(1-x0);
x(1) = x0^2;
c = 1;
f = zeros(1,N_obs);
for j = 2:N_ts
%     x(j) = 4*x(j-1)*(1-x(j-1)); % x values
    x(j) = x(j-1)^2; % x values
    if c <= N_obs && j == k(c)
%         f(c) = 4*(1-2*x(j))/R(c)*(Z(c)-x(j)); % f values
        f(c) = 1/R(c)*(Z(c)-x(j)); % f values
        c = c+1;
    end
end

lambda = zeros(1,N_obs);
lambda(N_obs) = f(N_obs);
for j = N_obs-1:-1:1
    in = k(j);
%     lambda(j) = 4*(1-2*x(in))*lambda(j+1) + f(j);
    lambda(j) = 2*x(in)*lambda(j+1) + f(j);
end

% dJ = -4*(1-2*x0)*lambda(1); % convergence critria [|dJ|<tol]
dJ = -2*x0*lambda(1); % convergence critria [|dJ|<tol]
nor = norm(dJ);

% Iteration for convergence of x0 value
N_it = 0; % iteration counter
while nor > tol
    
%     if N_it >= 20000
%         break;
%     end
    N_it = N_it + 1; % increment of iteration counter
    
    x0 = x0 - al*dJ; % updated value of x0
    % Calculating all x and f values based on x0 value from current update
%     x(1) = 4*x0*(1-x0);
    x(1) = x0^2;
    c = 1;
    f = zeros(1,N_obs);
    for j = 2:N_ts
%         x(j) = 4*x(j-1)*(1-x(j-1)); % x values
        x(j) = x(j-1)^2; % x values
        if c <= N_obs && j == k(c)
%             f(c) = 4*(1-2*x(j))/R(c)*(Z(c)-x(j)); % f values
            f(c) = 1/R(c)*(Z(c)-x(j)); % f values
            c = c+1;
        end
    end
    
    lambda = zeros(1,N_obs);
    lambda(N_obs) = f(N_obs);
    for j = N_obs-1:-1:1
        in = k(j);
%         lambda(j) = 4*(1-2*x(in))*lambda(j+1) + f(j);
        lambda(j) = 2*x(in)*lambda(j+1) + f(j);
    end
    
%     dJ = -4*(1-2*x0)*lambda(1); % convergence critria [|dJ|<tol]
    dJ = -2*x0*lambda(1); % convergence critria [|dJ|<tol]
    nor = norm(dJ);
    
    fprintf('Iteration# %i \t : \t x0= %f ; \t dJ= %f \n',N_it,x0,dJ);

end

fprintf('x0 = %f \n',x0);

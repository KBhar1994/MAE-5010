% MAE 5010 : HW2
% Problem# 1
clear all; clc;

%% Parameters
% Algorithm parameters
mu = 0;
sig_sq = 0.5;
N_ts = 50;
tol = 1e-4;
al = 1e-2;

% Model parameters
M = 1;
H = 1;

%% Generating twin experiment observations
x = ones(1,N_ts);
Z = zeros(1,N_ts);
noise = normrnd(mu,sig_sq,[1,length(x)]); % noise in the observation (Gaussian)
Z = x + noise; % generating noisy observation
O1 = [Z(1:N_ts)];
k1 = [1:N_ts];
O2 = [Z(1),Z(5:5:N_ts)];
k2 = [1 5:5:N_ts];
O3 = [Z(1),Z(10:10:N_ts)];
k3 = [1 10:10:N_ts];

%% Solution by (linear) 4D-VAR method
x0 = 0.01; % starting guess value for x0
tol = 1e-4; % tolerance for convergence
N_itr = []; % number of iteration needed for convergence for each observation set
X0 = []; % x0 values from minimization for each observation set

for i = 1:3
    
    clear x Z O W R lambda f r p bet rtr
    eval(['O = O' num2str(i) ';']);
    eval(['k = k' num2str(i) ';']);
    Z = O;
    N_obs = length(Z);
    
    W = sig_sq*eye(N_obs); % weight matrix in the cost function
    R = diag(W); % diagonals of the weight matrix
    
    c = 1;
    x = zeros(1,N_ts);
    f = zeros(1,N_obs);
    for j = 1:N_ts
        if j == 1
            x(j) = M*x0;
        else
            x(j) = M*x(j-1);
        end
        if c <= N_obs && j == k(c)
            f(c) = H'/R(c)*(Z(c)-H*x(j));
            c = c+1;
        end
    end
    
    lambda = zeros(1,N_obs);
    lambda(N_obs) = f(N_obs);
    for j = N_obs-1:-1:1
        lambda(j) = M'*lambda(j+1) + f(j);
    end
    
    dJ = -M'*lambda(1);
    nor = norm(dJ);
    
    A = 0;
    b = 0;
    for  j = 1:N_obs
        A = A + (M^j)'*H'/R(j)*H*M^j;
        b = b + (Z(j)'/R(j)*H*M^j)';
    end
    r = b - A*x0;
    p = r;
    
    % Iteration for convergence of x0 value
    N_it = 0; % iteration counter
    while nor > tol
        
        N_it = N_it + 1; % increment of iteration counter
        
%         % Conjugate-gradient algorithm for update of x0
%         rtr = r'*r;
%         al = rtr/(p'*A*p);
%         x0 = x0 + al*p;
%         r = r - al*p;
%         bet = r'*r/rtr;
%         p = r + bet*p;
        
        x0 = x0 - al*dJ; % gradient method for update of x0 
        
        c = 1;
        x = zeros(1,N_ts);
        f = zeros(1,N_obs);
        for j = 1:N_ts
            if j == 1
                x(j) = M*x0;
            else
                x(j) = M*x(j-1);
            end
            if c <= N_obs && j == k(c)
                f(c) = H'/R(c)*(Z(c)-H*x(j));
                c = c+1;
            end
        end
        
        lambda = zeros(1,N_obs);
        lambda(N_obs) = f(N_obs);
        for j = N_obs-1:-1:1
            lambda(j) = M'*lambda(j+1) + f(j);
        end
        
        dJ = -M'*lambda(1);
        nor = norm(dJ);
        
    end
    
    X0 = [X0 x0];
    N_itr = [N_itr N_it];
    
end

X0
N_itr

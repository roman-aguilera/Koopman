%% My code is a slight modification to the example code in the following video:
%% https://youtu.be/KAau5TBU0Sc?t=499

%% The lecture for theory behind the DMD algorithm is found here:
%% https://youtu.be/bYfGVQ1Sg98?t=1958
%% Kutz starts stepping through the algorithm at 25:00

%%

clear all, close all, clc

%% create a signal (training set)
xi  = linspace(-10, 10, 400); %400 measurements %vector of 400 data points from -10 to 10
t   = linspace(0, 2*pi, 200); %200 points in time (between 0 to 2pi)
dt = t(2) - t(1);  % dt = 
[Xgrid, T] = meshgrid(xi,t); % meshgrid builds an x direction and t direction % 2 matrices with dimensions 200x400 % Xgrid is 200 (length of t) rows, each row is a copy of xi % Tgrid is a 200x400 matrix, 400 (length of xi) columns, each column is a copy of t 
%goal here is to compose 2 signals, and then see if we can separate the signals
%create two spatio-temporal patterns
%this DMD example assumes that you have 2 distinct time signals
f1=sech(Xgrid+3).*(1*exp(1i*2.3*T));
f2=(sech(Xgrid).*tanh(Xgrid)).*(2*exp(1i*2.8*T));
f=f1+f2;


%% repeat signal creation for longer time horizon (test set)
xi_extended  = linspace(-10, 10, 400); %400 measurements
t_extended = linspace(0,8*pi, 800); %400 points in time (between 0 to 4pi)
dt_extended = t_extended(2) - t_extended(1);  %
[Xgrid_extended, T_extended] = meshgrid(xi_extended, t_extended); % meshgrid builds an x direction and t direction

f1_extended =sech(Xgrid_extended+3).*(1*exp(1i*2.3*T_extended));
f2_extended =(sech(Xgrid_extended).*tanh(Xgrid_extended)).*(2*exp(1i*2.8*T_extended));
f_extended_2 = f1_extended + f2_extended;

%% create plots for signal 1, signal 2, training set (signal1 + signal2), test set (signal1_extended + extended_signal2_extended) 

figure(1)
subplot(2,3,1), surfl(Xgrid, T, real(f1)); shading interp, colormap(gray), title('Signal 1') %plot signal 1
subplot(2,3,2), surfl(Xgrid, T, real(f2)); shading interp, colormap(gray), title('Signal 2') %plot signal 2

subplot(2,3,3), surfl(Xgrid, T, real(f)); shading interp, colormap(gray), title('Training set: F = Signal1 + Signal2') % plot training set (signal 1 + signal 2)
subplot(2,3,4), surfl(Xgrid_extended, T_extended, real(f_extended_2)); shading interp, colormap(gray), title('Test Set: Extended F') %plot test set


%% What would happen if simply did SVD on the data matrix???


[u,s,v] = svd(f.'); %perform PCA, made sure to transpose f

figure(2) %visualize svd
new_line = newline;
%title_new = "Normalization of eigenvalues (singlular values)";
%title_new = title_new + new_line  + "from SVD on full data matrix";
title_new = "Normalization of eigenvalues (singlular values)" + new_line  + "that come from SVD on full data matrix";

plot(diag(s)/sum(diag(s)), 'ro' ), title(title_new) %notice that first one takes up about 60 percent of variance, second one takes up about 40 percent of variance

figure(3) %take a look at what this gives in terms of time dynamics in terms of PCA %notice forst 2 modes are meaningful, any other modes after are junk
title("SVD on full data Matrix")
subplot(3,1,1), plot(real(u(:,1:2)), 'Linewidth', 2 ), title('left singular vecors U, from SVD on full data matrix ')
subplot(3,1,2), plot(real(v(:,1:2)), 'Linewidth', 2), title('right singular vectors Vs from SVD in full data matrix')
%notice SVD does not take into account time, so the 2 modes are mixed
%SVD goves you a basis which you can expand things in, but it's not the basis you want in the sense that it is not as interpretable (Note: by interpretable we mean in the sense that you can extract the 2 modes that you had back in the original system f=f1+f2 )

%SVD does not group together modes that oscilatte at same frequencies
%This highlights the philosophy behond the DMD algortihm: signals of the same frequency should be grouped together

%Mode 1 should be a hyperbolic secant
%Mode 2 should be a hyperbolic tangent


%% DMD Algorithm
% The outline of the steps is as follows
% STEP 1 : Do SVD on first window of data X1 
% STEP 2 : LOOK IN THE SIMILARITY TRANSFORM VARIABLE A_ORIGINAL (FROM THE EQUATION X2=A_ORIGINAL*X1).
            % BUILD A_TILDE BY PROJECTING A_ORIGINAL ONTO THE SVD COMPONENTS OF X1 (which are Ur, Sr, and Vr) 
            % A_TILDE IS THE LEAST SQUARE FIT MATRIX THAT TAKES YOU FROM ONE SNAPSHOT(X1) TO ANOTHER SNAPSHOT(X2).
            % A_TILDE IS ESSENTIALLY A SIMILARITY TRANSFORM WITH A_ORIGINAL, BUT NOW THE SIMILARITY TRANSFORM IS IN THE LOW RANK EMBEDDING SUBSPACE OF X1
% STEP 3 : GET EIGENVALUES AND EIGENVECTORS(A.K.A. EIGENFUNCTIONS) OF A_TILDE
% STEP 4 : PROJECT DMD MODES (EIGENVECTORS OF A_TILDE) ONTO ORIGINAL HIGH
            % DIMENSIONAL SPACE


X=f.'; %Make a Data Matrix X %transpose f so that rows are spatial elements, and columns are time snapshots (this is usually how DMD is presented)
%DMD breaks up your data matrix into 2 pieces X1 and X2
X1=X(:, 1:end-1); % the first piece of data is your data matrix, but excludes the LAST column (this is the first window in time)
X2=X(:,2:end); % the second piece of data is your data matrix, but excludes the FIRST column (this is the second window in time)

% STEP 1 OF DMD : SVD ON FIRST WINDOW OF DATA
[U,S,V] = svd(X1, 'econ'); %svd(X1, 'econ'); %Step 1 of DMD: do SVD on first window of data X1
r=2; %r-rank (2-rank) truncation to find low dimensional space to represetn data (notice that if we increase r to 3, the third phi in the plot is a "nonsmooth" vector)
Ur=U(:,1:r); %low dimensional subspace Ur is the rank reduced U (all rows, first r columns of U)
Sr=S(1:r,1:r); %rank reduced S (first r rows and first r columns of S)
Vr=V(:,1:r); %rank reduced V (all rows, first r olumns of V)

Atilde = Ur'*X2*Vr/Sr; % Step 2 OF DMD: LOOK IN THE SIMILARITY TRANSFORM VARIABLE %build Atilde (least square fit matrix) that takes you from one snapshot into another %its essentially a similarity transform with A, but now you're doing it in low rank subspace % Ur'*X2*Vr*inv(Sr); %Atilde=Ur'*X2*Vr/Sr; % Step 2 of DMD: LOOK IN THE SIMILARITY TRANSFORM VARIABLE %build Atilde (least square fit matrix) that takes you from one snapshot into another %its essentially a similarity transform with A, but now you're doing it in low rank subspace
[W,D] = eig(Atilde);   % Step 3 OF DMD: GET EIGENVALUES AND EIGENVECTORS %find eigen decomposition (eigenfunctions(eigenvectors) and eigenvalues) of Atilde
Phi = X2*Vr/Sr*W;      % Step 4 OF DMD: PROJECT BACK OUT ONTO REAL SPACE % Phi contains the DMD modes in the original measurement space coordinates% %X2*Vr*inv(Sr)*W; %Phi=X2*Vr/Sr*W; 

lambda = diag(D); %diagonal matrix ofeigenvalues of Atilde
omega =  log(lambda)/dt; %called omega in poncare space % this is trying to solve for omega in the equation lambda=e^omega*t

figure (4), 
subplot(3,1,2), hold on, plot(real(Phi), 'Linewidth', 2 ), title('Phi s') % add to first plot of figure 3 %purple and yellw are DMD modes


x1=X(:,1); %initial condition, aka first time snapshot at time zero
b=Phi\x1; %get the b vector which tells you how much of each mode is going on at a particular time %remember that phi*b = x0, therefore b = phi\x0


% essentially what DMD does is it takes a time sries and finds the best b's,
    % the best modes (phi's) and the best omegas (w's aka frequencies),
    %by which to represent your dynamical system as a linear one so that we
    %can approximate the dynamical system and its transition into the future, starting from an initial condition 

t_horizon_vec = t;
t_horizon_vec_extended = t_extended;
%t2=linspace(0,20*pi,200); %prediction horizon   
time_dynamics_extended = zeros(r, length(t_horizon_vec_extended) ); %we have r rows, and t columns


for iter=1:length(t_horizon_vec)    
    time_dynamics(:,iter)=(b.*exp(omega*t_horizon_vec(iter))); %space-by-time_matrix = space_matrix*exponent_time_matrix 
end

for iter=1:length(t_horizon_vec_extended)    
    time_dynamics_extended(:,iter)=(b.*exp(omega*t_horizon_vec_extended(iter))); %space-by-time_matrix = space_matrix*exponent_time_matrix 
end
    
X_dmd = Phi * time_dynamics; %DMD solution we multiply b*exp(w*t) with phi
X_dmd_extended =  Phi * time_dynamics_extended;

figure(1)
%subplot(2,2,4), surfl(Xgrid, T, real(X_dmd).'); shading interp, colormap(gray)
subplot(2,3,5), surfl(Xgrid_extended, T_extended, real(X_dmd_extended).'); shading interp, colormap(gray), title('Reconstructed F using DMD') %plot prediction





%% === Alg.1 + RDKF (All-data, CA-space, Rank Checks, Ahat/Bhat export) ===
% 입력: outSim.X(12xT), outSim.Thrust(8xT) 또는 Thrust_cmd(8xT), dt 또는 outSim.time
% 상태/출력: x = [phi theta psi phidot thetadot psidot z zdot]
% 참고: Alg.1의 (10)–(12), 블록 인덱싱, RDKF 업데이트는 논문 Algorithm 1을 따름. [p.5]
% 데이터 가정 체크: Assumption 2,3(필수) + 4(선택, prior state 없는 경우용) [p.3, p.6]
% (Duan et al., "Robust Data-Driven Kalman Filtering...", 2025)

clearvars -except outSim dt Ac Bc A_true B_true M_CA M_alloc MC; clc;

%% 0) 옵션 =====================================================================
% 데이터/전처리
USE_THRUST_CMD = false;     % true: Thrust_cmd, false: Thrust
UNWRAP_YAW     = false;     % yaw unwrap
USE_CA_SPACE   = true;      % 8ch -> 4ch(τx,τy,τz,Fz) 권장
RESTORE_8CH    = true;      % 4ch 식별 후 B(8ch) 복원 저장

% 세그먼트 스택(Alg.1) — Assumption-2: L >= max(n,m,p)  [p.3]
OVERLAP        = false;     % false: stride=L, true: stride=1
L_PAD          = 2;         % L = max(n,m,p) + L_PAD

% 랭크 체크(Assump-3,4)  [p.3, p.6]
CHECK_ASSUMP4  = true;      % prior state가 없을 때를 위한 rank(U)=2Lm도 리포트
RANK_TOL_SCALE = 100;       % 수치 랭크 tol = SCALE*eps*max(size(A))

% RDKF 파라미터 (Alg.1 Step 2, (12)) [p.5]
eps        = 1e-4;
psi_A      = 1.0;
psi_B      = 1.0;
lam_factor = 2.0;
USE_JOSEPH = false;

%% 1) 데이터 구성 ==============================================================
assert(isfield(outSim,'X'),'outSim.X (12xT) 필요');
if exist('dt','var') && ~isempty(dt)
    Ts = dt;
elseif isfield(outSim,'time') && numel(outSim.time)>1
    Ts = median(diff(outSim.time(:)));
else
    Ts = 0.005;
end
fprintf('Ts = %.6f s\n', Ts);

idx  = [7 9 11 8 10 12 5 6];  % [phi theta psi phidot thetadot psidot z zdot]
Xfull = outSim.X;             % 12 x T
Tfull = size(Xfull,2);

X_all = Xfull(idx,:).';       % T x n
if UNWRAP_YAW, X_all(:,3) = unwrap(X_all(:,3)); end
Y_all = X_all;                % y = x
n = size(X_all,2);  p = n;

% 입력(8ch) -> 필요 시 4ch 제어공간으로
if USE_THRUST_CMD
    U8 = outSim.Thrust_cmd;  assert(~isempty(U8),'outSim.Thrust_cmd 필요');
else
    U8 = outSim.Thrust;      assert(~isempty(U8),'outSim.Thrust 필요');
end
U_all = U8.';     % T x 8
m8 = size(U_all,2);

% NaN 방어
valid = all(~isnan([X_all U_all Y_all]),2);
X_all = X_all(valid,:);  U_all = U_all(valid,:);  Y_all = Y_all(valid,:);
T = size(X_all,1);  tt = (0:T-1)'*Ts;

% 8ch -> 4ch(τx,τy,τz,Fz) (권장)
if USE_CA_SPACE
    if ~(exist('M_CA','var')==1 && isequal(size(M_CA),[4 8]))
        if exist('M_alloc','var')==1 && isequal(size(M_alloc),[4 8])
            M_CA = M_alloc;
        else
            if ~exist('MC','var'), MC=struct; end
            if ~isfield(MC,'L'),  MC.L  = 0.2; end
            if ~isfield(MC,'cT'), MC.cT = 0.023; end
            M_CA = [ 2*MC.L,  MC.L,  -MC.L, -2*MC.L,  2*MC.L,   MC.L, -MC.L, -2*MC.L;  % roll
                      MC.L,   MC.L,   MC.L,   MC.L,   -MC.L,  -MC.L, -MC.L,   -MC.L;   % pitch
                     -MC.cT, -MC.cT,  MC.cT,  MC.cT,   MC.cT,  MC.cT, -MC.cT,  -MC.cT; % yaw
                      -1,     -1,     -1,     -1,      -1,     -1,    -1,      -1    ];% Fz
        end
    end
    U_all = U_all * M_CA.';   % T x 4  (u_ca = M_CA*u8)
    m = 4;
    inputNames = {'tau_x','tau_y','tau_z','Fz'};
else
    m = m8;
    inputNames = arrayfun(@(i)sprintf('u%d',i-1),1:m,'UniformOutput',false);
end
fprintf('Input std per channel: '); disp(std(U_all,0,1));

%% 2) 세그먼트 스택 (Alg.1) + 랭크 체크 =========================================
L = max([n,m,p]) + L_PAD;         % Assumption-2 [p.3]
stride = OVERLAP*1 + (~OVERLAP)*L;

% 전 구간 사용 (mask 없이 true)
mask_for_segments = true(T,1);
[Xs,Us,Ys,ks_used] = build_segments_with_mask(X_all,U_all,Y_all,L,stride,mask_for_segments);
Nseg = size(Xs,2);
assert(Nseg>=1,'세그먼트가 없습니다. (L/stride/T 확인)');

if L < max([n,m,p])
    warning('Assumption 2 위반: L(=%d) < max(n,m,p)=(%d). L_PAD를 늘리세요.', L, max([n,m,p]));
end

Theta  = [Xs; Us];                % (n+Lm) x Nseg   ← 논문 (11)의 [X;U]
Zsharp = Ys / Theta;              % = Ys * pinv(Theta)

% 블록 인덱싱 [p.5, (11)]
G1     = Zsharp(1:L*p,          1:n);
G2     = Zsharp(p+1:L*p+p,      1:n);
G3     = Zsharp(p+1:L*p+p,      n+1:n+m);
Csharp = G1(1:p,1:n);

% Assumption-3: rank([X;U]) = n + Lm  [p.3]
rTheta = numrank(Theta, RANK_TOL_SCALE);
fprintf('[Assump-3] rank([X;U]) = %d  (target %d = n+Lm). Nseg=%d\n', rTheta, n+L*m, Nseg);
if rTheta < (n+L*m)
    warning('Assumption 3 미달: rank([X;U]) < n+Lm. 입력 유발성/세그먼트 길이/겹침을 조절하세요.');
end

% G1의 col-rank 체크 (Theorem 1에서 중요)  [p.6]
rG1 = numrank(G1, RANK_TOL_SCALE);
fprintf('rank(G1) = %d (target n=%d), cond(G1)=%.3e, cond(Theta)=%.3e\n', rG1, n, cond(G1), cond(Theta));
if rG1 < n
    warning('G1가 full column rank 아님 → A#,B# 추정이 불안정할 수 있습니다.');
end

% (선택) Assumption-4: rank(U)=2Lm (prior state 없음 시)  [p.6]
if CHECK_ASSUMP4
    L2 = 2*L;
    if (size(X_all,1) >= L2+1)
        [~,Us2,~,ks2] = build_segments_with_mask(X_all,U_all,Y_all,L2,stride,true(T,1));
        rU2 = numrank(Us2, RANK_TOL_SCALE);
        fprintf('[Assump-4] rank(U) with 2L = %d (target %d = 2Lm), segments=%d\n', rU2, 2*L*m, numel(ks2));
        if rU2 < 2*L*m
            warning('Assumption 4 미달(참고): prior state 없이 진행하려면 입력 유발성 강화 필요.');
        end
    else
        fprintf('[Assump-4] 2L 세그먼트를 만들 데이터 길이가 부족하여 생략.\n');
    end
end

%% 3) A#, B# (Alg.1 (10)) ======================================================
Asharp = pinv(G1) * G2;                             % n x n
Bsharp = pinv_stable(G1, 1e-6*norm(G1,'fro')) * G3; % n x m
fprintf('[Alg.1] size(A#)=%dx%d, size(B#)=%dx%d\n', size(Asharp), size(Bsharp));

% (선택) 4ch->8ch 복원
Bsharp_8 = [];
if USE_CA_SPACE && RESTORE_8CH
    Bsharp_8 = Bsharp * M_CA;  % n x 8
end

%% 4) Q, R, λ ===================================================================
res = X_all(2:end,:) - (X_all(1:end-1,:)*Asharp.' + U_all(1:end-1,:)*Bsharp.');
Q = (size(res,1)>1)*cov(res,1) + (size(res,1)<=1)*1e-6*eye(n);
Q = (Q+Q')/2 + 1e-12*eye(n);

y_std = std(Y_all,0,1);  y_std(y_std==0)=1.0;
R = diag((1e-3*max(y_std,1e-6)).^2);
R = (R+R')/2 + 1e-12*eye(p);

% λ > || ε C#^T R^{-1} C# ||_2  [Alg.1 Step 2, p.5]
lam_min = norm(Csharp'*(R\Csharp), 2);
lam = lam_factor * eps * lam_min;
tries=0;
while true
    Rhat_test = R - (eps/lam)*(Csharp*Csharp');
    evmin = min(eig((Rhat_test+Rhat_test')/2));
    if evmin>1e-12 || tries>8, break; end
    lam = lam*1.5; tries=tries+1;
end
fprintf('lambda=%.3e (need > %.3e), bump tries=%d\n', lam, eps*lam_min, tries);

%% 5) RDKF (Alg.1 (12)) — 전 구간 한 번에 =======================================
xhat = zeros(T,n);
Ahat_hist = NaN(n,n,T);  Bhat_hist = NaN(n,m,T);
Ahat_last = NaN(n,n);     Bhat_last = NaN(n,m);

x = X_all(1,:).';   P = eye(n)*1e-1;   I = eye(n);
xhat(1,:) = x.';
for k = 1:T-1
    Pinv = inv(P + 1e-12*I);
    Phat = inv(Pinv + lam*(psi_A^2)*eps*I);
    Ahat = Asharp * (I - lam*(psi_A^2)*eps*Phat);
    Bhat = Bsharp - lam*(psi_B^2)*eps*(Phat*Bsharp);
    Rhat = R - (eps/lam)*(Csharp*Csharp'); Rhat=(Rhat+Rhat')/2 + 1e-12*eye(p);

    % KF step (y_{k+1})
    x_pred = Ahat*x + Bhat*U_all(k,:).';
    Pbar   = Asharp*Phat*Asharp' + Q;
    S      = Rhat + Csharp*Pbar*Csharp';
    K      = (Pbar*Csharp')/S;

    innov  = Y_all(k+1,:).' - Csharp*x_pred;
    x      = x_pred + K*innov;

    if USE_JOSEPH
        IKC = I - K*Csharp;
        P = IKC*Pbar*IKC' + K*Rhat*K';
    else
        P = Pbar - Pbar*Csharp'/S*Csharp*Pbar;
    end
    P=(P+P')/2;

    xhat(k+1,:) = x.';
    Ahat_hist(:,:,k+1) = Ahat;   Bhat_hist(:,:,k+1) = Bhat;
    Ahat_last = Ahat;            Bhat_last = Bhat;   % 마지막 값 저장
end

% RMSE
rmse_all = sqrt(mean((xhat - X_all).^2,1));
fprintf('RMSE (per state):\n'); disp(rmse_all);

%% 6) 저장/출력 ==================================================================
save('alg1_rdkf_all_result.mat', ...
     'Asharp','Bsharp','Bsharp_8','Csharp','xhat','rmse_all','Ts','L', ...
     'Ahat_hist','Bhat_hist','Ahat_last','Bhat_last','M_CA','USE_CA_SPACE');

% 숫자 CSV
writematrix(Asharp,     'Asharp.csv');
writematrix(Bsharp,     'Bsharp.csv');
writematrix(Ahat_last,  'Ahat_last.csv');   % A_hat 마지막 시점
writematrix(Bhat_last,  'Bhat_last.csv');   % B_hat 마지막 시점
writematrix(xhat,       'xhat.csv');
if ~isempty(Bsharp_8),  writematrix(Bsharp_8,'Bsharp_8.csv'); end

% 라벨 포함 테이블
stateNames = {'phi','theta','psi','phidot','thetadot','psidot','z','zdot'};
T_A  = array2table(Asharp,    'RowNames',stateNames,'VariableNames',stateNames);
T_B  = array2table(Bsharp,    'RowNames',stateNames,'VariableNames',inputNames);
T_Ah = array2table(Ahat_last, 'RowNames',stateNames,'VariableNames',stateNames);
T_Bh = array2table(Bhat_last, 'RowNames',stateNames,'VariableNames',inputNames);
writetable(T_A,  'Asharp_table.csv','WriteRowNames',true);
writetable(T_B,  'Bsharp_table.csv','WriteRowNames',true);
writetable(T_Ah, 'Ahat_last_table.csv','WriteRowNames',true);
writetable(T_Bh, 'Bhat_last_table.csv','WriteRowNames',true);

% 간단 플롯
t = tt;
figure('Name','Angles [rad]'); plot(t, X_all(:,1:3)); hold on; plot(t, xhat(:,1:3),'--');
grid on; xlabel('t [s]'); ylabel('rad'); legend({'\phi','\theta','\psi','\phî','\thetâ','\psî'});

figure('Name','Rates [rad/s]'); plot(t,X_all(:,4:6)); hold on; plot(t,xhat(:,4:6),'--');
grid on; xlabel('t [s]'); ylabel('rad/s'); legend({'\dot\phi','\dot\theta','\dot\psi','\dot\phî','\dot\thetâ','\dot\psî'});

figure('Name','z & zdot'); plot(t,X_all(:,7:8)); hold on; plot(t,xhat(:,7:8),'--');
grid on; xlabel('t [s]'); ylabel('[m],[m/s]'); legend({'z','\dot z','ẑ','\dot ẑ'});

%% ===== 로컬 함수들 =============================================================
function [Xs,Us,Ys,ks_used] = build_segments_with_mask(X,U,Y,L,stride,mask)
    n=size(X,2); m=size(U,2); p=size(Y,2);
    ks_all = 1:stride:(size(X,1)-L);
    keep = false(size(ks_all));
    for i=1:numel(ks_all)
        s = ks_all(i);
        if all(mask(s:s+L))   % y는 L+1포인트 → mask에 s..s+L 포함
            keep(i)=true;
        end
    end
    ks_used = ks_all(keep);
    Nseg = numel(ks_used);
    Xs = zeros(n,Nseg); Us=zeros(L*m,Nseg); Ys=zeros((L+1)*p,Nseg);
    for i=1:Nseg
        s = ks_used(i);
        Xs(:,i) = X(s,:).';
        Us(:,i) = reshape(U(s:s+L-1,:).', L*m, 1);
        Ys(:,i) = reshape(Y(s:s+L,:).',   (L+1)*p, 1);
    end
end

function r = numrank(A, scale)
    if nargin<2, scale = 100; end
    s = svd(A);
    tol = scale*eps(max(s))*max(size(A));
    r = sum(s > tol);
end

function Xdag = pinv_stable(X, tau)
    if nargin<2 || isempty(tau), tau = 1e-6*norm(X,'fro'); end
    [U,S,V] = svd(X,'econ'); s = diag(S);
    Xdag = V*diag(s./(s.^2 + tau^2))*U';
end

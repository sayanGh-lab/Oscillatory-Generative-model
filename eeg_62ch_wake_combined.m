% study with 2 channel
clear all
close all
fs=500;
dt = 1/fs;
n = 200;   
nepochs = 30; 
eta_omega = 0.0001; 
eta_alpha = 0.0001; 
eta_W = 0.0001;
eps = 0.5;  
eps_w = 0.001;
beta1=-20;
%%
tmpa = (2*pi*rand(n,1));
S = exp(1i*tmpa);
w = (S*S');
w = w - eye(n).*w;
A = abs(w);  
phi_ang = angle(w);
phi = phi_ang; % angle of lateral connection 
w = eps_w*A .*exp(1i*phi); % Lateral connection between oscilator W,
distance = 5;
A = nearbyA(n, eps_w, distance); 
A = A - eye(n)*eps_w;   % eye=identity matrix
phi0 = phi;
alpha = 0.2*ones(n,1);  % previously 0.2* % initiate the alpha ,alpha is the real feed fowrad weight form oscilator to o/p node
alpha0 = alpha;
omega = 0.5 + 128.5*rand(n,1);  % prev 39.5  % natural frequency of oscilator
omega = sort(omega, 1, 'ascend');
omega0 = omega;
omegaarr = zeros(n,nepochs);

%%% loading signal %%%
load('eeg_all_ch_wake') %
% chnl=24;
N = 5000;  % 10sec Data length

% filtering %
wp= [0.1 20] /250;    % pass band edge fre
ws=[0.01 100]/250;    % stop band  edge fre
del1=.3;
del2=0.03;
Rp = -20*log10(1-del1);        % pass band ripple
Rs=   -20*log10(del2);         % stop band attenuation
[n1,wn]=buttord(wp,ws,Rp,Rs)    % determine the order of the filter
[b,a] = butter(n1,wn);   % fulter coefficient
    freqz(b,a,250,500); title(' Butterworth bandpass Filter')  % frequency response
        lendata1=length(eeg_data);  % original length of extracted signal
filtered_signals1 = zeros(62,lendata1);
    for i=1:62
    
    filtered_signals1(i,:) = filter(b, a, eeg_data(i,:));
    filtered_signals1(i,:) = filtered_signals1(i,:)/max(abs(filtered_signals1(i,:)));
    filtered_signals1(i,:) = filtered_signals1(i,:) - mean(filtered_signals1(i,:));  % total signal
    end

signal=filtered_signals1;
sig_size=size(signal);
    chnl=sig_size(1);
    alpha_store=zeros(n,chnl);
s_store=zeros(N,chnl);
omega_store=zeros(n,chnl);
phi_store=zeros(n,chnl);

t =[0:1:N-1]*dt;
%%%% 1st phase of training %%%%
pteach_arr = signal*2.5; % paper itself D(t),teaching signal (EEG)
tharr = zeros(N, n);
rarr = zeros(N, n);

for nc=1:chnl
 pteach = pteach_arr(nc,:); 
    for ii = 1:nepochs
        th = 2*pi*rand(n,1); % z=r.e^(i.th),th=angle of oscilator
        r = ones(n,1);
%         fprintf('ch Epoch %d %d of %d\n',nc, ii, nepochs)
        line_length = 0;
        for i = 1:N
%             if mod(i,100)==1
%                 fprintf(repmat('\b',1,line_length))
%                 line_length = fprintf('Progress: %0.3f %%\n', i*100/N);
%             end
            s(i) = sum(alpha.*cos(th));%(Network predicted signal or reconstructed signal)
            e(i) = pteach(i) - s(i);   % ptech=desired signal(EEG Signal),s(i)= reconstructed signal at o/p summation node,e(i)=error signal
            
            r = r + ((1 + beta1*r.^2).*r + sum(A.*repmat(r',n,1).*cos(repmat(omega,1,n).*(-repmat(th,1,n)./repmat(omega,1,n) + repmat(th',n,1)./repmat(omega',n,1) + phi)),2) + eps*e(i)*cos(th))*dt;
            th = th + (omega + sum(A.*(repmat(r',n,1)./repmat(r,1,n)).*sin(repmat(omega,1,n).*(-repmat(th,1,n)./repmat(omega,1,n) + repmat(th',n,1)./repmat(omega',n,1) + phi)),2) - eps*e(i)*sin(th)./r)*dt;
            ind = find(r<0.2);
            r(ind) = ones(size(ind))*0.1;
           
            if sum(isnan(r))
                fprintf('R')
                pause
            end
            if sum(isnan(th))
                fprintf('TH')
                pause
            end           
            % Update omegas
            domega = -eta_omega*e(i)*sin(th);   % omega updating rule(natural fre of oscillator)
            omega  = omega + domega; % traing of omega,updating
            ind = find(omega < 0.1);
            omega(ind) = ones(size(ind))*0.1;  
            % update alpha
            dalpha = eta_alpha*e(i)*cos(th); % for freze the update of alpha eta_alpha=0
            alpha = alpha+dalpha;  % alpha is updating,(Connection between oscilator and o/p node)          
            W = A.*exp(1i*phi./repmat(omega',n,1)); % lateral connection among oscillator,which is complex
            % learning rule for lateral connection among oscillator
            delta_W = -A.*exp(1i*phi./repmat(omega',n,1)) + repmat(r,1,n)*repmat(r',n,1).^(omega*(1./omega')).*exp(1i*((repmat(th,1,n).*repmat(omega',n,1) - repmat(th',n,1).*repmat(omega,1,n))./repmat(omega',n,1)));
            W = W + eta_W*delta_W; % update of complex weight W ,which connects between oscilator,eta_w=Time constant
            phi = angle(W).*repmat(omega',n,1); % angle of lateral connection      
            tharr(i, :) = th;
            rarr(i,:) = r;
        end
        omegaarr(:,ii) = omega;        
    end
    s_store(:,nc)=s;
    omega_store(:,nc)=omega;
%     phi_store(:,:,nc)=phi;
    omegaarr = [omega0,omegaarr];
end
save eeg_1st_phase_of_training(62ch)


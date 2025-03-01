% % 2nd phase of training (without hidden layer wake%)
load('eeg_1st_phase_of_training(62ch)');

th1 = zeros(n,1);  % angle of oscillator newly defined
r1 = ones(n,1); % r -dynamics
tharr1 = zeros(N, n);
rarr1 = zeros(N, n);
fprintf('Running once\n');

for i=1:N    %b1=0.1

  r1 = r1 + ((1 +beta1*r1.^2).*r1 + sum(A.*repmat(r1',n,1).*cos(repmat(omega,1,n).*(-repmat(th1,1,n)./repmat(omega,1,n) + repmat(th1',n,1)./repmat(omega',n,1) + phi)),2) )*dt;
  th1 = th1 + (omega + sum(A.*(repmat(r1',n,1)./repmat(r1,1,n)).*sin(repmat(omega,1,n).*(-repmat(th1,1,n)./repmat(omega,1,n) + repmat(th1',n,1)./repmat(omega',n,1) + phi)),2) )*dt;
  tharr1(i,:) = th1; 
  rarr1(i,:) = r1;
end

% yt=s_store;
  Yd=pteach_arr'; %(Network predicted signal or reconstructed signal after 1st phase
for i = 1:chnl
    Yd(:,i) = Yd(:,i) - mean(Yd(:,i));
    Yd(:,i) = Yd(:,i)/max(abs(Yd(:,i)));
   end
eta_alpha1 = 0.00003; % coefficient of magnitude of alpha
eta_phi1 = 0.000001; % coefficient of angle
nn = 5000;    % previously it was 5000
lossVal = zeros(nn,1);
op1 = zeros(N,chnl);
Wa = rand(n,chnl).*exp(1i*2*pi*rand(n,chnl)); % complex feed foward weight
alpha1 = abs(Wa);% magnitude of complex feed foward weight(In paper Kij)
phi1 = angle(Wa); % angle of complex feed foward weight
fprintf('Optimizing\n');
RMSE=zeros(chnl,nn);
for  ii=1:nn
    ii
    op1 = real((rarr1.*exp(1i*tharr1))*(alpha1.*exp(1i*phi1)));  % rarr1=r dynamics form 1st phase,tharr1=Th(angle of oscilator)
    for i=1:chnl
    RMSE1(i,ii)=sqrt(mean((Yd(:,i) - op1(:,i)).^2));
    end

    dedalpha1 = zeros(n,chnl);
    dedphi1 = zeros(n,chnl);   
      
    for i =1:N    % N= length of the Data
        dedalpha1 = dedalpha1 + (-1)*(rarr1(i,:)'*(Yd(i,:) - op1(i,:))).*cos(repmat(tharr1(i,:)',1,chnl) + phi1); % magnitude of complex feed foward weight
        dedphi1 = dedphi1 + (-1)*(rarr1(i,:)'*(Yd(i,:) - op1(i,:))).*((-alpha1).*sin(repmat(tharr1(i,:)',1,chnl) + phi1));  
    end


    alpha1 = alpha1 - eta_alpha1*dedalpha1; % update of magnitude of alpha after 2nd phase(feed foward weight between oscilator to o/p)
    phi1 = phi1 - eta_phi1*dedphi1;% update of angle of alpha after 2nd phase

end
save network_op(without_hidden_layer)


%% main ploting
ch_name=["fp1","fp2","f7","f3","fz","f4","f8","fc5","fc1","fc2","fc6","t7", "c3","cz","c4","t8","cp5","cp1","cp2","cp6","p7","p3","pz","p4",...
    "p8","po9","o1","oz","o2","po10","af7","af3","af4","af8","f5","f1","f2","f6","ft9","ft7","fc3","fc4",...
    "ft8","ft10","c5","c1","c2","c6", "tp7","cp3","cpz","cp4","tp8","p5", "p1","p2","p6","po7","po3","poz","po4","po8"];
 
Ypp_f = zeros(257,chnl);
Ydd_f = zeros(257,chnl);
 
for i=1:chnl
    [Ypp_f(:,i),f]=pwelch(op1(:,i),hamming(500),0.50,[],500);% window specification 1s with hamming window,overlap 50%
   [ Ydd_f(:,i),f]=pwelch(Yd(:,i),hamming(500),0.50,[],500);% window specification 1s with hamming window,overlap 50%
end
for i=1:62
    figure(i)
subplot(2,2,[1,2]);
plot(t,Yd(:,i),'linewidth',1.4)
hold on 
plot(t,op1(:,i),'linewidth',1.4)
legend('Yd(t)','Yp(t)')


xlabel('Time (sec)');
ylabel('EEG(mV)')
title((ch_name(:,i)))



subplot(2,2,3);
plot(RMSE1(i,:),'linewidth',1.4)
xlabel('epoch')
ylabel('rmse error')

xlim([0 5000]);

title((ch_name(:,i)))
xlabel('epoch')
ylabel('rmse error')


subplot(2,2,4);
plot(f,10*log10(Ydd_f(:,i)))   % spectrum of teaching signal(10 sec-2nd part)
hold on
plot(f,10*log10(Ypp_f(:,i))) 

% plot(f,10*log10(Ydd_f(:,i)))   % spectrum of teaching signal(10 sec-2nd part)
xlabel('Frequency(Hertz)');
ylabel('Magnitude response db/Hz')
title((ch_name(:,i)));
xlim([0 20])
legend('Pr','Pd')

saveas(figure(i),fullfile('E:\figures\',['Spectra' num2str(i) '.jpeg']));

end

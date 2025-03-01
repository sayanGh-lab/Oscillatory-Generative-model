clc
clear all
load('eeg_1st_phase_of_training(62ch)')
N = 5000;
n_osci = 200; % no of oscillator

th2 = zeros(n_osci,1);  % angle of oscillator newly defined
r2 = ones(n_osci,1); % r -dynamics
tharr2 = zeros( n_osci,N);
rarr2 = zeros( n_osci,N);

    Zr = zeros(n_osci,N);
    Zi = zeros(n_osci,N);
    r2(:,1) = ones(n_osci,1);
    th2(:,1) = zeros(n_osci,1);

    %%% learning parameter %%%
etaWosr = 0.001;
etaWosi = 0.001;

% Yd = s_store;   % desired signal
   Yd=pteach_arr';
   for i = 1:chnl
    Yd(:,i) = Yd(:,i) - mean(Yd(:,i));
    Yd(:,i) = Yd(:,i)/max(abs(Yd(:,i)));
   end

for it = 1:N

 r2 = r2 + ((1 +beta1*r2.^2).*r2 + sum(A.*repmat(r2',n,1).*cos(repmat(omega,1,n).*(-repmat(th2,1,n)./repmat(omega,1,n) + repmat(th2',n,1)./repmat(omega',n,1) + phi)),2) )*dt;
  th2 = th2 + (omega + sum(A.*(repmat(r2',n,1)./repmat(r2,1,n)).*sin(repmat(omega,1,n).*(-repmat(th2,1,n)./repmat(omega,1,n) + repmat(th2',n,1)./repmat(omega',n,1) + phi)),2) )*dt;
  tharr2(:,it) = th2; 
  rarr2(:,it) = r2;

end
    Zr = real(rarr2.*exp(1i*tharr2));
    Zi = imag(rarr2.*exp(1i*tharr2));
        
    Z = Zr + 1i*Zi;

Kbs = 200;    %input dimension
dip = Z;

%% network arceteure
hln=100; % hidden layer neuron
W1bsr = rand(hln,Kbs);
W1bsi = rand(hln,Kbs);
W1bs = W1bsr + 1i*W1bsi;


ohl=chnl; % o/p dimension
W2bsr=rand(ohl,hln);
W2bsi=rand(ohl,hln);
W2bs=W2bsr+1i*W2bsi;
ahbs = 0.5;
aobs=0.5;
etaW1bs = 0.001;
etaW2bs = 0.001;

 nepochsbs = 5000; %no of epochs
 RMSE=zeros(chnl,nepochsbs);
 error1 = zeros(1,nepochsbs);
for nep1 = 1:nepochsbs
    nep1    
% Foreard propagation %%%
        nhbs = W1bs*dip;   % 
        nhbsr = real(nhbs); nhbsi = imag(nhbs);
        xhbsr = 2*sigmf(nhbsr,[ahbs,0]) - 1;
        xhbsi = 2*sigmf(nhbsi,[ahbs,0]) - 1;
        xhbs = xhbsr + 1i*xhbsi;      

       nobs = W2bs*xhbs;
       
        nobsr = real(nobs);
        nobsi = imag(nobs);
        ybsr=nobsr;
        ybsi=nobsi;
        ybsr = 2*sigmf(nobsr,[aobs,0]) - 1;
        ybsi = 2*sigmf(nobsi,[aobs,0]) - 1;
        ybs = ybsr + 1i*ybsi;   
ybs = ybsr ;  
ybs=ybs';
% error1(nep1) = error1(nep1) + norm(Yd' - ybs);        
for ii=1:chnl
    RMSE(ii,nep1)=sqrt(mean((Yd(:,ii) - ybs(:,ii)).^2));
    end
% backpropagnation
 
dW2bsr = (-1)*((Yd' - ybsr).*((aobs/2)*(1+ybsr).*(1-ybsr)))*xhbsr';
    
dW2bsi = ((Yd' - ybsr).*((aobs/2)*(1+ybsr).*(1-ybsr)))*xhbsi';                


        
dW1bsr= (-1)*((W2bsr'*((Yd' - ybsr).*((aobs/2)*(1+ybsr).*(1-ybsr)))).*((ahbs/2)*(1+xhbsr).*(1-xhbsr)))*real(dip)' ...
              + ((W2bsi'*((Yd' - ybsr).*((aobs/2)*(1+ybsr).*(1-ybsr)))).*((ahbs/2)*(1+xhbsi).*(1-xhbsi)))*imag(dip)';         

          dW1bsi = ((W2bsr'*((Yd' - ybsr).*((aobs/2)*(1+ybsr).*(1-ybsr)))).*((ahbs/2)*(1+xhbsr).*(1-xhbsr)))*imag(dip)' ...
         + ((W2bsi'*((Yd' - ybsr).*((aobs/2)*(1+ybsr).*(1-ybsr)))).*((ahbs/2)*(1+xhbsi).*(1-xhbsi)))*real(dip)';
  % Update
        W2bsr = W2bsr - etaW2bs*dW2bsr;
        W2bsi = W2bsi - etaW2bs*dW2bsi;
        W2bs = W2bsr + 1i*W2bsi;
        W1bsr = W1bsr - etaW1bs*dW1bsr;
        W1bsi = W1bsi - etaW1bs*dW1bsi;
        W1bs = W1bsr + 1i*W1bsi;
end
    op3=ybs;
save network_op
 %     % ploting
% ch_name=["fp1","fp2","f7","f3","fz","f4","f8","fc5","fc1","fc2","fc6","t7", "c3","cz","c4","t8","cp5","cp1","cp2","cp6","p7","p3","pz","p4",...
%     "p8","po9","o1","oz","o2","po10","af7","af3","af4","af8","f5","f1","f2","f6","ft9","ft7","fc3","fc4",...
%     "ft8","ft10","c5","c1","c2","c6", "tp7","cp3","cpz","cp4","tp8","p5", "p1","p2","p6","po7","po3","poz","po4","po8"];
% % 
% % 
% 
% Ypp_f = zeros(257,chnl);
% Ydd_f = zeros(257,chnl);
% % 
% for i=1:chnl
%     [Ypp_f(:,i),f]=pwelch(op3(:,i),hamming(500),0.50,[],500);% window specification 1s with hamming window,overlap 50%
%    [ Ydd_f(:,i),f]=pwelch(Yd(:,i),hamming(500),0.50,[],500);% window specification 1s with hamming window,overlap 50%
% end
% 
% % % % main ploting
% % for i=1:chnl
% %     figure(i)
% % subplot(2,2,[1,2]);
% % plot(t,Yd(:,i),'linewidth',1.4)
% % hold on 
% % plot(t,op3(:,i),'linewidth',1.4)
% % legend('Yd(t)','Yp(t)')
% % 
% % 
% % xlabel('Time (sec)');
% % ylabel('EEG(mV)')
% % title((ch_name(:,i)))
% % 
% % 
% % 
% % subplot(2,2,3);
% % plot(RMSE(i,:),'linewidth',1.4)
% % xlabel('epoch')
% % ylabel('rmse error')
% % 
% % xlim([0 nepochsbs]);
% % 
% % title((ch_name(:,i)))
% % xlabel('epoch')
% % ylabel('rmse error')
% % 
% % 
% % subplot(2,2,4);
% % plot(f,10*log10(Ypp_f(:,i)))   % spectrum of generated signal(20 sec)
% % hold on
% % plot(f,10*log10(Ydd_f(:,i)))   % spectrum of teaching signal(10 sec-2nd part)
% % xlabel('Frequency(Hertz)');
% % ylabel('Magnitude response db/Hz')
% % title((ch_name(:,i)));
% % xlim([0 20])
% % legend('Pr','Pd')
% % 
% % saveas(figure(i),fullfile('E:\figures\',['Spectra' num2str(i) '.jpeg']));
% % 
% % end

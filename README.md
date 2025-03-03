Instruction:1st run the code (eeg_48ch_wake_combined.m) with nearbyA.m function loaded in the same directory, with a sample time series (For example:   eeg_nrem_n3_all_ch.mat). Then network will learn frequency of oscillator. Also angle of lateral connection will be trained in this phase. Then store the updated frequency and phase values (angle of lateral conncetion) for further use.
 
 Hidden layer equation
(learning rates:   η_h  =0.001; η_o  =0.001).
(r_i ) ̇=(μ+〖〖βr〗_i〗^2 ) r_i+∑_(j=1     ∋ j≠i)^N▒〖A_ij r_j^(ω_i/ω_j ) cos ω_i (θ_j/ω_j -θ_i/ω_i +Ø_ij/(ω_i ω_j )) 〗          		(5)  
θ ̇_i=ω_i+∑_(j=1     ∋ j≠i)^N▒〖A_ij  (r_j^(ω_i/ω_j ))/r_i 〗 □(sin ω_i (θ_j/ω_j -θ_i/ω_i +Ø_ij/(ω_i ω_j )) )                 			(6)
Convert polar coordinate equation (r and θ ) to complex domain according to (z=〖re〗^iθ)
Z_(r )=〖real(rarr2*e〗^(i*tharr2)  )     							(7)
Z_(i )=〖imag(rarr2*e〗^(i*tharr2)  ) 							(8)
Z=Z_r+iZ_i           									(8a)
Where Z is the complex domain activation of hopf oscillator. Z_r is the real part of oscillator activation, Z_i is the imaginary part of oscillator activation.
Forward propagation:
W_lk^f1=W_(lk,R)^f1+iW_(lk,I)^f1    								(9)
Where W_lk^f1   is the feedforward weight between oscillatory layer to 1st hidden layer. These feedforward weights are initialized in complex domain. It has both real (W_(lk,R)^f1) and imaginary part (W_(lk,I)^f1).
W_ml^f2=W_(ml,R)^f2+iW_(ml,I)^f2    								(10)
Where W_ml^f2   is the feedforward weight between 1st hidden layer to output layer. These feedforward weights are initialized in complex domain. It has both real (W_(ml,R)^f2) and imaginary part (W_(ml,I)^f2).

n_l^Hf=∑_k▒〖W_lk^f1 Z_k 〗=∑_k▒(W_(lk,R)^f1 Z_(k,R)-W_(lk,I)^f1 Z_(k,I) ) +i∑_k▒〖(W_(lk,I)^f1 Z_(k,R)+W_(lk,R)^f1 Z_(k,I))〗    	(11)
Where n_l^Hf  is the product of complex oscillatory activation (Z_k) and W_lk^f1.(eqn. 12) It has both real(n_(l,R)^Hf=∑_k▒(W_(lk,R)^f1 Z_(k,R)-W_(lk,I)^f1 Z_(k,I) ) ) and imaginary part (n_(l,I)^Hf=∑_k▒〖(W_(lk,I)^f1 Z_(k,R)+W_(lk,R)^f1 Z_(k,I))〗).
n_l^Hf=n_(l,R)^Hf+in_(l,I)^Hf        								(12)
Both n_(l,R)^Hf and n_(l,I)^Hf are passed through tanh function separatedly. Real part of sigmoid activation (X_(l,R)^Hf=f_R^h (n_(l,R)^Hf ):real part of 1st tanh⁡activation  ) and (X_(l,I)^Hf=f_I^h (n_(l,I)^Hf )) (: imaginary part of 1st tanh activation) and X_l^Hf is the total activation of 1st hidden layer.(eqn. 13 & eqn, 14).
X_l^Hf=f_R^h (n_(l,R)^Hf )+if_I^h (n_(l,I)^Hf )    								(13)
X_l^Hf=X_(l,R)^Hf+iX_(l,I)^Hf    									(14)
Where n_m^o  is the product of 1st hidden layer activation (X_l^Hf) and W_ml^f2.(where W_ml^f2 is the 1st hidden to output node weight, W_(ml,R)^f2 , W_(ml,I)^f2   are the real and imaginary part of W_ml^f2)(eqn. 15) It has both real(n_(m,R)^o=∑_l▒(W_(ml,R)^f2 X_(l,R)^Hf-W_(ml,I)^f2 X_(l,I)^Hf ) ) and imaginary part (n_(m,I)^o=∑_l▒〖(W_(ml,I)^f2 X_(l,R)^Hf+W_(ml,R)^f2 X_(l,I)^Hf)〗).

n_m^o=∑_l▒〖W_ml^f2 X_l^Hf 〗=∑_l▒(W_(ml,R)^f2 X_(l,R)^Hf-W_(ml,I)^f2 X_(l,I)^Hf ) +i∑_l▒〖(W_(ml,I)^f2 X_(l,R)^Hf+W_(ml,R)^f2 X_(l,I)^Hf)〗     	(15)
n_m^o=n_(m,R)^o+in_(m,I)^o      								(16)
At output node (eqn. 17) we consider the only real part of n_m^o, which is n_(m,R)^o. And also at output layer we have tanh neuron.
Y_m=f_R^o (n_(m,R)^o )           									(17)
Backpropagation: The complex domain backpropagation has been adapted from [1].

Loss at every time step,
L(t)=1/2 (O_m (t)-Y_m (t))^2       							(18)
Where , O_m (t) is the desired (target) signal.
∂L/(∂W_(ml,R)^f2 )=(O_(m,R)-Y_(m,R) ) f_R^o' X_(l,R)^Hf        							(19)
(∂L(t))/(∂W_(ml,I)^f2 )=(O_(m,R)-Y_(m,R) ) f_R^o' X_(m,I)^Hf        							(20)
(∂L(t))/(∂W_(lk,R)^f1 )=(-1)∑_m▒〖(O_(m,R)-Y_(m,R) ) f_R^(o^' ) (W_(ml,R)^f2 f_R^(h^' ) Z_(k,R)-W_(ml,I)^f2 f_I^(h^' ) Z_(k,I))〗     		(21)
(∂L(t))/(∂W_(lk,I)^f1 )=∑_m▒〖(O_(m,R)-Y_(m,R) ) f_R^(o^' ) (W_(ml,R)^f2 f_R^(h^' ) Z_(k,I)+W_(ml,I)^f2 f_I^(h^' ) Z_(k,R))〗          			(22)
Rewrite the Activation function:
Sigmoidal activation function:
f(x,a_k,c_k )=1/(1+exp^((-a_k (x-b_k))) )          							(23)
Put , a_k=0.5,b_k=0,
2 f(x,a_k,c_k )-1=(1-exp^(-0.5x))/(1+exp^0.5x )=tanh⁡(x/2)       

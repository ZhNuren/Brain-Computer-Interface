% Here you can change the time interval / frequency range / channel and
% etc.
startup_bbci_toolbox();
ivals = {[81:382],[80:380],[78:380],[75:375],[80:382]}; 
%bandr = {[10 18],[9 13], [8 14], [10 17], [8 30], [5 30], [8 23], [23 30], [10,20], [5 14], [13 25]};
bandr = {[8 16],[10 18], [9 13], [8 14], [10 17], [8 30], [8 23], [20 30], [10,20], [13 25]};

load('BCI_MI_train_se1.mat')
load('BCI_MI_train_se1_Y.mat')
load('BCI_MI_test_se1.mat')
load('BCI_MI_test_se1_Y.mat')

fs=100; % sampling frequency, do not change
% Data point(time series) x channels x samples(trials) x subjects
%ch = {[1:66], [5,13,14,15,35,36,37,38,39,41], [1:30],[30:66]};
%ch = {[1:66], [1:20], [1:40], [5:45], [4:44], [5:60], [1:30],[30:66]};
%ch = {[1:66], [1:50], [5:45], [4:44], [5:60]};
%ch = {[1:66], [2:65], [3:64], [4:62], [5:66]};
ch = {[1:66], [5,13,14,15,35,36,37,38,39,41]};

[nT, nCH, nSample, nSub] = size(BCI_MI_train_se1(:,:,:,:)) 
%[nT, nCH, nSample, nSub] = size(BCI_MI_train_se1(:,:,:,:)) 

acc=[];

for sub=1:nSub
    list=[];
    trfv=[],tefv=[];
    ival = ivals{besttimelist(sub)};
    channel = ch{1};
    nCH = length(channel);
    for b = 1:length(bandr)
        band=bandr{b};
        epo = BCI_MI_train_se1(:,channel,:,sub); % Training data : 400 Time series x 66 channels x 100 trials 
%        epo = BCI_MI_train_se1(:,:,:,sub); % Training data : 400 Time series x 66 channels x 100 trials 

        epo_y = BCI_MI_train_se1_Y(:,sub); % Training label
    
        % Bandpass to the frequency band of interest
        [b,a]= butter(5, band/fs*2);        
        dat = permute(epo, [1 3 2]);
        dat = reshape(dat, [nT*nSample nCH]);
        dat = filter(b, a, dat);
        dat = reshape(dat, [nT nSample nCH]);
    
        % Time inteval selection
        dat = dat(ival, :, :);
    
        % CSP filtering     
        [nT2, nS2, nCH2] = size(dat)
        c1 = dat(:, epo_y == 1, :);
        c2 = dat(:, epo_y == 2, :);
        R1 = cov(reshape(c1, [nT2*50 nCH]));
        R2 = cov(reshape(c2, [nT2*50 nCH]));     
        
        [W, D] = eig(R2, R1+R2);
        CSP_W = W(:, [1:4, end-3:end]);    
        csp_c1 = reshape(c1, [nT2*50 nCH]) * CSP_W;
        csp_c2 = reshape(c2, [nT2*50 nCH]) * CSP_W;
        
        csp_c1 = permute(reshape(csp_c1, [nT2 50 8]), [1 3 2]);
        csp_c2 = permute(reshape(csp_c2, [nT2 50 8]), [1 3 2]);
        
        f_c1 = squeeze(log(var(csp_c1, 1)));
        f_c2 = squeeze(log(var(csp_c2, 1)));
        temp = [f_c1,f_c2];
        trfv = [trfv; temp];
            
        TrY(2, 1:50) = 1; TrY(1, 51:100) = 1;
        
 %% TEST data load
    tepo = BCI_MI_test_se1(:,channel,:,sub);
    %tepo = BCI_MI_test_se1(:,:,:,sub);

%     tepo_y = BCI_MI_test_se1_Y(:,sub); % the test label is hidden
    
    dat = permute(tepo, [1 3 2]);    
    dat = reshape(dat, [nT*100 nCH]);
    dat = filter(b, a, dat);
    dat = reshape(dat, [nT 100 nCH]);         
    
    dat = dat(ival, :, :);
    [nT2, nS2, nCH2] = size(dat)
    % Applying CSP
    csp_dat = reshape(dat, [nT2*nS2 nCH2]) * CSP_W;
    csp_dat = reshape(csp_dat, [nT2 nS2 8]);
    csp_dat = permute(csp_dat, [1 3 2]);
    temp = squeeze(log(var(csp_dat, 1)));
    tefv = [tefv;temp];

    end

    for l=1:size(trfv, 1)
        x1 = trfv(l,:);
        y1 = TrY(1,:);
        k = 3;    
        mi(l) = mi_discrete_cont(x1, y1, 3);
    end
    [a b] = sort(mi, 'descend'); 
    C = train_RLDAshrink(trfv, TrY);
    
    

    out= real( C.w'*tefv + repmat(C.b, [1 size(tefv,2)]) );    
    [a] = find(sign(out)==-1); [b] = find(sign(out)==1); 
    tm(a,1) = 2; tm(b,1) = 1;       
    Y_pred(:, sub) = tm;

end


% Note that you should submit the 'Y_pred' with 100 x 54 correponding to
% predictions of all trials over the all subject (54). 

 for i=1:54
    [a] = find(BCI_MI_test_se1_Y(:,i) == Y_pred(:,i));
    acc = [acc;length(a)/100];
 end
 mean(acc)
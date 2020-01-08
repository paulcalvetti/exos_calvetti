function SRLDSexample
close all

%% constant
    S=4; T=40; dh=5;

%% make some data
    [p,V,s,~]=chooseData(T,S,dh);

    figure('units','normalized','position',[0.05 0.4 0.95 0.1]);
    plot(1:T,V,'LineWidth',2,'MarkerSize',12); xlim([1 T]);
    drawnow;

    %% Ground truth    
    massReal=zeros(S,T);
    for t=1:T
        massReal(s(t),t)=1;
    end
    figure('units','normalized','position',[0.05 0.45 0.95 0.1]);
    imagesc(massReal);
    colormap('gray');
    xlim([1 T]);
    drawnow;

    %% Approximations
    numGaussians=[2 10];

    for n=1:length(numGaussians)
        [fapprox,Fapprox,wapprox,~]=filtering(p,V,numGaussians(n));
        [x,~]=RTSLinearSmoother(p,V,fapprox,Fapprox,wapprox,numGaussians(n));

        massApprox=zeros(S,T);
        for t=1:T
            for j=1:S
                massApprox(j,t)=sum(sum(x{t}(j,:,:),3),2);
            end
        end
        figure('units','normalized','position',[0.05 0.45 0.95 0.1]);
        imagesc(massApprox);
        colormap('gray');
        xlim([1 T]);
        drawnow;
    end;
    
    %% Exact Inference
    [fapprox,Fapprox,wapprox,~]=filtering(p,V,inf);
    [x,~]=RTSLinearSmoother(p,V,fapprox,Fapprox,wapprox,inf);

    massExact=zeros(S,T);
    for t=1:T
        for j=1:S
            massExact(j,t)=sum(sum(x{t}(j,:,:),3),2);
        end
    end
    figure('units','normalized','position',[0.05 0.45 0.95 0.1]);
    imagesc(massExact); colormap('gray'); xlim([1 T]);
    drawnow;
            
    keyboard           
end
function [p,V,s,H]=chooseData(T,S,dh)
    V(1)=inf;
    s(1)=0;
    missed=true;
    while min(V)<-10|| max(V)>25 || missed
        [p,V,s,H]=createData(T,S,dh);
        missed=false;
        for i=1:S
            if ~ismember(i,s)
                missed=true;
            end
        end
    end
end
function [p,V,s,H]=createData(T,S,dh)
%% constant
    ceq0=1;
    ceq1=2;

    for s=1:S
        p.B0{s}=randn(1,dh); p.mu0v{s}=randn; p.sig0v{s}=5;   p.mu0h{s}=randn(dh,1); p.sig0h{s}=randcov(dh); p.A{s}=randn(dh);
        p.B1{s}=randn(1,dh); p.mu1v{s}=randn; p.sig1v{s}=5;   p.mu1h{s}=randn(dh,1); p.sig1h{s}=randcov(dh);
        p.pstgstm1ctm1(s,s,ceq1)=0.999999; % if the process just flipped, it doesn't flip again now
        p.pstgstm1ctm1(s,s,ceq0)=rand/2+0.5; % if the process just flipped, it doesn't flip again now
        for sp=[1:s-1,s+1:S]
            p.pstgstm1ctm1(sp,s,ceq1)=(1-p.pstgstm1ctm1(s,s,ceq1))/(S-1);
            p.pstgstm1ctm1(sp,s,ceq0)=(1-p.pstgstm1ctm1(s,s,ceq0))/(S-1);
        end
        p.ps1(s)=rand;
        p.muh1{s}=randn(dh,1); p.sigh1{s}=randcov(dh);
    end
    p.ps1=p.ps1/sum(p.ps1);
    
    %dimensions
    dh=length(p.mu0h{1}); dv=length(p.mu0v{1});
    
    
    %initialise
    s=zeros(1,T); H=zeros(dh,T); V=zeros(dv,T);
    
    t=1;
    s(t)=randgen(p.ps1);
    H(:,t)=p.mu1h{s(t)}+chol(p.sig1h{s(t)})*randn(dh,1);
    V(:,t)=p.B1{s(t)}*H(:,t)+p.mu1v{s(t)}+chol(p.sig1v{s(t)})*randn(dv,1);
    
    t=2;
    s(t)=randgen(p.pstgstm1ctm1(:,s(t-1),1));
    if s(t)==s(t-1) % no reset
        H(:,t)=p.A{s(t)}*H(:,t-1)+p.mu0h{s(t)}+chol(p.sig0h{s(t)})*randn(dh,1);
        V(:,t)=p.B0{s(t)}*H(:,t)+p.mu0v{s(t)}+chol(p.sig0v{s(t)})*randn(dv,1);        
    else % reset
        H(:,t)=p.mu1h{s(t)}+chol(p.sig1h{s(t)})*randn(dh,1);
        V(:,t)=p.B1{s(t)}*H(:,t)+p.mu1v{s(t)}+chol(p.sig1v{s(t)})*randn(dv,1);
    end

    for t=3:T
        s(t)=randgen(p.pstgstm1ctm1(:,s(t-1),(s(t-1)~=s(t-2))+1));
        if s(t)==s(t-1) % no reset
            H(:,t)=p.A{s(t)}*H(:,t-1)+p.mu0h{s(t)}+chol(p.sig0h{s(t)})*randn(dh,1);
            V(:,t)=p.B0{s(t)}*H(:,t)+p.mu0v{s(t)}+chol(p.sig0v{s(t)})*randn(dv,1);        
        else % reset
            H(:,t)=p.mu1h{s(t)}+chol(p.sig1h{s(t)})*randn(dh,1);
            V(:,t)=p.B1{s(t)}*H(:,t)+p.mu1v{s(t)}+chol(p.sig1v{s(t)})*randn(dv,1);
        end
    end
end


%% FILTERING
function [f,F,w,alpha,loglik]=filtering(p,V,numgaussians)
%% constant
    ceq0=1;
    ceq1=2;

%% initialise
    T=length(V);
    dh=length(p.mu0h{1}); dv=length(p.mu0v{1});
    S=length(p.ps1);
    for t=1:T
        f{t}=zeros(dh,S,min(t+1,numgaussians));
        F{t}=zeros(dh,dh,S,min(t+1,numgaussians));
        w{t}=sparse(S,t+1,min(t+1,numgaussians));
        alpha{t}=zeros(S,1);
    end

    [st,~]=dbstack; fprintf([st(1).name ' n=%d\n'],numgaussians); clear st;
    tic
    
%% first time-step (t=1)
    t=1;

    %c1=1, reset
    for s=1:S
        [ft(:,s,1),Ft(:,:,s,1),logpvgsc]=...
            forwardUpdate(zeros(dh,1),zeros(dh,dh),V(:,t),zeros(dh,dh),p.B1{s},p.sig1h{s},p.sig1v{s},p.mu1h{s},p.mu1v{s});
        %output_ft = ft(:,s,1);

        %output_Ft = Ft(:,:,s,1);
        %output_log = logpvgsc;
        %input_paul = [zeros(dh,1),zeros(dh,dh),V(:,t),zeros(dh,dh),p.B1{s},p.sig1h{s},p.sig1v{s},p.mu1h{s},p.mu1v{s}];
        logp(s,1) = log(p.ps1(s))+logpvgsc;

        %c1=0, no reset
        [ft(:,s,2),Ft(:,:,s,2),logpvgsc]=...
            forwardUpdate(zeros(dh,1),zeros(dh,dh),V(:,t),zeros(dh,dh),p.B0{s},p.sigh1{s},p.sig0v{s},p.muh1{s},p.mu0v{s});
        logp(s,2) = log(p.ps1(s))+logpvgsc;
    end

    w{t}(:)=condexp(logp(:));

    % calculate likelihood 
    loglik=logsumexp(logp(:)',ones(1,2*S));    
    
    for s=1:S
        is{s}=find(w{t}(s,:));
        f{t}(:,s,:)=ft(:,s,is{s});
        F{t}(:,:,s,:)=Ft(:,:,s,is{s});
        alpha{t}(s)=sum(w{t}(s,:));
    end
    
%% remaining time-steps:
    fprintf('% 6.2f%%\n',0);
    factor=200;
    for t=2:T
        clear ft Ft;
        
        ft=zeros(dh,S,min(numgaussians+1,t+1));
        Ft=zeros(dh,dh,S,min(numgaussians+1,t+1));

        if mod(t,factor)==0; fprintf('\b\b\b\b\b\b\b\b% 6.2f%%\n',t*100/T); end;
        
        % have to calculate w
        clear logp;
        logp=ones(S,t+1)*-inf;
        
        for s=1:S
            %ct=1, reset
            [ft(:,s,1),Ft(:,:,s,1),logpvgscv]=...
                    forwardUpdate(zeros(dh,1),zeros(dh,dh),V(:,t),zeros(dh,dh),p.B1{s},p.sig1h{s},p.sig1v{s},p.mu1h{s},p.mu1v{s});

            pstm1ctm1(:,ceq1)=w{t-1}([1:s-1,s+1:S],1);
            pstm1ctm1(:,ceq0)=sum(w{t-1}([1:s-1,s+1:S],2:end),2);
            pstgstm1ctm1nots=reshape(p.pstgstm1ctm1(s,[1:s-1,s+1:S],:),S-1,2);
            
            % have to calculate w
            logp(s,1)=sum(log(pstm1ctm1(:)'*pstgstm1ctm1nots(:)))+logpvgscv; 

            %ct=0, no reset
            for n_i=1:length(is{s})
                % only get to index i+1 at time t from index i at time t-1
                [ft(:,s,n_i+1),Ft(:,:,s,n_i+1),logpvgsciv]=...
                        forwardUpdate(f{t-1}(:,s,n_i),F{t-1}(:,:,s,n_i),V(:,t),p.A{s},p.B0{s},p.sig0h{s},p.sig0v{s},p.mu0h{s},p.mu0v{s});

                i=is{s}(n_i);
                if i==1; ctm1=ceq1; else ctm1=ceq0; end

                logp(s,i+1)=sumlog([w{t-1}(s,i) p.pstgstm1ctm1(s,s,ctm1)])+logpvgsciv;
            end
        end
        
        wt=reshape(condexp(logp(:)),size(logp));
        
        for s=1:S
            is{s}=[1 is{s}+1];
            if (t>=numgaussians) % then drop a component
                [~,idrop]=min(wt(s,is{s}));
                Z=sum(wt(s,is{s}));
                is{s}=is{s}([1:idrop-1,idrop+1:numgaussians+1]);
                w{t}(s,:)=sparse(is{s},ones(1,numgaussians),wt(s,is{s}),t+1,1);
                w{t}(s,:)=w{t}(s,:)*Z/sum(w{t}(s,:)); % renormalise

                f{t}(:,s,:)=ft(:,s,[1:idrop-1,idrop+1:numgaussians+1]);
                F{t}(:,:,s,:)=Ft(:,:,s,[1:idrop-1,idrop+1:numgaussians+1]);
            else
                w{t}=wt;
                f{t}(:,s,:)=ft(:,s,is{s});
                F{t}(:,:,s,:)=Ft(:,:,s,is{s});
            end
            alpha{t}(s)=sum(w{t}(s,:));
        end
        
        % calculate likelihood 
        loglik=loglik+logsumexp(logp(:),ones(length(logp(:)),1));
    end
    
    toc
end
%% SMOOTHERS
function [x,beta]=RTSLinearSmoother(p,V,f,F,w,numgaussians)
%% constant
    ceq0=1;
    ceq1=2;

%% initialise
    T=length(V);
    dh=length(p.mu0h{1});
    S=length(p.ps1);

    [st,~]=dbstack; fprintf([st(1).name ' n=%d\n'],numgaussians); clear st;
    tic

%% t=T
    beta{T}=full(sum(w{T},2));

    for s=1:S   
        is{s}=find(squeeze(w{T}(s,:)));
        x{T}(s,1:length(is{s}),1)=full(w{T}(s,is{s}));
    
        meang{T}=zeros(dh,1);
        for n_i=1:length(is{s})
            gT{s,n_i,1}=f{T}(:,s,n_i);
            GT{s,n_i,1}=F{T}(:,:,s,n_i);
            meang{T}=meang{T}+x{T}(s,n_i,1)*gT{s,n_i,1};
        end
        jstp1{s}=1;
    end
    
    gtp1=gT;
    Gtp1=GT;
    istp1=is;
    
%% t<T
    fprintf('% 6.2f%%\n',0);

    factor=200;
    for t=T-1:-1:1
        meang{t}=zeros(dh,1);
        
        if mod(t,factor)==0; fprintf('\b\b\b\b\b\b\b\b% 6.2f%%\n',(T-t)*100/T); end;

        clear xt;
        for s=1:S
            is{s}=find(squeeze(w{t}(s,:)));

            for n_i=1:length(is{s})
                n_i_tp1=n_i+(istp1{s}(n_i)<=is{s}(n_i));

                % copy in from t+1
                if n_i_tp1<=length(istp1{s})
                    if is{s}(n_i)+1==istp1{s}(n_i_tp1)
                        xt(s,n_i,2:length(jstp1{s})+1)=x{t+1}(s,n_i_tp1,1:length(jstp1{s}));
                    end
                end
            end
            % calculate normalisation for jt=1
            Z(s)=p.pstgstm1ctm1(s,[1:s-1,s+1:S],ceq1)*w{t}([1:s-1,s+1:S],1) ...
                +   sum(p.pstgstm1ctm1(s,[1:s-1,s+1:S],ceq0)*w{t}([1:s-1,s+1:S],2:end));
            if istp1{s}(1)==1
                Z2(s)=sum(x{t+1}(s,1,:),3);
            else
                Z2(s)=0;
            end
        end
        for s=1:S
            
            % calculate j=1
            if is{s}(1)==1
                xt(s,1,1)=w{t}(s,1).*((p.pstgstm1ctm1([1:s-1,s+1:S],s,ceq1)'./Z([1:s-1,s+1:S]))*Z2([1:s-1,s+1:S])');
            end
            isbiggerthan1=find(is{s}>1);
            xt(s,isbiggerthan1,1)=w{t}(s,is{s}(isbiggerthan1)).*((p.pstgstm1ctm1([1:s-1,s+1:S],s,ceq0)'./Z([1:s-1,s+1:S]))*Z2([1:s-1,s+1:S])');
        end
        
        if numgaussians~=inf  % may need to drop something
            for s=1:S
                l_is=length(is{s});
                l_js=length(jstp1{s})+1;
                [i j v]=find(reshape(xt(s,:,1:l_js),l_is,l_js));
                [~,indices]=sort(v,'descend');
                num=min(length(indices),numgaussians); 
                indices=indices(1:num);
                xt(s,1:l_is,1:l_js)=full(sparse(i(indices),j(indices),v(indices)*sum(v)/sum(v(indices)),l_is,l_js));
            end
        end   
        %% now delete empty columns.
        x{t}=zeros(S,min(t+1,numgaussians),min(T-t+1,numgaussians));
        for s=1:S
            cols=find(squeeze(sum(xt(s,:,:),2)));
            x{t}(s,:,1:length(cols))=xt(s,:,cols);
            js{s}=[1 jstp1{s}+1]; js{s}=js{s}(cols);
        end
        
        
        beta{t}=full(sum(sum(x{t},3),2));
        meang{t}=zeros(dh,1);
                
        gt={}; Gt={};
        
        for s=1:S
            for n_j=1:length(js{s});
                jn=js{s}(n_j);
                if jn==1

                    for n_i=1:length(is{s});

                        gt{s,n_i,1}=f{t}(:,s,n_i);
                        Gt{s,n_i,1}=F{t}(:,:,s,n_i);

                        meang{t}=meang{t}+x{t}(s,n_i,1).*gt{s,n_i,1};
                    end

                else

                    n_j_tp1=1;
                    while jn-1~=jstp1{s}(n_j_tp1)
                        n_j_tp1=n_j_tp1+1;
                    end

                    for n_i=1:length(is{s});
                        if x{t}(s,n_i,n_j)>0
                            in=is{s}(n_i);

                            n_i_tp1=n_i+(istp1{s}(n_i)<=in);

                            [gt{s,n_i,n_j} Gt{s,n_i,n_j} ggpRGp]=backwardUpdate(gtp1{s,n_i_tp1,n_j_tp1},Gtp1{s,n_i_tp1,n_j_tp1},f{t}(:,s,n_i),F{t}(:,:,s,n_i),p.A{s},p.sig0h{s},p.mu0h{s}); %% EXPENSiVE

                            meang{t}=meang{t}+x{t}(s,n_i,n_j).*gt{s,n_i,n_j};
                        end
                    end
                end
            end
        end
        
        gtp1=gt;
        Gtp1=Gt;
        istp1=is;
        jstp1=js;
    end
    
    meang=cell2mat(meang);
    toc

    mean(meang,2)
end
%% STANDARD KALMAN UPDATES
function [fnew Fnew logpvgv]=forwardUpdate(f,F,v,A,B,CovH,CovV,meanH,meanV)
    %LDSFORWARDUPDATE Single Forward update for a Latent Linear Dynamical System (Kalman Filter)
    % [fnew Fnew logpvgv]=LDSforwardUpdate(f,F,v,A,B,CovH,CovV,meanH,meanV)
    %
    % inputs:
    % f : filterered mean p(h(t)|v(1:t))
    % F : filterered covariance p(h(t)|v(1:t))
    % v : observation v(t+1)
    % A : transition matrix
    % B : emission matrix
    % CovH : transition covariance
    % CovV : emission covariance
    % meanH : transition mean
    % meanV : emission mean
    %
    % Outputs:
    % fnew : : filterered mean p(h(t+1)|v(1:t+1))
    % Fnew : filterered covariance p(h(t+1)|v(1:t+1))
    % logpgvg : log p(v(t+1)|v(1:t))
    muh=A*f+meanH;

    muv=B*muh+meanV;
    Shh=A*F*A'+CovH;


    Svh=B*Shh;
    %Svv=B*Shh*B'+CovV;
    Svv=Svh*B'+CovV;
    del = v-muv;
    invSvvdel=Svv\del;

    fnew = muh+Svh'*invSvvdel;
    %Fnew=Shh-Svh'*(Svv\Svh); Fnew=0.5*(Fnew+Fnew');
    %K = Shh*B'/Svv; % Kalman Gain

    K = Svh'/Svv; % Kalman Gain

    tmp=eye(size(A))-K*B;
    Fnew = tmp*Shh*tmp'+K*CovV*K'; % Joseph's form 
    logpvgv = -0.5*del'*invSvvdel-0.5*logdet(2*pi*Svv); %% THiS iS WHERE THE TiME iS SPENT.
end
function [gnew Gnew Gpnew]=backwardUpdate(g,G,f,F,A,CovH,meanH)
    %LDSBACKWARDUPDATE Single Backward update for a Latent Linear Dynamical System (RTS smoothing update)
    % [gnew Gnew Gpnew]=LDSbackwardUpdate(g,G,f,F,A,CovH,meanH)
    %
    % inputs:
    % g : smoothed mean p(h(t+1)|v(1:T))
    % G : smoothed covariance p(h(t+1)|v(1:T))
    % f : filterered mean p(h(t)|v(1:t))
    % F : filterered covariance p(h(t)|v(1:t))
    % A : transition matrix
    % CovH : transition covariance
    % CovV : emission covariance
    % meanH : transition mean
    %
    % Outputs:
    % gnew : smoothed mean p(h(t)|v(1:T))
    % Gnew : smoothed covariance p(h(t)|v(1:T))
    % Gpnew : smoothed cross moment  <h_t h_{t+1}|v(1:T)>
    muh=A*f+meanH;
    Shtpt=A*F;
    %Shtptp=A*F*A'+CovH;
    Shtptp=Shtpt*A'+CovH;
    leftA = (Shtpt')/Shtptp;
    leftS = F - leftA*Shtpt;
    leftm = f - leftA*muh;
    gnew = leftA*g+leftm;
    leftAG=leftA*G;
    %Gnew = leftA*G*leftA'+leftS; Gnew=0.5*(Gnew+Gnew'); % could also use Joseph's form if desired
    %Gpnew = leftA*G+gnew*g'; % smoothed <h_t h_{t+1}>
    Gnew = leftAG*leftA'+leftS; Gnew=0.5*(Gnew+Gnew'); % could also use Joseph's form if desired
    Gpnew = leftAG+gnew*g'; % smoothed <h_t h_{t+1}>
end
%% UTILITY FUNCTIONS
function y=randcov(dh)
    x=0.001*randn(dh);
    y=triu(x)*triu(x)';
end
function y = randgen(p, m, n, s)
    % RANDGEN	Generates discrete random variables given the pdf
    %
    % Y = RANDGEN(P, M, N, S)
    %
    % Inputs :
    %	P : Relative likelihoods (or pdf) of symbols
    %	M : Rows    <Default = 1>
    %	N : Columns <Default = 1>
    %	S : Symbols <Default 1:length(P)>

    % Change History :
    % Date		Time		Prog	Note
    % 06-Jul-1997	 4:36 PM	ATC	Created under MATLAB 5.0.0.4064

    % ATC = Ali Taylan Cemgil,
    % Bogazici University, Dept. of Computer Eng. 80815 Bebek Istanbul Turkey
    % e-mail : cemgil@boun.edu.tr 
    if isempty(p); y=[]; return; end % no samples if p is empty (DB)
    if (nargin < 2) m = 1; end
    if (nargin < 3) n = 1;end
    if (nargin < 4) s = 1:length(p);end

    c = cumsum(p);
    c = c(:)'/c(end);
    N = m*n;
    u = rand(N,1);
    % for i=1:N, y(i) = length(find(c<y(i)))+1; end;

    y = sum( c(ones(N,1),:) < u(:, ones(length(c),1)) , 2 ) + 1;

    y = reshape(s(y), m, n);
end
function anew=logsumexp(a,b)
    %LOGSUMEXP Compute log(sum(exp(a).*b)) valid for large a
    % example: logsumexp([-1000 -1001 -998],[1 2 0.5])
    amax=max(a); A =size(a,1);
    anew = amax + log(sum(exp(a-repmat(amax,A,1)).*b));
end
function l=logdet(A)
    %LOGDET Log determinant of a positive definite matrix computed in a numerically more stable manner
    
    [u s v]=svd(A); 
    l=sum(log(diag(s)+1.0e-20));
end
function pnew=condexp(logp)
    %CONDEXP  Compute p\propto exp(logp);
    fprintf('size(logp) is %s\n', mat2str(size(logp)))

    pmax=max(logp,[],1); P =size(logp,1);
    pnew = condp(exp(logp-repmat(pmax,P,1)));
end
function pnew=condp(pin)
    %CONDP Make a conditional distribution from the matrix 
    % pnew=condp(pin)
    %
    % Input : pin  -- a positive matrix pin
    % Output:  matrix pnew such that sum(pnew,1)=ones(1,size(p,2))
    p = pin./max(pin(:));
    p = p+eps; % in case all unnormalised probabilities are zero
    pnew=p./repmat(sum(p,1),size(p,1),1);
end
function lx=sumlog(x)
    %SUMLOG sum(log(x)) with a cutoff at 10e-200
    cutoff=10e-200;x(find(x<cutoff))=cutoff; lx=sum(log(x));
end

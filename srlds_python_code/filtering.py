
from utility_functions import *




## FILTERING
#function [f,F,w,alpha,loglik]=filtering(p,V,numgaussians)
def filtering(p,V,numgaussians):


    ## constant
    ceq0=1
    ceq1=2


    ## initialise
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
## first time-step (t=1)
    t=1;

    #c1=1, reset
    for s=1:S
        [ft(:,s,1),Ft(:,:,s,1),logpvgsc]=...
            forwardUpdate(zeros(dh,1),zeros(dh,dh),V(:,t),zeros(dh,dh),p.B1{s},p.sig1h{s},p.sig1v{s},p.mu1h{s},p.mu1v{s});
        #output_ft = ft(:,s,1);

        #output_Ft = Ft(:,:,s,1);
        #output_log = logpvgsc;
        #input_paul = [zeros(dh,1),zeros(dh,dh),V(:,t),zeros(dh,dh),p.B1{s},p.sig1h{s},p.sig1v{s},p.mu1h{s},p.mu1v{s}];
        logp(s,1) = log(p.ps1(s))+logpvgsc;

        #c1=0, no reset
        [ft(:,s,2),Ft(:,:,s,2),logpvgsc]=...
            forwardUpdate(zeros(dh,1),zeros(dh,dh),V(:,t),zeros(dh,dh),p.B0{s},p.sigh1{s},p.sig0v{s},p.muh1{s},p.mu0v{s});
        logp(s,2) = log(p.ps1(s))+logpvgsc;
    end


    w{t}(:)=condexp(logp(:));

    # calculate likelihood
    loglik = logsumexp(logp(:)',ones(1,2*S));

    for s=1:S
        is {s} = find(w{t}(s,:));
        f{t}(:, s,:)=ft(:, s, is {s});
        F{t}(:,:, s,:)=Ft(:,:, s, is {s});
        alpha{t}(s) = sum(w{t}(s,:));
    end



    return [f,F,w,alpha,loglik]




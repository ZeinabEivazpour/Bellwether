function [G,P,Q,R,error1,error2,error,predicted_gain,true_gain] = csi(K,Y,m,centering,kappa,delta,tol)
% INPUT
% K  : kernel matrix n x n
% Y  : target vector n x d
% m  : maximal rank
% kappa : trade-off between approximation of K and prediction of Y (suggested: .99)
% centering : 1 if centering, 0 otherwise (suggested: 1)
% delta : number of columns of cholesky performed in advance (suggested: 40)
% tol : minimum gain at iteration (suggested: 1e-4)
%
% OUTPUT
% G : Cholesky decomposition -> K(P,P) is approximated by G*G'
% P : permutation matrix
% Q,R : QR decomposition of G (or center(G) if centering)
% error1 : tr(K-G*G')/tr(K) at each step of the decomposition
% error2 : ||Y-Q*Q'*Y||_F^2 / ||Y||_F^2 at each step of the decomposition
% predicted_gain : predicted gain before adding each column
% true_gain : actual gain after adding each column
%
% Copyright (c) Francis R. Bach, 2005.




n = size(K,1);
d = size(Y,2);
assert(size(Y,1)==n,'The targets have the wrong number of data points');

G = zeros(n,min(m+delta,n));    % Cholesky factor
diagK = diag(K);
P = 1:n;

Q = zeros(n,min(m+delta,n));                % Q part of the QR decomposition
R = zeros(min(m+delta,n),min(m+delta,n));   % R part of the QR decomposition
traceK = sum(diagK);
lambda = (1-kappa) / traceK ;
if centering, Y = Y - 1/n * repmat(sum(Y,1),n,1); end
sumY2 = sum(Y(:).^2);
mu = (kappa) / sumY2;
error1(1) = traceK;
error2(1) = sumY2;

k=0;            % current index of the Cholesky decomposition
kadv=0;         % current index of the look ahead steps
Dadv = diagK;
D = diagK;

% approximation cost cached quantities
A1 = zeros(n,1);
A2 = zeros(n,1);
A3 = zeros(n,1);
GTG = zeros(m);
QTY = zeros(m,d);
QTYYTQ = zeros(m,m);

% makes sure that delta is smaller than n
delta = min(delta,n);

% first performs delta steps of Cholesky and QR decomposition
for i=1:delta
    kadv = kadv + 1;
    % select best index
    diagmax = Dadv(kadv);
    jast = 1;
    for j=1:n-kadv+1
        if (Dadv(j+kadv-1)>diagmax / .99)
            diagmax = Dadv(j+kadv-1);
            jast = j;
        end
    end
    if diagmax<1e-12
        % all pivots are too close to zero, stops
        % this can only happen if the matrix has rank less than delta
        kadv = kadv - 1;
        break;
    else
        jast=jast+kadv-1;

        % permute indices
        P( [kadv jast] )        = P( [jast kadv] );
        Dadv( [kadv jast] )     = Dadv( [jast kadv] );
        D( [kadv jast] )        = D( [jast kadv] );
        A1( [kadv jast] )        = A1( [jast kadv] );
        G([kadv jast],1:kadv-1) = G([ jast kadv],1:kadv-1);
        Q([kadv jast],1:kadv-1) = Q([ jast kadv],1:kadv-1);

        % compute new Cholesky column
        G(kadv,kadv)=Dadv(kadv);
        G(kadv,kadv)=sqrt(G(kadv,kadv));
        newKcol = K(P(kadv+1:n),P(kadv));
        G(kadv+1:n,kadv)=1/G(kadv,kadv)*( newKcol - G(kadv+1:n,1:kadv-1)*(G(kadv,1:kadv-1))');

        % update diagonal
        Dadv(kadv+1:n) =  Dadv(kadv+1:n) - G(kadv+1:n,kadv).^2;
        Dadv(kadv) = 0;

        % performs QR
        if centering
            Gcol = G(:,kadv) - 1/n * repmat( sum(G(:,kadv) ,1),n,1 );
        else
            Gcol = G(:,kadv);
        end
        R(1:kadv-1,kadv) = Q(:,1:kadv-1)' * Gcol;
        Q(:,kadv) = Gcol - Q(:,1:kadv-1) * R(1:kadv-1,kadv);
        R(kadv,kadv) = norm(Q(:,kadv));
        Q(:,kadv) = Q(:,kadv) / R(kadv,kadv);

        % update cached quantities
        if centering
            GTG(1:kadv,kadv) = G(:,1:kadv)' * G(:,kadv);
        else
            GTG(1:kadv,kadv) = R(1:kadv,1:kadv)' * R(1:kadv,kadv);
        end
        GTG(kadv,1:kadv) = GTG(1:kadv,kadv)';
        QTY(kadv,:) = Q(:,kadv)' * Y(P,:);
        QTYYTQ(kadv,1:kadv) = QTY(kadv,:) * QTY(1:kadv,:)';
        QTYYTQ(1:kadv,kadv) = QTYYTQ(kadv,1:kadv)';

        % update costs
        A1(kadv:n) = A1(kadv:n) + GTG(kadv,kadv) * ( G(kadv:n,kadv).^2 );
        A1(kadv:n) = A1(kadv:n) + 2 * G(kadv:n,kadv) .* ( G(kadv:n,1:kadv-1) * GTG(1:kadv-1,kadv) );

    end
end

% compute remaining costs for all indices
A2 = sum( ( G(:,1:kadv) * ( R(1:kadv,1:kadv)' * QTY(1:kadv,:) ) ).^2 , 2);
A3 = sum( ( G(:,1:kadv) * R(1:kadv,1:kadv)' ).^2 , 2);


% start main loop
while k<m,

    k = k +1;

    % compute the gains in approximation for all remaining indices
    dJK = zeros(n-k+1,1);
    for i=1:n-k+1
        kast = k+i-1;
        if D(kast)<1e-12,
            % this column is already generated by already
            % selected columns -> cannot be selected
            dJK(i)=-1e100;
        else
            dJK(i) = A1(kast);
            if kast > kadv,
                % add eta
                dJK(i) = dJK(i) + D(kast)^2 - (D(kast) - Dadv(kast))^2;
            end
            dJK(i) = dJK(i) / D(kast);
        end
    end

    dJY = zeros(n-k+1,1);
    if kadv>k
        for i=1:n-k+1
            kast = k+i-1;
            if A3(kast) < 1e-12
                dJY(i) = 0;
            else
                dJY(i) = A2(kast) / A3(kast);
            end
        end
    end

    % select the best column
    dJ = lambda * dJK + mu * dJY;
    diagmax = -1;
    jast = 0;
    for j=1:n-k+1
        if D(j+k-1)>1e-12
            if dJ(j) > diagmax / 0.9
                diagmax = dJ(j);
                jast = j;
            end
        end
    end
    if jast==0,
        % no more good indices, exit
        k=k-1;
        break;
    end
    jast = jast + k - 1;
    predicted_gain(k) = diagmax;

    % performs one cholesky + QR step:
    % if new pivot not already selected, use pivot
    % otherwise, select new look ahead index that maximize Dadv

    if jast > kadv,
        newpivot = jast;
        jast = kadv + 1;
    else
        a = 1e-12;
        b = 0;
        for j=1:n-kadv
            if Dadv(j+kadv)>a/.99
                a=Dadv(j+kadv);
                b = j+kadv;
            end
        end
        if b==0, newpivot = 0;
        else newpivot = b;
        end
    end

    if newpivot > 0
        % performs steps

        kadv = kadv + 1;

        % permute
        P( [kadv newpivot] ) = P( [newpivot kadv] );
        Dadv( [kadv newpivot] ) = Dadv( [newpivot kadv] );
        D( [kadv newpivot] ) = D( [newpivot kadv] );
        A1( [kadv newpivot] ) = A1( [newpivot kadv] );
        A2( [kadv newpivot] ) = A2( [newpivot kadv] );
        A3( [kadv newpivot] ) = A3( [newpivot kadv] );
        G([kadv newpivot],1:kadv-1)=G([ newpivot kadv],1:kadv-1);
        Q([kadv newpivot],1:kadv-1)=Q([ newpivot kadv],1:kadv-1);

        % compute new Cholesky column
        G(kadv,kadv)=Dadv(kadv);
        G(kadv,kadv)=sqrt(G(kadv,kadv));
        newKcol = K(P(kadv+1:n),P(kadv));
        G(kadv+1:n,kadv)=1/G(kadv,kadv)*( newKcol - G(kadv+1:n,1:kadv-1)*(G(kadv,1:kadv-1))');

        % update diagonal
        Dadv(kadv+1:n) =  Dadv(kadv+1:n) - G(kadv+1:n,kadv).^2;
        Dadv(kadv) = 0;

        % performs QR
        if centering
            Gcol = G(:,kadv) - 1/n * repmat( sum(G(:,kadv) ,1),n,1 );
        else
            Gcol = G(:,kadv);
        end
        R(1:kadv-1,kadv) = Q(:,1:kadv-1)' * Gcol;
        Q(:,kadv) = Gcol - Q(:,1:kadv-1) * R(1:kadv-1,kadv);
        R(kadv,kadv) = norm(Q(:,kadv));
        Q(:,kadv) = Q(:,kadv) / R(kadv,kadv);

        % update the cached quantities
        if centering
            GTG(k:kadv,kadv) = G(:,k:kadv)' * G(:,kadv);
        else
            GTG(k:kadv,kadv) = R(1:kadv,k:kadv)' * R(1:kadv,kadv);
        end
        GTG(kadv,k:kadv) = GTG(k:kadv,kadv)';
        QTY(kadv,:) = Q(:,kadv)' * Y(P,:);
        QTYYTQ(kadv,k:kadv) = QTY(kadv,:) * QTY(k:kadv,:)';
        QTYYTQ(k:kadv,kadv) = QTYYTQ(kadv,k:kadv)';

        % update costs
        A1(kadv:n) = A1(kadv:n) + GTG(kadv,kadv) * ( G(kadv:n,kadv).^2 );
        A1(kadv:n) = A1(kadv:n) + 2 * G(kadv:n,kadv) .* ( G(kadv:n,k:kadv-1) * GTG(k:kadv-1,kadv) );
        A3(kadv:n) = A3(kadv:n) + G(kadv:n,kadv).^2 * sum( R(k:kadv,kadv).^2, 1 );
        temp = R(k:kadv,kadv)' * R(k:kadv,k:kadv-1);
        A3(kadv:n) = A3(kadv:n) + 2 *  G(kadv:n,kadv) .* ( G(kadv:n,k:kadv-1) * temp' );

        temp = R(k:kadv,kadv)' *  QTYYTQ(k:kadv,k:kadv);
        temp1 = temp * R(k:kadv,kadv) ;
        A2(kadv:n) = A2(kadv:n) + G(kadv:n,kadv).^2 * temp1;


        % temp = R(k:kadv,kadv)' *  QTYYTQ(k:kadv,k:kadv);
        temp2 = temp * R(k:kadv,k:kadv-1);
        A2(kadv:n) = A2(kadv:n) + 2 * G(kadv:n,kadv) .* ( G(kadv:n,k:kadv-1) * temp2' );
        % G(kadv:n,k:kadv-1) * R(k:kadv,kadv)' *  QTYYTQ(k:kadv,k:kadv) R(k:kadv,k:kadv-1)


    end

    % permute pivots in the Cholesky and QR decomposition between p,q
    p = k;
    q = jast;
    if p<q

        % store some quantities
        Gbef = G(:,p:q);
        Gbeftotal = G(:,k:kadv);
        GTGbef = GTG(p:q,p:q);
        QTYYTQbef = QTYYTQ(p:q,k:kadv);
        Rbef = R(p:q,p:q);
        Rbeftotal = R(k:kadv,k:kadv);
        tempG = eye(q-p+1,q-p+1);
        tempQ = eye(q-p+1,q-p+1);

        for s=q-1:-1:p
            % permute indices
            P( [s s+1] ) = P( [s+1 s] );
            Dadv( [s s+1] ) = Dadv( [s+1 s] );
            D( [s s+1] ) = D( [s+1 s] );
            A1( [s s+1] ) = A1( [s+1 s] );
            A2( [s s+1] ) = A2( [s+1 s] );
            A3( [s s+1] ) = A3( [s+1 s] );
            G([s s+1],1:kadv)=G([s+1 s], 1:kadv);
            Gbef([s s+1],:)=Gbef([s+1 s], :);
            Gbeftotal([s s+1],:)=Gbeftotal([s+1 s], :);
            Q([s s+1],1:kadv)=Q([s+1 s] ,1:kadv);


            % update decompositions
            [Q1,R1] = qr2( G(s:s+1,s:s+1)');
            G(:,s:s+1) = G(:,s:s+1) * Q1;
            G(s,s+1)=0;
            R(1:kadv,s:s+1) = R(1:kadv,s:s+1) * Q1;
            [Q2,R2] = qr2( R(s:s+1,s:s+1) );
            R(s:s+1,1:kadv) = Q2' * R(s:s+1,1:kadv);
            Q(:,s:s+1) = Q(:,s:s+1) * Q2;
            R(s+1,s)=0;

            % update relevant quantities
            nonchanged = [k:s-1 s+2:kadv ];
            GTG(nonchanged,s:s+1) = GTG(nonchanged,s:s+1) * Q1;
            GTG(s:s+1,nonchanged) = GTG(nonchanged,s:s+1)';
            GTG(s:s+1,s:s+1) = Q1' * GTG(s:s+1,s:s+1) * Q1;
            QTY(s:s+1,:) = Q2' * QTY(s:s+1,:);
            QTYYTQ(nonchanged,s:s+1) = QTYYTQ(nonchanged,s:s+1) * Q2;
            QTYYTQ(s:s+1,nonchanged) = QTYYTQ(nonchanged,s:s+1)';
            QTYYTQ(s:s+1,s:s+1) = Q2' * QTYYTQ(s:s+1,s:s+1) * Q2;
            tempG(:,s-p+1:s-p+2) = tempG(:,s-p+1:s-p+2) * Q1;
            tempQ(:,s-p+1:s-p+2) = tempQ(:,s-p+1:s-p+2) * Q2;
        end

        % update costs
        tempG = tempG(:,1);
        tempGG = GTGbef * tempG ;
        A1(k:n) = A1(k:n) - 2 * G(k:n,k) .* ( Gbef(k:n,:) * tempGG );                % between p and q -> different
        A1(k:n) = A1(k:n) - 2 * G(k:n,k) .* ( G(k:n,k:p-1) * GTG(k:p-1,k) );         % below p
        A1(k:n) = A1(k:n) - 2 * G(k:n,k) .* ( G(k:n,q+1:kadv) * GTG(q+1:kadv,k) );   % above q
        tempQ = tempQ(:,1);
        temp = G(k:n,q+1:kadv) * R(k,q+1:kadv)';
        temp = temp + G(k:n,k:p-1) * R(k,k:p-1)';
        temp = temp + Gbef(k:n,:) * Rbef' * tempQ;
        A3(k:n) = A3(k:n) - temp.^2 ;
        A2(k:n) = A2(k:n) + temp.^2 * QTYYTQ(k,k);
        temp2 = ( tempQ' * QTYYTQbef ) * Rbeftotal;
        A2(k:n) = A2(k:n) - 2 * temp .* ( Gbeftotal(k:n,:) * temp2' );

    else

                % update costs
        A1(k:n) = A1(k:n) - 2 * G(k:n,k) .* ( G(k:n,k:kadv) * GTG(k:kadv,k) );
        A3(k:n) = A3(k:n) - ( G(k:n,k:kadv) * R(k,k:kadv)' ).^2;
        temp = G(k:n,k:kadv) * R(k,k:kadv)';
        A2(k:n) = A2(k:n) + ( temp ).^2 * QTYYTQ(k,k);
        temp2 = QTYYTQ(k,k:kadv) * R(k:kadv,k:kadv);
        A2(k:n) = A2(k:n) - 2 * temp .* ( G(k:n,k:kadv) * temp2' );
    end

    % update diagonal and other quantities (A1,B1)
    D(k+1:n) =  D(k+1:n) - G(k+1:n,k).^2;
    D(k) = 0 ;
    A1(k:n) = A1(k:n) + GTG(k,k) * ( G(k:n,k).^2 );

    % compute errors and true gains
    temp2 =  Q(:,k)' * Y(P,:);
    temp2 = sum(temp2.^2);
    temp1 = sum( G(:,k).^2 );
    true_gain(k) = temp1 * lambda + temp2 * mu;
    error1(k+1) = error1(k) - temp1;
    error2(k+1) = error2(k) - temp2;
    if true_gain(k) < tol, break; end
end

% reduce dimensions of decomposition
G=G(:,1:k);
Q=Q(:,1:k);
R=R(1:k,1:k);

% compute and normalize errors
error = lambda * error1 + mu * error2;
error1 = error1 / traceK;
error2 = error2 / sumY2;



%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [Q,R] = qr2(M);
% QR decomposition for 2x2 matrices (this is to make sure that the C and
% Matlab implementations output exactly the same matrices
Q = zeros(2,2);
R = zeros(2,2);
x = sqrt( M(1,1)^2 + M(2,1)^2 );
R = x;
Q(:,1) = M(:,1) / x;
R(1,2) = Q(:,1)' * M(:,2);
Q(:,2) = M(:,2) - R(1,2) * Q(:,1);
R(2,2) = norm(Q(:,2));
Q(:,2) = Q(:,2) / R(2,2);

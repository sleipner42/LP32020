%%
load powercurve_V112
%%

N = 100;
my = 5;
E = exprnd(1/my,1,N);
U = rand(1,N);
e_inv = @(x) -log(1-x)/my;
E2 = arrayfun(e_inv, U);
histogram(E)
hold on
histogram(E2)
hold off
legend('True', 'Inverse')

%% Monte Carlo a)
N = 10e4;
lambda = 10.6;
k = 2.0;
W = wblrnd(lambda, k, [1, N]);
figure(1)
subplot(311)
histogram(W, 'Normalization', 'pdf')
hold on
y = 0:0.1:30;
f = wblpdf(y,lambda,k);
plot(y,f, 'LineWidth',1.5)
hold off

subplot(312)
plot(P, 'r--', W, feval(P,W))

subplot(313)
Power = feval(P,W);
histogram(Power, 'Normalization', 'pdf');

my = mean(Power)
sigma2 = var(Power);
diff = 1.96*sqrt(sigma2/N)
Imc = [my - diff, my + diff];


%% Truncated a)
a = 3;
b = 25;
U = rand(1,N);
U_fix = (wblcdf(b,lambda,k)-wblcdf(a,lambda,k)).*U + wblcdf(a, lambda,k);
W2 = wblinv(U_fix, lambda, k);

subplot(311)
histogram(W2, 'Normalization', 'pdf')

subplot(312)
plot(P, 'r--', W2, feval(P,W2))

subplot(313)
Power = feval(P,W2);
histogram(Power, 'Normalization', 'pdf');

my = mean(Power)
sigma2 = var(Power);
diff = 1.96*sqrt(sigma2/N)
Imc_t = [my - diff, my + diff];



%%
y = 0:0.1:30;
f = wblpdf(y,lambda,k);
u = f.*feval(P,y)';
subplot(211)
plot(y, u)
hold on
gmean = 13;
gvar = 4.5;
gscale = 2.8*10e5;
g = gscale*normpdf(y,gmean,gvar);
plot(y,g);
hold off

subplot(212)
d = u./g;
plot(y,d)


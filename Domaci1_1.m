clc, clear, close all

load dataset3.mat;

ulaz = pod(:, 1:2);
izlaz = pod(:, 3);

%% One-hot encoding
izlazOH = zeros(length(izlaz), 3);
izlazOH(izlaz == 1, 1) = 1;
izlazOH(izlaz == 2, 2) = 1;
izlazOH(izlaz == 3, 3) = 1;

%%
K1=ulaz(izlaz==1,:);
K2=ulaz(izlaz==2,:);
K3=ulaz(izlaz==3,:);

figure, hold all
plot(K1(:, 1), K1(:, 2), '*');
plot(K2(:, 1), K2(:, 2), 'o');
plot(K3(:, 1), K3(:, 2), 'd');

legend({'K1','K2', 'K3'});

%%
ulaz=ulaz';
izlazOH=izlazOH';

%% Podela na trening i test skup
rng(200);
N = length(izlaz);
ind = randperm(N);
indTrening = ind(1 : 0.8*N);
indTest = ind(0.8*N+1 : N);

ulazTrening = ulaz(:, indTrening);
izlazTrening = izlazOH(:, indTrening);

ulazTest = ulaz(:, indTest);
izlazTest = izlazOH(:, indTest);

%%
arhitektura = {[130 100 80],[6 5 4], [3 2]};
nets = [];

for k = 1 : length(arhitektura)
    net = patternnet(arhitektura{k});
    
    net.divideFcn = '';

    net.trainParam.epochs = 2000;
    net.trainParam.goal = 1e-3;
      
    net.trainParam.min_grad = 1e-4;
    
    net.layers{1}.transferFcn = 'poslin';
    net.layers{2}.transferFcn = 'poslin';
    if(k~=2) net.layers{3}.transferFcn = 'poslin';
    end
    
    net = train(net, ulazTrening, izlazTrening);
    
    nets{k} = net;
    
    break;
end

%%
for k = 1 : length(nets)
    pred = sim(nets{k}, ulazTest);
    figure, plotconfusion(izlazTest, pred);
    
end

%% Granica odlucivanja
for k = 1 : length(nets)

    Ntest = 500;
    ulazGO = [];
    x1 = linspace(-5, 5, Ntest);
    x2 = linspace(-5, 5, Ntest);

    for x11 = x1
        pom = [x11*ones(1, Ntest); x2];
        ulazGO = [ulazGO, pom];
    end

    predGO = sim(nets{k}, ulazGO);
    [vr, klasa] = max(predGO);

    K1go = ulazGO(:, klasa == 1);
    K2go = ulazGO(:, klasa == 2);
    K3go = ulazGO(:, klasa == 3);

    figure, hold all
    plot(K1go(1, :), K1go(2, :), '.')
    plot(K2go(1, :), K2go(2, :), '.')
    plot(K3go(1, :), K3go(2, :), '.')
    plot(K1(:, 1), K1(:, 2), 'bo')
    plot(K2(:, 1), K2(:, 2), 'r*')
    plot(K3(:, 1), K3(:, 2), 'yd')
    
end
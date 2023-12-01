clc, clear, close all

%% Ucitavanje podataka
rng(200)

pod = readtable('Weather.csv');

windGustDir = [pod.WindGustDir];
[~, ~, windGustDir] = unique(windGustDir);

windDir3pm = [pod.WindDir3pm];
[~, ~, windDir3pm] = unique(windDir3pm);

windDir9am = [pod.WindDir9am];
[~, ~, windDir9am] = unique(windDir9am);

rainToday = strcmpi([pod.RainToday], 'yes');

ulaz = [pod.MinTemp, pod.MaxTemp, pod.Rainfall, windGustDir, pod.WindGustSpeed, windDir9am, ...
    windDir3pm, pod.WindSpeed9am, pod.WindSpeed3pm, pod.Humidity9am, pod.Humidity3pm, ...
    pod.Pressure9am, pod.Pressure3pm, pod.Temp9am, pod.Temp3pm, rainToday];

izlaz = [pod.RainTomorrow];
izlaz = strcmpi(izlaz,'yes');

ulaz = ulaz';
izlaz = izlaz';

%% Prikaz raspodele odbiraka po klasama
figure
histogram(izlaz)
title('Podela odbiraka po klasama')
xlabel('Klase')
ylabel('Broj odbiraka')

%% Podela na klase
K1 = ulaz(:, izlaz == 1);
K2 = ulaz(:, izlaz == 0);

%% Izdvajanje skupova za trening, testiranje i validaciju
rng(200)

N1 = length(K1);
K1trening = K1(:, 1 : 0.7*N1);
K1test = K1(:, 0.7*N1+1 : 0.85*N1);
K1val = K1(:, 0.85*N1+1 : N1);

N2 = length(K2);
K2trening = K2(:, 1 : 0.7*N2);
K2test = K2(:, 0.7*N2+1 : 0.85*N2);
K2val = K2(:, 0.85*N2+1 : N2);

ulazTrening = [K1trening, K2trening];
izlazTrening = [ones(1, length(K1trening)), zeros(1, length(K2trening))];
ind = randperm(length(izlazTrening));
ulazTrening = ulazTrening(:, ind);
izlazTrening = izlazTrening(ind);

ulazTest = [K1test, K2test];
izlazTest = [ones(1, length(K1test)), zeros(1, length(K2test))];

ulazVal = [K1val, K2val];
izlazVal = [ones(1, length(K1val)), zeros(1, length(K2val))];

ulazSve = [ulazTrening, ulazVal];
izlazSve = [izlazTrening, izlazVal];

%% Krosvalidacija
rng(200)
arhitektura = {[15, 10, 8, 5], [17, 12, 7], [4, 5, 8, 6], [20, 15, 10]};
Pbest = 0;
Rbest = 0;
F1best = 0;

for reg = [0.2, 0.5, 0.8]
    for lr = [0.5, 0.1, 0.05]
        for arh = 1 : length(arhitektura)
            net = patternnet(arhitektura{arh});

            net.divideFcn = 'divideind';
            net.divideParam.trainInd = 1 : length(ulazTrening);
            net.divideParam.valInd = length(ulazTrening)+1 : length(ulazSve);
            net.divideParam.testInd = [];

            net.performParam.regularization = reg;

            net.trainFcn = 'trainrp'; 
            
            net.trainParam.lr = lr;
            net.trainParam.epochs = 1000;
            net.trainParam.goal = 1e-4;
            net.trainParam.min_grad = 1e-5;
            net.trainParam.max_fail = 20;
            net.trainParam.showWindow = false;

            [net, info] = train(net, ulazSve, izlazSve);

            pred = round(sim(net, ulazVal));

            [~, cm] = confusion(izlazVal, pred);
            P = cm(2,2)/(cm(2,2)+cm(1,2));
            R = cm(2,2)/(cm(2,2)+cm(1,1));
            F1 = 2*cm(2, 2)/(cm(2, 1)+cm(1, 2)+2*cm(2, 2));
            
            disp(['Reg = ' num2str(reg) ', LR = ' num2str(lr) ', epoch = ' num2str(info.best_epoch)])
            disp(['P = ' num2str(P) ', R = ' num2str(R) ', F1= ' num2str(F1)])
            disp(arhitektura{arh})
            disp('-------------------------------')

            if F1 > F1best
                Pbest = P;
                Rbest = R;
                F1best = F1;
                reg_best = reg;
                lr_best = lr;
                arh_best = arhitektura{arh};
                ep_best = info.best_epoch;
            end
        end
    end
end

%% Treniranje NM sa optimalnim parametrima (na celom trening + val skupu)
rng(200)
net = patternnet(arh_best);

net.divideFcn = '';

net.performParam.regularization = reg_best;

net.trainFcn = 'trainrp';

net.trainParam.lr = lr_best;

net.trainParam.epochs = ep_best;
net.trainParam.goal = 1e-4;

[net, info] = train(net, ulazSve, izlazSve);

%% Performanse NM
rng(200)
pred = sim(net, ulazSve);
figure, plotconfusion(izlazSve, pred);

pred2 = sim(net, ulazTest);
figure, plotconfusion(izlazTest, pred2);

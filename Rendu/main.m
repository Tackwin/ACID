load('data_visu.mat');

echalottes = C1visu;
oignons = C2visu;

mean_echa = mean(echalottes);
mean_oig = mean(oignons);

var_echa = cov(echalottes);
var_oig = cov(oignons);

figure('Name','Visualization');
scatter3(echalottes(:, 1), echalottes(:, 2), echalottes(:, 3));
hold on;
%scatter3(oignons(:, 1), oignons(:, 2), oignons(:, 3));

load('data_5.mat');

sizeC1 = 5;
sizeC2 = 5;

%probas a priori
pC1=sizeC1/(sizeC1+sizeC2);
pC2=sizeC2/(sizeC1+sizeC2);

% MAP (utilisation des proba a priori)

[modelC1] = trainModel(TrainC1);
[modelC2] = trainModel(TrainC2);
   
MAPResC1    =    MAPClassify(TestC1, modelC1, modelC2, pC1, pC2, '1', '2');
MAPResC2    =    MAPClassify(TestC2, modelC1, modelC2, pC1, pC2, '1', '2');
MAPC1Error = computeError(MAPResC1, '1');
MAPC2Error = computeError(MAPResC2, '2');
MAPError = (length(TestC1) * MAPC1Error + length(TestC2) *  MAPC2Error)/ (length(TestC1) + length(TestC2));

% Perceptron
W = [1; 1; 1; 1];
[W] = Perceptron(TrainC1', TrainC2',W);

PerResC1 = PerClassify(TestC1',W, '1', '2');
PerResC2 = PerClassify(TestC2',W, '1', '2');
PerC1Error = computeError(PerResC1, '1');
PerC2Error = computeError(PerResC2, '2');
PerError = (length(TestC1) * PerC1Error + length(TestC2) *  PerC2Error)/ (length(TestC1) + length(TestC2));


% Dimensions deux manuellement

TrainC1D2 = TrainC1(:, [1, 3]);
TrainC2D2 = TrainC2(:, [1, 3]);
TestC1D2 = TestC1(:, [1, 3]);
TestC2D2 = TestC2(:, [1, 3]);

[modelC1D2] = trainModel(TrainC1D2);
[modelC2D2] = trainModel(TrainC2D2);
       
% MAP (utilisation des proba a priori)
MAPResC1D2    =    MAPClassify(TestC1D2, modelC1D2, modelC2D2, pC1, pC2, '1', '2');
MAPResC2D2    =    MAPClassify(TestC2D2, modelC1D2, modelC2D2, pC1, pC2, '1', '2');
MAPC1ErrorD2 = computeError(MAPResC1D2, '1');
MAPC2ErrorD2 = computeError(MAPResC2D2, '2');
MAPErrorD2 = (length(TestC1D2) * MAPC1ErrorD2 + length(TestC2D2) *  MAPC2ErrorD2)/ (length(TestC1D2) + length(TestC2D2));

% Perceptron
WD2 = [1; 1; 1];
[WD2] = Perceptron(TrainC1D2', TrainC2D2',WD2);

PerResC1D2 = PerClassify(TestC1D2',WD2, '1', '2');
PerResC2D2 = PerClassify(TestC2D2',WD2, '1', '2');
PerC1ErrorD2 = computeError(PerResC1D2, '1');
PerC2ErrorD2 = computeError(PerResC2D2, '2');
PerErrorD2 = (length(TestC1D2) * PerC1ErrorD2 + length(TestC2D2) *  PerC2ErrorD2)/ (length(TestC1D2) + length(TestC2D2));


% ACP
FullData = vertcat(TrainC1, TrainC2);
V = CalculACP(FullData);
V = V(:, 1:2);

TrainC1ACP = Projection(V, TrainC1);
TrainC2ACP = Projection(V, TrainC2);
TestC1ACP = Projection(V, TestC1);
TestC2ACP = Projection(V, TestC2);

[modelC1ACP] = trainModel(TrainC1ACP);
[modelC2ACP] = trainModel(TrainC2ACP);
       
% MAP (utilisation des proba a priori)
MAPResC1ACP =    MAPClassify(TestC1ACP, modelC1ACP, modelC2ACP, pC1, pC2, '1', '2');
MAPResC2ACP =    MAPClassify(TestC2ACP, modelC1ACP, modelC2ACP, pC1, pC2, '1', '2');
MAPC1ErrorACP = computeError(MAPResC1ACP, '1');
MAPC2ErrorACP = computeError(MAPResC2ACP, '2');
MAPErrorACP = (length(TestC1ACP) * MAPC1ErrorACP + length(TestC2ACP) *  MAPC2ErrorACP)/ (length(TestC1ACP) + length(TestC2ACP));

% Perceptron
WACP = [1; 1; 1];
[WACP] = Perceptron(TrainC1ACP', TrainC2ACP', WACP);

PerResC1ACP = PerClassify(TestC1ACP',WACP, '1', '2');
PerResC2ACP = PerClassify(TestC2ACP',WACP, '1', '2');
PerC1ErrorACP = computeError(PerResC1ACP, '1');
PerC2ErrorACP = computeError(PerResC2ACP, '2');
PerErrorACP = (length(TestC1ACP) * PerC1ErrorACP + length(TestC2ACP) *  PerC2ErrorACP)/ (length(TestC1ACP) + length(TestC2ACP));

hold off;

figure('Name','3D, 2D, ACP for 5 data points (First MAP then Perceptron)');
Label = categorical({'3D','2D', 'ACP'});
Label = reordercats(Label,{'3D','2D', 'ACP'});
bar(Label, [MAPError, PerError; MAPErrorD2, PerErrorD2; MAPErrorACP, PerErrorACP]);

load('data_100.mat');

sizeC1 = 100;
sizeC2 = 100;

%probas a priori
pC1=sizeC1/(sizeC1+sizeC2);
pC2=sizeC2/(sizeC1+sizeC2);

% MAP (utilisation des proba a priori)

[modelC1] = trainModel(TrainC1);
[modelC2] = trainModel(TrainC2);
   
MAPResC1    =    MAPClassify(TestC1, modelC1, modelC2, pC1, pC2, '1', '2');
MAPResC2    =    MAPClassify(TestC2, modelC1, modelC2, pC1, pC2, '1', '2');
MAPC1Error = computeError(MAPResC1, '1');
MAPC2Error = computeError(MAPResC2, '2');
MAPError = (length(TestC1) * MAPC1Error + length(TestC2) *  MAPC2Error)/ (length(TestC1) + length(TestC2));

% Perceptron
W = [1; 1; 1; 1];
[W] = Perceptron(TrainC1', TrainC2',W);

PerResC1 = PerClassify(TestC1',W, '1', '2');
PerResC2 = PerClassify(TestC2',W, '1', '2');
PerC1Error = computeError(PerResC1, '1');
PerC2Error = computeError(PerResC2, '2');
PerError = (length(TestC1) * PerC1Error + length(TestC2) *  PerC2Error)/ (length(TestC1) + length(TestC2));


% Dimensions deux manuellement

TrainC1D2 = TrainC1(:, [1, 3]);
TrainC2D2 = TrainC2(:, [1, 3]);
TestC1D2 = TestC1(:, [1, 3]);
TestC2D2 = TestC2(:, [1, 3]);

[modelC1D2] = trainModel(TrainC1D2);
[modelC2D2] = trainModel(TrainC2D2);
       
% MAP (utilisation des proba a priori)
MAPResC1D2    =    MAPClassify(TestC1D2, modelC1D2, modelC2D2, pC1, pC2, '1', '2');
MAPResC2D2    =    MAPClassify(TestC2D2, modelC1D2, modelC2D2, pC1, pC2, '1', '2');
MAPC1ErrorD2 = computeError(MAPResC1D2, '1');
MAPC2ErrorD2 = computeError(MAPResC2D2, '2');
MAPErrorD2 = (length(TestC1D2) * MAPC1ErrorD2 + length(TestC2D2) *  MAPC2ErrorD2)/ (length(TestC1D2) + length(TestC2D2));

% Perceptron
WD2 = [1; 1; 1];
[WD2] = Perceptron(TrainC1D2', TrainC2D2',WD2);

PerResC1D2 = PerClassify(TestC1D2',WD2, '1', '2');
PerResC2D2 = PerClassify(TestC2D2',WD2, '1', '2');
PerC1ErrorD2 = computeError(PerResC1D2, '1');
PerC2ErrorD2 = computeError(PerResC2D2, '2');
PerErrorD2 = (length(TestC1D2) * PerC1ErrorD2 + length(TestC2D2) *  PerC2ErrorD2)/ (length(TestC1D2) + length(TestC2D2));


% ACP
FullData = vertcat(TrainC1, TrainC2);
V = CalculACP(FullData);
V = V(:, 1:2);

TrainC1ACP = Projection(V, TrainC1);
TrainC2ACP = Projection(V, TrainC2);
TestC1ACP = Projection(V, TestC1);
TestC2ACP = Projection(V, TestC2);

[modelC1ACP] = trainModel(TrainC1ACP);
[modelC2ACP] = trainModel(TrainC2ACP);
       
% MAP (utilisation des proba a priori)
MAPResC1ACP =    MAPClassify(TestC1ACP, modelC1ACP, modelC2ACP, pC1, pC2, '1', '2');
MAPResC2ACP =    MAPClassify(TestC2ACP, modelC1ACP, modelC2ACP, pC1, pC2, '1', '2');
MAPC1ErrorACP = computeError(MAPResC1ACP, '1');
MAPC2ErrorACP = computeError(MAPResC2ACP, '2');
MAPErrorACP = (length(TestC1ACP) * MAPC1ErrorACP + length(TestC2ACP) *  MAPC2ErrorACP)/ (length(TestC1ACP) + length(TestC2ACP));

% Perceptron
WACP = [1; 1; 1];
[WACP] = Perceptron(TrainC1ACP', TrainC2ACP', WACP);

PerResC1ACP = PerClassify(TestC1ACP',WACP, '1', '2');
PerResC2ACP = PerClassify(TestC2ACP',WACP, '1', '2');
PerC1ErrorACP = computeError(PerResC1ACP, '1');
PerC2ErrorACP = computeError(PerResC2ACP, '2');
PerErrorACP = (length(TestC1ACP) * PerC1ErrorACP + length(TestC2ACP) *  PerC2ErrorACP)/ (length(TestC1ACP) + length(TestC2ACP));

hold off;

figure('Name','3D, 2D, ACP for 100 data points (First MAP then Perceptron)');
Label = categorical({'3D','2D', 'ACP'});
Label = reordercats(Label,{'3D','2D', 'ACP'});
bar(Label, [MAPError, PerError; MAPErrorD2, PerErrorD2; MAPErrorACP, PerErrorACP]);



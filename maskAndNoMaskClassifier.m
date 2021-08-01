%% Carico il Dataset locale e divido le immagini tra training set e test set
imds = imageDatastore('/Users/giovannisacco/Documents/MATLAB/Progetto_ML_real/dataset_mask', 'LabelSource', 'foldernames', 'IncludeSubfolders',true);

[imdsTrain, imdsValidation, imdsTest] = splitEachLabel(imds,0.6,0.1,'randomized');

%% Transfer learning della rete preaddestrata di partenza, Alexnet. 
% AlexNet è addestrato su più di un milione di immagini e può classificare 
% le immagini in 1000 categorie di oggetti. Di conseguenza, il modello 
% ha appreso rappresentazioni ricche di funzionalità 
% per un'ampia gamma di immagini.

net = alexnet;
inputSize = net.Layers(1).InputSize;

% Estraggo tutti i layer della rete fatta eccezione per gli ultimi 3 in
% quanto questi ultimi sono addestrati per il riconoscimento di 1000 classi
% mentre devono essere messi a punto per il nuovo problema di
% classificiazione
layersTransfer = net.Layers(1:end-3);

% Estraggo il numero di classi previste per il nuovo problema di
% classficiazione (nel nostro caso saranno 2, with_mask & without_mask
numClasses = numel(categories(imdsTrain.Labels));

% Modifico i layer di Alexnet per affrontare la nuova classificazione 
% sostituendo gli ultimi tre livelli con un layer fully connected 
% (calibrato sulle nuove classi trovate prima),un layer softmax e un layer
% di output per la classificazione.
% Per apprendere più velocemente nei nuovi livelli rispetto ai livelli 
% lasciati invariati da Alexnet, aumento i valori WeightLearnRateFactor e
% BiasLearnRateFactor del livello fully connected.
layers = [
    layersTransfer
    fullyConnectedLayer(numClasses,'WeightLearnRateFactor',20,'BiasLearnRateFactor',20)
    softmaxLayer
    classificationLayer];

%% Manipolazione delle immagini
% La rete richiede immagini di input di dimensione (227 227 3), ma le 
% immagini nel nostro Dataset hanno dimensioni diverse. Utilizzo allora
% l' AugmentedImageDatastore per ridimensionarle TUTTE automaticamente. 
% Specifico poi delle operazioni di Argumentation
% aggiuntive da eseguire SOLO sulle immagini di addestramento: 
% capovolgo randomicamente le immagini di addestramento lungo l'asse 
% verticale e le traduco in modo casuale fino a 30 pixel in orizzontale e
% in verticale. L'aumento dei dati così prodotto aiuta a prevenire 
% l'overfitting della rete e a memorizzare i dettagli 
% esatti delle immagini di addestramento.

pixelRange = [-30 30];
imageAugmenter = imageDataAugmenter( ...
    'RandXReflection',true, ...
    'RandXTranslation',pixelRange, ...
    'RandYTranslation',pixelRange);

augimdsTrain = augmentedImageDatastore(inputSize,imdsTrain, ...
    'DataAugmentation',imageAugmenter);

augimdsValidation = augmentedImageDatastore(inputSize,imdsValidation);
augimdsTest = augmentedImageDatastore(inputSize,imdsTest);

%% Imposto le opzioni di addestramento e addestro la rete
% 'Congelo' le feature dei primi livelli della rete preaddestrata. 
% Per rallentare l'apprendimento nei leayer ricevuti as is da Alexnet 
% imposto la velocità di apprendimento iniziale su un valore basso. 
% Ricordo che nel passaggio precedente ho aumentato i fattori di velocità 
% di apprendimento per il livello fully connected per accelerare 
% l'apprendimento nei nuovi livelli finali. 
% Questa combinazione di impostazioni della velocità di apprendimento 
% determina quindi un apprendimento veloce solo nei nuovi livelli e un 
% apprendimento più lento negli altri livelli. 
% Infine nel caso del transfer learning, non è necessario effettuare il
% training per tante epoche per questo ne imposto solo 6.

options = trainingOptions('sgdm', ...
    'MiniBatchSize',10, ...
    'MaxEpochs',6, ...
    'InitialLearnRate',1e-4, ...
    'Shuffle','every-epoch', ...
    'ValidationData',augimdsValidation, ...
    'ValidationFrequency',3, ...
    'Verbose',false, ...
    'Plots','training-progress');

% Addestro la 'nuova' rete
netTransfer = trainNetwork(augimdsTrain,layers,options);

%% Classifico immagini del test set con la 'nuova' rete appena addestrata

[YPred,scores] = classify(netTransfer,augimdsTest);

% Mostro i risultati ottenuti su alcune immagini del dataset
idx = randperm(numel(imdsTest.Files),4);
figure
for i = 1:4
    subplot(2,2,i)
    I = readimage(imdsTest,idx(i));
    imshow(I)
    label = YPred(idx(i));
    title(string(label));
end

%% Calcolo l'accuracy rispetto al dataset

YTest = imdsTest.Labels;
accuracy = mean(YPred == YTest);
disp(accuracy);

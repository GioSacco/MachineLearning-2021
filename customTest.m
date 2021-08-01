%% Test con immagine scelta dall'utente

% Seleziono l'immagine tra quelle presenti sulla memoria del dispositivo
[filename, pathname] = uigetfile('*.*', 'Select the input grayscale image');
filewithpath = strcat(pathname, filename);
originalImage = imread(filewithpath);

% Effettuo il resize dell'immagine
resizedImage = imresize(originalImage, [227 227]);

% Effettuo la classificazione e mostro il risultato
figure;
imshow(resizedImage);
label = classify(netTransfer, resizedImage);
title(char(label));
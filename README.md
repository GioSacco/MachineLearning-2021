# Machine Learning 2020/2021 - Giovanni Sacco

## Progetto per l'esame di Machine Learning A.A. 2020/2021

- Il progetto consiste nella creazione di un classificatore che, ricevuta in input un'immagine di un uomo/donna, verifichi la presenza o meno della mascherina.
- Come Dataset ho utilizzato quello disponibile al seguente link https://www.kaggle.com/dhruvmak/face-mask-detection 

## Tecnologia e metodologia utilizzata

- Per lo sviluppo ho utilizzato Matlab e, nello specifico i seguenti Toolbox:
	- Deep Learning Toolbox
	- Deep Learning Toolbox Model for AlexNet Network
- Sono parito dalla rete AlexNet, preaddestrata a riconosce 1000 classi differenti di immagini, e l'ho appositamente customizzata per ottimizzarne i risultati sul problema di classificazione affrontato per il progetto. Nel dettaglio, per implementare il Transfer Learning, ho effettuato le seguenti operazioni:
	- ho eliminato gli ultimi 3 layer di Alexnet, addestrati a riconoscere 1000 differenti classi di immagini;
	- ho aggiunto i seguenti layer:
		- layer fully connected (calibrato sulle sole due classi - with_mask e without_mask - relative al nostro problema): inoltre, per apprendere più velocemente nei nuovi layer rispetto ai layer lasciati invariati da Alexnet, ho aumentato i valori WeightLearnRateFactor e BiasLearnRateFactor di tale layer;
		- layer softmax
		- layer di output per la classificazione.
- Successivamente ho effettuato delle attività di ottimizzazione sulle immagini del Dataset:
	- ho utilizzato l'AugmentedImageDatastore per ridimensionare tutte le immagini (training, validation, test) in quanto la rete richiede immagini di input di dimensione (227 227 3);
	- ho poi rielaborato le immagini del solo training set:
		- ho capovolto randomicamente le immagini lungo l'asse verticale e le traduco in modo casuale fino a 30 pixel in orizzontale e in verticale;
		- l'aumento dei dati così prodotto aiuta difatti a prevenire l'overfitting della rete e a memorizzare i dettagli esatti delle immagini di addestramento.
- Prima di procedere con l'addestramento 'congelo' le feature dei primi livelli della rete preaddestrata. Per rallentare l'apprendimento nei leayer ricevuti as is da Alexnet imposto quindi la velocità di apprendimento iniziale su un valore basso. Ricordo che nel passaggio precedente ho aumentato i fattori di velocità di apprendimento per il livello fully connected per accelerare l'apprendimento nei nuovi livelli finali. Questa combinazione di impostazioni della velocità di apprendimento determina quindi un apprendimento veloce solo nei nuovi layer e un apprendimento più lento negli altri. Infine nel caso del Transfer Learning, non essendo necessario effettuare il training per tante epoche, ne imposto solo 6.

## Risultati ottenuti

- Valutata la rete su test set l'accuracy ottenuta è del 0.9924.

- Ho creato infine un secondo file 'customTest.m' tramite il quale è possile sottomettere immagini selezionate manulamente dalla memoria del calcolatore.
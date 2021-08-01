# Machine Learning 2020/2021 - Giovanni Sacco

## Progetto per l'esame di Machine Learning A.A. 2020/2021

- Il progetto consiste nella creazione di un classificatore che, ricevuta in input un'immagine di un uomo/donna, verifichi la presenza o meno della mascherina.
- Come Dataset ho utilizzato quello disponibile al seguente link https://www.kaggle.com/dhruvmak/face-mask-detection 
- Per lo sviluppo ho utilizzato Matlab e, nello specifico i seguenti Toolbox:
	- Deep Learning Toolbox
	- Deep Learning Toolbox Model for AlexNet Network
- Sono parito dalla rete AlexNet, preaddestrata a riconosce 1000 classi differenti di immagini, e l'ho appositamente customizzata per ottimizzarne i risultati sul problema di classificazione affrontato per il progetto. Nel dettaglio, per implementare il Transfer Learning, ho effettuato le seguenti operazioni:
	- ho eliminato gli ultimi 3 layer di Alexnet, addestrati a riconoscere 1000 differenti classi di immagini;
	- ho aggiunto i seguenti layer:
		- layer fully connected (calibrato sulle sole due classi - with_mask e without_mask - relative al nostro problema);
		- layer softmax
		- layer di output per la classificazione.
- 
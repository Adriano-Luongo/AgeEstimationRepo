# Stima dell'età apparente su dataset VGGFACE2

Questa repository contiene il codice relativo al documento "Contest di visione artificiale - Gruppo 12"
I partecipanti sono:

Adriano Luongo  a.luongo45@studenti.unisa.it

Simone Cristofaro s.cristofaro3@studenti.unisa.it

Elena Guzzo e.guzzo6@studenti.unisa.it

Carmen Fucile c.fucile@studenti.unisa.it

In particolare il seguente README fornisce le indicazioni richieste quali

- istruzioni per riprodurre l'allenamento
- istruzioni per riprodurre il test

Dal momento che il modello della rete superava le dimensioni massime consentite da GitHub quest'ultimo è stato caricato al seguente link drive:
*https://drive.google.com/file/d/1A5coeNbuVIu8Mu5nmBzlWvWiFuo8_IYk/view?usp=sharing*

## Allenamento
Per l'allenamento abbiamo studiato il codice offerto dal framework presente a  https://github.com/MiviaLab e l'abbiamo modificato per adattarlo alle nostre esigenze.
In particolare, per effettuare l'allenamento in maniera rapida si è scelto di rappresentare il dataset mediante l'utilizzo dei TFrecords e di utilizzare tali dati tramite l'API di Tensorflow. Per questo motivo, è necessario avere a disposizione, per effettuare l'allenamento, il dataset in tale formato e dividerlo, come indicato dal codice, tra le cartelle (train,val,test) in /dataset/data/vggface2_data . La convenzione sui nomi dei tfrecord segue questa espressione: partition.reduced(random)_*.tfrecord dove l'asterisco rappresenta il numero del tfrecord. 
Di seguito sono posti i link per il download dei file tfrecords da noi utilizzati.

Train:
1) https://drive.google.com/file/d/1u5In7vy-SgLMJvAjZG1lIJCTzvUX6QQE/view?usp=sharing
2) https://drive.google.com/file/d/1kLLdQfdV7qKHgNQSJtOubK0m7MbKqWJf/view?usp=sharing
3) https://drive.google.com/file/d/1zf-AdezJvpCCTZ8ixU6XBNjGl2edNxKI/view?usp=sharing
4) https://drive.google.com/file/d/1bS47YKJ8LsEF3vSc6zRY6su1CT9oNjoG/view?usp=sharing
5) https://drive.google.com/file/d/1FVoZqMnXjWJYs1nE92lkL4o5eyrvQ0iu/view?usp=sharing
6) https://drive.google.com/file/d/1_9d8pRIb9nR7Zrphx2CSpw0LiMFozebj/view?usp=sharing
Train_aggiuntivo:
7) https://drive.google.com/file/d/1QYgrO14RB_hyse9mmD_kf83U8aJCNSGi/view?usp=sharing
8) https://drive.google.com/file/d/1PPSl6sXzvP8tdNqeDCf5xLGz8DcXdS-l/view?usp=sharing
9) https://drive.google.com/file/d/176Hn78xQ7Cg4SQ4QNkm0L9Sg6R2cVBZp/view?usp=sharing

Validation:
https://drive.google.com/file/d/1LjJ7qi9r0zb-qzjiFHpls9v8IXEYu1GF/view

Test:
https://drive.google.com/file/d/1ftBUGUGPV4AL2sOEhNwMZE41NmGe08xP/view

Una volta sistemati i file nelle apposite cartelle per allenare la rete è necessario, far partire lo script <code>train.py</code> dalla directory training. 
Dal momento che lo spazio di memoria massimo offerto da Colab, piattaforma usata per allenare la rete, è insufficiente a conservare tutti i tfrecords creati da noi, è stato necessario dividere l'allenamento in fasi successive, alternando gruppi di file da 3 tfrecord ciascuno. Dal momento che ogni Tfrecord contiene 100.000 immagini, abbiamo utilizzato gruppi da 300.000 immagini per ogni fase di training.

La prima fase di allenamento, quindi le prime 10 epoche sono state effettuate con i primi 3 tfrecord, a partire dal set di pesi pre-allenati sul dataset VGGFace1. Il comando utilizzato è il seguente 
```bash
!python3 train.py --net resnet50 --augmentation default --preprocessing vggface2 --optimizer adam --pretraining vggface --batch 128 --lr 0.001 --training-epochs 10 --dir "/path/to/save/results"
```
Successivamente l'allenamento è stato ripreso dalla 9 epoca, ultima epoca in cui il modello è migliorato, cambiando però il set di immagini su cui far allenare la rete.
Il comando lanciato è il seguente
```bash
!python3 train.py --net resnet50 --augmentation default --preprocessing vggface2 --optimizer adam --resume True --resumepath 'checkpoint.09.hdf5' --batch 128 --lr 0.001 --training-epochs 20 --dir "/path/to/save/results"
```

L'allenamento è stato poi ripreso dall'epoca 19, l'ultima epoca in cui il modello è migliorato. Da questo punto sono state eseguite 5 epoche sul primo set d'immagini e altre 5 sul secondo. Il comando lanciato è il seguente per il primo set 
```bash
!python3 train.py --net resnet50 --augmentation default --preprocessing vggface2 --optimizer adam --resume True --resumepath 'checkpoint.19.hdf5' --batch 128 --lr 0.001 --training-epochs 25 --dir "/path/to/save/results"
```
E questo per il secondo 

```bash
!python3 train.py --net resnet50 --augmentation default --preprocessing vggface2 --optimizer adam --resume True --resumepath 'checkpoint.25.hdf5' --batch 128 --lr 0.001 --training-epochs 30 --dir "/path/to/save/results"
```

Successivamente, avendo ancora un po' di tempo a disposizione è stato caricato un set di 300k immagini aggiuntive e l'allenamento è proseguito con questo nuovo set d'immagini.
Il comando: 
```bash
!python3 train.py --net resnet50 --augmentation default --preprocessing vggface2 --optimizer adam --resume True --resumepath 'checkpoint.30.hdf5' --batch 128 --lr 0.001 --training-epochs 40 --dir "/path/to/save/results"
```
L'allenamento termina alla 38-esima epoca per mancanza di tempo. L'ultimo miglioramento si ottiene all'epoca 36. 

## Test
Per effettuare il test della rete e quindi generare il file csv richiesto, è necessario come detto precedentemente popolare la cartella dataset/data/vggface2_data/test con il file tfrecord costruito ad hoc per il test. 
Successivamente il comando seguente viene lanciato

```bash
!python3 train.py --mode "test" --net resnet50 --preprocessing vggface2  --testweights 'checkpoint.36.hdf5' --batch 128 
```
La batch size è impostata per decidere il numero di elementi da dare alla rete per la prediction ad ogni step. 


## Fonti esterne
Il codice all'interno della repository ha fatto utilizzo di codice presente in queste repository:
* VGGFace: https://github.com/rcmalli/keras-vggface
* BaseFramework: https://github.com/MiviaLab/GenderRecognitionFramework




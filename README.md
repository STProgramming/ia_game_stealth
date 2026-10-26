# Shadow Protocol RL

## Avvio rapido del gioco

Il repository include il frontend, il server Python e i modelli finali V7.1
per guardie e SWAT. Per avviare il gioco su Windows con Python 3.11 o successivo:

```powershell
git clone https://github.com/STProgramming/ia_game_stealth.git
cd ia_game_stealth
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt -r requirements_frontend.txt
python game_server.py
```

Apri <http://127.0.0.1:8765> nel browser. Il server trova automaticamente
`models_v7_1_guard/guard_v7_1_final.zip` e
`models_v7_1_swat/swat_v7_1_final.zip`.

I checkpoint intermedi, l'ambiente virtuale e i dati delle partite in `.runtime/`
rimangono locali. Il frontend non richiede una compilazione. Per verificare il progetto:

```powershell
python -m pytest tests -q
node --test tests/test_frontend.cjs tests/test_radar.cjs
```

Node.js serve soltanto per i test del frontend. Le sezioni seguenti descrivono
l'architettura e gli strumenti per l'addestramento.

Questa cartella è la prima versione dell'IA **che apprende davvero** per le guardie di Shadow Protocol.

## Architettura

La policy PPO NON controlla direttamente `x/y` della guardia.

La rete sceglie un intento tattico:

| ID | Azione |
|---:|---|
| 0 | HOLD |
| 1 | PATROL |
| 2 | INVESTIGATE_SOUND |
| 3 | GO_LAST_SEEN |
| 4 | INTERCEPT |
| 5 | SEARCH_LAST_SEEN |
| 6 | PROTECT_NEAREST_INTEL |
| 7 | FLANK_LEFT |
| 8 | FLANK_RIGHT |
| 9 | PURSUE |
| 10 | ARREST |
| 11 | CALL_REINFORCEMENTS |

Il movimento vero viene poi eseguito da A*.

Questa separazione è importante: la rete impara **cosa fare**, mentre collisioni,
muri e pathfinding restano deterministici.

## Niente telepatia

`ShadowProtocolEnv._sense()` è l'unico punto autorizzato a leggere la posizione
reale del player.

Quando il player non è visibile, la policy vede soltanto:

- ultima posizione realmente vista;
- età di quella memoria;
- velocità osservata mentre era visibile;
- rumore con errore;
- stato degli intel che la guardia ha realmente controllato;
- stato di allerta;
- se sono già stati chiamati rinforzi.

La posizione nascosta del player non viene inserita né nell'osservazione né in
`info`.

## Installazione Windows

Consigliato Python 3.11 / 3.12 / 3.13.

```powershell
cd shadow_protocol_rl

python -m venv .venv
.\.venv\Scripts\Activate.ps1

python -m pip install --upgrade pip
pip install -r requirements.txt
```

## 1. Verifica subito l'ambiente

```powershell
python demo_random.py
```

Poi:

```powershell
python -m pytest tests -q
```

Se non hai pytest:

```powershell
pip install pytest
```

## 2. Primo training

Per una prova breve:

```powershell
python train.py --timesteps 50000
```

Per iniziare a vedere qualcosa di più serio:

```powershell
python train.py --timesteps 300000
```

Per training più lungo:

```powershell
python train.py --timesteps 1000000
```

Output:

```text
models/
  best/best_model.zip
  checkpoints/
  eval/
  tensorboard/
  shadow_guard_ppo.zip
```

## 3. Valuta la guardia

```powershell
python evaluate.py --model models/best/best_model.zip --episodes 50
```

Guarda soprattutto:

- captures;
- escapes;
- reward medio;
- distribuzione delle azioni.

Se la policy usa sempre la stessa azione, non è ancora una buona IA.

## 4. Esporta per il browser

```powershell
python export_onnx.py --model models/best/best_model.zip
```

Ottieni:

```text
models/shadow_guard_policy.onnx
```

È il file che poi possiamo caricare nel gioco JavaScript tramite ONNX Runtime Web.

## Reward attuale

Il reward è volutamente semplice.

Premia:

- uso coerente di un rumore realmente percepito;
- perseguimento dell'ultima posizione vista;
- mantenimento del contatto;
- arresto di un player che si arrende;
- chiamata rinforzi quando giustificata;
- cattura.

Penalizza:

- azioni che richiedono informazione non disponibile;
- chiamare rinforzi senza motivo;
- tempo perso;
- fuga completa del player.

Non assegna mai un premio in base al fatto che la guardia si stia avvicinando
alla **posizione nascosta corrente** del player. Questo è essenziale per non
insegnarle la telepatia.

## Player bot

Ogni episodio estrae uno stile:

- `stealth`
- `runner`
- `deceiver`
- `mixed`

Quindi la guardia non si allena contro una sola traiettoria.

Il passo successivo può aggiungere:

- fumogeni;
- diversivi;
- corpi;
- sangue;
- arresto in coppia;
- furto intel con scoperta differita;
- SWAT;
- curriculum learning.

## Multi-agent

`shadow_ai/multi_agent_env.py` contiene già lo scaffold PettingZoo.

L'obiettivo successivo è:

```text
GuardPolicy condivisa
   ├── guard_0 -> osservazione locale 0
   ├── guard_1 -> osservazione locale 1
   ├── guard_2 -> osservazione locale 2
   └── guard_3 -> osservazione locale 3
```

Stessa rete, conoscenze diverse.

Poi possiamo introdurre un blackboard radio separato, dove un agente comunica
soltanto informazioni che possiede davvero.

## Nota

Non aspettarti una guardia intelligente dopo 5.000 step.

L'ambiente è progettato per training progressivo. Prima dobbiamo verificare che:

1. impari il contatto visivo;
2. impari ad andare all'ultima posizione;
3. impari quando investigare un rumore;
4. impari protezione/intercetto;
5. soltanto dopo passiamo a più agenti.


## Diversivi e apprendimento nelle partite interattive

In guardia normale un diversivo provoca un'indagine locale. Se lo sentono due
guardie, si comunicano chi controlla e chi resta in sorveglianza. In guardia
alta una coppia raggiunge punti distinti per controllare e coprire il fianco.
Il secondo dispositivo che attira la stessa guardia genera cautela generale
persistente, richiede una scorta disponibile e aumenta l'attenzione delle altre
guardie. I tre impulsi dello stesso dispositivo valgono come un solo evento.
La guardia alta conseguente a un avvistamento resta attiva dopo la perdita del
contatto; il reset azzera la memoria dei diversivi della partita.

Se una guardia vede direttamente il lancio, raggiunge subito la sorgente stimata
con un approccio tattico. Più testimoni si distribuiscono tra controllo e copertura
del fianco, poi cercano nella zona di partenza. Il punto stimato deriva solo dal
lancio osservato: non segue il giocatore nascosto e gli impulsi del dispositivo
non spostano l'indagine verso il punto di atterraggio. Il rilevamento rispetta
campo visivo, distanza, muri e stato operativo della guardia.

Ogni cattura, neutralizzazione o uscita vittoriosa salva un JSON in
`.runtime/episodes/`, con osservazioni della policy, azioni tattiche, eventi dei
diversivi ed esito. Il server esegue in background un piccolo aggiornamento dei
pesi basato sull'esito: ricompensa positiva alla sicurezza per cattura, negativa
per fuga, con baseline di valore ed entropia. Questo aggiornamento incrementale
non equivale a un nuovo rollout PPO: usa anche gli intenti deterministici del
runtime e non garantisce un miglioramento dopo ogni singola partita.

Il checkpoint originale viene preservato; ogni episodio produce un checkpoint
separato e aggiorna `guard_latest.zip` o `swat_latest.zip`. Il modello aggiornato
viene caricato alla prossima partita iniziata dopo il completamento del lavoro.
L'interfaccia mostra addestramento in corso, completato oppure fallito; i dettagli
del fallimento restano nello schema e nello stato `learning` del server. Un reset
manuale di una partita non conclusa non avvia l'addestramento. Le osservazioni
registrate contengono soltanto le informazioni disponibili alla policy.

## Spotting e risposta tattica nel gioco interattivo

Un avvistamento parziale dal 25% provoca verifica locale e solitaria. La guardia
chiede un buddy soltanto se sono già successi eventi sospetti, il nemico è stato
confermato in precedenza oppure la stessa guardia ha avuto almeno due episodi
distinti di spotting all'80%. Un unico avvistamento prolungato conta una volta;
per iniziare un nuovo episodio la fiducia deve scendere sotto il 15%. Dopo
1,2 secondi di radio una sola pattuglia disponibile raggiunge il punto stimato.
La chiamata si interrompe se il mittente viene neutralizzato. Il sospetto visivo
parziale non manda tutta la struttura in cautela. Al contatto confermato la guardia
chiama prima un collega operativo entro otto unità di distanza e raggiungibile
con un percorso di massimo dodici unità. Se manca un collega disponibile, oppure
il supporto assegnato viene neutralizzato, parte la chiamata alla SWAT. Le
telecamere mantengono l'invio diretto della SWAT. Una sentinella con contatto
visivo verifica o insegue anche quando la policy propone di restare ferma.

Ogni cono interpola il proprio indice di spotting: azzurro allo 0%, giallo al
50%, rosso al 100%, anche per telecamere e SWAT. Il colore non dipende
dall'allarme globale e il sospetto degli altri agenti non lo modifica.

La SWAT di rinforzo corre più rapidamente verso i rapporti ricevuti e può
seguire la rotta delle pattuglie ancora in ricerca. Se l'ultimo punto noto è
nelle stanze chiuse archivio o caveau, la squadra sceglie una porta comune,
attende in posizioni distinte, usa una flashbang e un fumogeno, poi controlla
gli angoli con percorsi distribuiti. Le granate hanno scorte limitate; il fumo
blocca la vista senza bloccare il movimento e la flash rallenta temporaneamente
il giocatore nel raggio dell'esplosione, rispettando i muri. Le guardie libere
sorvegliano gli intel ancora presenti e l'esfiltrazione; quelle in contatto
continuano a verificare o inseguire. Anche la modalità SWAT dedicata usa
movimento e rotazioni più rapidi. Il reset azzera chiamate, piani ed effetti.

Accovacciarsi usa **C**, evitando le scorciatoie del browser con Ctrl. Il player
diventa verde quando può stordire un nemico alle spalle e giallo se possiede anche
un coltello per una neutralizzazione letale. Durante la presa silenziosa resta
verde; muri e distanza impediscono prese non valide. Salute e stamina
sono indicate da due barre sotto ogni personaggio. I colpi delle guardie consumano
salute, le neutralizzazioni letali azzerano quella del bersaglio.

La corsa consuma stamina anche per guardie e SWAT. Il recupero inizia dopo due
secondi senza corsa ed è graduale. Ogni ciclo di corsa di almeno mezzo secondo e
recupero aggiunge l'8% di fatica, fino al 40%: la velocità cala e soltanto un riposo
prolungato dopo aver riempito la stamina rimuove gradualmente la fatica. Dopo
l'esaurimento bisogna recuperare almeno il 35% prima di correre nuovamente.
Quando viene scoperto, il player ha 2,5 secondi di corsa al 165% della velocità
normale di sprint, consumando più stamina. Gli aggiornamenti dello stesso contatto
non rinnovano lo scatto; un nuovo episodio dopo almeno 12 secondi senza contatto
confermato può riattivarlo.

Trovare un cadavere o un compagno stordito aumenta la reattività visiva e riduce
il tempo di puntamento delle guardie da 0,95 a 0,35 secondi. La guardia richiede
copertura, controlla tatticamente il punto e chiama due guardie ordinarie dopo
due secondi di radio. Il ritrovamento da solo non invia la SWAT e non comunica
la posizione nascosta del giocatore.

## Equipaggiamento da raccogliere e sparatorie

Il player inizia con il proprio coltello, senza armi da fuoco, munizioni o granate.
Il coltello appartiene solo al player e non compare sulla mappa o nel bottino.
Le prime due missioni contengono quattro rifornimenti raggiungibili: un caricatore nei servizi,
bende, una postazione medica e un fumogeno. Pistole e munizioni non sono
distribuite sulla mappa: le armi si recuperano dalle unità neutralizzate.
La terza aggiunge una protezione biologica con respiratore e una provetta sterile,
entrambe fuori dalla zona contaminata.
**E** raccoglie gli oggetti vicini,
rispettando i muri, e prende anche gli intel. Ogni rifornimento si consuma una
volta; il reset ripristina la mappa e azzera l'inventario.

**F tenuto** permette di soffocare e stordire una guardia da dietro. Una guardia neutralizzata lascia la
propria arma, le munizioni rimaste e un caricatore; la SWAT lascia anche le granate
non utilizzate. Il bottino rimane a terra finché non viene raccolto con E.
**F** richiede un coltello. **H**, oppure il tasto destro del mouse, spara con
l'arma raccolta nella direzione dello sguardo: ogni colpo consuma munizioni e può ferire o
abbattere il primo nemico sulla traiettoria, ma non attraversa i muri. **Q** lancia il bossolo o caricatore selezionato nello zaino.
Lo sguardo segue il movimento e mantiene la direzione quando il player si ferma.
Il mouse serve a selezionare lo zaino e non orienta più la visuale.
Seleziona granate e oggetti con il mouse e tocca **E** per usarli. Le bende raccolte entrano nello zaino; le postazioni mediche rimangono nella mappa.

Guardie e SWAT controllano la linea di tiro: se un collega operativo si trova
tra il tiratore e il player, trattengono il fuoco e si spostano lateralmente
verso un angolo raggiungibile e libero. I lati preferiti alternano tra i membri
della squadra; durante lo spostamento il tiratore resta rivolto verso l'ultimo
punto realmente osservato. La stessa protezione vale nella modalità SWAT
dedicata. Gli spari producono un richiamo sonoro per i nemici vicini.

## Ferite e scie di sangue

I proiettili ricevuti aprono ferite: il player perde gradualmente salute e vigore,
anche da fermo. La velocità effettiva aumenta il sanguinamento: correre e usare
lo scatto lascia più gocce e accelera il consumo di salute e stamina. La fatica
rallenta il player e una ferita aperta ritarda il recupero. La benda (seleziona nello zaino e tocca **E**) ferma il sanguinamento dopo **1 secondo** senza recuperare salute.
La postazione medica (tieni **E** da vicino) richiede **4 secondi**, recupera gradualmente il **10% della
salute massima durante l’applicazione**, poi un ulteriore **25% nei successivi
2 minuti**, fino al limite della salute massima. Ferma il sanguinamento al
completamento. **E breve** raccoglie le bende; **E tenuto** usa la postazione fissa: si trovano sulla mappa,
e i kit medici non vengono lasciati dalle sentinelle neutralizzate. Movimento,
azioni di combattimento o un nuovo colpo interrompono l’applicazione; l’oggetto
iniziato viene consumato e il recupero differito parte solo completando il kit.

Guardie e SWAT, perso il contatto visivo, seguono le nuove tracce realmente
visibili. I muri e il cono visivo limitano ciò che possono scoprire: non ricevono
la posizione nascosta del player. Le tracce esaminate sono note alla squadra e
non avviano altre indagini; nuove gocce possono prolungare l'inseguimento. Alla
fine della scia l'unità cerca attorno all'ultima traccia. Il sangue rimasto dopo
un'uccisione ravvicinata continua a lasciare impronte senza ferire il player.

## Zaino, comandi e protezioni

La salute e il vigore del player sono sempre visibili nel pannello in alto a
sinistra. Lo zaino in basso mostra icone, quantità e selezione: clicca un
oggetto o un’arma con il mouse. **E breve** raccoglie un oggetto portatile vicino
(o un intel), altrimenti usa l’oggetto selezionato. Le postazioni mediche sono
fisse nell’infermeria: avvicinati e tieni **E** durante i 4 secondi di cura. Le
bende si trovano nei punti di primo soccorso; un fumogeno è nel settore sicurezza,
un caricatore nei servizi. Armi e ulteriori oggetti provengono dal bottino nemico.
Le azioni mediche non usano più J o K.

**C** cammina lentamente e accovaccia. **Spazio** esegue un tuffo di circa
3,2 celle, bloccato dai muri, con tonfo udibile entro 12 celle, costo del 28%
di vigore e pausa di 1,2 secondi tra tuffi. **F breve** usa il coltello: da dietro
taglia la gola, frontalmente toglie il 25% della salute e blocca la sentinella
per 0,3 secondi: durante la reazione al colpo non può muoversi, sparare o
catturare. Poi può contrattaccare se ha salute e vigore. Costa il 22% del
vigore e ha una pausa di 0,6 secondi.
**F tenuto** soffoca da dietro per 3 secondi: consuma il 15% iniziale e il 20%
per secondo; movimento, rilascio o mancanza di energia lo interrompono.

Le barre di salute, vigore e corazza dei nemici appaiono solo quando il player
li osserva direttamente, entro 12 celle e 90° nella direzione dello sguardo,
con visuale libera da muri e fumo. La SWAT parte con corazza consumabile: il
fucile ne distrugge il 32% a colpo, la pistola il 16%, con una piccola quota
di danno alla salute. Esaurita la corazza i colpi infliggono danno completo.
Lo scudo orientato verso il tiratore blocca salute e corazza, ma costa al
portatore il 30% del vigore per colpo di fucile o il 20% per colpo di pistola;
i colpi sui fianchi o da dietro aggirano lo scudo.
Lo scudo blindato è un rettangolo davanti al soldato con una feritoia visiva:
il portatore vede attraverso, con cono ridotto da 104° a 84° e portata ridotta del 15%.
Il player non ha barre sotto il corpo: salute e vigore restano nel pannello fisso.
Durante un inseguimento, una guardia rimasta ferma per 0,8 secondi ricalcola
il percorso e tenta di riprendere il movimento rispettando i muri.

La mappa giocabile rappresenta l’atrio di un centro multinazionale di ricerca
chimica: reception, direzione, archivio, laboratori, sicurezza, infermeria e
deposito campioni, con porte e corridoi collegati. La modalità SWAT usa una
planimetria compatta dello stesso tipo; le mappe di addestramento restano invariate.

Il fumogeno crea una nube di raggio 3 celle per 8 secondi. Dentro o attraverso
la nube, la portata visiva scende al 5%: si può ancora vedere da molto vicino.
Guardie e SWAT scelgono un’uscita raggiungibile usando la geometria della mappa
e interrompono il puntamento per liberarsi dal fumo. La nube consuma il 35%
del vigore massimo al secondo e ritarda il recupero. A vigore esaurito provoca
una perdita di salute del 2,5% al secondo, limitata complessivamente al 10%
della salute massima per partita. Le stesse regole valgono per il player.
Chi è esausto cammina e non può correre o sparare; riprende le azioni armate
dopo aver recuperato almeno il 35% del vigore.

La guardia principale e tutte le sentinelle usano lo stesso motore di azioni:
spotting, inseguimento, ricerca locale, indagine sui rumori, prove e sangue,
supporto radio, difesa degli intel e dell’esfiltrazione, uscita dal fumo,
spostamento per liberare la linea di tiro, puntamento e fuoco. La rete della
guardia principale propone intenzioni tramite la stessa interfaccia disponibile
alle altre guardie (`policy_action`); le reazioni di sicurezza, i limiti del vigore
e le informazioni realmente osservate valgono per tutte. Percorso, memoria,
decisioni dell’IA ed equipaggiamento possono differire tra le unità.
Durante l'allerta, le guardie in movimento corrono consumando il 14% del
vigore al secondo; se esaurite rallentano fino al recupero. Lo sparo di pistola
costa l'8% del vigore e richiede energia sufficiente. La SWAT ha una riserva di
vigore pari a 2,5 volte quella delle guardie: a barra piena può correre per
circa 18 secondi, con un consumo del 5,6% al secondo. Per sparo consuma il
3,2% con la pistola, il 6,4% con la canna liscia, l'1,8% con il mitra e il
2% con il mitra a una mano; una granata costa il 12% della sua barra.
La SWAT mantiene lo stesso passo di corsa tra risposta, ricerca e ingresso
nelle stanze. Le brevi soste per controllare gli angoli non aggiungono fatica;
il consumo di corsa avviene solo durante uno spostamento effettivo. Ferite,
esaurimento e accecamento continuano a influenzare il movimento.

Quando una sentinella controlla da vicino un punto intel vuoto, chiede via
radio se qualche collega lo ha preso. Dopo 1,2 secondi arriva la risposta
negativa e la sicurezza passa in CAUTELA, senza chiamare rinforzi o SWAT e
senza conoscere la posizione del player. Se non ci sono colleghi disponibili,
la mancanza di conferme porta ugualmente in cautela. Il controllo richiede
visuale libera da muri e fumo; una chiamata interrotta dalla neutralizzazione
dell'osservatore può essere ripresa da un'altra guardia. Un'allerta di
combattimento già attiva resta tale.

Entro 1,35 celle, guardie e SWAT tentano un attacco corpo a corpo contro un
player armato, anche di coltello. Gli effetti dipendono dall'attaccante:

| Attaccante | Preparazione | Salute persa | Vigore perso dal player | Infortunio | Oggetto caduto | Paura | Costo per il nemico |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Guardia | 0,3 s | 12% | 25% | 15% | 25% | 0% | 40% |
| SWAT | 0,4 s | 22% | 40% | 30% | 35% | 0% | 20% |
| SWAT con scudo | 0,75 s | 32% | 60% | 55% | 50% | 60% | 26% |

Il costo viene pagato all'inizio, anche se il colpo viene evitato o interrotto.
La pausa tra gli attacchi è di 1,6 secondi. Chi è stordito, esausto o accecato
non può iniziare un attacco. Premendo F durante la preparazione, anche sul
fotogramma dell'impatto, la coltellata può disarmare il nemico: l'arma e le
munizioni cadono e possono essere raccolte. Il colpo frontale contro il bash
dello scudo viene bloccato e non lo disarma. Un tuffo con SPAZIO può schivare
il colpo. L'impatto spinge a terra per 0,75 secondi, seguiti da 0,65 secondi
per rialzarsi automaticamente; i muri arrestano la spinta. Durante caduta e
risalita il player non può muoversi o attaccare.

L'oggetto perso resta nel punto della caduta e si recupera con E dopo essersi rialzati.
Per gli oggetti in quantità cade una sola unità; le armi da fuoco cadono
insieme alla scorta di munizioni. Il resto dello zaino rimane intatto.
La velocità del player è ridotta del 6% in entrambe le modalità, camminando,
correndo e muovendosi accovacciato.

L'infortunio persiste anche passando alla missione successiva e si cura solo
completando i 4 secondi di un medikit, portatile o in postazione. Riposo e
bende non lo curano. Riduce il passo del 55% oltre alle penalità di salute
e fatica, impedisce corsa e tuffi e consente 60 secondi complessivi di
accovacciamento fino alla cura. Zoppicare produce rumore udibile entro
5 celle anche da accovacciati.

La paura da colpo di scudo dà una scarica di adrenalina per 6 secondi,
inverte le direzioni e ogni 1,5 secondi può provocare un'azione involontaria
(12%) oppure il tentativo di avvicinarsi a una guardia e arrendersi (2%).
Per calmarla occorre mangiare uno snack e mantenere salute e vigore pieni
per 3 secondi consecutivi. Gli snack si raccolgono sulla mappa e si mangiano
selezionandoli nello zaino e premendo E. Il pannello mostra paura,
infortunio e tempo di accovacciamento rimasto.

I nemici che hanno ferito il player, perdendolo di vista, si spostano verso
una postazione medica per sorvegliarla. L'uso, anche interrotto, lascia tracce:
quando un nemico le vede sul posto deduce il passaggio del player e cerca
nei dintorni della postazione, senza ricevere la sua posizione attuale.

La SWAT usa fucili a canna liscia e mitra, oltre a flashbang e fumogeni.
Il portatore di scudo alterna tra pistola e mitra a una mano, meno preciso.
Le armi hanno portata, dispersione, danno, cadenza e consumo di vigore propri
e si possono recuperare dai nemici. La SWAT senza scudo dispone anche di
una granata: la lancia solo a distanza, con visuale libera e senza compagni
vicini al bersaglio. Le granate recuperate si lanciano con E, hanno una
miccia di 1,5 secondi e un raggio di 3 celle segnalato sulla mappa; i muri
bloccano i danni e l'esplosione può colpire anche chi le ha lanciate.

## Campagna in tre missioni

Il pulsante per il quadro successivo conserva tutti gli altri oggetti dello
zaino, inclusi coltello, cure, granate e oggetti di missione; la pistola e
tutte le munizioni vengono smarrite. I caricatori vuoti e i bossoli usati
come diversivi rimangono nello zaino.

1. **Atrio e reception:** uffici amministrativi e sala congressi; recupera la
   scheda di sicurezza dall’ufficio e raggiungi l’uscita verso R&D.
2. **Uffici R&D:** recupera il dossier cartaceo e le prove digitali prima
   dell’uscita verso i laboratori. La scheda rimane nello zaino.
3. **Laboratori:** raccogli la protezione biologica (indossata automaticamente
   con E) e la provetta sterile. Il campione nella zona segnalata si preleva
   con E solo con entrambe; la provetta viene consumata. Il contatto con
   l’agente fittizio senza protezione azzera la salute in circa 0,25 secondi.

Ogni missione usa una planimetria diversa. Le pattuglie passano da 2 a 4 a 7
sentinelle, oltre alla guardia principale; le telecamere da 2 a 4 a 6 e lo
spotting diventa progressivamente più rapido. Completa gli obiettivi, raggiungi
l’esfiltrazione e premi **Invio** per la missione successiva. La terza chiude
la campagna. L’interfaccia mostra obiettivo e requisiti del prelievo.


Il player indossa una tuta tattica con un indicatore circolare e una freccia direzionale sulla mappa.
La tuta aumenta la velocita di movimento nelle modalita giocabili; le guardie
in CAUTELA si muovono il 40% piu velocemente rispetto alla stessa azione normale.
Il takedown da dietro raggiunge 2 celle e richiede sempre una visuale libera.
**C** alterna accovacciamento e posizione eretta con una singola pressione.
Dopo un **tuffo con Spazio**, il player resta a terra: premi nuovamente
**Spazio** oppure usa i comandi di movimento per rialzarti
(0,65 secondi prima di poter riprendere il movimento).


Le coltellate mostrano la lama e un arco di movimento, anche a vuoto. Gli impatti
contro nemici, muri e ostacoli solidi producono rumore udibile entro 6 celle;
il colpo a vuoto consuma vigore ma non genera rumore di impatto. A vigore zero
il player ha il fiatone: la respirazione e udibile entro 5 celle, anche da fermo,
e continua fino al recupero dall'esaurimento. I coni delle guardie che indagano
su un rumore diventano gialli; un avvistamento confermato li porta al rosso.


Uno sparo del player attiva l'allerta generale: tutte le guardie ricevono la
posizione dello sparo, alzano la guardia e la sicurezza avvia la risposta.
Arriva solo la SWAT: una pattuglia di rinforzo ancora in attesa viene annullata,
e la scoperta di corpi durante la risposta SWAT non ne chiama altre.
L'uso del fumogeno porta invece a CAUTELA con sorveglianza rafforzata e coni
gialli, senza chiamare la SWAT da solo. Il fumogeno non abbassa un'allerta
 gia attiva; armi scariche o fumogeni esauriti non cambiano lo stato.

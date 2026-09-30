# Rapport d'évaluation

**Manuscrit :** « From Affect to Cognition: A Neuro-Fuzzy Emotion Recognition Framework based on the OCC Appraisal Model » (`occ_fuzzy_llm_conference.tex`)
**Version révisée proposée :** `occ_fuzzy_recognition_conference.tex` / `.pdf`

## 1. Résumé du manuscrit

Le papier propose de reconnaître les émotions à partir de leur cause cognitive (théorie OCC) plutôt qu'à partir de leur profil affectif (VAD). Un encodeur RoBERTa prédit la désirabilité de l'événement (D), le caractère louable de l'action de l'agent (P) et l'agent responsable. Un système flou de Mamdani applique ensuite des règles OCC pour produire l'émotion, son intensité et une explication. Le travail se présente comme l'alternative OCC au système CAT2-NFI (VAD + logique floue de type 2) des mêmes auteurs.

## 2. Avis général

L'idée est pertinente et bien positionnée. Distinguer les émotions par leur cause (qui est responsable, une norme a-t-elle été violée) plutôt que par leur valence est un argument fort. La continuité avec CAT2-NFI donne aussi une trajectoire de recherche cohérente. En l'état, le manuscrit ne peut cependant pas être accepté : il n'y a aucune expérience, le résumé affirme des résultats non mesurés, et la formalisation n'est pas assez précise pour être reproduite.

**Recommandation : rejet en l'état, avec encouragement à resoumettre** après exécution des expériences. La version révisée corrige la conception et le protocole. Elle reste à compléter avec des mesures réelles, mais une fois les résultats obtenus elle me semble au niveau d'une conférence indexée.

## 3. Points majeurs

1. **Résultats affirmés sans expériences.** Le résumé affirme que « les évaluations expérimentales démontrent » un gain d'OCC sur VAD. Or le papier ne contient ni tableau ni chiffre, et la section expérimentale se limite à « des datasets tels qu'EmoWOZ ». C'est rédhibitoire devant tout comité. Dans la révision, chaque valeur non mesurée est un emplacement rouge (`\tbd`, `\authnote`), et le résumé ne revendique plus aucun résultat.

2. **« 22 émotions OCC » : c'est impossible avec D, P et l'agent.** Les émotions de perspective (espoir, peur, soulagement, déception, satisfaction, peur confirmée) exigent une variable de probabilité et de confirmation. Les émotions liées au sort d'autrui (se réjouir pour, pitié, jubilation, ressentiment) exigent la désirabilité pour l'autre et l'appréciation de l'autre. Les émotions d'attraction (amour, haine) exigent l'attrait d'un objet. Avec (D, P, agent), on atteint au plus **10 types** : Joy, Distress, Pride, Shame, Admiration, Reproach, Gratification, Remorse, Gratitude, Anger. La révision le dit explicitement et donne la base complète de 52 règles (Tableau II).

3. **La supervision du module 1 n'est pas définie.** Le papier dit « optimisé par MSE », sans dire d'où viennent les cibles D et P. EmoWOZ n'annote ni D ni P. Si ces cibles étaient dérivées des étiquettes d'émotion, l'évaluation serait circulaire : le système retrouverait ses propres cibles. La révision entraîne le module 1 sur les 21 notes d'évaluation cognitive d'enVENT (Troiano et al., 2023), avec des formules explicites (Éq. 3), et n'utilise jamais les étiquettes d'émotion.
   - **Difficulté que j'ai vérifiée dans le questionnaire d'enVENT :** les items de normes sont unipolaires (« l'événement a heurté mes principes », « les actions ont violé des lois ou des normes »). Ils mesurent le blâme, pas l'éloge.
   - La révision construit donc P de façon explicite : c'est le mérite d'un résultat désirable obtenu par une personne responsable sans violer de norme.
   - Cette construction doit être validée sur environ 200 jugements humains bipolaires (note rouge dans le papier).

4. **Système flou sous-spécifié et en partie incohérent.**
   - Les fonctions d'appartenance ne sont pas données (« trapézoïdales et triangulaires qui se chevauchent »).
   - Une seule règle est montrée.
   - L'agent entre par une indicatrice nette, `min(...) × 𝟙(A = Self)`, ce qui contredit la motivation floue : l'incertitude sur le responsable est perdue.
   - Aucune règle de décision n'est donnée. Le centroïde fournit une intensité, pas une classe ; que fait-on quand plusieurs émotions se déclenchent ?
   - La tête « agent » est un classifieur, mais le papier dit l'optimiser par MSE.

   La révision apporte :
   - les paramètres des partitions (Tableau I), des partitions de Ruspini vérifiées numériquement ;
   - des degrés d'agent souples, $\mu_a = \pi_a$ ;
   - la séparation entre l'activation $\alpha_e$ (le poids de la preuve) et l'intensité $I_e$ ;
   - une règle de décision avec un seuil θ et une correspondance fixe avec les étiquettes des corpus (Tableau III).

5. **Protocole expérimental insuffisant.** Aucun jeu de données précis, aucune correspondance entre étiquettes et types OCC, seulement deux baselines, ni ablation ni test statistique.
   - Surtout, la comparaison avec VAD-Fuzzy n'est pas équitable si CAT2-NFI est évalué comme dans son papier. Sur EmoBank, les classes y sont dérivées des scores VAD de référence, ce qui avantage mécaniquement le VAD.
   - La révision évalue sur **enVENT** (dans le domaine) et sur **EmoWOZ** (hors domaine, sans entraînement sur ses étiquettes). Les étiquettes d'EmoWOZ sont justement définies à partir d'OCC (valence × déclencheur × conduite).
   - Elle croise la représentation (VAD ou OCC) avec la mise en correspondance (règles floues ou régression logistique apprise), et ajoute un classifieur boîte noire comme référence.
   - Elle introduit une mesure ciblée : le taux de confusion entre émotions de même valence (SVC).
   - Elle ajoute des ablations (inférence nette, agent dur), un bootstrap apparié, le test de McNemar avec correction de Holm, et une corrélation de l'intensité avec l'intensité déclarée.

6. **L'explicabilité est revendiquée mais jamais évaluée.** La révision distingue deux choses. La fidélité est garantie par construction : l'explication cite la règle qui a produit la décision. La plausibilité, elle, doit être mesurée : deux annotateurs vérifient les appréciations énoncées sur 100 explications, avec le κ de Cohen.

7. **L'exemple de motivation n'est pas étayé.** Le papier affirme, sans référence, que Sadness et Remorse sont proches en VAD. La révision le remplace par une paire minimale, calculée avec le système : le même résultat (manquer le mariage de sa sœur) causé par une tempête, un chauffeur fautif ou sa propre erreur. On obtient respectivement Distress, Anger et Remorse, c'est-à-dire sadness, anger et guilt/shame.

## 4. Points mineurs

- **Références erronées :**
  - Hofmann et al. (2020) a paru à COLING, pas dans IEEE TAC, et son titre est « Appraisal theories for emotion classification in text ».
  - enVENT a paru dans *Computational Linguistics* 49(1), 2023, et non à ACL, sous un autre titre.
  - FLAME a paru dans la revue *Autonomous Agents and Multi-Agent Systems*, pas à la conférence AAMAS.
  - Picard date de 1997.
  - « LATIS (2024) » n'est pas un lieu de publication.
  - Les pages manquent partout.
- **Terminologie :** « compound OCC emotions » désigne, dans OCC, uniquement Gratification, Remorse, Gratitude et Anger. Par ailleurs, les circonstances ne sont pas un « agent » au sens OCC.
- **Figure 1 :** la flèche bidirectionnelle entre la fuzzification et les règles n'a pas de sens. L'étiquette « Defuzzification » est placée sur la mauvaise flèche.
- `\bibliographystyle{IEEEtran}` est inutile avec un environnement `thebibliography` écrit à la main.

## 5. Point de vigilance pour le futur papier journal (VAD + flou + LLM contre OCC + flou + LLM)

Pour que la comparaison entre VAD et OCC soit recevable, les deux systèmes doivent être évalués sur des étiquettes catégorielles humaines, indépendantes des deux représentations. Ce n'est pas le cas d'EmoBank tel qu'il est utilisé dans CAT2-NFI, puisque les classes y sont dérivées du VAD de référence. Le protocole de la version conférence (enVENT et EmoWOZ, même machinerie floue de type 1, mêmes sondes linéaires) est conçu pour être repris tel quel dans le journal.

## 6. Liste des éléments à compléter par les auteurs (notes rouges)

1. Exécuter les expériences et remplir les Tableaux IV et V, ainsi que la section Résultats.
2. Valider la construction de P sur environ 200 jugements humains bipolaires.
3. Donner les effectifs par classe après filtrage des étiquettes non couvertes d'enVENT.
4. Remplacer les hyperparamètres par ceux réellement utilisés.
5. Vérifier que les prototypes NRC-VAD de VAD-T1FIS sont cohérents avec la construction de CAT2-NFI.
6. Décrire les annotateurs de l'étude sur les explications.
7. Remplacer les valeurs illustratives de l'exemple par les sorties réelles, et ajouter les prédictions de B1 et B2.
8. Compléter la référence de CAT2-NFI (lieu, année ou « under review »).
9. Ajouter une phrase de résultats dans le résumé et dans la conclusion.

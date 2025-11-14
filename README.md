<!-- <div align="center"> -->

# DocsBlip
## Was ist das hier?
Es geht hier um den Versuch, LiLT-Embeddings [1] ins Llama2 [2] einzubetten, um dokumenbasierte Generation zu ermöglichen.

## Was ist die Idee dahinter?
Wir bauen zwischen dem vortrainierten LiLT und dem Llama2 einen Adapter (BLIP2) ein [3]. Dies tut:
- Die LiLT-Embeddings in eine kurzere Sequenz, heißt Query-Tokens, zusammenfassen, wenn sie zu lange sind. Im Defaultmodus hat die Sequence eine Länge von 32.
- Die LiLT-Embeddings auf die Text-Ebene zu pushen, und somit soll das LLM diese auch verstehen.

Der Adapter besteht hauptsächlich aus BERT. Dazu kommt aber noch die learnable Query-Tokens. Er nimmt die LiLT Embeddings und die Query-Tokens, verarbeitet ihre Relation, und gibt die "zusammengefasste" LiLT-Embeddings.

## Wie wird das gelernt?
Wir behalten das Wissen vom LiLT und Llama2, indem wir diese beide während des Trainings frieren. Der Adapter wird für VQA-Generierung in zwei Schritten trainiert.
1. Im ersten Schritt lernt er die multimodale Anpassung mit drei Losses
    - Contrastive Loss zwischen Dokument und Antwort
    - Matching-Loss zwischen Dokument und Antwort
    - Caption-Generation Loss
2. Im zweiten Schritt lernt er hauptsächlich die Generation mit dem Next-Token Prediction Ansatz (Cross-Entropy über die Dictionary).

## Datensatz
DocVQA ist ein Datensatz für Visuelle Fragebeantwortung (VQA) auf Dokumentenbildern.

<!-- ## Literatur

| Name     | Paper                                        | Category |
|----------|----------------------------------------------|----------|
| DocVQA   | DocVQA: A Dataset for VQA on Document Images | dataset  |
| HyperDoc |                                              | model    |
| Sammlungen | https://github.com/ym-xu/awesome-document-understanding-datasets  | repo    |
| DIVE-Doc | https://github.com/JayRay5/DIVE-Doc/blob/main/data/docvqa/utils.py  | repo    |
| inspect-ai | https://github.com/JayRay5/DIVE-Doc  | repo    | -->

## TODOs:
- [x] Demo
- [x] Datenvorverarbeitung
- [x] Trainings- und Eval.-Pipeline
- [x] OPT2 oder Vicuna Chat statt BART verwenden
- [x] FT mit kontrastivem Loss und LM-Loss
- [x] Attention-Schichten statt Proj. Schicht

## Literatur
[1] Wang, Jiapeng, Lianwen Jin and Kai Ding. “LiLT: A Simple yet Effective Language-Independent Layout Transformer for Structured Document Understanding.” ArXiv abs/2202.13669 (2022): n. pag.
[2] Touvron, Hugo, Louis Martin, Kevin R. Stone, Peter Albert, Amjad Almahairi, Yasmine Babaei, Niko-lay Bashlykov, Soumya Batra, Prajjwal Bhargava, Shruti Bhosale, Daniel M. Bikel, Lukas Blecher, Cris-tian Cantón Ferrer, Moya Chen, Guillem Cucurull, David Esiobu, Jude Fernandes, Jeremy Fu, Wenyin Fu, Brian Fuller, Cynthia Gao, Vedanuj Goswami, Naman Goyal, Anthony S. Hartshorn, Saghar Hosseini, Rui Hou, Hakan Inan, Marcin Kardas, Viktor Kerkez, Madian Khabsa, Isabel M. Kloumann, Artem Korenev, Punit Singh Koura, Marie-Anne Lachaux, Thibaut Lavril, Jenya Lee, Diana Liskovich, Yinghai Lu, Yuning Mao, Xavier Martinet, Todor Mihaylov, Pushkar Mishra, Igor Molybog, Yixin Nie, Andrew Poulton, Jeremy Reizenstein, Rashi Rungta, Kalyan Saladi, Alan Schelten, Ruan Silva, Eric Michael Smith, R. Subramanian, Xia Tan, Binh Tang, Ross Taylor, Adina Williams, Jian Xiang Kuan, Puxin Xu, Zhengxu Yan, Iliyan Zarov, Yuchen Zhang, Angela Fan, Melissa Hall Melanie Kambadur, Sharan Narang, Aur'elien Rodriguez, Robert Stojnic, Sergey Edunov and Thomas Scialom. “Llama 2: Open Foundation and Fine-Tuned Chat Models.” ArXiv abs/2307.09288 (2023): n. pag.
[3] Junnan Li, Dongxu Li, Silvio Savarese, and Steven Hoi. 2023. BLIP-2: bootstrapping language-image pre-training with frozen image encoders and large language models. In Proceedings of the 40th International Conference on Machine Learning (ICML'23), Vol. 202. JMLR.org, Article 814, 19730–19742.
[4] Mathew, Minesh, Dimosthenis Karatzas, R. Manmatha and C. V. Jawahar. “DocVQA: A Dataset for VQA on Document Images.” 2021 IEEE Winter Conference on Applications of Computer Vision (WACV) (2020): 2199-2208.
# auto-stop-tar
Code and data for paper "When to Stop Reviewing in Technology-Assisted Reviews"


## Table of contents

- Data
    * CLEF TAR 2017
    * CLEF TAR 2018
    * CLEF TAR 2019
- Code
    - Baselines
        * AutoTar knee
        * SCAL
    - AutoStop
        * Ranking module
        * Sampling module
        * Estimiation module
       
- [Citation](#citation)


## Data  

- CLEF TAR 2017
It contains 42 topics selected from the complete topics by the organisor (see [CLEF 2017 Technologically Assisted Reviews in Empirical Medicine Overview](https://pure.strath.ac.uk/ws/portalfiles/portal/71285524/Kanoulas_etal_CEUR_2017_CLEF_2017_technologically_assisted_reviews_in_empirical_medicine_overview.pdf))
    * `topics` contains topics file in the format
        ```
          Topic: CD007394 

          Title: Galactomannan detection for invasive aspergillosis in immunocompromised patients 

          Query: 
          “Aspergillus"[MeSH]
          "Aspergillosis"[MeSH]
          ... 

          Pids: 
              24514094 
              24505528 
        ```
    * `abs_qrels` contains qrel files in the TREC format.
        ```
        qid dummy pid rel_label
        ```
    * `plain_texts` contains abstract of each pid in the format 
        ```
        pid::::text
        ```
- CLEF TAR 2018
It contains 30 new topics released in CLEF TAR 2018 (see [CLEF 2018 Technologically Assisted Reviews in Empirical Medicine Overview](http://ceur-ws.org/Vol-2125/invited_paper_6.pdf))
- CLEF TAR 2019
It contains 31 new topics released in CLEF TAR 2019 (see [CLEF 2019 Technologically Assisted Reviews in Empirical Medicine Overview](http://ceur-ws.org/Vol-2380/paper_250.pdf))
 
## Code

- AutoTar knee

  AutoTar is proposed in paper *Autonomy and Reliability of Continuous Active Learning for Technology-Assisted Review*.
  The key component of *AutoTar knee* is a knee detection method, which is proposed in paper *"Kneedle" in a Haystack: Detecting Knee Points in System Behavior*.

  We implemented the method ourselves.

- SCAL 

  SCAL is p


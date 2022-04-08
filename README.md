# HitAnalyzer

Recipe to compile:
```
cmsrel CMSSW_10_6_26
cd CMSSW_10_6_26/src/
cmsenv
git cms-init
git remote add mmusich git@github.com:mmusich/cmssw.git
git checkout from-CMSSW_10_6_26_muoncaldev3
git cms-addpkg SimG4Core/MagneticField TrackPropagation/Geant4e
mkdir Analysis
cd Analysis/
git clone -b from-CMSSW_10_6_26_muoncaldev3 git@github.com:erc-asymow/HitAnalyzer.git
scramv1 b -j 20
```

plldesigner
===========


A pythonic tool for PLL design and exploration (focused in PLLs implemented in
  hardware). More information can be found in
  [Phase Unlocked](http://jfosorio.github.io/). The final propose of this
  project is to have a complete design tool for PLL's (Phase-locked loops).
  It proposes a class that allows to:
* Analyze the loop stability
* Specify the noise sources and calculate the overall noise
* Specify non-linearities in the loop and simulate the transient response
  (VCO, PFD-CP)

Status
======

Currently a first version is available The first two bullets are already cover.
Regarding the third bullet I have been thinking in different alternatives:
generate veriloga code, generate modelica code, generate pure C++ or Cython
code.


Development plan
================


- [ ] **Phase noise class**. Create the class structure to specify the noise, the
  pnoise object
  - [x] Noise, specially for oscillator can be specified as
$f_n/fm^{-n}+f_{n=1}/fm^{-n+1}+$
  - [x] over rule the addition operation
  - [x] Interpolate
  - [x] Generate a model of the data (extrapolate)
  - [x] Create plots with asymptotic values (Postpone)
  - [ ] Create the object with phase vs t point
  - [ ] Plot several noise sources and the resultant (Postpone)
- [ ] **LTI model of the PLL**
  - [x] Second order approximation
  - [ ] Phase margin plot
  - [ ] Timing vs phase  margin and error
  - [ ] phase noise optimization
- [ ] **Design routines**
  - [x] Given fc and R (or the DN)  calculate the filter
  - [ ]Specify

- [ ] **Time domain simulator**. Time domain simulation that can be easily
  configuration using class. This is a set of routines and/or classes to create
  a simulator in the time domain.
  - [ ] First version with a fix step (CP is a big impulse as proposed by
    Perrot)

- [ ] **Documentation**
  - [ ] **Notebooks**
    - [x] Sigma-delta modulator
    - [ ] Analog PLL class
    - [ ] phase noise
    - [ ]
  - [ ] Webpage

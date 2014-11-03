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
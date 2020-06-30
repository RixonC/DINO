# DINO
Distributed Newton-Type Optimization Method.

This method is from https://arxiv.org/abs/2006.03694.
For simplicity and compatibility, we use all-reduce operations instead of broadcast and then reduce. 
For the other methods that we compared to, code can be found at https://github.com/RixonC/DINGO.

This code requires `Python 3`, `PyTorch`, `Torchvision`, `Scipy` and `Matplotlib`.

### Authors
1. Rixon Crane. School of Mathematics and Physics, University of Queensland, Australia. Email: r.crane(AT)uq.edu.au
2. Fred Roosta. School of Mathematics and Physics, University of Queensland, Australia, and International Computer Science Institute, Berkeley, USA. Email: fred.roosta(AT)uq.edu.au

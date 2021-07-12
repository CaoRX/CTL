# "Contracting Arbitrary Tensor Networks"
# PhysRevLett. 125. 060503.

# the framework of tensor contraction:
# 1. decompose each tensor into an MPS, each bond corresponding to a tensor
# 2. contract MPSes by first moving the bond to one end, and then contract
# 3. In the new tensor network, double edges should be merged by first move them to neighbors, then merge
# 4. iterate until one scalar remains

# the functionality we want:
# 1. decompose a tensor to MPS, in canonical form

# "A Practical Introduction to Tensor Networks: MPS and PEPS"
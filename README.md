# ISBI challenges
Playful attempts at ISBI challenge problems past and present

I will be reproducing past winners (and runners up!) and having a crack at any future challenges.

I wills start with reproducing the [classic Unet model](https://arxiv.org/pdf/1505.04597v1.pdf) that was applied to [ISBI 2012](http://brainiac2.mit.edu/isbi_challenge/home)

The second implementation is not an ISB challenge but deep image sequence interpolation inspired by https://arxiv.org/pdf/2002.11616.pdf.

This implements frame feature interpolation based on deformable sampling.  Images in a sequence are transformed into a learnt feature space where they are registered and fused. This fused image is then transformed back into intensity space to generate a temporal (actually z-stack for my data) interpolation.
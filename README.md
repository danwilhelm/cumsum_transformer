# How a Transformer Computes the Cumulative Sum Sign
**Author:** Dan Wilhelm, dan@danwilhelm.com

We investigate how a one-layer attention+feed-forward transformer computes the cumulative sum. I've written the investigation conversationally, providing numerous examples and insights.

Of particular interest, we:
1. design a 38-weight attention-only circuit with smaller loss than the provided model;
2. manually remove the MLP and rewire a trained circuit, retaining 100% accuracy;
3. prove that an equally-attended attention block is equivalent to a single linear projection (of the prior-input mean!); and
4. provide an independent transformer implementation to make it easier to modify the internals.


This is my proposed solution to a monthly puzzle authored by Callum McDougall! You may find more information about the challenge and monthly problem series here:
- [Problem GitHub page](https://github.com/callummcdougall/ARENA_2.0/tree/main/chapter1_transformers/exercises/monthly_algorithmic_problems/november23_cumsum)
- [ARENA page](https://arena-ch1-transformers.streamlit.app/Monthly_Algorithmic_Problems)
- [eindex dependency (needed to run the provided model)](https://github.com/callummcdougall/eindex/tree/main)


## Table of Contents
0. Introduction
1. All 24 embedding channels directly encode token sign and magnitude
2. Attention softmax equally attends to each token
3. Equally-divided attention computes the expanding mean
4. Feed-forward network "cleans up" the signal
5. What about the zero unembed pattern?
6. Surgically removing the MLP, retaining 100% accuracy
7. Designing a 38-weight attention-only cumsum circuit
8. Appendix A. Rewriting two linear transforms as one
9. Appendix B. Designing a 38-weight circuit with skip connections

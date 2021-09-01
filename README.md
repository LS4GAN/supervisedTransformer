# supervisedTransformer
Supervised Toyzero image translation with Transformer

## current structure:
1. `transformer.py`: Containing the main body of the full transformer model without embedding and generator (embedding and generator ared supposed to be plugins).
2. `supervised_transformer_toyzero.ipynb`: currently containing data loading and embedding and generator; needs to be separated into granular

## To-do:
1. separate the really long `supervised_transformer_toyzero.ipynb` into parts:
    - data loading aparatus;
    - embedding and generator;

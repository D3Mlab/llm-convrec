"""
   Taken from  https://github.com/D3Mlab/rir/blob/main/prefernce_matching/statics.py
   
   Dictonaries to look up the name of BERT_MODELS and TOEKNIZER_MODELS.
"""

BERT_MODELS = {
    'TF-IDF':'TF-IDF',
    # 'Finetuned_subsampled':'./fine_tune_msmarco_ii/out/checkpoint-17280',
    #  'VFT29000':'./fine_tune_iii/VANILLA_10_512_200/checkpoint-29000',
    'CONDENSER': 'Luyu/condenser',
    'COCONDENSERMARCO': 'Luyu/co-condenser-marco',
    'TASB': "sebastian-hofstaetter/distilbert-dot-tas_b-b256-msmarco",
    'VANILLA': 'bert-base-uncased',
    'FTVANILLA': './data/local-tf-checkpoint',
    # 'MSContrastive': './contrastive/MSMARCO_16_512_6000/checkpoint-4056',
    # 'MFT20000':'./fine_tune_iii/MSMARCO_16_512_200/checkpoint-20000'
    'contriever': "facebook/contriever",
    'contrievermsmarco': "facebook/contriever-msmarco",

}

TOEKNIZER_MODELS = {
    'TF-IDF':'TF-IDF',
    'COCONDENSERMARCO': 'Luyu/co-condenser-marco',
    'CONDENSER': 'Luyu/condenser',
    'TASB': "sebastian-hofstaetter/distilbert-dot-tas_b-b256-msmarco",
    'VANILLA': 'bert-base-uncased',
    'contriever': "facebook/contriever",
    'contrievermsmarco': "facebook/contriever-msmarco",
    'FTVANILLA': 'bert-base-uncased',
    # 'MSContrastive': "sebastian-hofstaetter/distilbert-dot-tas_b-b256-msmarco",
    # 'Finetuned_subsampled':"sebastian-hofstaetter/distilbert-dot-tas_b-b256-msmarco",
    # 'VFT29000':'bert-base-uncased',
    # 'MFT20000':"sebastian-hofstaetter/distilbert-dot-tas_b-b256-msmarco",
}

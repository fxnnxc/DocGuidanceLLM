import re 
from tqdm import tqdm 
from datasets import load_dataset
from datasets import Dataset, DatasetDict

def get_wikitext(name, cache_dir,  shorten=None, **kwargs):
    assert name in ['wikitext-103-v1', 'wikitext-2-v1'] 
    raw_datasets = load_dataset('EleutherAI/wikitext_document_level',   
                            name=name,
                            cache_dir=cache_dir if cache_dir != "none" else None,
                    )
    dataset = raw_datasets['train'] 
    dataset = dataset.rename_columns({'page': 'text'})    
    pattern = re.compile("=.*=")
    def get_source(example):
        page = example['text']
        full = re.findall(pattern, page)[0]
        if shorten:
            page = example['text'][:shorten]
        source = full.strip().split("=")[1]
        return {'text':page, 'id': source,}
    dataset = dataset.map(get_source)        
    
    return dataset

def get_selected_wiki_page_data_dict(wiki_dataset, selected_pages, segment_length, max_segements=10):
    page_text_dict = {}
    full_pages = wiki_dataset['id']
    for page in tqdm(selected_pages):
        idx = full_pages.index(page)
        text = wiki_dataset[idx]['text']
        text = text.replace("\n", " ")
        text = text.split()
        if len(text) < segment_length:
            training_texts = [" ".join(text)]    
        else:
            training_texts = [" ".join(text[j:j+segment_length]) 
                              for j in range(0, len(text)-segment_length, segment_length//2)][:max_segements]
        page_text_dict[page] = Dataset.from_dict({'text':training_texts})
        
    return DatasetDict(page_text_dict)
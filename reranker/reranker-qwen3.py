# Requires transformers>=4.51.0
import torch
from transformers import AutoModel, AutoTokenizer, AutoModelForCausalLM
import time

MAX_LENGTH = 8192

print("Starting reranker...")

def format_instruction(instruction, query, doc):
    if instruction is None:
        instruction = 'Given a web search query, retrieve relevant passages that answer the query'
    output = "<Instruct>: {instruction}\n<Query>: {query}\n<Document>: {doc}".format(instruction=instruction,query=query, doc=doc)
    return output

def build_inputs(queries, documents, instruction=None):
    """
    Build full text prompts and tokenize them in one call
    using padding=True and truncation.
    """
    assert len(queries) == len(documents), "queries and documents must be same length"

    texts = []  
    for q, d in zip(queries, documents):  
        core = format_instruction(instruction, q, d)  
        full_text = prefix + core + suffix  
        texts.append(full_text)  

    # Single tokenizer call: tokenization + padding  
    encoded = tokenizer(  
        texts,  
        padding=True,  
        truncation="longest_first",  
        max_length=MAX_LENGTH,  
        return_tensors="pt",  
    )  

    # Move to model device  
    encoded = {k: v.to(model.device) for k, v in encoded.items()}  
    return encoded  

@torch.no_grad()
def compute_logits(inputs, **kwargs):
    batch_scores = model(**inputs).logits[:, -1, :]
    true_vector = batch_scores[:, token_true_id]
    false_vector = batch_scores[:, token_false_id]
    batch_scores = torch.stack([false_vector, true_vector], dim=1)
    batch_scores = torch.nn.functional.log_softmax(batch_scores, dim=1)
    scores = batch_scores[:, 1].exp().tolist()
    return scores

print("Downloading model")
start = time.perf_counter()

tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-Reranker-0.6B", padding_side='left')
model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-Reranker-0.6B").eval()
# We recommend enabling flash_attention_2 for better acceleration and memory saving.
# model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-Reranker-0.6B", torch_dtype=torch.float16, attn_implementation="flash_attention_2").cuda().eval()

end = time.perf_counter()  
elapsed = end - start  
print(f"Downloaded in {elapsed:.2f} seconds")

token_false_id = tokenizer.convert_tokens_to_ids("no")
token_true_id = tokenizer.convert_tokens_to_ids("yes")

prefix = "<|im_start|>system\nJudge whether the Document meets the requirements based on the Query and the Instruct provided. Note that the answer can only be \"yes\" or \"no\".<|im_end|>\n<|im_start|>user\n"
suffix = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
prefix_tokens = tokenizer.encode(prefix, add_special_tokens=False)
suffix_tokens = tokenizer.encode(suffix, add_special_tokens=False)
        
task = 'Given a web search query, retrieve relevant passages that answer the query'

queries = ["What is the capital of China?",
    "Explain gravity",
    "what is a constituent entity?",
    "what is a constituent entity?",
    "What is the capital of China?",
    "Explain gravity",
    "what is a constituent entity?",
    "what is a constituent entity?",
    "What is the capital of China?",
    "Explain gravity",
    "what is a constituent entity?",
    "what is a constituent entity?",
    "What is the capital of China?",
    "Explain gravity",
    "what is a constituent entity?",
    "what is a constituent entity?",
    "What is the capital of China?",
    "Explain gravity",
    "what is a constituent entity?",
    "what is a constituent entity?"
]

documents = [
    "The capital of China is Beijing.",
    "Gravity is a force that attracts two bodies towards each other. It gives weight to physical objects and is responsible for the movement of planets around the sun.",
    "The term Constituent Entity defines those Group Entities that are subject to the GloBE Rules. For example, a Group Entity must be a Constituent Entity before it can be treated as an LTCE and subject to charge under the IIR or UTPR, under Chapters 2 to 5 of the rules.",
    "a place of business or activity",
    "The capital of China is Beijing.",
    "Gravity is a force that attracts two bodies towards each other. It gives weight to physical objects and is responsible for the movement of planets around the sun.",
    "The term Constituent Entity defines those Group Entities that are subject to the GloBE Rules. For example, a Group Entity must be a Constituent Entity before it can be treated as an LTCE and subject to charge under the IIR or UTPR, under Chapters 2 to 5 of the rules.",
    "a place of business or activity",
    "The capital of China is Beijing.",
    "Gravity is a force that attracts two bodies towards each other. It gives weight to physical objects and is responsible for the movement of planets around the sun.",
    "The term Constituent Entity defines those Group Entities that are subject to the GloBE Rules. For example, a Group Entity must be a Constituent Entity before it can be treated as an LTCE and subject to charge under the IIR or UTPR, under Chapters 2 to 5 of the rules.",
    "a place of business or activity",
    "The capital of China is Beijing.",
    "Gravity is a force that attracts two bodies towards each other. It gives weight to physical objects and is responsible for the movement of planets around the sun.",
    "The term Constituent Entity defines those Group Entities that are subject to the GloBE Rules. For example, a Group Entity must be a Constituent Entity before it can be treated as an LTCE and subject to charge under the IIR or UTPR, under Chapters 2 to 5 of the rules.",
    "a place of business or activity",
    "The capital of China is Beijing.",
    "Gravity is a force that attracts two bodies towards each other. It gives weight to physical objects and is responsible for the movement of planets around the sun.",
    "The term Constituent Entity defines those Group Entities that are subject to the GloBE Rules. For example, a Group Entity must be a Constituent Entity before it can be treated as an LTCE and subject to charge under the IIR or UTPR, under Chapters 2 to 5 of the rules.",
    "a place of business or activity"
]

print("Download complete. Running reranker...")
start = time.perf_counter()

pairs = [format_instruction(task, query, doc) for query, doc in zip(queries, documents)]

# Tokenize the input texts
inputs = build_inputs(queries, documents, instruction=task) 
scores = compute_logits(inputs)

end = time.perf_counter()  
elapsed = end - start  
print(f"Reranker ran in {elapsed:.2f} seconds")

print("scores: ", scores)
from transformers import BertTokenizer, BertModel
import torch
from sklearn.metrics.pairwise import cosine_similarity

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')

# A method that gets the BERT embedding for a given text. This will be used for both the pre-set embeddings and the user input.
def get_bert_embedding(text):
    inputs = tokenizer(text, return_tensors='pt')
    outputs = model(**inputs)
    return outputs.last_hidden_state.mean(dim=1).detach().numpy()

# A method that gets the closest pre-set feature to a given user embedding.
def get_closest_feature(embeddings, user_embedding):
    user_embedding_2d = user_embedding.reshape(1, -1)

    # Debugging: Print out similarity scores for each feature
    for feature, embedding in embeddings.items():
        score = cosine_similarity(embedding.reshape(1, -1), user_embedding_2d)[0][0]
        print(f"Similarity with {feature}: {score}")

    return max(embeddings, key=lambda feature:
    cosine_similarity(embeddings[feature].reshape(1, -1), user_embedding_2d)[0][0])

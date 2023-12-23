import TextWeaver
import pandas as pd

df = {'text': ["This is a test sentence. This is another test sentence.", "This is a second test sentence. This is another second test sentence."]}
df = pd.DataFrame(df)
garment = TextWeaver.Garment()

garment.corpus = garment.load_corpus_pandas(df)

print(garment.get_sentences())
# Tests

from TextWeaver import Fabric

text = "This is a test sentence. This is another test sentence."

fabric = Fabric(text)

print("####################################\nNAMED ENTITIES\n####################################\n", fabric.get_named_entities())
print("####################################\nPARTS OF SPEECH\n####################################\n", fabric.get_pos())
print("####################################\nSENTENCES\n####################################\n", fabric.get_sentences())
print("####################################\nLEMMAS\n####################################\n", fabric.get_lemmas())
print("####################################\nSTEMS\n####################################\n", fabric.get_stems())
print("####################################\nSTOP WORDS\n####################################\n", fabric.remove_stopwords())
print("####################################\nSENTIMENT\n####################################\n", fabric.get_sentiment())

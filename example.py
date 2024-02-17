# This is an example of how to use the TextWeaver module.
from TextWeaver import Fabric

# Create a Fabric object
fabric0 = Fabric("This is a test sentence. This is another test sentence. This is me testing a sentence.")

# Create another Fabric object
fabric1 = Fabric("The quick brown fox jumps over the lazy dog.") 

fabric0.assign_codes("Testing", "test sentence")
fabric0.assign_codes("Testing", "testing")

fabric1.assign_codes("Animals", "fox")
fabric1.assign_codes("Animals", "dog")

print(fabric0.codes)

print(fabric0.find_themes("Testing"))

print(fabric1.find_themes("Animals"))

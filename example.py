from TextWeaver import Fabric

fabric = Fabric("This is a test sentence. This is another test sentence. This is me testing a sentence.")

fabric.assign_codes("Testing", "test sentence")

fabric.assign_codes("Testing", "testing")

print(fabric.codes)

print(fabric.find_themes("Testing"))

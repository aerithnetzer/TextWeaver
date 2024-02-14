from TextWeaver import Fabric

fabric = Fabric("This is a test sentence. This is another test sentence. This is me testing a sentence.")

fabric.assign_codes("Child Theme Testing", "test sentence")

fabric.assign_codes("Child Theme Testing", "testing")

fabric.assign_codes("Child child theme test", 'testing')

print(fabric.codes)

print(fabric.make_child_theme("Parent Theme Types of Testing", "Child Theme Testing"))
print(fabric.make_child_theme('Child Theme Testing', "Child child theme test"))

fabric.make_theme_graph()

print(fabric.codes)
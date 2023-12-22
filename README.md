# TextWeaver 👕📖

TextWeaver is a text-focused qualitative analysis tool. This is in _very_ early development, so I wouldn't try to do anything serious with it yet. But, you can use it to perform basic text analysis over a large corpus of text.

## Structure and Philosophy

The TextWeaver project contains what is essentially two classes; fabric and garment. You can think of garment as a collection of fabrics. When running a method on the garment object, you are actually just running the same method on many fabrics. As a result of this philosophy, you can think of the garment class as a corpus of text, while a fabric class is an instance of text within the corpus.

### Why Fabrics and Garments?

Partly to avoid confusion when coding with terms like "text" and "corpus," especially with more involved procedures like codes.

## What's Next?

I would like to get the TextWeaver package to a stable state where it can do everything that the major for-profit text analysis tools can do. Then, I can extend to other sources, like audio and images (maybe with a package like AudioWeaver or PictureWeaver). Even farther down the line, I would like to orchestrate all of these packages into a unified GUI so that people don't have to interact with this package programmatically.

## Getting Started

As an example for how the code is meant to run, you can use `example.py`; it is reproduced below.

```python
# Import the TextWeaver library
from TextWeaver import Fabric

# Create a Fabric object, consisting of a string of text.
fabric = Fabric("This is a test sentence. This is another test sentence. This is me testing a sentence.")

# Assign codes to the fabric object
fabric.assign_codes("Testing", "test sentence")

# Assign another code
fabric.assign_codes("Testing", "testing")

# Print the codes
print(fabric.codes)

# Print the text as a string with the substrings that match values found in codes highlighted in green
print(fabric.find_themes("Testing"))
```

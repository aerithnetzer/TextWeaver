# FILEPATH: /Users/aerith/warlock/TextWeaver/tests.py
# Unit Tests

import unittest
from TextWeaver import Fabric, Garment

class TestFabric(unittest.TestCase):
    def setUp(self):
        self.text = "This is a test sentence. This is another test sentence."
        self.fabric = Fabric(self.text)

    def test_get_named_entities(self):
        result = self.fabric.get_named_entities()
        self.assertIsInstance(result, list)

    def test_get_pos(self):
        result = self.fabric.get_pos()
        self.assertIsInstance(result, list)

    def test_get_sentences(self):
        result = self.fabric.get_sentences()
        self.assertEqual(result, ["This is a test sentence.", "This is another test sentence."])

    def test_get_lemmas(self):
        result = self.fabric.get_lemmas()
        self.assertIsInstance(result, list)

    def test_get_stems(self):
        result = self.fabric.get_stems()
        self.assertIsInstance(result, list)

    def test_remove_stopwords(self):
        result = self.fabric.remove_stopwords()
        self.assertIsInstance(result, list)

    def test_get_sentiment(self):
        result = self.fabric.get_sentiment()
        self.assertIsInstance(result, dict)

    def test_assign_codes(self):
        self.fabric.assign_codes("Testing", "test sentence")
        result = self.fabric.codes
        self.assertIsInstance(result, dict)
        self.assertIn("Testing", result)
        self.assertEqual(result["Testing"], ["test sentence"])
        
    def test_get_codes(self):
        result = self.fabric.find_codes("test sentence")
        self.assertIsInstance(result, list)

    def test_move_code(self):
        # See if key "Test" exists and has a subkey "Testing" with the value "test sentence"
        self.fabric.assign_codes("Testing", "Test sentence")
        result = self.fabric.move_code("Testing", "Test")
        self.assertEqual(result["Test"]["Testing"], ["Test sentence"])
        # Check if the old key "Testing" is gone
        self.assertNotIn("Testing", result)

class TestGarment(unittest.TestCase):
    def setUp(self):
        self.garment = Garment(directory="tests/test_directory")

    def test_get_sentiment(self):
        result = self.garment.get_sentiment()
        self.assertIsInstance(result, dict)

if __name__ == '__main__':
    unittest.main()

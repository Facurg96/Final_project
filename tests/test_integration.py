import unittest

import requests
import json


class TestIntegration(unittest.TestCase):
    def test_index(self):
        response = requests.request(
            "GET",
            "http://0.0.0.0/",
        )
        self.assertEqual(response.status_code, 200)

    def test_analyze(self):
        form_data={
            'product_name':"car",
            'product_description':"chevrolet truck",
            'price':1000}
        headers = {}
        payload = {}
        response = requests.request(
            "POST",
            "http://0.0.0.0/analyze",
            headers=headers,
            data=(form_data),
            
        )
        self.assertEqual(response.status_code, 200)
        #print(response.text)
        f = open("tests/example.txt")
        new_f=f.read()
        self.assertEqual(response.text, str(new_f))
        """self.assertEqual(len(data.keys()), 3)
        self.assertEqual(data["success"], True)
        self.assertEqual(data["prediction"], "Eskimo_dog")
        self.assertAlmostEqual(data["score"], 0.9346, 5)"""


if __name__ == "__main__":
    unittest.main()

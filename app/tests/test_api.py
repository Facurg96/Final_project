
from unittest import TestCase
from unittest.mock import patch
from fastapi.testclient import TestClient

import app


client = TestClient(app)





class TestIntegration(TestCase):
    def setUp(self):
        self.client = TestClient(app)

    def test_index(self):
        response = self.client.get("/")
        self.assertEqual(response.status_code, 200)
        
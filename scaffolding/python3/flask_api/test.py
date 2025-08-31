import unittest
from api import app


class TestAPI(unittest.TestCase):
    def setUp(self):
        self.app = app.test_client()

    def tearDown(self):
        pass

    def test_get_key_values(self):
        response = self.app.get('/api/v1/key_values')
        self.assertEqual(response.status_code, 200)
        expected_key_values = {"k1": "v1", "k2": "v2"}
        self.assertEqual(response.json, expected_key_values)


if __name__ == '__main__':
    unittest.main()

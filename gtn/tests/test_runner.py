import unittest

from gtn import runner


class TestRunner(unittest.TestCase):
    def setUp(self) -> None:
        self.instance = runner.Runner()

    def tearDown(self) -> None:
        del self.instance

    def test_load_data(self):
        self.instance.load_data()

    def test_preprocess(self):
        self.fail()

    def test_run(self):
        self.fail()

    def test_create_config(self):
        self.fail()


if __name__ == '__main__':
    unittest.main()

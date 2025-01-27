# Provide tools for writing and executing automated tests for Python.
import unittest
from dataset import *
from dataloader import *


# Testcase
class TestCase(unittest.TestCase):
    # Create Testcases
    def setUp(self):
        self.datasets = [

            # Case 1: Small integer number
            (np.array([[1, 2], [3, 4]]), np.array([0, 1])),

            # Case 2: Float number
            (np.array([[1.1, 2.2], [3.3, 4.4]]), np.array([0.5, 1.5])),

            # Case 3: Short string
            (np.array(["a", "b", "c"]), np.array([1, 2, 3])),

            # Case 4: Long string
            (np.array(["x" * 100, "y" * 100, "z" * 100]), np.array([100, 200, 300])),

            # Case 5: Empty
            # reshape(0, 2): Transforming the array into a 2-dimensional array with 0 rows and 2 columns
            (np.array([]).reshape(0, 2), np.array([])),

            # Case 6: One batch
            # Creating a 2-dimensional array with number 42 and creating a 1-dimensional array with number 99
            (np.array([[42]]), np.array([99])),

            # Case 7: Big data
            # np.random.rand(10000, 20): Creating an array with 10000 rows and 20 columns with
            # values between 0 and 1
            # np.random.randint(0, 10, size=10000): Creating an array with 1 row and 0 columns with
            # 10000 values between 0 and 1000
            (np.random.rand(10000, 20), np.random.randint(0, 1000, size=10000)),

            # Case 8: Vector
            (np.array([1, 2, 3]), np.array([4, 5, 6])),
        ]

    # Testing
    def test_01_tensor_dataset(self):
        for i, (data, labels) in enumerate(self.datasets):
            # enumerate: getting index and value
            # Finding mistake in testcases
            with self.subTest(f"Test case {i + 1}"):
                tensor_dataset = TensorDataset(data, labels)

                # Check quantity
                # assertEqual: Compare two values and check if they are equal
                self.assertEqual(tensor_dataset.length(), len(data))

                # Check data and label
                # assertEqual: Check if true, return true

                self.assertTrue(
                    np.array_equal(tensor_dataset.dataset, data)
                    if isinstance(data, np.ndarray)
                    else tensor_dataset.dataset == data
                )

                self.assertTrue(
                    np.array_equal(tensor_dataset.labels, labels)
                    if isinstance(labels, np.ndarray)
                    else tensor_dataset.labels == labels
                )

                # Check if not empty
                if len(data) > 0:
                    for index in range(len(data)):
                        data_label = tensor_dataset.getimage(index)
                        self.assertTrue(np.array_equal(data_label.image_data, data[index]))
                        self.assertEqual(data_label.image_label, labels[index])

    # Check index if it is out of bound and exception handling
    def test_02_index_out_of_bounds(self):
        for i, (data, labels) in enumerate(self.datasets):
            # Check the length of data = the length of labels ; data is not empty
            if len(data) == len(labels) and len(data) > 0:
                with self.subTest(f"Test case {i + 1}"):
                    tensor_dataset = TensorDataset(data, labels)
                    # Check the index if index < 0 or the length of data is too Long
                    with self.assertRaises(IndexError):
                        tensor_dataset.getimage(-1)
                    with self.assertRaises(IndexError):
                        tensor_dataset.getimage(len(data))


if __name__ == "__main__":
    unittest.main()

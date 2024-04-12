import unittest
from runner import CudaRunner  # Make sure to import correctly from your actual module

class TestCudaRunner(unittest.TestCase):
    def setUp(self):
        # Sample input file name (not used in tests, but required for initialization)
        self.runner = CudaRunner(input_file='test_input.cu', output_file='test_output.ptx', result_file='test_results.txt')

    def test_parse_output(self):
        # Comma-separated simulated standard output from CUDA execution command
        simulated_output = """
        ld,0,0,0,0x1f4,10
        st,0,0,0,0x208,11
        ld,0,0,0,0x1fc,10
        ld,0,1,0,0x1f4,12
        st,0,0,0,0x200,11
        ld,0,0,0,0x1f4,10
        ld,0,1,0,0x1f8,12
        st,0,0,0,0x204,11
        ld,0,0,1,0x200,10
        """
        # Adjust the expected_data based on the new simulated output format
        expected_data = [
            ('0', '0', '0', '10', 0, 'ld', '0x1f4'),
            ('0', '0', '0', '11', 0, 'st', '0x208'),
            ('0', '0', '0', '10', 1, 'ld', '0x1fc'),
            ('0', '1', '0', '12', 0, 'ld', '0x1f4'),
            ('0', '0', '0', '11', 1, 'st', '0x200'),
            ('0', '0', '0', '10', 2, 'ld', '0x1f4'),
            ('0', '1', '0', '12', 1, 'ld', '0x1f8'),
            ('0', '0', '0', '11', 2, 'st', '0x204'),
            ('0', '0', '1', '10', 0, 'ld', '0x200'),
        ]

        parsed_data = self.runner.parse_output(simulated_output.strip())  # Remove leading/trailing newlines

        # Assert that the parsed data matches the expected data
        self.assertEqual(parsed_data, expected_data)

if __name__ == '__main__':
    unittest.main()
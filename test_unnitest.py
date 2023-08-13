import unittest
import os
import subprocess
from Deepl_reproduction.pipeline.data_loading import load_all_data, load_data_to_front_database, load_data

class Test(unittest.TestCase):
    """
    The goal of this class is to implement unnitest
    and check everything commited makes sense
    """
    def test_image_build_run(self)->None:
        """
        The goal of this function is to test
        whether the construction and running
        of the docker image works well

        Arguments:
            -None
        Returns:
            -None
        """

        result = subprocess.run(['docker', 'build', '-t', 'front_database_image:latest', '-f', 'Dockerfile-front-database', '.'], capture_output=True, text=True)
        result_run = subprocess.run(['docker', 'run','-d', '-p', '3306:3306', '--name', 'local-mysql-container', 'front_database_image'], capture_output=True, text=True)
        result_images = subprocess.run(['docker', 'images'], capture_output=True, text=True)
        
        self.assertEqual(result.returncode, 0)
        self.assertEqual(result_run.returncode, 0)

    def test_front_database_data(self)->None:
        """
        The goal of this function is to test
        whether the data loaded in the front
        database is accurate

        Arguments:
            -None
        Returns:
            -None
        """

        subprocess.run(['docker', 'build', '-t', 'front_database_image:latest', '-f', 'Dockerfile-front-database', '.'], capture_output=True, text=True)
        subprocess.run(['docker', 'run','-d', '-p', '3306:3306', '--name', 'local-mysql-container', 'front_database_image'], capture_output=True, text=True)
        load_data_to_front_database()
        df_front_database=load_data()
        


if __name__ == "__main__":
    unittest.main()
import unittest
import os
import subprocess
from Deepl_reproduction.ETL.transform.traduction import translate_text
from Deepl_reproduction.pipeline.data_loading import load_all_data, load_data_to_front_database, load_data

class Test(unittest.TestCase):
    """
    The goal of this class is to implement unnitest
    and check everything commited makes sense
    """

    def test_traduction_function(self)->None:
        """
        The goal of this function is to test
        the traduction function works and 
        is able to deliver accuracte results 
        
        Arguments:
            -None
        Returns:
            -None
        """

        languages=["FR", "JA", "EN"]
        sentence_to_translate=["Hey, how are you ?", 
                               "Bonjour, comment ça va ?",
                               "Hola, ¿cómo estás?", 
                               "こんにちは、お元気ですか?"]
        
        sentence_translated=[translate_text(sentence,target_lang=target_lang) for sentence in sentence_to_translate for target_lang in languages]
        is_all_sentence=all([isinstance(sentence,str) for sentence in sentence_translated])

        self.assertTrue(is_all_sentence)

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

        result = subprocess.run(['docker', 'build', '-t', 'front_database_image:latest', '-f', 'Dockerfile-front-database', '.'], capture_output=True, text=True)
        result_run = subprocess.run(['docker', 'run','-d', '-p', '3306:3306', '--name', 'local-mysql-container', 'front_database_image'], capture_output=True, text=True)
        result_images = subprocess.run(['docker', 'images'], capture_output=True, text=True)
        
        
        load_data_to_front_database()
        df_front_database=load_data()
        
        self.assertEqual(result.returncode, 0)
        self.assertEqual(result_run.returncode, 0)
        self.assertFalse(df_front_database.empty)
        self.assertIn("french",df_front_database.columns)
        self.assertIn("english",df_front_database.columns)
        self.assertEqual(2, df_front_database.shape[1])


if __name__ == "__main__":
    unittest.main()
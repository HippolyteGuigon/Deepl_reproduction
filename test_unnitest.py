import unittest
import os
import subprocess
import sys
from google.cloud import storage
from Deepl_reproduction.ETL.transform.traduction import translate_text
from Deepl_reproduction.pipeline.data_loading import (
    load_data_to_front_database,
    load_data,
)
from Deepl_reproduction.model.model_loading import load_model

client = storage.Client.from_service_account_json(
    "deepl_api_key.json", project="deepl-reprodution"
)

sys.path.insert(0, "./Deepl_reproduction")


class Test(unittest.TestCase):
    """
    The goal of this class is to implement unnitest
    and check everything commited makes sense
    """

    def test_traduction_function(self) -> None:
        """
        The goal of this function is to test
        the traduction function works and
        is able to deliver accuracte results

        Arguments:
            -None
        Returns:
            -None
        """

        languages = ["FR", "JA", "EN-GB", "ES"]
        sentence_to_translate = {
            "Hey, how are you ?": "EN",
            "Bonjour, comment ça va ?": "FR",
            "Hola, ¿cómo estás?": "ES",
            "こんにちは、お元気ですか?": "JA",
        }

        sentence_translated = [
            translate_text(sentence, target_lang=target_lang)
            for sentence in sentence_to_translate.keys()
            for target_lang in languages
            if sentence_to_translate[sentence] != target_lang
        ]
        is_all_sentence = all(
            [isinstance(sentence, str) for sentence in sentence_translated]
        )

        self.assertTrue(is_all_sentence)

    def test_front_database_data(self) -> None:
        """
        The goal of this function is to test
        whether the data loaded in the front
        database is accurate

        Arguments:
            -None
        Returns:
            -None
        """

        result = subprocess.run(
            [
                "docker",
                "build",
                "-t",
                "front_database_image:latest",
                "-f",
                "Dockerfile-front-database",
                ".",
            ],
            capture_output=True,
            text=True,
        )
        result_run = subprocess.run(
            [
                "docker",
                "run",
                "-d",
                "-p",
                "3306:3306",
                "--name",
                "local-mysql-container",
                "front_database_image",
            ],
            capture_output=True,
            text=True,
        )

        load_data_to_front_database(kaggle_length=50000)
        df_front_database = load_data()

        self.assertEqual(result.returncode, 0)
        self.assertEqual(result_run.returncode, 0)
        self.assertFalse(df_front_database.empty)
        self.assertIn("french", df_front_database.columns)
        self.assertIn("english", df_front_database.columns)
        self.assertEqual(2, df_front_database.shape[1])

    def test_check_model_loading(self) -> None:
        """
        The goal of this function is to
        check the ability to load a model
        from google cloud storage

        Arguments:
            -None
        Returns:
            -None
        """

        sys.path.insert(0, "Deepl_reproduction")
        bucket = client.get_bucket("english_deepl_bucket")
        blobs = bucket.list_blobs()
        file_names = [blob.name for blob in blobs]
        file_names = [blob.name for blob in blobs if blob.name != "bpe.model"]
        loss_gcp = [
            name.replace("deepl_english_model_loss_", "").replace(".pth.tar", "")
            for name in file_names
        ]
        loss_gcp = [float(loss.replace("_", ".")) for loss in loss_gcp]
        min_loss_gcp = min(loss_gcp)
        best_model_name = [
            name for name in file_names if str(min_loss_gcp).replace(".", "_") in name
        ][0]
        best_model_path = os.path.join(
            os.getcwd(), "Deepl_reproduction/model", best_model_name
        )
        best_model = load_model()

        self.assertTrue(
            os.path.exists(os.path.join("Deepl_reproduction/model", best_model_path))
        )
        self.assertIsInstance(best_model, dict)


if __name__ == "__main__":
    unittest.main()

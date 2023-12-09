# csci-567-group-project

# To reproduce results
Please run the `concatenated_network.ipynb` file with the following modifications:
- Delete the second code block (this was left from use on Colab)
- Change all the file locations in the `read_csv` function to wherever you are holding the `{train, dev, test}_{textmodel, voice}.csv` files

This will produce the results for the concatenated model. The text and voice models were trained separately in `text_processing.ipynb` and `voice.py`

These models were used in `text_vector.py` and `voice_vector.py` to create the concatenated data present in the repo. Unfortunately, our full dataset is way too large to include in this repo, so we can not offer the data to reproduce the concatenated vectors.

The dataset we used was sourced from https://affective-meld.github.io/, but we did some preprocessing of the audio to make all the .mp3 files into .wav files

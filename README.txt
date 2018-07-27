Requirements:
    1. TensorFlow 1.8
    2. tqdm
    3. nltk (with punkt resource for word tokenize)
    4. glove.840B.300d.txt (if embed.pkl is present in the root directory of this project, then the glove.840B.300d.txt
                            is not required, as embed.pkl is a filtered version of the original file)

Usage:
    Training:
        python3 main.py [commandline options, see predict.py for details]

    Best setting training (the best setting is picked by choosing the overall highest val acc on the standard story cloze task val set):
        python3 main.py --near 1 --random 5 --backward 1 --nn vanilla --preEmbedding

    To reproduce the best result on original story cloze test:
        python3 main.py --near 1 --random 5 --backward 1 --nn vanilla --preEmbedding --loadModel --testModel --testTag test

        the output file is "original_test_results.csv"

    To reproduce the result on eth test set:
        python3 main.py --near 1 --random 5 --backward 1 --nn vanilla --preEmbedding --loadModel --testModel --testTag eth

        the output file is "ETH_test_results.csv"

Attention:
    The code will generate a pkl file in data folder, when reproducing results of the best setting, please use the pkl file
    in the link below, as different machines may generate different vocabularies because of encodings. The pkl file below is
    generated on Leonhard.

For generating negative samples:
    1. Please see the folder "random_backward" for generating random and backward negative samples;
    2. Please see the folder "nearest" for generating nearest ending negative samples.


Link for trained model, "embed.pkl" and data folder: https://polybox.ethz.ch/index.php/s/YZYXhQrpAEUgSj7 (Please unzip the included zip file. There will be a folder named "project2", move everything in the folder to the root directory of the project.)

Contact: Zhao Meng, zhmeng@student.ethz.ch


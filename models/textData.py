import os
from collections import Counter
import random
from tqdm import tqdm
from models.data_utils import Sample, Batch
import nltk
import csv
import numpy as np
import pickle as p
class TextData:
    def __init__(self, args):
        self.args = args

        #note: use 20k most frequent words
        self.UNK_WORD = 'unk'
        self.PAD_WORD = '<pad>'

        # list of batches
        self.train_batches = []
        self.val_batches = []
        self.test_batches = []

        self.word2id = {}
        self.id2word = {}

        self.train_samples = None
        self.valid_samples = None
        self.test_samples = None

        self.train_samples, self.valid_samples, self.test_samples = self._create_data()

        self.preTrainedEmbedding = None
        # [num_batch, batch_size, maxStep]
        self.train_batches = self._create_batch(self.train_samples)
        self.val_batches = self._create_batch(self.valid_samples)

        # note: test_batches is none here
        self.test_batches = self._create_batch(self.test_samples)

        self.preTrainedEmbedding = self.create_embeddings()

    def create_embeddings(self):
        print('Creating pretrained embeddings!')
        words = self.word2id.keys()

        glove_embed = {}

        if os.path.exists('embed.pkl'):
            with open('embed.pkl', 'rb') as file:
                glove_embed = p.load(file)
        else:
            with open(self.args.embedFile, 'r') as glove:
                lines = glove.readlines()
                for line in tqdm(lines):
                    splits = line.split()
                    word = splits[0]
                    if len(splits) > 301:
                        word = ''.join(splits[0:len(splits)-300])
                        splits[1:] = splits[len(splits)-300:]
                    if word not in words:
                        continue
                    embed = [float(s) for s in splits[1:]]
                    glove_embed[word] = embed

            with open('embed.pkl', 'wb') as file:
                p.dump(glove_embed, file)

        embeds = []
        for word_id in range(len(self.id2word)):
            word = self.id2word[word_id]
            if word in glove_embed.keys():
                embed = glove_embed[word]
            else:
                embed = glove_embed[self.UNK_WORD]
                self.word2id[word] = self.word2id[self.UNK_WORD]
            embeds.append(embed)

        embeds = np.asarray(embeds)


        return embeds


    def getVocabularySize(self):
        assert len(self.word2id) == len(self.id2word)
        return len(self.word2id)

    def _create_batch(self, all_samples, tag='test'):
        all_batches = []
        if tag == 'train':
            random.shuffle(all_samples)
        if all_samples is None:
            return all_batches

        num_batch = len(all_samples)//self.args.batchSize + 1
        for i in range(num_batch):
            samples = all_samples[i*self.args.batchSize:(i+1)*self.args.batchSize]

            if len(samples) == 0:
                continue

            batch = Batch(samples)
            all_batches.append(batch)

        return all_batches

    def _create_samples(self, file_path):
        with open(file_path, 'r') as file:
            all_samples = []
            length = len(file.readlines())
            file.seek(0)
            reader = csv.reader(file)
            for idx, line in enumerate(tqdm(reader, total=length)):
                if idx == 0:
                    continue

                # 6 sentences, the first 4 are contexts, last 2 candidates
                sentences = line[1:7]

                # 6 sentences, the first 4 are contexts, last 2 candidates
                label = int(line[-1]) - 1
                assert label == 0 or label == 1

                sample_sentences = []
                sample_input = []
                sample_length = []
                for i, sentence in enumerate(sentences):
                    words = nltk.word_tokenize(sentence)
                    words = words[0:self.args.maxSteps]

                    sample_length.append(len(words))

                    while len(words) < self.args.maxSteps:
                        words.append(self.PAD_WORD)
                    sample_sentences.append(words)

                    word_ids = []
                    for word in words:
                        if word in self.word2id.keys():
                            word_ids.append(self.word2id[word])
                        else:
                            word_ids.append(self.word2id[self.UNK_WORD])
                    sample_input.append(word_ids)

                # label = 0: the 5th sentence is correct
                # label = 1: the 6th sentence is correct

                sample = Sample(input_=sample_input, sentences=sample_sentences, length=sample_length,
                            label=label)

                all_samples.append(sample)

        return all_samples

    def map_to_id(self, sentences):
        inputs = []
        padded_sentences = []
        length = []
        for sentence in sentences:
            words = nltk.word_tokenize(sentence)
            length.append(len(words))
            while len(words) < self.args.maxSteps:
                words.append(self.PAD_WORD)

            word_ids = []
            for word in words:
                if word in self.word2id.keys():
                    word_ids.append(self.word2id[word])
                else:
                    #print(word)
                    word_ids.append(self.word2id[self.UNK_WORD])
            inputs.append(word_ids)
            padded_sentences.append(words)

        return inputs, padded_sentences, length

    def _create_train_samples(self, file_path, tag='near'):
        if tag == 'near':
            n_fake = self.args.near
        elif tag == 'random':
            n_fake = self.args.random
        else:
            n_fake = self.args.backward

        if n_fake == 0:
            return []

        with open(file_path, 'r', encoding='ISO-8859-1') as file:
            all_samples = []
            length = len(file.readlines())
            file.seek(0)
            reader = csv.reader(file)
            for idx, line in enumerate(tqdm(reader, total=length)):
                if idx == 0:
                    continue
                # 4 context sentences and 1 correct sentences
                context_sentences = line[0:5]
                context_inputs, context_sentences, context_length = self.map_to_id(context_sentences)

                # the 5th sentence is always the correct sentence in our examples
                label = 0

                fake_sentences = line[5:]
                fake_inputs, fake_sentences, fake_length = self.map_to_id(fake_sentences)


                fake_inputs = fake_inputs[0:n_fake]
                fake_sentences = fake_sentences[0:n_fake]
                fake_length = fake_length[0:n_fake]

                for i, fake in enumerate(fake_sentences):
                    sample_input = context_inputs + [fake_inputs[i]]
                    sample_sentences = context_sentences + [fake]
                    sample_length = context_length + [fake_length[i]]
                    sample = Sample(input_=sample_input, sentences=sample_sentences, length=sample_length,
                            label=label)

                    all_samples.append(sample)

        return all_samples

    def _create_data(self):

        near_path = os.path.join(self.args.dataDir, self.args.nearFile)
        random_path = os.path.join(self.args.dataDir, self.args.randomFile)
        backward_path = os.path.join(self.args.dataDir, self.args.backwardFile)

        val_path = os.path.join(self.args.dataDir, self.args.valFile)
        test_path = os.path.join(self.args.dataDir, self.args.testFile)



        print('Building vocabularies')
        self.word2id, self.id2word = self._build_vocab(near_path, val_path, test_path)

        print('Building val samples!')
        val_samples = self._create_samples(val_path)
        print('Building test samples!')
        test_samples = self._create_samples(test_path)


        print('Building nearest samples!')
        near_samples = self._create_train_samples(near_path, tag='near')

        print('Building random samples!')
        random_samples = self._create_train_samples(random_path, tag='random')

        print('Building backward samples!')
        backward_samples = self._create_train_samples(backward_path, tag='backward')

        train_samples = near_samples + random_samples + backward_samples

        random.shuffle(train_samples)


        return train_samples, val_samples, test_samples

    @staticmethod
    def _read_words(filename, all_words=None, tag='train'):
        if all_words is None:
            all_words = []

        with open(filename, 'r', encoding='ISO-8859-1') as file:
            length = len(file.readlines())
            file.seek(0)
            reader = csv.reader(file)
            for idx, line in enumerate(tqdm(reader, total=length)):
                if idx == 0:
                    continue

                assert len(line) == 8 or len(line) == 11 or len(line) == 9
                # 6 sentences, the first 4 are contexts, last 2 candidates
                if tag != 'train':
                    sentences = line[1:7]
                else:
                    sentences = line[:]

                # remove double quotes
                for i, sentence in enumerate(sentences):
                    words = nltk.word_tokenize(sentence)
                    all_words.extend(words)

        return all_words

    def _build_vocab(self, train_path, val_path, test_path):
        all_words = self._read_words(train_path, tag='train')
        all_words = self._read_words(val_path, all_words=all_words, tag = 'val')
        all_words = self._read_words(test_path, all_words=all_words, tag = 'test')

        counter = Counter(all_words)

        count_pairs = sorted(counter.items(), key=lambda x: (-x[1], x[0]))
        #print(count_pairs[300000])
        # keep the most frequent vocabSize words, including the special tokens
        # -1 means we have no limits on the number of words
        if self.args.vocabSize != -1:
            count_pairs = count_pairs[0:self.args.vocabSize-2]


        count_pairs.append((self.UNK_WORD, 100000))
        count_pairs.append((self.PAD_WORD, 100000))

        if self.args.vocabSize != -1:
            assert len(count_pairs) == self.args.vocabSize

        words, _ = list(zip(*count_pairs))
        word_to_id = dict(zip(words, range(len(words))))

        id_to_word = {v: k for k, v in word_to_id.items()}

        print(len(word_to_id))
        print(len(id_to_word))

        return word_to_id, id_to_word

    def get_batches(self, tag='train'):
        if tag == 'train':
            return self._create_batch(self.train_samples, tag='train')
        elif tag == 'val':
            return self.val_batches
        else:
            return self.test_batches

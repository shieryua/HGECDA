import numpy as np
import torch
from torch.utils.data import Dataset
from collections import deque
np.random.seed(100)
num_walks_per_node = 100

class DataReader:
    NEGATIVE_TABLE_SIZE = 1000   # 1e6

    def __init__(self, dataset, min_count, care_type):

        self.negativesone = dict()
        self.negatives = []
        self.discards = []
        self.negpos = 0
        self.care_type = care_type
        self.word2id = dict()
        self.id2word = dict()
        self.word2idC = dict()
        self.word2idD = dict()
        self.word2idM = dict()
        self.sentences_count = 0
        self.token_count = 0
        self.word_frequency = dict()
        self.inputFileName = dataset.fn
        self.hasnone = False
        self.read_words(min_count)
        self.initTableNegatives()
        # self.initTableDiscards()

    def read_words(self, min_count):
        word_frequency = dict()
        for line in open(self.inputFileName):  # gb18030 , encoding="gbk"  line = ['BCRC-3', 'miR-182-5p', 'bladder-cancer', 'miR-146b-3p', 'hsa_circ_0071662']
            line = line.split()
            if len(line) > 1:
                self.sentences_count += 1
                for word in line:
                    # print('wwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwww',word)
                    if len(word) > 0:
                        self.token_count += 1
                        word_frequency[word] = word_frequency.get(word, 0) + 1    # 统计列表中每个元素出现次数
                        if self.token_count % 1000000 == 0:     # 1000000
                            print("Read " + str(int(self.token_count / 1000000)) + "M words.")         # 1000000
        # print(len(word_frequency.keys()))

        wid = 0
        cc = 0
        for w, c in word_frequency.items():    #  可遍历的键值 # w :节点名字 c: 节点在整个游走中出现频率
            if c < min_count:
                continue
            if w == "CS":
                cc = c
                self.hasnone = True
                continue
            self.word2id[w] = wid
            self.id2word[wid] = w
            self.word_frequency[wid] = c
            wid += 1
        if self.hasnone == True:
            self.word2id['CS'] = wid
            self.word_frequency[wid] = cc
            print("Total embeddings: " + str(len(self.word2id) - 1))
        self.word_count = len(self.word2id)
        print("wid",wid)
        print("frequency",len(word_frequency))
        print("Total embeddings0: " + str(len(self.word2id) )) 

    # def initTableDiscards(self):
    #     # get a frequency table for sub-sampling. Note that the frequency is adjusted by
    #     # sub-sampling tricks.
    #     t = 0.0001
    #     f = np.array(list(self.word_frequency.values())) / self.token_count
    #     self.discards = np.sqrt(t / f) + (t / f)

    def initTableNegatives(self):
        # get a table for negative sampling, if word with index 2 appears twice, then 2 will be listed
        # in the table twice.
        #-------------------------------
        # for i in self.word_frequency:
        #     if self.word_frequency[i] > 2*num_walks_per_node:
        #         self.negativesone[i]= i
        # self.negatives = np.array(list(self.negativesone.values()))
        # np.random.shuffle(self.negatives)
        # pow_frequency = np.array(list(self.negativesone.values()))**1
        # words_pow = sum(pow_frequency)
        # ratio = pow_frequency / words_pow
        # self.sampling_prob = ratio
        #-------------------------------
        pow_frequency = np.array(list(self.word_frequency.values())) ** 0.75
        words_pow = sum(pow_frequency)
        ratio = pow_frequency / words_pow
        count = np.round(ratio * DataReader.NEGATIVE_TABLE_SIZE)
        for wid, c in enumerate(count):
            # if wid == (len(self.word2id) - 1) and self.hasnone==True:
            #     continue
            self.negatives += [wid] * int(c)
        self.negatives = np.array(self.negatives)
        np.random.shuffle(self.negatives)
        self.sampling_prob = ratio

    def getNegatives(self, UV, size):  # TODO check equality with target
        if self.care_type == 0:
            response = self.negatives[self.negpos:self.negpos + size]
            self.negpos = (self.negpos + size) % len(self.negatives)  #len(negative) = 100000006
            if len(response) != size:
                return np.concatenate((response, self.negatives[0:self.negpos]))
        return response

class EmbeddingDataset(Dataset):
    def __init__(self, data, window_size):
        # read in data, window_size and input filename
        self.data = data
        self.window_size = window_size
        self.input_file = open(data.inputFileName)  # ISO-8859-1 , encoding="gbk"

    def __len__(self):
        # return the number of walks
        return self.data.sentences_count   # 整个游走里的路径个数

    def __getitem__(self, idx):
        # return the list of pairs (center, context, 5 negatives)
        while True:
            L = 0
            line = self.input_file.readline()   #  游走路径读入
            if not line:
                self.input_file.seek(0, 0)    # 移动文件的读取指针到指定位置
                line = self.input_file.readline()

            if len(line) > 1:
                L += 1
                words = line.split()  # 一条路径里包含的节点
                pair_catch = []
                # if L <= num_walks_per_node*514:
                #     if len(words) > 1:
                #         word_ids = [self.data.word2id[w] for w in words if
                #                     w in self.data.word2id ]   # 这个随机条件确实除去了一部分节点 and np.random.rand() < self.data.discards[self.data.word2id[w]]
                #         word_idsC = [word_ids[c] for c in range(len(word_ids)) if c ==0 or c==4]
                #         word_idsD = [word_ids[d] for d in range(len(word_ids)) if d == 2 ]
                #         word_idsM = [word_ids[m] for m in range(len(word_ids)) if m == 1 or m == 3]
                #     for u in word_idsC:
                #         for v in word_idsD:
                #             pair_catch.append((v, u, self.data.getNegatives(v,1)))
                # elif L <= num_walks_per_node*(514+62):
                #     if len(words) > 1:
                #         word_ids = [self.data.word2id[w] for w in words if
                #                     w in self.data.word2id]  # 这个随机条件确实除去了一部分节点 and np.random.rand() < self.data.discards[self.data.word2id[w]]
                #         word_idsD = [word_ids[d] for d in range(len(word_ids)) if d == 0 or d == 4]
                #         word_idsC = [word_ids[c] for c in range(len(word_ids)) if c == 2]
                #         word_idsM = [word_ids[m] for m in range(len(word_ids)) if m == 1 or m == 3]
                #     for u in word_idsC:
                #         for v in word_idsD:
                #             pair_catch.append((u, v, self.data.getNegatives(u,1)))
                # if L <= num_walks_per_node*(514):
                #     if len(words) > 1:
                #         word_ids = [self.data.word2id[w] for w in words if
                #                     w in self.data.word2id]  # 这个随机条件确实除去了一部分节点 and np.random.rand() < self.data.discards[self.data.word2id[w]]
                #         word_idsC1 = [word_ids[c1] for c1 in range(len(word_ids)) if c1 == 0]
                #         word_idsC2 = [word_ids[c2] for c2 in range(len(word_ids)) if c2 == 2]
                #         word_idsM = [word_ids[m] for m in range(len(word_ids)) if m == 1]
                #     for u in word_idsC1:
                #         for v in word_idsC2:
                #             pair_catch.append((u, v, self.data.getNegatives(u,1)))
                # elif L <= num_walks_per_node*(514+62):
                #     if len(words) > 1:
                #         word_ids = [self.data.word2id[w] for w in words if
                #                     w in self.data.word2id]  # 这个随机条件确实除去了一部分节点 and np.random.rand() < self.data.discards[self.data.word2id[w]]
                #         word_idsD1 = [word_ids[d1] for d1 in range(len(word_ids)) if d1 == 0]
                #         word_idsD2 = [word_ids[d2] for d2 in range(len(word_ids)) if d2 == 2]
                #         word_idsM = [word_ids[m] for m in range(len(word_ids)) if m == 1]
                #     for u in word_idsD1:
                #         for v in word_idsD2:
                #             pair_catch.append((u, v, self.data.getNegatives(u,1)))
                # elif L <= num_walks_per_node*(2*514+62):
                #     if len(words) > 1:
                #         word_ids = [self.data.word2id[w] for w in words if
                #                     w in self.data.word2id]  # 这个随机条件确实除去了一部分节点 and np.random.rand() < self.data.discards[self.data.word2id[w]]
                #         word_idsD1 = [word_ids[d1] for d1 in range(len(word_ids)) if d1 == 0 or d1 == 3]
                #         word_idsD2 = [word_ids[d2] for d2 in range(len(word_ids)) if d2 == 2 or d2== 1]
                #         # word_idsM = [word_ids[m] for m in range(len(word_ids)) if m == 1]
                #     for u in word_idsD1:
                #         for v in word_idsD2:
                #             pair_catch.append((u, v, self.data.getNegatives(u,1)))
                # else:
                #     if len(words) > 1:
                #         word_ids = [self.data.word2id[w] for w in words if
                #                     w in self.data.word2id]  # 这个随机条件确实除去了一部分节点 and np.random.rand() < self.data.discards[self.data.word2id[w]]
                #         word_idsD1 = [word_ids[d1] for d1 in range(len(word_ids)) if d1 == 0 or d1 == 3]
                #         word_idsD2 = [word_ids[d2] for d2 in range(len(word_ids)) if d2 == 2 or d2== 1]
                #         # word_idsM = [word_ids[m] for m in range(len(word_ids)) if m == 1]
                #     for u in word_idsD1:
                #         for v in word_idsD2:
                #             pair_catch.append((v, u, self.data.getNegatives(u,1)))
                if len(words) > 1:
                    word_ids = [self.data.word2id[w] for w in words if
                                w in self.data.word2id ]
                    for i, u in enumerate(word_ids):

                        for j, v in enumerate(
                                word_ids[max(i - self.window_size, 0):i + self.window_size+1]):
                            assert u < self.data.word_count
                            assert v < self.data.word_count
                            if i == j:
                                continue
                            pair_catch.append((u, v, self.data.getNegatives(v,1)))
                return pair_catch


    @staticmethod
    def collate(batches):
        # print('1231234',np.array(batches).shape)
        # print('12312345',np.array(batches[0]).shape)
        all_u = [u for batch in batches for u, _, _ in batch if len(batch) > 0]
        all_v = [v for batch in batches for _, v, _ in batch if len(batch) > 0]
        all_neg_v = [neg_v for batch in batches for _, _, neg_v in batch if len(batch) > 0]

        return torch.LongTensor(all_u), torch.LongTensor(all_v), torch.LongTensor(all_neg_v)
# -----------------------------------------------------------------------------------------------------------------
# class EmbeddingDataset(Dataset):
#     def __init__(self, data, window_size):
#         # read in data, window_size and input filename
#         self.data = data
#         self.window_size = window_size
#         self.input_file = open(data.inputFileName)  # ISO-8859-1 , encoding="gbk"
#
#     def __len__(self):
#         # return the number of walks
#         return self.data.sentences_count   # 整个游走里的路径个数
#
#     def __getitem__(self, idx):
#         # return the list of pairs (center, context, 5 negatives)
#         while True:
#             line = self.input_file.readline()   #  游走路径读入
#             if not line:
#                 self.input_file.seek(0, 0)    # 移动文件的读取指针到指定位置
#                 line = self.input_file.readline()
#
#             if len(line) > 1:
#                 words = line.split()  # 一条路径里包含的节点
#
#
#                 if len(words) > 1:
#                     word_ids = [self.data.word2id[w] for w in words if
#                                 w in self.data.word2id ]   # 这个随机条件确实除去了一部分节点 and np.random.rand() < self.data.discards[self.data.word2id[w]]
#                     # print('idsids',word_ids)
#                     pair_catch = []
#                     for i, u in enumerate(word_ids):
#
#                         for j, v in enumerate(
#                                 word_ids[max(i - self.window_size, 0):i + self.window_size]):
#                             assert u < self.data.word_count
#                             assert v < self.data.word_count
#
#
#                             if i == j:
#                                 continue
#                             pair_catch.append((u, v, self.data.getNegatives(v,1)))
#                     return pair_catch
#
#
#     @staticmethod
#     def collate(batches):
#         # print('1231234',np.array(batches).shape)
#         # print('12312345',np.array(batches[0]).shape)
#         all_u = [u for batch in batches for u, _, _ in batch if len(batch) > 0]
#         all_v = [v for batch in batches for _, v, _ in batch if len(batch) > 0]
#         all_neg_v = [neg_v for batch in batches for _, _, neg_v in batch if len(batch) > 0]
#
#         return torch.LongTensor(all_u), torch.LongTensor(all_v), torch.LongTensor(all_neg_v)
# -----------------------------------------------------------------------------------------------------------------























class HuffmanNode:
    def __init__(self, word_id, frequency):
        self.word_id = word_id  # 叶子结点存词对应的id, 中间节点存中间节点id
        self.frequency = frequency  # 存单词频次
        self.left_child = None
        self.right_child = None
        self.father = None
        self.Huffman_code = []  # 霍夫曼码（左1右0）
        self.path = []  # 根到叶子节点的中间节点id

class HuffmanTree:
    def __init__(self, wordid_frequency_dict):
        self.word_count = len(wordid_frequency_dict)  # 单词数量
        self.wordid_code = dict()
        self.wordid_path = dict()
        self.root = None
        unmerge_node_list = [HuffmanNode(wordid, frequency) for wordid, frequency in
                             wordid_frequency_dict.items()]  # 未合并节点list
        self.huffman = [HuffmanNode(wordid, frequency) for wordid, frequency in
                        wordid_frequency_dict.items()]  # 存储所有的叶子节点和中间节点
        # 构建huffman tree
        self.build_tree(unmerge_node_list)
        # 生成huffman code
        self.generate_huffman_code_and_path()

    def merge_node(self, node1, node2):
        sum_frequency = node1.frequency + node2.frequency
        mid_node_id = len(self.huffman)  # 中间节点的value存中间节点id
        father_node = HuffmanNode(mid_node_id, sum_frequency)
        if node1.frequency >= node2.frequency:
            father_node.left_child = node1
            father_node.right_child = node2
        else:
            father_node.left_child = node2
            father_node.right_child = node1
        self.huffman.append(father_node)
        return father_node

    def build_tree(self, node_list):
        while len(node_list) > 1:
            i1 = 0  # 概率最小的节点
            i2 = 1  # 概率第二小的节点
            if node_list[i2].frequency < node_list[i1].frequency:
                [i1, i2] = [i2, i1]
            for i in range(2, len(node_list)):
                if node_list[i].frequency < node_list[i2].frequency:
                    i2 = i
                    if node_list[i2].frequency < node_list[i1].frequency:
                        [i1, i2] = [i2, i1]
            father_node = self.merge_node(node_list[i1], node_list[i2])  # 合并最小的两个节点
            if i1 < i2:
                node_list.pop(i2)
                node_list.pop(i1)
            elif i1 > i2:
                node_list.pop(i1)
                node_list.pop(i2)
            else:
                raise RuntimeError('i1 should not be equal to i2')
            node_list.insert(0, father_node)  # 插入新节点
        self.root = node_list[0]

    def generate_huffman_code_and_path(self):
        stack = [self.root]
        while len(stack) > 0:
            node = stack.pop()
            # 顺着左子树走
            while node.left_child or node.right_child:
                code = node.Huffman_code
                path = node.path
                node.left_child.Huffman_code = code + [1]
                node.right_child.Huffman_code = code + [0]
                node.left_child.path = path + [node.word_id]
                node.right_child.path = path + [node.word_id]
                # 把没走过的右子树加入栈
                stack.append(node.right_child)
                node = node.left_child
            word_id = node.word_id
            word_code = node.Huffman_code
            word_path = node.path
            self.huffman[word_id].Huffman_code = word_code
            self.huffman[word_id].path = word_path
            # 把节点计算得到的霍夫曼码、路径  写入词典的数值中
            self.wordid_code[word_id] = word_code
            self.wordid_path[word_id] = word_path

    # 获取所有词的正向节点id和负向节点id数组
    def get_all_pos_and_neg_path(self):
        positive = []  # 所有词的正向路径数组
        negative = []  # 所有词的负向路径数组
        for word_id in range(self.word_count):
            pos_id = []  # 存放一个词 路径中的正向节点id
            neg_id = []  # 存放一个词 路径中的负向节点id
            for i, code in enumerate(self.huffman[word_id].Huffman_code):
                if code == 1:
                    pos_id.append(self.huffman[word_id].path[i])
                else:
                    neg_id.append(self.huffman[word_id].path[i])
            positive.append(pos_id)
            negative.append(neg_id)
        return positive, negative

class InputData:
    def __init__(self, input_file_name, min_count):
        self.input_file_name = input_file_name
        self.input_file = open(self.input_file_name)  # 数据文件
        self.min_count = min_count  # 要淘汰的低频数据的频度
        self.wordId_frequency_dict = dict()  # 词id-出现次数 dict
        self.word_count = 0  # 单词数（重复的词只算1个）
        self.word_count_sum = 0  # 单词总数 （重复的词 次数也累加）
        self.sentence_count = 0  # 句子数
        self.id2word_dict = dict()  # 词id-词 dict
        self.word2id_dict = dict()  # 词-词id dict
        self._init_dict()  # 初始化字典
        self.huffman_tree = HuffmanTree(self.wordId_frequency_dict)  # 霍夫曼树
        self.huffman_pos_path, self.huffman_neg_path = self.huffman_tree.get_all_pos_and_neg_path()
        self.word_pairs_queue = deque()
        # 结果展示
        print('Word Count is:', self.word_count)
        print('Word Count Sum is', self.word_count_sum)
        print('Sentence Count is:', self.sentence_count)
        print('Tree Node is:', len(self.huffman_tree.huffman))

    def _init_dict(self):
        word_freq = dict()
        # 统计 word_frequency
        for line in self.input_file:
            line = line.strip().split(' ')  # 去首尾空格
            self.word_count_sum += len(line)
            self.sentence_count += 1
            for word in line:
                try:
                    word_freq[word] += 1
                except:
                    word_freq[word] = 1
        word_id = 0
        # 初始化 word2id_dict,id2word_dict, wordId_frequency_dict字典
        for per_word, per_count in word_freq.items():
            if per_count < self.min_count:  # 去除低频
                self.word_count_sum -= per_count
                continue
            self.id2word_dict[word_id] = per_word
            self.word2id_dict[per_word] = word_id
            self.wordId_frequency_dict[word_id] = per_count
            word_id += 1
        self.word_count = len(self.word2id_dict)

    # 获取mini-batch大小的 正采样对 (Xw,w) Xw为上下文id数组，w为目标词id。上下文步长为window_size，即2c = 2*window_size
    def get_batch_pairs(self, batch_size, window_size):
        while len(self.word_pairs_queue) < batch_size:
            for _ in range(10000):  # 先加入10000条，减少循环调用次数
                self.input_file = open(self.input_file_name, encoding="utf-8")
                sentence = self.input_file.readline()
                if sentence is None or sentence == '':
                    continue
                wordId_list = []  # 一句中的所有word 对应的 id
                for word in sentence.strip().split(' '):
                    try:
                        word_id = self.word2id_dict[word]
                        wordId_list.append(word_id)
                    except:
                        continue
                # 寻找正采样对 (context(w),w) 加入正采样队列
                for i, wordId_w in enumerate(wordId_list):
                    context_ids = []
                    for j, wordId_u in enumerate(wordId_list[max(i - window_size, 0):i + window_size + 1]):
                        assert wordId_w < self.word_count
                        assert wordId_u < self.word_count
                        if i == j:  # 上下文=中心词 跳过
                            continue
                        elif max(0, i - window_size + 1) <= j <= min(len(wordId_list), i + window_size - 1):
                            context_ids.append(wordId_u)
                    if len(context_ids) == 0:
                        continue
                    self.word_pairs_queue.append((context_ids, wordId_w))
        result_pairs = []  # 返回mini-batch大小的正采样对
        for _ in range(batch_size):
            result_pairs.append(self.word_pairs_queue.popleft())
        return result_pairs

    def get_pairs(self, pos_pairs):
        neg_word_pair = []
        pos_word_pair = []
        for pair in pos_pairs:
            pos_word_pair += zip([pair[0]] * len(self.huffman_pos_path[pair[1]]), self.huffman_pos_path[pair[1]])
            neg_word_pair += zip([pair[0]] * len(self.huffman_neg_path[pair[1]]), self.huffman_neg_path[pair[1]])
        return pos_word_pair, neg_word_pair

    # 估计数据中正采样对数，用于设定batch
    def evaluate_pairs_count(self, window_size):
        return self.word_count_sum * (2 * window_size - 1)  - (self.sentence_count - 1) * (1 + window_size) * window_size


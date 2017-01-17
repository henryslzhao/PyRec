class InputData:
    def __init__(self, train_file, test_file):
        self.uid_hash, self.item_hash = self.__load_uid_item(train_file, test_file)
        self.train_tuple = {uid:{} for uid in self.uid_hash.values()}
        self.test_tuple = {uid:{} for uid in self.uid_hash.values()}
        self.__get_train_tuple(train_file)
        self.__get_test_tuple(test_file)

    def __get_train_tuple(self, train_file):
        train = open(train_file, 'r')
        for line in train.readlines():
            line = line.strip().split()
            r_uid = int(line[0])
            r_item = int(line[1])
            rating = float(line[2])
            uid = self.uid_hash[r_uid]
            item = self.item_hash[r_item]
            try:
                self.train_tuple[uid][item] = rating
            except:
                print 'cannot load train data'
        train.close()

    def __get_test_tuple(self, test_file):
        test = open(test_file, 'r')
        for line in test.readlines():
            line = line.strip().split()
            r_uid = int(line[0])
            r_item = int(line[1])
            rating = float(line[2])
            uid = self.uid_hash[r_uid]
            item = self.item_hash[r_item]
            try:
                self.test_tuple[uid][item] = rating
            except:
                print 'cannot load train data'
        test.close()

    def __load_uid_item(self, train_file, test_file):
        train = open(train_file, 'r')
        test = open(test_file, 'r')
        uid_hash = {}
        item_hash = {}
        for line in train.readlines():
            line = line.split()
            uid = int(line[0])
            item = int(line[1])
            uid_hash[uid] = uid_hash.get(uid, int(len(uid_hash)))
            item_hash[item] = item_hash.get(item, int(len(item_hash)))
        for line in test.readlines():
            line = line.split()
            uid = int(line[0])
            item = int(line[1])
            uid_hash[uid] = uid_hash.get(uid, int(len(uid_hash)))
            item_hash[item] = item_hash.get(item, int(len(item_hash)))
        train.close()
        test.close()
        return uid_hash, item_hash
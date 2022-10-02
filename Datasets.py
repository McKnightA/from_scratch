import pandas as pd
import numpy as np
import os


class Dataset:
    def __init__(self):
        self.train = {}
        self.validation = {}
        self.test = {}
        self.data = {"trn": self.train, "val": self.validation, "tst": self.test}

        self.timeseries = False

        self.rng = np.random.default_rng(42069)

    def get_data(self):
        raise NotImplementedError("the get_data method for this dataset has not yet been implemented.\n"
                                  "go figure out who fucked up")

    def shuffle_data(self):
        raise NotImplementedError("the shuffle_data method for this dataset has not yet been implemented.\n"
                                  "go figure out who fucked up")


class HW5Part1(Dataset):
    def __init__(self):
        super().__init__()
        self.name = "Homework Highs"
        self.timeseries = True
        self.f_set_size = 1

    def get_data(self):
        trn = pd.read_csv("raw_data/first_data/train.csv")
        val = pd.read_csv("raw_data/first_data/validation.csv")
        tst = pd.read_csv("raw_data/first_data/test.csv")

        trn["High"] = (trn["High"] - trn["High"].mean()) / trn["High"].std()
        val["High"] = (val["High"] - val["High"].mean()) / val["High"].std()
        tst["High"] = (tst["High"] - tst["High"].mean()) / tst["High"].std()

        for set_, set__ in zip([trn, val, tst], ["trn", "val", "tst"]):
            doot = np.array(set_["High"])
            self.data[str(set__)]["X"] = [np.expand_dims(doot[:-1], axis=-1)]
            self.data[str(set__)]["Y"] = [np.expand_dims(doot[1:], axis=-1)]

        return self.data

    def shuffle_data(self):
        pass


class Goog(Dataset):
    def __init__(self):
        super().__init__()
        self.name = "Google Highs"
        self.timeseries = True
        self.f_set_size = 1

    def get_data(self):
        raw = pd.read_csv("raw_data/stonk/GOOG.csv")

        highs = raw["High"].to_numpy()

        trn = highs[:int(len(highs)*6/8)]
        val = highs[int(len(highs)*6/8):int(len(highs)*7/8)]
        tst = highs[int(len(highs)*7/8):]

        trn = (trn - np.mean(trn)) / np.std(trn)
        val = (val - np.mean(val)) / np.std(val)
        tst = (tst - np.mean(tst)) / np.std(tst)

        for set_, set__ in zip([trn, val, tst], ["trn", "val", "tst"]):
            self.data[str(set__)]["X"] = [np.expand_dims(set_[:-1], axis=-1)]
            self.data[str(set__)]["Y"] = [np.expand_dims(set_[1:], axis=-1)]

        return self.data


class GoogCat(Dataset):
    def __init__(self):
        super().__init__()
        self.name = "Categorical (up/down) Google Highs"
        self.timeseries = True
        self.f_set_size = 1
        self.classes = 2

    def get_data(self):
        raw = pd.read_csv("raw_data/stonk/GOOG.csv")

        highs = raw["High"].to_numpy()

        trn = highs[:int(len(highs)*6/8)]
        val = highs[int(len(highs)*6/8):int(len(highs)*7/8)]
        tst = highs[int(len(highs)*7/8):]

        trn = (trn - np.mean(trn)) / np.std(trn)
        val = (val - np.mean(val)) / np.std(val)
        tst = (tst - np.mean(tst)) / np.std(tst)

        for set_, set__ in zip([trn, val, tst], ["trn", "val", "tst"]):
            dat = np.expand_dims(set_[:-1], axis=-1)
            self.data[str(set__)]["X"] = [dat]

            lab = np.clip(dat < np.expand_dims(set_[1:], axis=-1), .1, .9)
            self.data[str(set__)]["Y"] = [lab]

        return self.data


class Hashrate(Dataset):
    def __init__(self):
        super().__init__()
        self.name = "BTC Hashrate"
        self.timeseries = True
        self.f_set_size = 1

    def get_data(self):
        raw = pd.read_csv("raw_data/ghost/hash-rate-btc-24h.csv")

        highs = raw["value"].to_numpy()

        trn = highs[:int(len(highs)*6/8)]
        val = highs[int(len(highs)*6/8):int(len(highs)*7/8)]
        tst = highs[int(len(highs)*7/8):]

        trn = (trn - np.mean(trn)) / np.std(trn)
        val = (val - np.mean(val)) / np.std(val)
        tst = (tst - np.mean(tst)) / np.std(tst)

        for set_, set__ in zip([trn, val, tst], ["trn", "val", "tst"]):
            self.data[str(set__)]["X"] = [np.expand_dims(set_[:-1], axis=-1)]
            self.data[str(set__)]["Y"] = [np.expand_dims(set_[1:], axis=-1)]

        return self.data


class ActiveAdults(Dataset):
    def __init__(self):
        super().__init__()
        self.name = "Activity recognition with healthy older people using a batteryless wearable sensor"
        self.timeseries = True
        self.f_set_size = 3
        self.classes = 4

    def get_data(self):
        thing = list(os.walk("raw_data/Datasets_Healthy_Older_People"))

        raw = []
        for i in range(len(thing)):
            if i != 0:
                for j in list(thing[i])[2]:
                    if "READ" not in j and i == 1:
                        raw.append(
                            np.genfromtxt("raw_data/Datasets_Healthy_Older_People/S1_Dataset/" + j, delimiter=','))

                    if "READ" not in j and i == 2:
                        raw.append(
                            np.genfromtxt("raw_data/Datasets_Healthy_Older_People/S2_Dataset/" + j, delimiter=','))

        data = []
        label = []
        for seq in raw:
            d = []
            l = []
            for i in range(len(seq)):
                d.append(seq[i][1:4])
                l.append(int(seq[i][-1] - 1))  # the labels are 1 indexed so i'm shifting them to 0 index

            data.append(np.array(d))
            label.append(np.eye(self.classes)[l])

        trn = (data[:int(len(data) * 6 / 8)], label[:int(len(label) * 6 / 8)])
        val = (data[int(len(data) * 6 / 8):int(len(data) * 7 / 8)], label[int(len(label) * 6 / 8):int(len(label) * 7 / 8)])
        tst = (data[int(len(data) * 7 / 8):], label[int(len(label) * 7 / 8):])

        for set_, set__ in zip([trn, val, tst], ["trn", "val", "tst"]):
            self.data[str(set__)]["X"] = set_[0]
            self.data[str(set__)]["Y"] = set_[1]

        return self.data


class MNIST(Dataset):
    def __init__(self):
        super().__init__()
        self.name = "hand drawn digits"
        self.timeseries = False
        self.f_set_size = 784

    def get_data(self):
        xtr = np.load("raw_data/mnist/train_x.npy")
        ytr = np.load("raw_data/mnist/train_y.npy")
        xts = np.load("raw_data/mnist/test_x.npy")
        yts = np.load("raw_data/mnist/test_y.npy")

        # flattening since I don't have convolutional layers
        xtr = np.reshape(xtr, (len(xtr), self.f_set_size))
        xts = np.reshape(xts, (len(xts), self.f_set_size))

        # normalize

        # create validation set
        shuff = np.arange(len(xtr))
        self.rng.shuffle(shuff)
        xtr = xtr[shuff]
        ytr = ytr[shuff]
        xva = xtr[50000:]
        yva = ytr[50000:]
        xtr = xtr[:50000]
        ytr = ytr[:50000]

        # store x y pairs in self.data
        self.data["trn"]["X"] = xtr
        self.data["trn"]["Y"] = ytr
        self.data["val"]["X"] = xva
        self.data["val"]["Y"] = yva
        self.data["tst"]["X"] = xts
        self.data["tst"]["Y"] = yts

        return self.data
